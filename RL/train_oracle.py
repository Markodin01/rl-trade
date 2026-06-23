#!/usr/bin/env python
"""
Oracle PF Training - Upper Bound Experiment

Trains RL agent with PERFECT particle filter (using lookahead).
This establishes the upper bound: if RL can't succeed with perfect
information, the strategy itself is flawed.

Usage:
    python train_oracle.py --episodes 1500
    python train_oracle.py --episodes 100 --quick  # Quick test
"""

import argparse
import os
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

from env import CryptoTradingEnvLongShort
from agent import DuelingDQNAgent
from utils import AdvancedLogger


def load_data_with_oracle(data_dir="data/train"):
    """Load data + oracle PF features"""
    print("Loading data with Oracle PF...")

    norm_path = os.path.join(data_dir, "norm_train_1h.npy")
    raw_path = os.path.join(data_dir, "raw_train_1h.npy")
    oracle_path = os.path.join(data_dir, "oracle_pf_train_1h.npy")

    if not os.path.exists(oracle_path):
        raise FileNotFoundError(
            f"Oracle PF not found: {oracle_path}\n"
            "Run: python preprocess_with_oracle.py"
        )

    norm = np.load(norm_path)
    raw = np.load(raw_path)
    oracle = np.load(oracle_path)

    print(f"✅ Norm: {norm.shape}")
    print(f"✅ Raw: {raw.shape}")
    print(f"✅ Oracle PF: {oracle.shape}")
    print()

    return norm, raw, oracle


def main():
    parser = argparse.ArgumentParser(description='Train with Oracle PF (perfect information)')
    parser.add_argument('--episodes', type=int, default=1500)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--episode-len', type=int, default=200)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--discrete-sizing', action='store_true',
                        help='Option 2: agent picks bet size via action (7-action space)')
    args = parser.parse_args()

    if args.quick and args.episodes > 200:
        args.episodes = 100

    print("="*80)
    print("🔮 ORACLE PF TRAINING - Perfect State Estimation")
    print("="*80)
    print(f"Episodes:       {args.episodes}")
    print(f"Episode length: {args.episode_len} hours")
    print(f"Reward mode:    tail (jackpot hunting)")
    print(f"Oracle PF:      ENABLED (8 perfect features)")
    print("="*80)
    print()

    # Load data
    try:
        norm, raw, oracle = load_data_with_oracle()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return

    print(f"Loaded {len(norm):,} candles with Oracle PF")
    print(f"Oracle PF features:")
    print(f"  [0] will_jackpot_24h")
    print(f"  [1] will_jackpot_100h")
    print(f"  [2] will_jackpot_200h")
    print(f"  [3] time_to_next_tail")
    print(f"  [4] in_cascade")
    print(f"  [5] optimal_entry_score")
    print(f"  [6] regime")
    print(f"  [7] criticality")
    print()

    # Create environment WITH oracle PF
    env = CryptoTradingEnvLongShort(
        norm, raw,
        oracle_pf=oracle,  # ⭐ ORACLE ENABLED
        init_balance=10_000,
        fee_pct=0.001,
        episode_len=args.episode_len,
        random_start=True,
        lookback=10,
        drawdown_limit=0.40,
        reward_mode="tail",
        discrete_sizing=args.discrete_sizing,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment:")
    print(f"   State dimension:  {state_dim} (70 base + 8 oracle PF)")
    print(f"   Action dimension: {action_dim}")
    print(f"   Episode length:   {args.episode_len} hours")
    print(f"   Reward mode:      {env.reward_mode.upper()}")
    print()

    # Create agent
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = DuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=128,
        lr=5e-4,
        gamma=0.997,  # VC: high discount so far-future tail payoffs aren't discounted
                      # to ~0 (0.99^150=0.22 vs 0.997^150=0.64) -> enables holding for
                      # 100-200h moves instead of scalping immediate wiggles.
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        per_alpha=0.6,
        per_capacity=50_000
    )

    # Logger
    logger = AdvancedLogger()
    returns = []
    jackpots = []
    oracle_jackpots_captured = []  # Track when oracle said "jackpot" and we captured
    best_mean_50 = -1e9
    breakthrough_episode = None
    early_terminations = 0

    print()
    print("="*80)
    print("STARTING ORACLE TRAINING")
    print("="*80)
    print()

    start_time = time.time()

    # Per-STEP epsilon schedule. FIX: the old per-episode 0.995 decay needed ~600
    # episodes just to reach the 0.05 floor, so a 200-300 episode run stayed 38-100%
    # random the whole time -> forced ~45 trades/episode and pure fee bleed. Decay
    # linearly over 70% of the total step budget so exploitation actually happens.
    total_steps = max(1, args.episodes * args.episode_len)
    eps_decay_steps = int(0.7 * total_steps)
    global_step = 0
    agent.epsilon = 1.0

    # Training loop
    for epi in tqdm(range(args.episodes), desc="Training"):
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        steps = 0
        episode_jackpot = False

        # Track oracle predictions for this episode
        oracle_predicted_jackpot = False

        while not done:
            mask = env._valid_mask()
            action = agent.act(obs, mask)

            # Check oracle prediction (before step). Feature 1 is jp_signed_100h
            # in [-1,1]; |value|>=1 means a real >15% directional move is coming.
            if env.oracle_pf is not None:
                oracle_features = env.oracle_pf[env.t, :]
                if abs(oracle_features[1]) >= 0.99:
                    oracle_predicted_jackpot = True

            prev_obs_np = obs.cpu().numpy()[0]
            next_obs, reward, done, info = env.step(action)
            next_mask = env._valid_mask()  # valid actions for the resulting state
            next_obs_np = np.asarray(next_obs, dtype=np.float32)

            agent.remember(prev_obs_np, action, float(reward),
                           next_obs_np, float(done), next_mask)

            obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            steps += 1

            # Per-step epsilon anneal
            global_step += 1
            frac = min(1.0, global_step / eps_decay_steps)
            agent.epsilon = max(agent.epsilon_min,
                                1.0 - frac * (1.0 - agent.epsilon_min))

            # Training updates (start early!)
            # For quick test: start immediately
            # For full run: start after 20 episodes
            warmup = 20 if args.episodes >= 1000 else 5
            if epi >= warmup and len(agent.memory) >= 256:
                for _ in range(2):
                    agent.replay(256)

        # Episode complete
        portfolio = info['portfolio_value']
        portfolio_return = (portfolio / env.init_balance) - 1.0
        returns.append(portfolio_return)

        # Track jackpots
        if portfolio_return > 0.15:
            jackpots.append(epi)
            episode_jackpot = True
            logger.logger.info(f"🎰 JACKPOT @ Episode {epi+1}: {portfolio_return*100:.2f}%")

            # Did oracle predict it?
            if oracle_predicted_jackpot:
                oracle_jackpots_captured.append(epi)
                logger.logger.info(f"   ✅ Oracle predicted this jackpot!")

        # Track early terminations
        if steps < env.episode_len:
            early_terminations += 1

        # (epsilon now annealed per-step inside the rollout loop)

        # Target network update
        if (epi + 1) % 20 == 0:
            agent.update_target()

        # Log episode
        ep_info = env.get_episode_data()
        logger.store_episode_summary(ep_info)
        logger.log_episode_details(epi + 1, ep_info, threshold=0.03)

        # Checkpoint saves
        if (epi + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(logger.run_dir, f"model_oracle_ep_{epi+1}.pth")
            agent.save(checkpoint_path)
            logger.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

        # Evaluation every 50 episodes
        if (epi + 1) % 50 == 0:
            recent_ret = returns[-50:] if len(returns) >= 50 else returns
            avg_50 = np.mean(recent_ret)
            median_50 = np.median(recent_ret)
            # compounded total return over the recent window (the metric that matters
            # for a fat-tail strategy -- win rate is EXPECTED to fall as it concentrates)
            compounded_50 = float(np.prod([1 + r for r in recent_ret]) - 1.0)
            win_rate = np.mean([r > 0 for r in recent_ret])
            early_rate = early_terminations / 50.0

            # Jackpot metrics
            recent_jackpots = [j for j in jackpots if j >= epi - 49]
            jackpot_rate = len(recent_jackpots) / 50.0

            # Oracle capture rate
            recent_oracle_captures = [j for j in oracle_jackpots_captured if j >= epi - 49]
            oracle_capture_rate = len(recent_oracle_captures) / max(len(recent_jackpots), 1)

            # Recent episode metrics
            recent_eps = logger.all_episodes[-50:]
            avg_trades = np.mean([ep.get('total_trades', 0) for ep in recent_eps])
            avg_sharpe = np.mean([ep.get('sharpe_ratio', 0) for ep in recent_eps])

            logger.logger.info(f"\n{'='*60}")
            logger.logger.info(f"Episode {epi+1}/{args.episodes}")
            logger.logger.info(f"{'='*60}")
            logger.logger.info(f"  Avg return (last 50):    {avg_50:.2%}  (median {median_50:.2%})")
            logger.logger.info(f"  Compounded (last 50):    {compounded_50:.1%}  <-- key metric")
            logger.logger.info(f"  Win rate (last 50):      {win_rate:.2%}  (expected to fall w/ tail-hunting)")
            logger.logger.info(f"  Jackpot rate (last 50):  {jackpot_rate:.1%} ({len(recent_jackpots)}/50)")
            logger.logger.info(f"  Oracle capture rate:     {oracle_capture_rate:.1%} of jackpots")
            logger.logger.info(f"  Avg trades per ep:       {avg_trades:.1f}")
            logger.logger.info(f"  Avg Sharpe:              {avg_sharpe:.3f}")
            logger.logger.info(f"  Early termination:       {early_rate:.1%}")
            logger.logger.info(f"  Epsilon:                 {agent.epsilon:.3f}")
            logger.logger.info(f"  Time elapsed:            {(time.time() - start_time)/60:.1f} min")
            logger.logger.info(f"{'='*60}")

            early_terminations = 0

        # Breakthrough detection
        if (epi + 1) % 10 == 0:
            avg_50 = np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)
            if avg_50 > best_mean_50 * 1.15 and breakthrough_episode is None:
                best_mean_50 = avg_50
                breakthrough_episode = epi + 1
                logger.logger.info(f"\n🎯 BREAKTHROUGH @ Episode {breakthrough_episode}")
                logger.logger.info(f"   50-ep average: {avg_50:.2%}")

    print("\n" + "="*80)
    print("ORACLE TRAINING COMPLETE")
    print("="*80)

    # Save final model
    final_path = os.path.join(logger.run_dir, f"model_oracle_final_{int(time.time())}.pth")
    agent.save(final_path)
    logger.logger.info(f"\n✅ Final model saved: {final_path}")

    # Generate plots
    logger.logger.info("Generating plots...")
    logger.plot_results(returns, [])

    # Analyze best episodes
    returns_array = np.array(returns)
    best_indices = np.argsort(returns_array)[-10:][::-1]
    worst_indices = np.argsort(returns_array)[:5]
    compare_episodes = list(best_indices + 1) + list(worst_indices + 1)
    logger.plot_episode_comparison(compare_episodes)

    logger.generate_final_report(returns)

    # Summary
    training_time = (time.time() - start_time) / 60
    oracle_capture_rate_total = len(oracle_jackpots_captured) / max(len(jackpots), 1)

    summary = {
        "mode": "oracle_pf",
        "episodes": args.episodes,
        "episode_length_hours": args.episode_len,
        "reward_mode": "tail",
        "oracle_pf_enabled": True,
        "training_time_minutes": float(training_time),
        "avg_return_all": float(np.mean(returns)),
        "avg_return_last_50": float(np.mean(returns[-50:])),
        "avg_return_last_100": float(np.mean(returns[-100:])) if len(returns) >= 100 else float(np.mean(returns)),
        "best_return": float(np.max(returns)),
        "worst_return": float(np.min(returns)),
        "win_rate": float(np.mean([r > 0 for r in returns])),
        "sharpe_ratio": float(np.mean(returns) / (np.std(returns) + 1e-8)),
        "total_jackpots": len(jackpots),
        "jackpot_rate": float(len(jackpots) / args.episodes),
        "oracle_jackpots_captured": len(oracle_jackpots_captured),
        "oracle_capture_rate": float(oracle_capture_rate_total),
        "breakthrough_episode": breakthrough_episode,
        "final_epsilon": float(agent.epsilon),
        "model_path": final_path,
    }

    with open(os.path.join(logger.run_dir, "oracle_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("ORACLE TRAINING SUMMARY")
    print("="*80)
    print(f"  Training time:         {training_time:.1f} minutes")
    print(f"  Episodes completed:    {args.episodes}")
    print(f"  Avg return (all):      {summary['avg_return_all']*100:.3f}%")
    print(f"  Avg return (last 50):  {summary['avg_return_last_50']*100:.3f}%")
    print(f"  Win rate:              {summary['win_rate']*100:.1f}%")
    print(f"  Total jackpots:        {summary['total_jackpots']} ({summary['jackpot_rate']*100:.1f}%)")
    print(f"  Oracle capture rate:   {summary['oracle_capture_rate']*100:.1f}%")
    print(f"     (Agent captured {len(oracle_jackpots_captured)}/{len(jackpots)} jackpots)")
    print("="*80)

    # Interpretation
    print("\n" + "="*80)
    print("ORACLE EXPERIMENT RESULTS")
    print("="*80)

    if summary['avg_return_last_50'] > 0.03:
        print("✅ SUCCESS: RL can exploit tails with perfect information!")
        print(f"   Upper bound established: {summary['avg_return_last_50']*100:.2f}% avg return")
        print("\n   Next step: Build real PF to approximate oracle")
    elif summary['avg_return_last_50'] > 0:
        print("⚠️  MARGINAL: RL shows some learning but not strong")
        print("   May need more training or better reward tuning")
    else:
        print("❌ FAILURE: RL cannot exploit even with perfect information")
        print("   Strategy may be flawed or need different approach")

    print("="*80)
    print(f"\nFull logs: {logger.run_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
