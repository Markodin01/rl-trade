#!/usr/bin/env python
"""
Tail-Optimized Training Script
================================

Trains RL agent optimized for tail-event capture based on empirical analysis:
- 75.4% temporal clustering (vs 40% expected)
- 15.6% jackpot rate at 100h
- 22.9% jackpot rate at 200h (OPTIMAL)
- 29.6% average jackpot return
- 2.03 gain-to-pain ratio

Usage:
    python train_tail.py --episodes 1500                    # Full training
    python train_tail.py --episodes 100 --quick             # Quick test
    python train_tail.py --episodes 1500 --data-path ../data
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


def load_preprocessed_data(data_dir="data"):
    """Load preprocessed .npy files"""
    train_dir = os.path.join(data_dir, "train")

    # Try multiple locations
    search_paths = [
        train_dir,
        data_dir,
        ".",
    ]

    for path in search_paths:
        norm_path = os.path.join(path, "norm_train_1h.npy")
        raw_path = os.path.join(path, "raw_train_1h.npy")

        if os.path.exists(norm_path) and os.path.exists(raw_path):
            print(f"✅ Loading data from: {path}")
            norm = np.load(norm_path)
            raw = np.load(raw_path)
            return norm, raw

    raise FileNotFoundError(
        "Could not find norm_train_1h.npy and raw_train_1h.npy\n"
        f"Searched: {search_paths}\n"
        "Run preprocess.py first to generate training data"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Train RL agent with tail-optimized rewards'
    )
    parser.add_argument('--episodes', type=int, default=1500,
                       help='Number of training episodes')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (uses --episodes or 100)')
    parser.add_argument('--data-path', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--episode-len', type=int, default=200,
                       help='Episode length in hours (default: 200, optimal from analysis)')
    parser.add_argument('--save-every', type=int, default=100,
                       help='Save checkpoint every N episodes')

    args = parser.parse_args()

    # Quick mode override
    if args.quick and args.episodes > 200:
        args.episodes = 100
        print("⚡ Quick mode: reducing to 100 episodes")

    print("="*80)
    print("🎯 TAIL-OPTIMIZED RL TRAINING")
    print("="*80)
    print(f"Episodes:      {args.episodes}")
    print(f"Episode length: {args.episode_len} hours (~{args.episode_len/24:.1f} days)")
    print(f"Reward mode:    tail (jackpot hunting)")
    print(f"Data path:     {args.data_path}")
    print("="*80)
    print()

    # Load data
    print("Loading preprocessed data...")
    try:
        norm, raw = load_preprocessed_data(args.data_path)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return

    print(f"✅ Loaded {len(norm):,} hourly candles")
    print(f"   Features: {norm.shape[1]} normalized")
    print(f"   OHLC: {raw.shape[1]} columns")
    print()

    # Create environment with TAIL-OPTIMIZED settings
    env = CryptoTradingEnvLongShort(
        norm, raw,
        init_balance=10_000,
        fee_pct=0.001,
        episode_len=args.episode_len,      # 200h optimal from analysis
        random_start=True,
        lookback=10,
        drawdown_limit=0.40,               # Allow jackpot hunting
        reward_mode="tail"                  # ⭐ TAIL-OPTIMIZED REWARDS
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment created:")
    print(f"   State dimension:  {state_dim}")
    print(f"   Action dimension: {action_dim}")
    print(f"   Episode length:   {args.episode_len} hours")
    print(f"   Drawdown limit:   {env.drawdown_limit*100:.0f}%")
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
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        per_alpha=0.6,
        per_capacity=50_000
    )

    # Logger
    logger = AdvancedLogger()
    returns = []
    jackpots = []  # Track jackpots (>15%)
    best_mean_50 = -1e9
    breakthrough_episode = None
    early_terminations = 0

    print()
    print("="*80)
    print("STARTING TRAINING - TAIL MODE")
    print("="*80)
    print()

    # Training loop
    start_time = time.time()

    for epi in tqdm(range(args.episodes), desc="Training"):
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        steps = 0
        episode_jackpot = False

        while not done:
            mask = env._valid_mask()
            action = agent.act(obs, mask)

            next_obs, reward, done, info = env.step(action)
            next_mask = env._valid_mask()
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            agent.remember(obs.cpu().numpy()[0], action, float(reward),
                         next_obs.cpu().numpy()[0], float(done), next_mask)

            obs = next_obs
            steps += 1

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

        # Track jackpots (>15% return)
        if portfolio_return > 0.15:
            jackpots.append(epi)
            episode_jackpot = True
            logger.logger.info(f"🎰 JACKPOT @ Episode {epi+1}: {portfolio_return*100:.2f}%")

        # Track early terminations
        if steps < env.episode_len:
            early_terminations += 1

        # Epsilon decay (start early!)
        # For quick test: decay from start
        # For full run: decay after warmup
        if epi >= warmup:
            agent.decay_epsilon()

        # Target network update
        if (epi + 1) % 20 == 0:
            agent.update_target()

        # Log episode
        ep_info = env.get_episode_data()
        logger.store_episode_summary(ep_info)
        logger.log_episode_details(epi + 1, ep_info, threshold=0.03)

        # Checkpoint saves
        if (epi + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(logger.run_dir, f"model_ep_{epi+1}.pth")
            agent.save(checkpoint_path)
            logger.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

        # Evaluation every 50 episodes
        if (epi + 1) % 50 == 0:
            avg_50 = np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)
            win_rate = np.mean([r > 0 for r in returns[-50:]]) if len(returns) >= 50 else np.mean([r > 0 for r in returns])
            early_rate = early_terminations / 50.0

            # Jackpot metrics
            recent_jackpots = [j for j in jackpots if j >= epi - 49]
            jackpot_rate = len(recent_jackpots) / 50.0

            # Recent episode metrics
            recent_eps = logger.all_episodes[-50:]
            avg_trades = np.mean([ep.get('total_trades', 0) for ep in recent_eps])
            avg_sharpe = np.mean([ep.get('sharpe_ratio', 0) for ep in recent_eps])

            logger.logger.info(f"\n{'='*60}")
            logger.logger.info(f"Episode {epi+1}/{args.episodes}")
            logger.logger.info(f"{'='*60}")
            logger.logger.info(f"  Avg return (last 50): {avg_50:.2%}")
            logger.logger.info(f"  Win rate (last 50):   {win_rate:.2%}")
            logger.logger.info(f"  Jackpot rate (last 50): {jackpot_rate:.1%} ({len(recent_jackpots)}/50)")
            logger.logger.info(f"  Avg trades per ep:    {avg_trades:.1f}")
            logger.logger.info(f"  Avg Sharpe:           {avg_sharpe:.3f}")
            logger.logger.info(f"  Early termination:    {early_rate:.1%}")
            logger.logger.info(f"  Epsilon:              {agent.epsilon:.3f}")
            logger.logger.info(f"  Buffer size:          {len(agent.memory):,}")
            logger.logger.info(f"  Time elapsed:         {(time.time() - start_time)/60:.1f} min")
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
    print("TRAINING COMPLETE - Generating Analysis")
    print("="*80)

    # Save final model
    final_path = os.path.join(logger.run_dir, f"model_final_{int(time.time())}.pth")
    agent.save(final_path)
    logger.logger.info(f"\n✅ Final model saved: {final_path}")

    # Generate plots
    logger.logger.info("Generating training plots...")
    logger.plot_results(returns, [])

    # Analyze best episodes
    logger.logger.info("Analyzing top performing episodes...")
    returns_array = np.array(returns)
    best_indices = np.argsort(returns_array)[-10:][::-1]
    worst_indices = np.argsort(returns_array)[:5]

    compare_episodes = list(best_indices + 1) + list(worst_indices + 1)
    logger.plot_episode_comparison(compare_episodes)

    # Generate final report
    logger.generate_final_report(returns)

    # Summary
    training_time = (time.time() - start_time) / 60

    summary = {
        "episodes": args.episodes,
        "episode_length_hours": args.episode_len,
        "reward_mode": "tail",
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
        "breakthrough_episode": breakthrough_episode,
        "final_epsilon": float(agent.epsilon),
        "model_path": final_path,
        "avg_trades_per_episode": float(np.mean([ep.get('total_trades', 0) for ep in logger.all_episodes])),
        "avg_sharpe_per_episode": float(np.mean([ep.get('sharpe_ratio', 0) for ep in logger.all_episodes])),
    }

    with open(os.path.join(logger.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"  Training time:        {training_time:.1f} minutes")
    print(f"  Episodes completed:   {args.episodes}")
    print(f"  Avg return (all):     {summary['avg_return_all']*100:.3f}%")
    print(f"  Avg return (last 50): {summary['avg_return_last_50']*100:.3f}%")
    print(f"  Best return:          {summary['best_return']*100:.3f}%")
    print(f"  Worst return:         {summary['worst_return']*100:.3f}%")
    print(f"  Win rate:             {summary['win_rate']*100:.1f}%")
    print(f"  Sharpe ratio:         {summary['sharpe_ratio']:.3f}")
    print(f"  Total jackpots:       {summary['total_jackpots']} ({summary['jackpot_rate']*100:.1f}%)")
    print(f"  Avg trades per ep:    {summary['avg_trades_per_episode']:.1f}")
    print(f"  Breakthrough episode: {breakthrough_episode or 'None'}")
    print("="*80)

    # Jackpot analysis
    if len(jackpots) > 0:
        print("\n" + "="*80)
        print("JACKPOT ANALYSIS (>15% returns)")
        print("="*80)
        jackpot_returns = [returns[j] for j in jackpots]
        print(f"  Total jackpots:   {len(jackpots)}")
        print(f"  Jackpot rate:     {len(jackpots)/args.episodes*100:.2f}% of episodes")
        print(f"  Avg jackpot:      {np.mean(jackpot_returns)*100:.2f}%")
        print(f"  Max jackpot:      {np.max(jackpot_returns)*100:.2f}%")
        print(f"  First jackpot:    Episode {jackpots[0]+1}")
        print(f"  Last jackpot:     Episode {jackpots[-1]+1}")

        # Compare to empirical data
        empirical_rate = 0.156  # 15.6% from your analysis at 100h
        empirical_rate_200h = 0.229  # 22.9% from your analysis at 200h
        print(f"\n  Target rate (empirical 200h): {empirical_rate_200h*100:.1f}%")
        if len(jackpots)/args.episodes > empirical_rate_200h * 0.5:
            print(f"  ✅ Agent is capturing jackpots!")
        else:
            print(f"  ⚠️  Agent is under-capturing jackpots (needs more training?)")

    print("\n" + "="*80)
    print(f"Full analysis available in: {logger.run_dir}")
    print("="*80)

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"1. Analyze episodes:")
    print(f"   python analyze_episodes.py {logger.run_dir} --top 10 --plot")
    print(f"\n2. Check for coasting:")
    print(f"   python analyze_episodes.py {logger.run_dir} --coasting")
    print(f"\n3. Test out-of-sample (if you have test data):")
    print(f"   python test_oos.py {final_path}")
    print("="*80)


if __name__ == "__main__":
    main()
