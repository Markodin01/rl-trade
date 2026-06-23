#!/usr/bin/env python
"""
IMPROVED Training Script:
    - Sharpe-based reward mode (fixes "1-trade coasting")
    - Step-by-step logging
    - Better episode analysis
    - Comparative visualizations

Usage:
    python train_improved.py --reward sharpe     # Recommended: Sharpe-based
    python train_improved.py --reward pnl        # Old approach
    python train_improved.py --reward hybrid     # Combination
    python train_improved.py --quick             # Quick test (100 episodes)
"""

import argparse
import os
import time
import json
import numpy as np
import torch
from tqdm import tqdm

from env import CryptoTradingEnvLongShort
from agent import DuelingDQNAgent
from utils import AdvancedLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test (100 episodes)')
    parser.add_argument('--reward', type=str, default='sharpe', 
                       choices=['sharpe', 'pnl', 'hybrid'],
                       help='Reward mode: sharpe (continuous), pnl (sparse), hybrid')
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--analyze-best', type=int, default=5,
                       help='Number of best/worst episodes to analyze in detail')
    args = parser.parse_args()
    
    # Configuration
    if args.episodes:
        episodes = args.episodes
    else:
        episodes = 100 if args.quick else 1000
    
    print("="*80)
    print("CRYPTO RL TRAINING - IMPROVED VERSION")
    print("="*80)
    print(f"Episodes: {episodes}")
    print(f"Reward mode: {args.reward.upper()}")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    try:
        norm = np.load("data/processed/train/norm_train_1h.npy")
        raw = np.load("data/processed/train/raw_train_1h.npy")
    except FileNotFoundError:
        try:
            norm = np.load("norm_train_1h.npy")
            raw = np.load("raw_train_1h.npy")
        except FileNotFoundError:
            print("ERROR: Could not find training data files!")
            return
    
    print(f"✅ Loaded {len(norm):,} hourly candles")
    print()
    
    # Create environment with IMPROVED settings
    env = CryptoTradingEnvLongShort(
        norm, raw,
        init_balance=10_000,
        fee_pct=0.001,
        episode_len=500,
        random_start=True,
        lookback=10,
        drawdown_limit=0.30,
        reward_mode=args.reward  # NEW!
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment created:")
    print(f"   State dimension: {state_dim}")
    print(f"   Action dimension: {action_dim}")
    print(f"   Reward mode: {args.reward}")
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
    
    # IMPROVED Logger
    logger = AdvancedLogger()
    returns = []
    best_mean_50 = -1e9
    breakthrough_episode = None
    
    early_terminations = 0
    
    print()
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    # Training loop
    for epi in tqdm(range(episodes), desc="Training"):
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        steps = 0
        
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
            
            # Training updates
            if epi >= 100 and len(agent.memory) >= 256:
                for _ in range(2):
                    agent.replay(256)
        
        # Episode complete
        portfolio = info['portfolio_value']
        portfolio_return = (portfolio / env.init_balance) - 1.0
        returns.append(portfolio_return)
        
        # Track early terminations
        if steps < env.episode_len:
            early_terminations += 1
        
        # Epsilon decay
        if epi >= 100:
            agent.decay_epsilon()
        
        # Target network update
        if (epi + 1) % 20 == 0:
            agent.update_target()
        
        # IMPROVED: Log episode with step details
        ep_info = env.get_episode_data()
        logger.store_episode_summary(ep_info)
        logger.log_episode_details(epi + 1, ep_info, threshold=0.03)
        
        # Evaluation every 50 episodes
        if (epi + 1) % 50 == 0:
            avg_50 = np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)
            win_rate = np.mean([r > 0 for r in returns[-50:]]) if len(returns) >= 50 else np.mean([r > 0 for r in returns])
            early_rate = early_terminations / 50.0
            
            # Calculate average trades and Sharpe
            recent_eps = logger.all_episodes[-50:]
            avg_trades = np.mean([ep.get('total_trades', 0) for ep in recent_eps])
            avg_sharpe = np.mean([ep.get('sharpe_ratio', 0) for ep in recent_eps])
            
            logger.logger.info(f"\n{'='*60}")
            logger.logger.info(f"Episode {epi+1}/{episodes}")
            logger.logger.info(f"{'='*60}")
            logger.logger.info(f"  Avg return (last 50): {avg_50:.2%}")
            logger.logger.info(f"  Win rate (last 50): {win_rate:.2%}")
            logger.logger.info(f"  Avg trades per ep: {avg_trades:.1f}")
            logger.logger.info(f"  Avg Sharpe: {avg_sharpe:.3f}")
            logger.logger.info(f"  Early termination rate: {early_rate:.1%}")
            logger.logger.info(f"  Epsilon: {agent.epsilon:.3f}")
            logger.logger.info(f"  Buffer size: {len(agent.memory):,}")
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
    
    # Save model
    final_path = os.path.join(logger.run_dir, f"model_final_{int(time.time())}.pth")
    agent.save(final_path)
    logger.logger.info(f"\n✅ Model saved: {final_path}")
    
    # Generate plots
    logger.logger.info("Generating training plots...")
    logger.plot_results(returns, [])
    
    # IMPROVED: Analyze best and worst episodes
    logger.logger.info("Analyzing top performing episodes...")
    returns_array = np.array(returns)
    best_indices = np.argsort(returns_array)[-args.analyze_best:][::-1]
    worst_indices = np.argsort(returns_array)[:args.analyze_best]
    
    compare_episodes = list(best_indices + 1) + list(worst_indices + 1)
    logger.plot_episode_comparison(compare_episodes)
    
    # Generate final report
    logger.generate_final_report(returns)
    
    # Summary
    summary = {
        "episodes": episodes,
        "reward_mode": args.reward,
        "avg_return_all": float(np.mean(returns)),
        "avg_return_last_50": float(np.mean(returns[-50:])),
        "avg_return_last_100": float(np.mean(returns[-100:])) if len(returns) >= 100 else float(np.mean(returns)),
        "best_return": float(np.max(returns)),
        "worst_return": float(np.min(returns)),
        "win_rate": float(np.mean([r > 0 for r in returns])),
        "sharpe_ratio": float(np.mean(returns) / (np.std(returns) + 1e-8)),
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
    for k, v in summary.items():
        if isinstance(v, float):
            if 'return' in k or 'rate' in k:
                print(f"  {k}: {v*100:.3f}%" if abs(v) < 1 else f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    print("="*80)
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    print("\nTop 5 Episodes:")
    top_indices = np.argsort(returns)[-5:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        ep_data = logger.all_episodes[idx]
        print(f"  {rank}. Episode {idx+1}: {returns[idx]:.2%} return")
        print(f"     Trades: {ep_data.get('total_trades', 0)} | " +
              f"Win Rate: {ep_data.get('positive_trades', 0)/max(ep_data.get('total_trades', 1), 1)*100:.1f}% | " +
              f"Sharpe: {ep_data.get('sharpe_ratio', 0):.3f}")
    
    print("\nWorst 5 Episodes:")
    bottom_indices = np.argsort(returns)[:5]
    for rank, idx in enumerate(bottom_indices, 1):
        ep_data = logger.all_episodes[idx]
        print(f"  {rank}. Episode {idx+1}: {returns[idx]:.2%} return")
        print(f"     Trades: {ep_data.get('total_trades', 0)} | " +
              f"Max DD: {ep_data.get('max_drawdown', 0)*100:.1f}% | " +
              f"Sharpe: {ep_data.get('sharpe_ratio', 0):.3f}")
    
    # Trading behavior analysis
    print("\n" + "="*80)
    print("TRADING BEHAVIOR ANALYSIS")
    print("="*80)
    all_trades = [ep.get('total_trades', 0) for ep in logger.all_episodes]
    print(f"Average trades per episode: {np.mean(all_trades):.1f}")
    print(f"Min trades: {np.min(all_trades)}")
    print(f"Max trades: {np.max(all_trades)}")
    print(f"Median trades: {np.median(all_trades):.1f}")
    
    # Check for "coasting" behavior (episodes with 1-3 trades)
    coasting_episodes = [i for i, t in enumerate(all_trades) if 1 <= t <= 3]
    if coasting_episodes:
        coasting_returns = [returns[i] for i in coasting_episodes]
        print(f"\nEpisodes with 1-3 trades (potential coasting): {len(coasting_episodes)} ({len(coasting_episodes)/len(all_trades)*100:.1f}%)")
        print(f"  Avg return from coasting episodes: {np.mean(coasting_returns):.2%}")
        print(f"  Win rate from coasting episodes: {np.mean([r > 0 for r in coasting_returns])*100:.1f}%")
    
    print("\n" + "="*80)
    print(f"Full analysis available in: {logger.run_dir}")
    print("="*80)

if __name__ == "__main__":
    main()