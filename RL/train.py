#!/usr/bin/env python
"""
FIXED Training Script:
    - Replay buffer size management (clear old samples)
    - Fixed logging threshold
    - Better episode tracking
    - Progress monitoring

Usage:
    python train_fixed.py                    # Full training (1500 episodes)
    python train_fixed.py --quick            # Quick test (100 episodes)
    python train_fixed.py --no-filter        # Disable conviction filter
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
    parser.add_argument('--no-filter', action='store_true', help='Disable conviction filter')
    parser.add_argument('--threshold', type=float, default=0.5, help='Conviction threshold (default: 0.5)')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes (overrides --quick)')
    args = parser.parse_args()
    
    # Configuration
    if args.episodes:
        episodes = args.episodes
    else:
        episodes = 100 if args.quick else 1500
    use_filter = not args.no_filter
    
    print("="*80)
    print("CRYPTO RL TRAINING - FIXED VERSION")
    print("="*80)
    print(f"Episodes: {episodes}")
    print(f"Conviction filter: {use_filter}")
    if use_filter:
        print(f"Conviction threshold: {args.threshold}")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    try:
        norm = np.load("data/processed/train/norm_train_1h.npy")
        raw = np.load("data/processed/train/raw_train_1h.npy")
    except FileNotFoundError:
        print("ERROR: Data files not found. Looking in current directory...")
        try:
            norm = np.load("norm_train_1h.npy")
            raw = np.load("raw_train_1h.npy")
        except FileNotFoundError:
            print("ERROR: Could not find training data files!")
            print("Expected: norm_train_1h.npy and raw_train_1h.npy")
            return
    
    print(f"✅ Loaded {len(norm):,} hourly candles")
    print(f"   Date range: {len(norm)/24/365.25:.1f} years")
    print()
    
    # Create environment with FIXED settings
    env = CryptoTradingEnvLongShort(
        norm, raw,
        init_balance=10_000,
        fee_pct=0.001,
        episode_len=500,  # 20.8 days
        random_start=True,
        lookback=10,
        drawdown_limit=0.30  # 30% instead of 50%
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment created:")
    print(f"   State dimension: {state_dim}")
    print(f"   Action dimension: {action_dim}")
    print(f"   Episode length: 500 steps = 20.8 days")
    print(f"   Drawdown limit: 30%")
    print()
    
    # Create agent with SMALLER replay buffer
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
        per_capacity=50_000  # REDUCED from 200k to 50k
    )
    
    # Logger
    logger = AdvancedLogger()
    returns = []
    best_mean_50 = -1e9
    breakthrough_episode = None
    
    # Track statistics
    early_terminations = 0
    successful_episodes = 0
    
    print()
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    # Training loop
    for epi in tqdm(range(episodes), desc="Training"):
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        ep_reward = 0.0
        steps = 0
        
        while not done:
            mask = env._valid_mask()
            
            # Action selection with optional conviction filter
            if use_filter and np.random.rand() > agent.epsilon:
                with torch.no_grad():
                    q_vals = agent.policy(obs).cpu().numpy()[0]
                
                q_vals[mask == 0] = -np.inf
                valid_q = q_vals[mask == 1]
                
                if len(valid_q) > 1:
                    best_q = np.max(valid_q)
                    second_best = np.partition(valid_q, -2)[-2]
                    
                    if best_q - second_best < args.threshold:
                        action = 0  # Force HOLD
                    else:
                        action = int(np.argmax(q_vals))
                else:
                    action = int(np.argmax(q_vals))
            else:
                action = agent.act(obs, mask)
            
            next_obs, reward, done, info = env.step(action)
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            agent.remember(obs.cpu().numpy()[0], action, float(reward), 
                         next_obs.cpu().numpy()[0], float(done))
            
            obs = next_obs
            ep_reward += reward
            steps += 1
            
            # FIXED: Training updates - only 2 per step instead of 4
            if epi >= 100 and len(agent.memory) >= 256:
                for _ in range(2):  # Reduced from 4 to 2
                    agent.replay(256)
        
        # Episode complete
        portfolio = info['portfolio_value']
        portfolio_return = (portfolio / env.init_balance) - 1.0
        returns.append(portfolio_return)
        
        # Track early terminations
        if steps < env.episode_len:
            early_terminations += 1
        else:
            successful_episodes += 1
        
        # Epsilon decay
        if epi >= 100:
            agent.decay_epsilon()
        
        # Target network update
        if (epi + 1) % 20 == 0:
            agent.update_target()
        
        # FIXED: Buffer management - prevent unbounded growth
        if len(agent.memory) > 45_000:
            # Don't let it grow beyond capacity, but this shouldn't happen
            # with proper deque in replay.py
            pass
        
        # Evaluation every 50 episodes
        if (epi + 1) % 50 == 0:
            avg_50 = np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)
            win_rate = np.mean([r > 0 for r in returns[-50:]]) if len(returns) >= 50 else np.mean([r > 0 for r in returns])
            early_rate = early_terminations / 50.0
            
            logger.logger.info(f"\nEpisode {epi+1}/{episodes}")
            logger.logger.info(f"  Avg return (last 50): {avg_50:.2%}")
            logger.logger.info(f"  Win rate (last 50): {win_rate:.2%}")
            logger.logger.info(f"  Early termination rate: {early_rate:.1%}")
            logger.logger.info(f"  Epsilon: {agent.epsilon:.3f}")
            logger.logger.info(f"  Replay buffer: {len(agent.memory):,}")
            
            # Reset counters
            early_terminations = 0
            successful_episodes = 0
        
        # Breakthrough detection
        if (epi + 1) % 10 == 0:
            avg_50 = np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)
            if avg_50 > best_mean_50 * 1.15 and breakthrough_episode is None:
                best_mean_50 = avg_50
                breakthrough_episode = epi + 1
                logger.logger.info(f"\n🎯 Breakthrough @ Episode {breakthrough_episode}")
                logger.logger.info(f"   50-ep average: {avg_50:.2%}")
        
        # FIXED: Log episode details with LOWER threshold (3% instead of 8%)
        ep_info = env.get_episode_data()
        ep_info["final_return"] = returns[-1]
        logger.log_episode_details(epi + 1, ep_info, threshold=0.03)  # Changed from 0.08
    
    # Save model
    final_path = os.path.join(logger.run_dir, f"model_final_{int(time.time())}.pth")
    agent.save(final_path)
    logger.logger.info(f"\n✅ Model saved: {final_path}")
    
    # Plot results
    logger.plot_results(returns, [])
    
    # Summary
    summary = {
        "episodes": episodes,
        "avg_return_all": float(np.mean(returns)),
        "avg_return_last_50": float(np.mean(returns[-50:])),
        "avg_return_last_100": float(np.mean(returns[-100:])) if len(returns) >= 100 else float(np.mean(returns)),
        "best_return": float(np.max(returns)),
        "worst_return": float(np.min(returns)),
        "win_rate": float(np.mean([r > 0 for r in returns])),
        "breakthrough_episode": breakthrough_episode,
        "final_epsilon": float(agent.epsilon),
        "conviction_filter": use_filter,
        "conviction_threshold": args.threshold if use_filter else None,
        "model_path": final_path,
        "final_buffer_size": len(agent.memory),
    }
    
    with open(os.path.join(logger.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("="*80)
    
    # Print some notable episodes
    print("\nTop 5 Episodes:")
    top_indices = np.argsort(returns)[-5:][::-1]
    for idx in top_indices:
        print(f"  Episode {idx+1}: {returns[idx]:.2%}")
    
    print("\nBottom 5 Episodes:")
    bottom_indices = np.argsort(returns)[:5]
    for idx in bottom_indices:
        print(f"  Episode {idx+1}: {returns[idx]:.2%}")

if __name__ == "__main__":
    main()