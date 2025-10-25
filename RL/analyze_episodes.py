#!/usr/bin/env python
"""
Episode Analysis Tool - Inspect step-by-step behavior

Usage:
    python analyze_episodes.py training_logs/run_YYYYMMDD_HHMMSS --episode 42
    python analyze_episodes.py training_logs/run_YYYYMMDD_HHMMSS --top 5
    python analyze_episodes.py training_logs/run_YYYYMMDD_HHMMSS --coasting
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def load_episode(run_dir: str, episode_num: int) -> pd.DataFrame:
    """Load episode step log as DataFrame."""
    csv_path = os.path.join(run_dir, "episodes", f"ep_{episode_num:04d}_steps.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Episode {episode_num} not found at {csv_path}")
    return pd.read_csv(csv_path)

def analyze_episode_behavior(df: pd.DataFrame, episode_num: int):
    """Detailed analysis of episode behavior."""
    print("="*80)
    print(f"EPISODE {episode_num} BEHAVIOR ANALYSIS")
    print("="*80)
    
    # Basic stats
    print(f"\nTotal steps: {len(df)}")
    print(f"Final portfolio: ${df['total_portfolio'].iloc[-1]:.2f}")
    print(f"Return: {(df['total_portfolio'].iloc[-1] / 10000 - 1)*100:+.2f}%")
    print(f"Total reward: {df['step_reward'].sum():.2f}")
    
    # Action distribution
    print("\nAction Distribution:")
    action_counts = df['action'].value_counts()
    for action, count in action_counts.items():
        pct = count / len(df) * 100
        print(f"  {action:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Position analysis
    print("\nPosition Timeline:")
    positions = df[df['position_opened'] | df['position_closed']]
    for idx, row in positions.iterrows():
        print(f"  Step {row['step']:3d}: {row['action']:12s} -> {row['position_after']:7s}", end="")
        if row['pnl'] != 0:
            print(f" | PnL: ${row['pnl']:+8.2f}", end="")
        print()
    
    # Reward analysis
    print(f"\nReward Statistics:")
    print(f"  Mean reward: {df['step_reward'].mean():.4f}")
    print(f"  Max reward: {df['step_reward'].max():.4f}")
    print(f"  Min reward: {df['step_reward'].min():.4f}")
    print(f"  Std reward: {df['step_reward'].std():.4f}")
    
    # Identify "big win" syndrome
    big_rewards = df[df['step_reward'] > 1.0]
    if len(big_rewards) > 0:
        print(f"\n⚠️  BIG REWARD EVENTS: {len(big_rewards)}")
        for idx, row in big_rewards.iterrows():
            print(f"  Step {row['step']:3d}: Reward={row['step_reward']:+.2f} | Action={row['action']}")
        
        # Check behavior after big win
        if len(big_rewards) > 0:
            first_big_win = big_rewards.index[0]
            after_win = df.iloc[first_big_win+1:first_big_win+20]
            if len(after_win) > 0:
                holds_after = (after_win['action'] == 'HOLD').sum()
                print(f"\n  Behavior after first big win:")
                print(f"    HOLDs in next 20 steps: {holds_after}/20 ({holds_after/20*100:.0f}%)")
    
    # Portfolio evolution
    print(f"\nPortfolio Evolution:")
    print(f"  Start: ${df['total_portfolio'].iloc[0]:.2f}")
    print(f"  Peak: ${df['total_portfolio'].max():.2f}")
    print(f"  End: ${df['total_portfolio'].iloc[-1]:.2f}")
    
    # Drawdown
    peak = df['total_portfolio'].cummax()
    drawdown = (peak - df['total_portfolio']) / peak
    max_dd = drawdown.max()
    print(f"  Max Drawdown: {max_dd*100:.2f}%")

def plot_episode(df: pd.DataFrame, episode_num: int, output_path: str = None):
    """Create detailed episode visualization."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # 1. Portfolio value
    ax = axes[0]
    ax.plot(df['step'], df['total_portfolio'], linewidth=2, color='darkblue')
    ax.axhline(10000, color='r', ls='--', alpha=0.5, label='Initial')
    
    # Mark trades
    opens = df[df['position_opened']]
    closes = df[df['position_closed']]
    ax.scatter(opens['step'], opens['total_portfolio'], c='green', s=100, 
              marker='^', label='Open', zorder=5)
    ax.scatter(closes['step'], closes['total_portfolio'], c='red', s=100, 
              marker='v', label='Close', zorder=5)
    
    ax.set_title(f"Episode {episode_num}: Portfolio Value", fontsize=14, fontweight='bold')
    ax.set_xlabel("Step")
    ax.set_ylabel("Portfolio ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Cumulative reward
    ax = axes[1]
    ax.plot(df['step'], df['cumulative_reward'], linewidth=2, color='green')
    ax.axhline(0, color='r', ls='--', alpha=0.5)
    ax.fill_between(df['step'], df['cumulative_reward'], alpha=0.3, color='green')
    ax.set_title("Cumulative Reward", fontsize=14, fontweight='bold')
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    
    # 3. Step rewards
    ax = axes[2]
    colors = ['red' if r < 0 else 'green' for r in df['step_reward']]
    ax.bar(df['step'], df['step_reward'], color=colors, alpha=0.6)
    ax.axhline(0, color='black', ls='-', linewidth=1)
    ax.set_title("Step Rewards", fontsize=14, fontweight='bold')
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    
    # 4. Position and balance composition
    ax = axes[3]
    ax.plot(df['step'], df['liquid'], linewidth=2, label='Liquid ($)', color='blue')
    ax.plot(df['step'], df['position_value'], linewidth=2, label='Position ($)', color='orange')
    ax.set_title("Balance Composition", fontsize=14, fontweight='bold')
    ax.set_xlabel("Step")
    ax.set_ylabel("Value ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Plot saved: {output_path}")
    else:
        plt.show()
    
    plt.close()

def find_coasting_episodes(run_dir: str, max_trades: int = 3):
    """Find episodes with 1-3 trades (coasting behavior)."""
    episodes_dir = os.path.join(run_dir, "episodes")
    if not os.path.exists(episodes_dir):
        print(f"Episodes directory not found: {episodes_dir}")
        return []
    
    coasting = []
    for fname in os.listdir(episodes_dir):
        if fname.endswith("_detailed.json"):
            with open(os.path.join(episodes_dir, fname)) as f:
                data = json.load(f)
                trades = data.get('total_trades', 0)
                if 1 <= trades <= max_trades:
                    ep_num = int(fname.split('_')[1])
                    ret = (data.get('final_portfolio', 10000) / 10000 - 1) * 100
                    coasting.append({
                        'episode': ep_num,
                        'trades': trades,
                        'return': ret,
                        'positive': data.get('positive_trades', 0),
                    })
    
    return sorted(coasting, key=lambda x: x['return'], reverse=True)

def main():
    parser = argparse.ArgumentParser(description='Analyze RL episode behavior')
    parser.add_argument('run_dir', help='Path to training run directory')
    parser.add_argument('--episode', type=int, help='Analyze specific episode')
    parser.add_argument('--top', type=int, help='Analyze top N episodes by return')
    parser.add_argument('--bottom', type=int, help='Analyze bottom N episodes by return')
    parser.add_argument('--coasting', action='store_true', 
                       help='Find episodes with 1-3 trades')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.run_dir):
        print(f"ERROR: Run directory not found: {args.run_dir}")
        return
    
    print("="*80)
    print("EPISODE ANALYSIS TOOL")
    print("="*80)
    print(f"Run directory: {args.run_dir}\n")
    
    # Analyze specific episode
    if args.episode:
        df = load_episode(args.run_dir, args.episode)
        analyze_episode_behavior(df, args.episode)
        
        if args.plot:
            output = os.path.join(args.run_dir, "analysis", 
                                f"ep_{args.episode:04d}_detailed.png")
            os.makedirs(os.path.dirname(output), exist_ok=True)
            plot_episode(df, args.episode, output)
    
    # Analyze top N
    elif args.top:
        # Load all episode summaries
        episodes_dir = os.path.join(args.run_dir, "episodes")
        all_episodes = []
        
        for fname in os.listdir(episodes_dir):
            if fname.endswith("_detailed.json"):
                with open(os.path.join(episodes_dir, fname)) as f:
                    data = json.load(f)
                    ep_num = int(fname.split('_')[1])
                    ret = (data.get('final_portfolio', 10000) / 10000 - 1)
                    all_episodes.append((ep_num, ret))
        
        all_episodes.sort(key=lambda x: x[1], reverse=True)
        top_eps = all_episodes[:args.top]
        
        print(f"TOP {args.top} EPISODES:\n")
        for rank, (ep_num, ret) in enumerate(top_eps, 1):
            print(f"{rank}. Episode {ep_num}: {ret*100:+.2f}%")
        
        print("\nDetailed analysis:")
        for ep_num, ret in top_eps:
            df = load_episode(args.run_dir, ep_num)
            analyze_episode_behavior(df, ep_num)
            print()
    
    # Analyze bottom N
    elif args.bottom:
        episodes_dir = os.path.join(args.run_dir, "episodes")
        all_episodes = []
        
        for fname in os.listdir(episodes_dir):
            if fname.endswith("_detailed.json"):
                with open(os.path.join(episodes_dir, fname)) as f:
                    data = json.load(f)
                    ep_num = int(fname.split('_')[1])
                    ret = (data.get('final_portfolio', 10000) / 10000 - 1)
                    all_episodes.append((ep_num, ret))
        
        all_episodes.sort(key=lambda x: x[1])
        bottom_eps = all_episodes[:args.bottom]
        
        print(f"BOTTOM {args.bottom} EPISODES:\n")
        for rank, (ep_num, ret) in enumerate(bottom_eps, 1):
            print(f"{rank}. Episode {ep_num}: {ret*100:+.2f}%")
        
        print("\nDetailed analysis:")
        for ep_num, ret in bottom_eps:
            df = load_episode(args.run_dir, ep_num)
            analyze_episode_behavior(df, ep_num)
            print()
    
    # Find coasting episodes
    elif args.coasting:
        coasting = find_coasting_episodes(args.run_dir)
        
        print(f"EPISODES WITH 1-3 TRADES (COASTING): {len(coasting)}\n")
        
        if coasting:
            print("Top coasters by return:")
            for item in coasting[:10]:
                print(f"  Episode {item['episode']:4d}: {item['trades']} trades | "
                      f"{item['return']:+6.2f}% | {item['positive']}/{item['trades']} wins")
            
            print("\nAverage return from coasters:")
            avg_ret = np.mean([c['return'] for c in coasting])
            win_rate = np.mean([c['positive']/max(c['trades'],1) for c in coasting])
            print(f"  {avg_ret:+.2f}% with {win_rate*100:.1f}% win rate")
            
            # Analyze one example
            if coasting:
                example = coasting[0]
                print(f"\nDetailed analysis of best coaster (Episode {example['episode']}):")
                df = load_episode(args.run_dir, example['episode'])
                analyze_episode_behavior(df, example['episode'])

if __name__ == "__main__":
    main()