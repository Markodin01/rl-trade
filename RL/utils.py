#!/usr/bin/env python
"""
IMPROVED Logger - Step-by-step tracking and analysis
"""

import logging
import os
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class AdvancedLogger:
    """
    Enhanced logging with step-by-step tracking.
    
    Directory layout:
        training_logs/
            run_YYYYMMDD_HHMMSS/
                main.log
                evaluation_results.json
                episodes/
                    ep_00xx_detailed.json
                    ep_00xx_steps.csv
                wins/
                    ep_00xx_profit_yy.pct.json
                losses/
                    ep_00xx_loss_yy.pct.json
                analysis/
                    training_results.png
                    episode_comparison.png
    """
    def __init__(self, base_dir="training_logs"):
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"run_{self.run_timestamp}")
        self.wins_dir = os.path.join(self.run_dir, "wins")
        self.losses_dir = os.path.join(self.run_dir, "losses")
        self.episodes_dir = os.path.join(self.run_dir, "episodes")
        self.analysis_dir = os.path.join(self.run_dir, "analysis")
        
        for d in [self.wins_dir, self.losses_dir, self.episodes_dir, self.analysis_dir]:
            os.makedirs(d, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(os.path.join(self.run_dir, "main.log"))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        self.logger.handlers.clear()
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info(f"Run directory created: {self.run_dir}")
        self.eval_results = []
        
        # Track all episodes for analysis
        self.all_episodes = []

    # -----------------------------------------------------------------------
    def log_episode_details(self, episode_num: int, episode_data: dict, 
                           threshold: float = 0.03):
        """
        Log episode with step-by-step detail.
        Now saves both JSON summary AND CSV of steps.
        """
        ret = episode_data.get("final_portfolio", 0) / 10000 - 1  # Assuming 10k start
        
        # Save to wins/losses if significant
        if abs(ret) >= threshold:
            is_win = ret > 0
            target_dir = self.wins_dir if is_win else self.losses_dir
            return_str = f"{abs(ret) * 100:.1f}pct"
            fname = f"ep_{episode_num:04d}_{'profit' if is_win else 'loss'}_{return_str}.json"
            path = os.path.join(target_dir, fname)
            
            with open(path, "w") as f:
                json.dump(self._to_native(episode_data), f, indent=2)
            self.logger.info(f"  Detailed log saved: {fname}")
        
        # ALWAYS save to episodes directory with step log
        ep_json = os.path.join(self.episodes_dir, f"ep_{episode_num:04d}_detailed.json")
        with open(ep_json, "w") as f:
            json.dump(self._to_native(episode_data), f, indent=2)
        
        # Save step log as CSV for easy analysis
        if 'step_log' in episode_data and len(episode_data['step_log']) > 0:
            df = pd.DataFrame(episode_data['step_log'])
            csv_path = os.path.join(self.episodes_dir, f"ep_{episode_num:04d}_steps.csv")
            df.to_csv(csv_path, index=False)
            
            # Also save a human-readable summary
            self._save_episode_summary(episode_num, episode_data, df)
    
    # -----------------------------------------------------------------------
    def _save_episode_summary(self, episode_num: int, episode_data: dict, 
                             step_df: pd.DataFrame):
        """Generate human-readable episode summary."""
        summary_path = os.path.join(self.episodes_dir, 
                                   f"ep_{episode_num:04d}_summary.txt")
        
        with open(summary_path, "w") as f:
            f.write("="*80 + "\n")
            f.write(f"EPISODE {episode_num} SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Overall stats
            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"  Final Portfolio: ${episode_data['final_portfolio']:.2f}\n")
            f.write(f"  Final Balance: ${episode_data['final_balance']:.2f}\n")
            f.write(f"  Position Value: ${episode_data.get('final_position_value', 0):.2f}\n")
            
            ret = (episode_data['final_portfolio'] / 10000 - 1) * 100
            f.write(f"  Return: {ret:+.2f}%\n")
            f.write(f"  Max Drawdown: {episode_data['max_drawdown']*100:.2f}%\n")
            f.write(f"  Sharpe Ratio: {episode_data.get('sharpe_ratio', 0):.3f}\n\n")
            
            # Trade stats
            f.write("TRADING ACTIVITY:\n")
            f.write(f"  Total Trades: {episode_data['total_trades']}\n")
            f.write(f"  Positive Trades: {episode_data['positive_trades']}\n")
            win_rate = episode_data['positive_trades'] / max(episode_data['total_trades'], 1) * 100
            f.write(f"  Win Rate: {win_rate:.1f}%\n")
            f.write(f"  Long Trades: {episode_data['long_trades']} ")
            f.write(f"({episode_data['positive_long_trades']} wins)\n")
            f.write(f"  Short Trades: {episode_data['short_trades']} ")
            f.write(f"({episode_data['positive_short_trades']} wins)\n\n")
            
            # Step analysis
            f.write("STEP ANALYSIS:\n")
            f.write(f"  Total Steps: {len(step_df)}\n")
            
            action_counts = step_df['action'].value_counts()
            f.write("  Action Distribution:\n")
            for action, count in action_counts.items():
                pct = count / len(step_df) * 100
                f.write(f"    {action}: {count} ({pct:.1f}%)\n")
            
            # Reward distribution
            f.write(f"\n  Total Reward: {step_df['step_reward'].sum():.2f}\n")
            f.write(f"  Avg Reward: {step_df['step_reward'].mean():.4f}\n")
            f.write(f"  Max Reward: {step_df['step_reward'].max():.2f}\n")
            f.write(f"  Min Reward: {step_df['step_reward'].min():.2f}\n\n")
            
            # Position periods
            position_changes = step_df[step_df['position_opened'] | step_df['position_closed']]
            if len(position_changes) > 0:
                f.write("POSITION TIMELINE:\n")
                for idx, row in position_changes.iterrows():
                    f.write(f"  Step {row['step']:3d}: {row['action']:12s} ")
                    f.write(f"-> {row['position_after']:7s} ")
                    if row['pnl'] != 0:
                        f.write(f"(PnL: ${row['pnl']:+.2f})")
                    f.write("\n")
            
            f.write("\n" + "="*80 + "\n")
    
    # -----------------------------------------------------------------------
    def _to_native(self, o):
        """Convert numpy types to native Python for JSON serialization."""
        if isinstance(o, dict):
            return {k: self._to_native(v) for k, v in o.items()}
        if isinstance(o, list):
            return [self._to_native(i) for i in o]
        if isinstance(o, (np.integer, np.floating)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    # -----------------------------------------------------------------------
    def log_evaluation(self, episode_num: int, eval_metrics: dict):
        self.eval_results.append({"episode": episode_num, **eval_metrics})
        eval_path = os.path.join(self.run_dir, "evaluation_results.json")
        with open(eval_path, "w") as f:
            json.dump(self.eval_results, f, indent=2)

    # -----------------------------------------------------------------------
    def plot_results(self, returns, eval_history):
        """Enhanced plotting with additional insights."""
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        
        # 1 - Raw returns
        ax = axes[0, 0]
        ax.plot(returns, alpha=0.6, linewidth=0.8)
        ax.axhline(0, color='r', ls='--', alpha=0.3)
        ax.set_title("Portfolio Return per Episode", fontsize=14, fontweight='bold')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return %")
        ax.grid(alpha=0.3)
        
        # 2 - Rolling average
        ax = axes[0, 1]
        window = 50
        rolling = pd.Series(returns).rolling(window).mean()
        ax.plot(rolling, linewidth=2, color='darkblue')
        ax.axhline(0, color='r', ls='--', alpha=0.3)
        ax.set_title(f"Rolling Avg Return ({window} eps)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return %")
        ax.grid(alpha=0.3)
        
        # 3 - Return distribution
        ax = axes[0, 2]
        ax.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(0, color='r', ls='--', linewidth=2)
        ax.axvline(np.mean(returns), color='g', ls='--', linewidth=2, label=f'Mean: {np.mean(returns):.3f}')
        ax.set_title("Return Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Return %")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4 - Cumulative performance
        ax = axes[1, 0]
        cumulative = np.cumsum(returns)
        ax.plot(cumulative, linewidth=2, color='darkgreen')
        ax.axhline(0, color='r', ls='--', alpha=0.3)
        ax.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='green')
        ax.set_title("Cumulative Performance", fontsize=14, fontweight='bold')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Return %")
        ax.grid(alpha=0.3)
        
        # 5 - Win rate rolling
        ax = axes[1, 1]
        win = [1 if r > 0 else 0 for r in returns]
        win_roll = pd.Series(win).rolling(window).mean()
        ax.plot(win_roll, linewidth=2, color='purple')
        ax.axhline(0.5, color='r', ls='--', alpha=0.3)
        ax.set_title(f"Rolling Win Rate ({window} eps)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Win Rate")
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3)
        
        # 6 - Evaluation metrics (if any)
        ax = axes[1, 2]
        if eval_history:
            eps = [e["episode"] for e in eval_history]
            ret = [e["mean_return"] for e in eval_history]
            sharpe = [e["mean_sharpe"] for e in eval_history]
            ax_twin = ax.twinx()
            ax.plot(eps, ret, 'b-o', label='Mean Return', linewidth=2)
            ax_twin.plot(eps, sharpe, 'g-s', label='Sharpe', linewidth=2)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Mean Return %", color='b')
            ax_twin.set_ylabel("Sharpe", color='g')
            ax.set_title("Evaluation Metrics", fontsize=14, fontweight='bold')
            lines, labs = ax.get_legend_handles_labels()
            lines2, labs2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines + lines2, labs + labs2, loc='upper left')
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Evaluation Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title("Evaluation Metrics", fontsize=14, fontweight='bold')
        
        # 7 - Volatility of returns
        ax = axes[2, 0]
        vol = pd.Series(returns).rolling(window).std()
        ax.plot(vol, linewidth=2, color='orange')
        ax.set_title(f"Rolling Volatility ({window} eps)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Std Dev of Returns")
        ax.grid(alpha=0.3)
        
        # 8 - Max drawdown evolution
        ax = axes[2, 1]
        if len(self.all_episodes) > 0:
            drawdowns = [ep.get('max_drawdown', 0) for ep in self.all_episodes]
            ax.plot(drawdowns, linewidth=1, alpha=0.6)
            dd_roll = pd.Series(drawdowns).rolling(window).mean()
            ax.plot(dd_roll, linewidth=2, color='red', label=f'Rolling Avg')
            ax.set_title("Max Drawdown per Episode", fontsize=14, fontweight='bold')
            ax.set_xlabel("Episode")
            ax.set_ylabel("Max Drawdown %")
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Drawdown Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title("Max Drawdown per Episode", fontsize=14, fontweight='bold')
        
        # 9 - Trading activity
        ax = axes[2, 2]
        if len(self.all_episodes) > 0:
            trades = [ep.get('total_trades', 0) for ep in self.all_episodes]
            trades_roll = pd.Series(trades).rolling(window).mean()
            ax.plot(trades_roll, linewidth=2, color='brown')
            ax.set_title(f"Avg Trades per Episode ({window} rolling)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Episode")
            ax.set_ylabel("Number of Trades")
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Trade Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title("Trading Activity", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        out_path = os.path.join(self.analysis_dir, "training_results.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Training plot saved to {out_path}")
    
    # -----------------------------------------------------------------------
    def plot_episode_comparison(self, episode_nums: list, max_plots: int = 6):
        """
        Plot side-by-side comparison of specific episodes.
        Useful for comparing best vs worst episodes.
        """
        episode_nums = episode_nums[:max_plots]
        n_episodes = len(episode_nums)
        
        if n_episodes == 0:
            return
        
        fig, axes = plt.subplots(n_episodes, 3, figsize=(18, 5*n_episodes))
        if n_episodes == 1:
            axes = axes.reshape(1, -1)
        
        for idx, ep_num in enumerate(episode_nums):
            # Load episode data
            csv_path = os.path.join(self.episodes_dir, f"ep_{ep_num:04d}_steps.csv")
            if not os.path.exists(csv_path):
                continue
            
            df = pd.DataFrame(pd.read_csv(csv_path))
            
            # Plot 1: Portfolio value over time
            ax = axes[idx, 0]
            ax.plot(df['step'], df['total_portfolio'], linewidth=2)
            ax.axhline(10000, color='r', ls='--', alpha=0.3, label='Initial')
            ax.set_title(f"Episode {ep_num}: Portfolio Value")
            ax.set_xlabel("Step")
            ax.set_ylabel("Portfolio ($)")
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Plot 2: Cumulative reward
            ax = axes[idx, 1]
            ax.plot(df['step'], df['cumulative_reward'], linewidth=2, color='green')
            ax.axhline(0, color='r', ls='--', alpha=0.3)
            ax.set_title(f"Episode {ep_num}: Cumulative Reward")
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.grid(alpha=0.3)
            
            # Plot 3: Actions taken
            ax = axes[idx, 2]
            action_counts = df['action'].value_counts()
            ax.bar(range(len(action_counts)), action_counts.values, 
                  tick_label=action_counts.index, color='steelblue')
            ax.set_title(f"Episode {ep_num}: Action Distribution")
            ax.set_xlabel("Action")
            ax.set_ylabel("Count")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        out_path = os.path.join(self.analysis_dir, "episode_comparison.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Episode comparison saved to {out_path}")
    
    # -----------------------------------------------------------------------
    def store_episode_summary(self, episode_data: dict):
        """Store episode data for later analysis."""
        self.all_episodes.append(episode_data)
    
    # -----------------------------------------------------------------------
    def generate_final_report(self, returns: list):
        """Generate comprehensive final report."""
        report_path = os.path.join(self.run_dir, "FINAL_REPORT.txt")
        
        with open(report_path, "w") as f:
            f.write("="*80 + "\n")
            f.write("TRAINING RUN FINAL REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Run ID: {self.run_timestamp}\n")
            f.write(f"Total Episodes: {len(returns)}\n\n")
            
            f.write("OVERALL STATISTICS:\n")
            f.write(f"  Mean Return: {np.mean(returns)*100:.3f}%\n")
            f.write(f"  Median Return: {np.median(returns)*100:.3f}%\n")
            f.write(f"  Std Return: {np.std(returns)*100:.3f}%\n")
            f.write(f"  Best Return: {np.max(returns)*100:.3f}%\n")
            f.write(f"  Worst Return: {np.min(returns)*100:.3f}%\n")
            f.write(f"  Win Rate: {np.mean([r > 0 for r in returns])*100:.1f}%\n")
            
            # Sharpe
            if len(returns) > 1:
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
                f.write(f"  Overall Sharpe: {sharpe:.3f}\n")
            
            f.write("\n")
            
            # Quartile analysis
            q1, q2, q3 = np.percentile(returns, [25, 50, 75])
            f.write("QUARTILE ANALYSIS:\n")
            f.write(f"  Q1 (25th percentile): {q1*100:.3f}%\n")
            f.write(f"  Q2 (50th percentile): {q2*100:.3f}%\n")
            f.write(f"  Q3 (75th percentile): {q3*100:.3f}%\n")
            f.write("\n")
            
            # Best/worst episodes
            best_eps = np.argsort(returns)[-10:][::-1]
            worst_eps = np.argsort(returns)[:10]
            
            f.write("TOP 10 EPISODES:\n")
            for rank, ep_idx in enumerate(best_eps, 1):
                f.write(f"  {rank}. Episode {ep_idx+1}: {returns[ep_idx]*100:+.3f}%\n")
            
            f.write("\nWORST 10 EPISODES:\n")
            for rank, ep_idx in enumerate(worst_eps, 1):
                f.write(f"  {rank}. Episode {ep_idx+1}: {returns[ep_idx]*100:+.3f}%\n")
            
            f.write("\n" + "="*80 + "\n")
        
        self.logger.info(f"Final report saved to {report_path}")