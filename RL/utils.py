#!/usr/bin/env python
"""
Helper classes used by train.py - logger + a quick plotter.
"""

import logging
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class AdvancedLogger:
    """
    Directory layout:
        training_logs/
            run_YYYYMMDD_HHMMSS/
                main.log
                evaluation_results.json
                wins/
                    ep_00xx_profit_yy.pct.json
                losses/
                    ep_00xx_loss_yy.pct.json
    """
    def __init__(self, base_dir="training_logs"):
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"run_{self.run_timestamp}")
        self.wins_dir = os.path.join(self.run_dir, "wins")
        self.losses_dir = os.path.join(self.run_dir, "losses")
        os.makedirs(self.wins_dir, exist_ok=True)
        os.makedirs(self.losses_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # file + console handlers
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

    # -----------------------------------------------------------------------
    def log_episode_details(self, episode_num: int, episode_data: dict, threshold: float = 0.08):
        ret = episode_data["final_return"]
        if abs(ret) < threshold:
            return

        is_win = ret > 0
        target_dir = self.wins_dir if is_win else self.losses_dir
        # file name eg. ep_0010_profit_12.3pct.json
        return_str = f"{abs(ret) * 100:.1f}pct"
        fname = f"ep_{episode_num:04d}_{'profit' if is_win else 'loss'}_{return_str}.json"
        path = os.path.join(target_dir, fname)

        # JSON can't serialise np types - convert
        def _to_native(o):
            if isinstance(o, dict):
                return {k: _to_native(v) for k, v in o.items()}
            if isinstance(o, (np.integer, np.floating)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o

        with open(path, "w") as f:
            json.dump(_to_native(episode_data), f, indent=2)
        self.logger.info(f"  Detailed log saved: {fname}")

    # -----------------------------------------------------------------------
    def log_evaluation(self, episode_num: int, eval_metrics: dict):
        self.eval_results.append({"episode": episode_num, **eval_metrics})
        eval_path = os.path.join(self.run_dir, "evaluation_results.json")
        with open(eval_path, "w") as f:
            json.dump(self.eval_results, f, indent=2)

    # -----------------------------------------------------------------------
    def plot_results(self, returns, eval_history):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        # 1 - raw returns
        ax = axes[0, 0]
        ax.plot(returns, alpha=0.6)
        ax.axhline(0, color="r", ls="--", alpha=0.3)
        ax.set_title("Portfolio Return per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return %")
        # 2 - rolling average
        ax = axes[0, 1]
        window = 50
        rolling = pd.Series(returns).rolling(window).mean()
        ax.plot(rolling, linewidth=2)
        ax.axhline(0.5, color="r", ls="--", alpha=0.3)
        ax.set_title(f"Rolling Avg ({window} eps)")
        ax.set_xlabel("Episode")
        # 3 - return distribution
        ax = axes[0, 2]
        ax.hist(returns, bins=40, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="r", ls="--")
        ax.set_title("Return distribution")
        # 4 - cumulative
        ax = axes[1, 0]
        ax.plot(np.cumsum(returns), linewidth=2)
        ax.axhline(0, color="r", ls="--", alpha=0.3)
        ax.set_title("Cumulative Return")
        # 5 - evaluation metrics (if any)
        if eval_history:
            ax = axes[1, 1]
            eps = [e["episode"] for e in eval_history]
            ret = [e["mean_return"] for e in eval_history]
            sharpe = [e["mean_sharpe"] for e in eval_history]
            ax_twin = ax.twinx()
            ax.plot(eps, ret, "b-o", label="Mean Return")
            ax_twin.plot(eps, sharpe, "g-s", label="Sharpe")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Mean Return %", color="b")
            ax_twin.set_ylabel("Sharpe", color="g")
            lines, labs = ax.get_legend_handles_labels()
            lines2, labs2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines + lines2, labs + labs2, loc="upper left")
        # 6 - win-rate rolling
        ax = axes[1, 2]
        win = [1 if r > 0 else 0 for r in returns]
        win_roll = pd.Series(win).rolling(window).mean()
        ax.plot(win_roll, linewidth=2)
        ax.axhline(0.5, color="r", ls="--", alpha=0.3)
        ax.set_title(f"Rolling Win Rate ({window} eps)")
        ax.set_xlabel("Episode")
        ax.set_ylim([0, 1])
        plt.tight_layout()
        out_path = os.path.join(self.run_dir, "training_results.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        self.logger.info(f"Plot saved to {out_path}")