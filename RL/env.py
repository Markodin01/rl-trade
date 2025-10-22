#!/usr/bin/env python
"""
CryptoTradingEnvLongShort
    - Vectorised Gymnasium env.
    - Observation = 66 features (your earlier count).
    - Action space = 5: [HOLD, LONG, CLOSE_LONG, SHORT, CLOSE_SHORT]
    - Curriculum is baked directly into the step() method (3 phases).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

log = logging.getLogger(__name__)

def _calc_volatility(series: np.ndarray, win: int = 20) -> np.ndarray:
    """Fast numpy rolling volatility (log returns)."""
    log_ret = np.log(series[1:] / series[:-1])
    vol = np.empty_like(series)
    vol[:] = np.nan
    for i in range(win, len(series)):
        vol[i] = np.std(log_ret[i - win:i]) * np.sqrt(252)
    return vol

class CryptoTradingEnvLongShort(gym.Env):
    def __init__(
        self,
        norm: np.ndarray,                    # (T, 66) - normalised features
        raw: np.ndarray,                     # (T, 4) - raw OHLC
        init_balance: float = 10_000,
        fee_pct: float = 0.001,              # 0.1 % per trade
        episode_len: int = 500,
        random_start: bool = True,
        lookback: int = 10,
    ):
        super().__init__()
        self.norm = norm.astype(np.float32)
        self.raw = raw.astype(np.float32)
        self.T = self.norm.shape[0]

        # ----------- hyper-params
        self.init_balance = float(init_balance)
        self.fee = float(fee_pct)
        self.episode_len = int(episode_len)
        self.random_start = bool(random_start)
        self.lookback = int(lookback)

        # ----------- action / obs spaces
        self.action_space = spaces.Discrete(5)    # 0 HOLD,1 LONG,2 CLOSE_LONG,3 SHORT,4 CLOSE_SHORT
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(66,), dtype=np.float32
        )

        # ----------- internal state
        self.reset()

    # -----------------------------------------------------------------------
    # Helper: build a full observation (no pandas)
    # -----------------------------------------------------------------------
    def _build_obs(self) -> np.ndarray:
# ---- market part (first 7 columns of norm)
        market = self.norm[self.t]                           # (66,)

        # ---- position part (9 numbers)
        price = self.raw[self.t, 3]                          # close
        pos_type = float(self.position)                      # -1/0/1
        pos_value = (abs(self.btc) * price) / self.init_balance if self.btc > 0 else 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized = (price - self.entry_price) / self.entry_price if self.position == 1 else (self.entry_price - price) / self.entry_price
        else:
            unrealized = 0.0
        portfolio = (self.balance + (self.btc * price) if self.position != 0 else self.balance)
        total_ret = (portfolio / self.init_balance) - 1.0
        time_in_pos = (self.t - getattr(self, "pos_enter", self.t)) / 100.0
        vol = self.norm[self.t, -1]                          # we stored volatility as the last column
        win_rate = self.positive_trades / max(self.total_trades, 1)
        act_rate = self.trades / 100.0
        short_ratio = self.short_trades / max(self.total_trades, 1)

        pos_vec = np.array([
            pos_type, pos_value, unrealized, total_ret,
            time_in_pos, vol, win_rate, act_rate, short_ratio
        ], dtype=np.float32)

        # ---- historical OHLC context (lookback * 4)
        start = max(0, self.t - self.lookback)
        hist = self.norm[start:self.t, :4].flatten()
        if hist.shape[0] < self.lookback * 4:
            pad = np.zeros(self.lookback * 4 - hist.shape[0], dtype=np.float32)
            hist = np.concatenate([pad, hist])

        # ---- momentum (5)
        cur = self.raw[self.t, 3]
        mom = []
        for step in [1, 3, 6, 12, 24]:
            idx = max(0, self.t - step)
            mom.append((cur - self.raw[idx, 3]) / self.raw[idx, 3])
        mom = np.array(mom, dtype=np.float32)

        # ---- final concatenation (66)
        return np.concatenate([market, pos_vec, hist, mom]).astype(np.float32)

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------
    def reset(self):
        self.balance = self.init_balance
        self.position = 0                    # -1 short, 0 flat, 1 long
        self.btc = 0.0
        self.entry_price = 0.0
        self.trades = 0
        self.positive_trades = 0
        self.total_trades = 0
        self.long_trades = self.short_trades = self.positive_long_trades = self.positive_short_trades = 0

        # - start index
        if self.random_start:
            self.t = np.random.randint(0, self.T - self.episode_len)
        else:
            self.t = 0
        self.pos_enter = self.t
        self.curric_step = 0    # episode-internal curriculum step counter

        return self._build_obs()

    # -----------------------------------------------------------------------
    # Action-mask (separate from observation)
    # -----------------------------------------------------------------------
    def _valid_mask(self) -> np.ndarray:
        mask = np.zeros(5, dtype=np.float32)    # HOLD, LONG, CLOSE_LONG, SHORT, CLOSE_SHORT
        if self.position == 0:
            mask[[0, 1, 3]] = 1
        elif self.position == 1:
            mask[[0, 2]] = 1
        else:  # short
            mask[[0, 4]] = 1
        return mask

    # -----------------------------------------------------------------------
    # STEP
    # -----------------------------------------------------------------------
    def step(self, action: int):
        # ---- illegal actions → big penalty
        mask = self._valid_mask()
        if mask[action] == 0:
            return self._build_obs(), -20.0, False, {}

        # ---- basic price movement
        cur_price = self.raw[self.t, 3]                      # close at step t
        nxt_price = self.raw[self.t + 1, 3]                  # close at step t+1
        price_change = (nxt_price - cur_price) / cur_price

        reward = 0.0

        # ----- ACTION LOGIC (keep the same economic semantics you already had)
        if action == 0:  # HOLD
            # reward = position * price_change * 500 (already built in your original code)
            reward += price_change * self.position * 500

        elif action == 1 and self.position == 0:  # OPEN LONG
            spend = self.balance / (1 + self.fee)
            self.btc = spend / cur_price
            self.balance = 0.0
            self.entry_price = cur_price
            self.position = 1
            self.pos_enter = self.t
            self.trades += 1

        elif action == 2 and self.position == 1:  # CLOSE LONG
            gross = self.btc * nxt_price
            fee = gross * self.fee
            pnl = gross - fee - (self.entry_price * self.btc)
            reward += pnl / self.init_balance * 2000
            self.balance = gross - fee
            self.btc = 0.0
            self.position = 0
            self.entry_price = 0.0
            self.trades += 1
            self.positive_trades += int(pnl > 0)
            self.long_trades += 1
            self.positive_long_trades += int(pnl > 0)

        elif action == 3 and self.position == 0:  # OPEN SHORT
            spend = self.balance / (1 + self.fee)
            self.btc = spend / cur_price                     # BTC short-owed
            self.balance = 0.0
            self.entry_price = cur_price
            self.position = -1
            self.pos_enter = self.t
            self.trades += 1

        elif action == 4 and self.position == -1:  # CLOSE SHORT
            gross = self.btc * nxt_price
            fee = gross * self.fee
            pnl = self.entry_price * self.btc - (gross + fee)
            reward += pnl / self.init_balance * 2000
            self.balance = pnl
            self.btc = 0.0
            self.position = 0
            self.entry_price = 0.0
            self.trades += 1
            self.positive_trades += int(pnl > 0)
            self.short_trades += 1
            self.positive_short_trades += int(pnl > 0)

        # -----------------------------------------------------------------------
        # Curriculum hook - three phases based on steps inside this episode
        # -----------------------------------------------------------------------
        self.curric_step += 1
        if self.curric_step < 200:                           # Phase 0 - trend-bonus
            # We assume the last column of 'norm' is the EMA_5 (you can change the index if you add more columns)
            ema = self.norm[self.t, -2]  # EMA_5 is second-last column (you can adjust)
            if (ema < 0 and self.position == -1) or (ema > 0 and self.position == 1):
                reward += 1.0                                # extra signal for being "on the trend"
        elif self.curric_step < 400:                         # Phase 1 - volatility + draw-down penalty
            vol = self.norm[self.t, -1]                      # volatility column
            reward -= 0.4 * max(0, vol - 0.02)               # penalise when volatility > 2 %
            # extra penalty for missing >1 % moves when flat
            if self.position == 0 and abs(price_change) > 0.01:
                reward -= 0.5

        # else Phase 2 - realistic (nothing extra, the reward already contains PnL & penalty)

        # -----------------------------------------------------------------------
        # Episode termination handling
        # -----------------------------------------------------------------------
        self.t += 1
        done = False
        if self.t >= self.T - 2 or (self.t - self.pos_enter) >= self.episode_len:
            done = True
            # forced liquidation if we are still in a position
            if self.position != 0:
                if self.position == 1:
                    gross = self.btc * nxt_price
                    fee = gross * self.fee
                    pnl = gross - fee - (self.entry_price * self.btc)
                else:
                    gross = self.btc * nxt_price
                    fee = gross * self.fee
                    pnl = self.entry_price * self.btc - (gross + fee)
                reward += pnl / self.init_balance * 2000
                self.balance = self.balance + pnl if self.position == -1 else pnl - fee
                self.btc = 0.0
                self.position = 0

        # clip rewards (helps TD-target stability)
        reward = float(np.tanh(reward))

        # info dict - you can enrich it later if you want.
        info = {
            "portfolio_value": self.balance + (self.btc * nxt_price if self.position != 0 else 0),
            "position": self.position,
            "reward": reward,
        }
        return self._build_obs(), reward, done, info