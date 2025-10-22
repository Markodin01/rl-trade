#!/usr/bin/env python
"""
CryptoTradingEnvLongShort - FOR 1-HOUR AGGREGATED DATA
    - Works with 16 market features (not 7)
    - Observation dimension: 70 (16 + 9 + 40 + 5)
    - Position sizing, drawdown limit, proper info dict
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

log = logging.getLogger(__name__)

class CryptoTradingEnvLongShort(gym.Env):
    def __init__(
        self,
        norm: np.ndarray,                    # (T, 16) - normalised hourly features
        raw: np.ndarray,                     # (T, 4) - raw OHLC
        init_balance: float = 10_000,
        fee_pct: float = 0.001,              # 0.1 % per trade
        episode_len: int = 500,              # 500 hours = 20.8 days
        random_start: bool = True,
        lookback: int = 10,                  # 10 hours lookback
        drawdown_limit: float = 0.5,         # -50% max drawdown
    ):
        super().__init__()
        self.norm = norm.astype(np.float32)
        self.raw = raw.astype(np.float32)
        self.T = self.norm.shape[0]

        # Validate input shapes
        assert self.norm.shape[1] == 16, f"Expected 16 market features, got {self.norm.shape[1]}"
        assert self.raw.shape[1] == 4, f"Expected 4 OHLC columns, got {self.raw.shape[1]}"

        # ----------- hyper-params
        self.init_balance = float(init_balance)
        self.fee = float(fee_pct)
        self.episode_len = int(episode_len)
        self.random_start = bool(random_start)
        self.lookback = int(lookback)
        self.drawdown_limit = float(drawdown_limit)

        # ----------- action / obs spaces
        self.action_space = spaces.Discrete(5)
        
        # Observation: market(16) + pos(9) + hist(40) + mom(5) = 70
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(70,), dtype=np.float32
        )

        # ----------- internal state
        self.reset()

    # -----------------------------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        """Build 70-dim observation for 1-hour data."""
        
        # ---- market part (16 features from norm)
        market = self.norm[self.t, :16]                      # (16,)

        # ---- position part (9 features)
        price = self.raw[self.t, 0]                          # close
        pos_type = float(self.position)
        pos_value = (abs(self.btc) * price) / self.init_balance if self.btc != 0 else 0.0
        
        if self.position != 0 and self.entry_price > 0:
            unrealized = (price - self.entry_price) / self.entry_price if self.position == 1 \
                         else (self.entry_price - price) / self.entry_price
        else:
            unrealized = 0.0
            
        portfolio = self.balance + (self.btc * price if self.position != 0 else 0)
        total_ret = (portfolio / self.init_balance) - 1.0
        time_in_pos = (self.t - self.pos_enter) / 100.0
        vol = self.norm[self.t, 11]                          # volatility is 12th feature (index 11)
        win_rate = self.positive_trades / max(self.total_trades, 1)
        act_rate = self.trades / 100.0

        pos_vec = np.array([
            pos_type, pos_value, unrealized, total_ret,
            time_in_pos, vol, win_rate, act_rate, 
            self.short_trades / max(self.total_trades, 1)
        ], dtype=np.float32)

        # ---- historical OHLC context (lookback * 4 = 40)
        start = max(0, self.t - self.lookback)
        hist = self.raw[start:self.t, :4].flatten()
        if hist.shape[0] < self.lookback * 4:
            pad = np.zeros(self.lookback * 4 - hist.shape[0], dtype=np.float32)
            hist = np.concatenate([pad, hist])

        # ---- momentum (5 features)
        cur = self.raw[self.t, 0]  # close
        mom = []
        for step in [1, 6, 24, 72, 168]:  # 1h, 6h, 1d, 3d, 1w
            idx = max(0, self.t - step)
            mom.append((cur - self.raw[idx, 0]) / self.raw[idx, 0])
        mom = np.array(mom, dtype=np.float32)

        # ---- final concatenation (16 + 9 + 40 + 5 = 70)
        return np.concatenate([market, pos_vec, hist, mom]).astype(np.float32)

    # -----------------------------------------------------------------------
    def reset(self):
        self.balance = self.init_balance
        self.position = 0
        self.btc = 0.0
        self.entry_price = 0.0
        self.trades = 0
        self.positive_trades = 0
        self.total_trades = 0
        self.long_trades = self.short_trades = 0
        self.positive_long_trades = self.positive_short_trades = 0

        if self.random_start:
            self.t = np.random.randint(0, max(1, self.T - self.episode_len))
        else:
            self.t = 0
        self.pos_enter = self.t
        self.curric_step = 0
        
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_portfolio = []

        return self._build_obs()

    # -----------------------------------------------------------------------
    def _valid_mask(self) -> np.ndarray:
        mask = np.zeros(5, dtype=np.float32)
        if self.position == 0:
            mask[[0, 1, 3]] = 1
        elif self.position == 1:
            mask[[0, 2]] = 1
        else:
            mask[[0, 4]] = 1
        return mask

    # -----------------------------------------------------------------------
    def _get_position_size(self) -> float:
        """Volatility-based position sizing (50-95%)."""
        vol = self.norm[self.t, 11]  # volatility feature
        alloc_pct = np.clip(1.0 - (vol / 0.15) * 0.45, 0.50, 0.95)
        return alloc_pct

    # -----------------------------------------------------------------------
    def step(self, action: int):
        # ---- illegal actions
        mask = self._valid_mask()
        if mask[action] == 0:
            return self._build_obs(), -20.0, False, self._get_info()

        # ---- price movement
        cur_price = self.raw[self.t, 0]
        nxt_price = self.raw[self.t + 1, 0]
        price_change = (nxt_price - cur_price) / cur_price

        reward = 0.0

        # ----- ACTIONS with position sizing
        if action == 0:  # HOLD
            reward += price_change * self.position * 500

        elif action == 1 and self.position == 0:  # OPEN LONG
            alloc_pct = self._get_position_size()
            spend = self.balance * alloc_pct / (1 + self.fee)
            self.btc = spend / cur_price
            self.balance -= spend * (1 + self.fee)
            self.entry_price = cur_price
            self.position = 1
            self.pos_enter = self.t
            self.trades += 1
            self.total_trades += 1

        elif action == 2 and self.position == 1:  # CLOSE LONG
            gross = self.btc * nxt_price
            fee = gross * self.fee
            pnl = gross - fee - (self.entry_price * self.btc)
            reward += pnl / self.init_balance * 2000
            self.balance += gross - fee
            self.btc = 0.0
            self.position = 0
            self.entry_price = 0.0
            self.positive_trades += int(pnl > 0)
            self.long_trades += 1
            self.positive_long_trades += int(pnl > 0)

        elif action == 3 and self.position == 0:  # OPEN SHORT
            alloc_pct = self._get_position_size()
            spend = self.balance * alloc_pct / (1 + self.fee)
            self.btc = spend / cur_price
            self.balance -= spend * (1 + self.fee)
            self.entry_price = cur_price
            self.position = -1
            self.pos_enter = self.t
            self.trades += 1
            self.total_trades += 1

        elif action == 4 and self.position == -1:  # CLOSE SHORT
            gross = self.btc * nxt_price
            fee = gross * self.fee
            pnl = self.entry_price * self.btc - (gross + fee)
            reward += pnl / self.init_balance * 2000
            self.balance += pnl
            self.btc = 0.0
            self.position = 0
            self.entry_price = 0.0
            self.positive_trades += int(pnl > 0)
            self.short_trades += 1
            self.positive_short_trades += int(pnl > 0)

        # -----------------------------------------------------------------------
        # Curriculum (adjusted for hourly timeframe)
        # -----------------------------------------------------------------------
        self.curric_step += 1
        if self.curric_step < 200:  # First ~8 days
            ema_short = self.norm[self.t, 4]   # EMA_5
            ema_long = self.norm[self.t, 5]    # EMA_25
            trend = ema_short - ema_long
            if (trend < 0 and self.position == -1) or (trend > 0 and self.position == 1):
                reward += 1.0
        elif self.curric_step < 400:  # Days 8-17
            vol = self.norm[self.t, 11]
            reward -= 0.4 * max(0, vol - 0.02)
            if self.position == 0 and abs(price_change) > 0.01:
                reward -= 0.5

        # -----------------------------------------------------------------------
        # Drawdown limit
        # -----------------------------------------------------------------------
        portfolio = self.balance + (self.btc * nxt_price if self.position != 0 else 0)
        if portfolio < self.init_balance * self.drawdown_limit:
            reward -= 50.0
            done = True
            log.warning(f"Drawdown limit reached: portfolio={portfolio:.2f}")
        else:
            done = False

        # -----------------------------------------------------------------------
        # Episode end
        # -----------------------------------------------------------------------
        self.t += 1
        if self.t >= self.T - 2 or (self.t - self.pos_enter) >= self.episode_len:
            done = True
            if self.position != 0:
                if self.position == 1:
                    gross = self.btc * nxt_price
                    fee = gross * self.fee
                    pnl = gross - fee - (self.entry_price * self.btc)
                else:
                    gross = self.btc * nxt_price
                    fee = gross * self.fee
                    pnl = self.entry_price * self.btc - (gross + fee)
                
                self.balance += pnl if self.position == -1 else gross - fee
                self.btc = 0.0
                self.position = 0

        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_portfolio.append(portfolio)

        reward = float(np.tanh(reward))

        return self._build_obs(), reward, done, self._get_info()

    # -----------------------------------------------------------------------
    def _get_info(self) -> dict:
        portfolio = self.balance + (self.btc * self.raw[min(self.t, self.T-1), 0] if self.position != 0 else 0)
        portfolio_return = (portfolio / self.init_balance) - 1.0
        
        if len(self.episode_rewards) > 1:
            returns = np.array(self.episode_rewards)
            sharpe_ratio = returns.mean() / (returns.std() + 1e-8) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0.0
        
        return {
            "portfolio_value": portfolio,
            "portfolio_return": portfolio_return,
            "position": self.position,
            "total_trades": self.total_trades,
            "positive_trades": self.positive_trades,
            "sharpe_ratio": sharpe_ratio,
            "balance": self.balance,
            "btc_held": self.btc,
        }

    # -----------------------------------------------------------------------
    def get_episode_data(self) -> dict:
        portfolio = self.balance + (self.btc * self.raw[min(self.t, self.T-1), 0] if self.position != 0 else 0)
        
        return {
            "total_trades": self.total_trades,
            "positive_trades": self.positive_trades,
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "positive_long_trades": self.positive_long_trades,
            "positive_short_trades": self.positive_short_trades,
            "final_portfolio": portfolio,
            "final_balance": self.balance,
            "actions": self.episode_actions,
            "rewards": self.episode_rewards,
            "portfolio_history": self.episode_portfolio,
        }