#!/usr/bin/env python
"""
CryptoTradingEnvLongShort - FIXED VERSION
    - Proper position sizing (no over-leveraging)
    - Balance protection (can't go negative)
    - Better reward shaping
    - Fixed drawdown calculation
    - More reasonable curriculum
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
        drawdown_limit: float = 0.30,        # -30% max drawdown (more reasonable)
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
        
        # FIXED: Use position_value instead of recalculating
        pos_value = self.position_value / self.init_balance if self.position != 0 else 0.0
        
        if self.position != 0 and self.entry_price > 0:
            unrealized = (price - self.entry_price) / self.entry_price if self.position == 1 \
                         else (self.entry_price - price) / self.entry_price
        else:
            unrealized = 0.0
            
        portfolio = self._get_portfolio_value()
        total_ret = (portfolio / self.init_balance) - 1.0
        time_in_pos = min((self.t - self.pos_enter) / 100.0, 1.0)  # Cap at 1.0
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
        self.position_size = 0.0  # USD value of position
        self.position_value = 0.0  # Current value
        self.btc = 0.0
        self.entry_price = 0.0
        self.trades = 0
        self.positive_trades = 0
        self.total_trades = 0
        self.long_trades = self.short_trades = 0
        self.positive_long_trades = self.positive_short_trades = 0
        
        # Track peak portfolio for drawdown
        self.peak_portfolio = self.init_balance
        self.max_drawdown = 0.0

        if self.random_start:
            self.t = np.random.randint(0, max(1, self.T - self.episode_len))
        else:
            self.t = 0
        
        self.episode_start = self.t  # NEW: Track episode start (never changes)
        self.pos_enter = self.t      # Track position start (resets on new position)
        self.curric_step = 0
        
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_portfolio = []

        return self._build_obs()

    # -----------------------------------------------------------------------
    def _valid_mask(self) -> np.ndarray:
        mask = np.zeros(5, dtype=np.float32)
        if self.position == 0:
            mask[[0, 1, 3]] = 1  # HOLD, OPEN_LONG, OPEN_SHORT
        elif self.position == 1:
            mask[[0, 2]] = 1  # HOLD, CLOSE_LONG
        else:
            mask[[0, 4]] = 1  # HOLD, CLOSE_SHORT
        return mask

    # -----------------------------------------------------------------------
    def _get_position_size(self) -> float:
        """FIXED: Conservative position sizing (30-60% based on volatility)."""
        vol = self.norm[self.t, 11]  # volatility feature
        
        # More conservative: 30-60% instead of 50-95%
        # Lower volatility = larger position
        alloc_pct = np.clip(0.60 - (vol / 0.10) * 0.30, 0.30, 0.60)
        return alloc_pct

    # -----------------------------------------------------------------------
    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        if self.t >= self.T:
            price = self.raw[self.T - 1, 0]
        else:
            price = self.raw[self.t, 0]
        
        if self.position != 0:
            if self.position == 1:
                # Long: portfolio = balance + btc * price
                return self.balance + (self.btc * price)
            else:
                # Short: portfolio = balance + position_size + unrealized_pnl
                # Unrealized PnL = position_size - (btc * current_price)
                # We sold BTC for position_size, currently would need to buy back at btc*price
                current_buyback_cost = self.btc * price
                unrealized_pnl = self.position_size - current_buyback_cost
                # Balance already has: initial - (position_size + open_fee)
                # So portfolio = balance + position_size + unrealized_pnl
                return self.balance + self.position_size + unrealized_pnl
        else:
            return self.balance

    # -----------------------------------------------------------------------
    def step(self, action: int):
        # ---- illegal actions
        mask = self._valid_mask()
        if mask[action] == 0:
            return self._build_obs(), -10.0, False, self._get_info()

        # ---- price movement
        cur_price = self.raw[self.t, 0]
        next_price = self.raw[min(self.t + 1, self.T - 1), 0]
        price_change = (next_price - cur_price) / cur_price

        reward = 0.0

        # ----- ACTIONS with FIXED position sizing
        if action == 0:  # HOLD
            # Reward for holding in trending position
            if self.position != 0:
                pnl_pct = price_change * self.position
                reward += pnl_pct * 100  # Scale by 100 for better gradient

        elif action == 1 and self.position == 0:  # OPEN LONG
            alloc_pct = self._get_position_size()
            available = self.balance * alloc_pct
            
            # Account for fees
            cost = available * (1 + self.fee)
            
            if cost > self.balance:
                # Can't afford the position
                reward = -5.0
            else:
                self.btc = available / cur_price
                self.balance -= cost
                self.entry_price = cur_price
                self.position = 1
                self.position_size = available
                self.position_value = available
                self.pos_enter = self.t
                self.trades += 1
                self.total_trades += 1
                reward = -0.1  # Small penalty for opening position

        elif action == 2 and self.position == 1:  # CLOSE LONG
            gross = self.btc * next_price
            fee = gross * self.fee
            net = gross - fee
            pnl = net - self.position_size
            
            self.balance += net
            reward = (pnl / self.init_balance) * 1000  # Scale up for learning
            
            self.positive_trades += int(pnl > 0)
            self.long_trades += 1
            self.positive_long_trades += int(pnl > 0)
            
            # Reset position
            self.btc = 0.0
            self.position = 0
            self.entry_price = 0.0
            self.position_size = 0.0
            self.position_value = 0.0

        elif action == 3 and self.position == 0:  # OPEN SHORT
            alloc_pct = self._get_position_size()
            available = self.balance * alloc_pct
            
            # Account for fees
            fee_cost = available * self.fee
            total_cost = available + fee_cost
            
            if total_cost > self.balance:
                reward = -5.0
            else:
                # For short: we "sell" BTC we don't have
                # We set aside capital equal to the short size
                self.btc = available / cur_price
                self.balance -= total_cost  # Remove capital + fee from balance
                self.entry_price = cur_price
                self.position = -1
                self.position_size = available  # Track the USD value we shorted
                self.position_value = available
                self.pos_enter = self.t
                self.trades += 1
                self.total_trades += 1
                reward = -0.1

        elif action == 4 and self.position == -1:  # CLOSE SHORT
            # Buy back BTC to close short
            buyback_cost = self.btc * next_price
            buyback_fee = buyback_cost * self.fee
            total_buyback = buyback_cost + buyback_fee
            
            # Short PnL = (sold at entry) - (bought at current)
            # We sold BTC for self.position_size, now buying back for total_buyback
            pnl = self.position_size - total_buyback
            
            # Return: original capital + PnL - opening fee (already deducted)
            # balance currently has: initial - (position_size + open_fee)
            # we need to add back: position_size + pnl
            self.balance += self.position_size + pnl
            
            reward = (pnl / self.init_balance) * 1000
            
            self.positive_trades += int(pnl > 0)
            self.short_trades += 1
            self.positive_short_trades += int(pnl > 0)
            
            # Reset position
            self.btc = 0.0
            self.position = 0
            self.entry_price = 0.0
            self.position_size = 0.0
            self.position_value = 0.0

        # -----------------------------------------------------------------------
        # FIXED: Simplified curriculum (less aggressive)
        # -----------------------------------------------------------------------
        self.curric_step += 1
        
        # Early stage: encourage trend following
        if self.curric_step < 100:  # First ~4 days
            if self.position != 0:
                trend_match = (price_change > 0 and self.position == 1) or \
                             (price_change < 0 and self.position == -1)
                if trend_match:
                    reward += 0.5
        
        # Later stage: penalize excessive trading
        elif self.curric_step > 200:
            if action != 0:  # Any action except HOLD
                reward -= 0.2

        # -----------------------------------------------------------------------
        # FIXED: Drawdown calculation and termination
        # -----------------------------------------------------------------------
        portfolio = self._get_portfolio_value()
        
        # Update peak
        if portfolio > self.peak_portfolio:
            self.peak_portfolio = portfolio
        
        # Calculate drawdown from peak
        current_drawdown = (self.peak_portfolio - portfolio) / self.peak_portfolio
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Terminate if drawdown exceeds limit
        if current_drawdown >= self.drawdown_limit:
            reward -= 100.0
            done = True
            log.warning(f"Drawdown limit reached: portfolio={portfolio:.2f}, drawdown={current_drawdown:.1%}")
        else:
            done = False
        
        # FIXED: Check for negative balance (should never happen, but safety check)
        if self.balance < 0:
            log.error(f"Negative balance: {self.balance:.2f}")
            reward -= 200.0
            done = True

        # -----------------------------------------------------------------------
        # Episode end
        # -----------------------------------------------------------------------
        self.t += 1
        steps_in_episode = self.t - self.episode_start
        if self.t >= self.T - 2 or steps_in_episode >= self.episode_len:
            done = True
            # Force close position at episode end
            if self.position != 0:
                if self.position == 1:
                    gross = self.btc * next_price
                    fee = gross * self.fee
                    self.balance += gross - fee
                else:  # Short
                    cost = self.btc * next_price
                    fee = cost * self.fee
                    pnl = self.position_size - (cost + fee)
                    self.balance += pnl
                
                self.btc = 0.0
                self.position = 0

        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_portfolio.append(portfolio)

        # Clip reward for stable training
        reward = float(np.clip(reward, -10, 10))

        return self._build_obs(), reward, done, self._get_info()

    # -----------------------------------------------------------------------
    def _get_info(self) -> dict:
        portfolio = self._get_portfolio_value()
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
            "max_drawdown": self.max_drawdown,
        }

    # -----------------------------------------------------------------------
    def get_episode_data(self) -> dict:
        portfolio = self._get_portfolio_value()
        
        return {
            "total_trades": self.total_trades,
            "positive_trades": self.positive_trades,
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "positive_long_trades": self.positive_long_trades,
            "positive_short_trades": self.positive_short_trades,
            "final_portfolio": portfolio,
            "final_balance": self.balance,
            "max_drawdown": self.max_drawdown,
            "actions": self.episode_actions,
            "rewards": self.episode_rewards,
            "portfolio_history": self.episode_portfolio,
        }