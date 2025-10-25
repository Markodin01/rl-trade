#!/usr/bin/env python
"""
CryptoTradingEnvLongShort - IMPROVED VERSION
    - Fixed reward structure (Sharpe-based continuous rewards)
    - Step-by-step logging
    - Corrected position accounting
    - Better incentives for active trading
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

log = logging.getLogger(__name__)

class CryptoTradingEnvLongShort(gym.Env):
    def __init__(
        self,
        norm: np.ndarray,
        raw: np.ndarray,
        init_balance: float = 10_000,
        fee_pct: float = 0.001,
        episode_len: int = 500,
        random_start: bool = True,
        lookback: int = 10,
        drawdown_limit: float = 0.30,
        reward_mode: str = "sharpe",  # NEW: "sharpe", "pnl", "hybrid"
    ):
        super().__init__()
        self.norm = norm.astype(np.float32)
        self.raw = raw.astype(np.float32)
        self.T = self.norm.shape[0]

        assert self.norm.shape[1] == 16
        assert self.raw.shape[1] == 4

        # Hyperparams
        self.init_balance = float(init_balance)
        self.fee = float(fee_pct)
        self.episode_len = int(episode_len)
        self.random_start = bool(random_start)
        self.lookback = int(lookback)
        self.drawdown_limit = float(drawdown_limit)
        self.reward_mode = reward_mode

        # Action/obs spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(70,), dtype=np.float32
        )

        # Step-by-step logging
        self.step_log = []
        
        self.reset()

    # -----------------------------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        """Build 70-dim observation."""
        market = self.norm[self.t, :16]
        
        # Position features
        price = self.raw[self.t, 0]
        pos_type = float(self.position)
        
        # FIXED: Separate liquid vs position value
        liquid_pct = self.balance / self.init_balance
        
        if self.position != 0:
            position_pct = self.position_value / self.init_balance
            if self.entry_price > 0:
                unrealized = (price - self.entry_price) / self.entry_price if self.position == 1 \
                             else (self.entry_price - price) / self.entry_price
            else:
                unrealized = 0.0
        else:
            position_pct = 0.0
            unrealized = 0.0
            
        portfolio = self._get_portfolio_value()
        total_pct = portfolio / self.init_balance
        time_in_pos = min((self.t - self.pos_enter) / 100.0, 1.0)
        vol = self.norm[self.t, 11]
        win_rate = self.positive_trades / max(self.total_trades, 1)
        trade_rate = self.total_trades / max(self.t - self.episode_start, 1)

        pos_vec = np.array([
            pos_type, liquid_pct, position_pct, unrealized, total_pct,
            time_in_pos, vol, win_rate, trade_rate
        ], dtype=np.float32)

        # Historical OHLC
        start = max(0, self.t - self.lookback)
        hist = self.raw[start:self.t, :4].flatten()
        if hist.shape[0] < self.lookback * 4:
            pad = np.zeros(self.lookback * 4 - hist.shape[0], dtype=np.float32)
            hist = np.concatenate([pad, hist])

        # Momentum
        cur = self.raw[self.t, 0]
        mom = []
        for step in [1, 6, 24, 72, 168]:
            idx = max(0, self.t - step)
            mom.append((cur - self.raw[idx, 0]) / self.raw[idx, 0])
        mom = np.array(mom, dtype=np.float32)

        return np.concatenate([market, pos_vec, hist, mom]).astype(np.float32)

    # -----------------------------------------------------------------------
    def reset(self):
        self.balance = self.init_balance
        self.position = 0  # 0=neutral, 1=long, -1=short
        self.position_size = 0.0
        self.position_value = 0.0
        self.btc = 0.0
        self.entry_price = 0.0
        self.trades = 0
        self.positive_trades = 0
        self.total_trades = 0
        self.long_trades = self.short_trades = 0
        self.positive_long_trades = self.positive_short_trades = 0
        
        self.peak_portfolio = self.init_balance
        self.max_drawdown = 0.0

        if self.random_start:
            self.t = np.random.randint(0, max(1, self.T - self.episode_len))
        else:
            self.t = 0
        
        self.episode_start = self.t
        self.pos_enter = self.t
        
        # NEW: Track returns for Sharpe calculation
        self.episode_returns = []
        self.last_portfolio = self.init_balance
        
        # Step logging
        self.step_log = []

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
        """Conservative position sizing (30-60% based on volatility)."""
        vol = self.norm[self.t, 11]
        alloc_pct = np.clip(0.60 - (vol / 0.10) * 0.30, 0.30, 0.60)
        return alloc_pct

    # -----------------------------------------------------------------------
    def _get_portfolio_value(self) -> float:
        """FIXED: Correct portfolio calculation."""
        if self.t >= self.T:
            price = self.raw[self.T - 1, 0]
        else:
            price = self.raw[self.t, 0]
        
        if self.position == 0:
            # No position: portfolio = balance
            return self.balance
        
        elif self.position == 1:
            # Long position: portfolio = balance + current_value_of_btc
            btc_value = self.btc * price
            return self.balance + btc_value
        
        else:  # self.position == -1 (SHORT)
            buyback_cost = self.btc * price
            unrealized_pnl = self.position_size - buyback_cost
            # For portfolio calc: balance already has the short proceeds removed
            # So portfolio = balance + position_size (what we'd get back if we closed)
            return self.balance + self.position_size  # ✅ CORRECT

    # -----------------------------------------------------------------------
    def _calculate_reward(self, action: int, pnl: float = 0.0, 
                         illegal: bool = False) -> float:
        """
        NEW: Improved reward calculation with multiple modes.
        
        Modes:
        - "sharpe": Continuous Sharpe-like reward (encourages consistent gains)
        - "pnl": Direct PnL-based (your current approach)
        - "hybrid": Combination of both
        """
        if illegal:
            return -10.0
        
        # Calculate portfolio change
        current_portfolio = self._get_portfolio_value()
        portfolio_change = (current_portfolio - self.last_portfolio) / self.init_balance
        self.episode_returns.append(portfolio_change)
        self.last_portfolio = current_portfolio
        
        if self.reward_mode == "sharpe":
            # Sharpe-based: reward consistent incremental gains
            # This prevents "one big trade then coast" problem
            base_reward = portfolio_change * 1000  # FIXED: Better scaling to reasonable range
            
            # Bonus for maintaining good Sharpe
            if len(self.episode_returns) > 10:
                recent = self.episode_returns[-10:]
                mean_ret = np.mean(recent)
                std_ret = np.std(recent) + 1e-8
                sharpe = mean_ret / std_ret
                base_reward += sharpe * 0.5
            
            # Small penalty for inaction when we should be trading
            if action == 0 and self.position == 0:
                # Being neutral and doing nothing
                base_reward -= 0.005  # FIXED: 10x smaller
            
            # Penalty for excessive trading
            if action != 0:
                base_reward -= 0.002  # FIXED: 10x smaller
            
            return float(np.clip(base_reward, -2, 2))  # FIXED: Tighter)
        
        elif self.reward_mode == "pnl":
            # Your current approach (causes the 1-trade problem)
            if action in [2, 4]:  # Close position
                return (pnl / self.init_balance) * 1000
            else:
                return -0.1 if action != 0 else 0.0
        
        else:  # hybrid
            # Combine both approaches
            pnl_reward = (pnl / self.init_balance) * 100 if action in [2, 4] else 0.0
            sharpe_reward = portfolio_change * 50
            
            # Weight toward Sharpe for earlier episodes
            steps = self.t - self.episode_start
            sharpe_weight = max(0.3, 1.0 - steps / self.episode_len)
            
            total = sharpe_weight * sharpe_reward + (1 - sharpe_weight) * pnl_reward
            
            # Small action penalty
            if action != 0:
                total -= 0.05
            
            return float(np.clip(total, -10, 10))

    # -----------------------------------------------------------------------
    def step(self, action: int):
        """IMPROVED: Better accounting and logging."""
        
        # --- Pre-step state for logging ---
        pre_balance = self.balance
        pre_position = self.position
        pre_portfolio = self._get_portfolio_value()
        
        # --- Action validation ---
        mask = self._valid_mask()
        if mask[action] == 0:
            reward = self._calculate_reward(action, illegal=True)
            
            # Log this step
            self._log_step(action, reward, pre_balance, pre_position, 
                          pre_portfolio, pnl=0.0, illegal=True)
            
            return self._build_obs(), reward, False, self._get_info()

        # --- Price movement ---
        cur_price = self.raw[self.t, 0]
        next_price = self.raw[min(self.t + 1, self.T - 1), 0]
        
        pnl = 0.0
        position_opened = False
        position_closed = False

        # --- EXECUTE ACTIONS ---
        if action == 0:  # HOLD
            pass

        elif action == 1 and self.position == 0:  # OPEN LONG
            alloc_pct = self._get_position_size()
            available = self.balance * alloc_pct
            cost = available * (1 + self.fee)
            
            if cost > self.balance:
                reward = self._calculate_reward(action, illegal=True)
                self._log_step(action, reward, pre_balance, pre_position,
                              pre_portfolio, pnl=0.0, illegal=True)
                return self._build_obs(), reward, False, self._get_info()
            
            self.btc = available / cur_price
            self.balance -= cost
            self.entry_price = cur_price
            self.position = 1
            self.position_size = available
            self.position_value = available
            self.pos_enter = self.t
            self.trades += 1
            self.total_trades += 1
            position_opened = True

        elif action == 2 and self.position == 1:  # CLOSE LONG
            gross = self.btc * next_price
            fee = gross * self.fee
            net = gross - fee
            pnl = net - self.position_size
            
            self.balance += net
            self.positive_trades += int(pnl > 0)
            self.long_trades += 1
            self.positive_long_trades += int(pnl > 0)
            
            # Reset position
            self.btc = 0.0
            self.position = 0
            self.entry_price = 0.0
            self.position_size = 0.0
            self.position_value = 0.0
            position_closed = True

        elif action == 3 and self.position == 0:  # OPEN SHORT
            alloc_pct = self._get_position_size()
            available = self.balance * alloc_pct
            fee_cost = available * self.fee
            total_cost = available + fee_cost
            
            if total_cost > self.balance:
                reward = self._calculate_reward(action, illegal=True)
                self._log_step(action, reward, pre_balance, pre_position,
                              pre_portfolio, pnl=0.0, illegal=True)
                return self._build_obs(), reward, False, self._get_info()
            
            self.btc = available / cur_price
            self.balance -= total_cost
            self.entry_price = cur_price
            self.position = -1
            self.position_size = available
            self.position_value = available
            self.pos_enter = self.t
            self.trades += 1
            self.total_trades += 1
            position_opened = True

        elif action == 4 and self.position == -1:  # CLOSE SHORT
            buyback_cost = self.btc * next_price
            buyback_fee = buyback_cost * self.fee
            total_buyback = buyback_cost + buyback_fee
            pnl = self.position_size - total_buyback
            
            self.balance += self.position_size + pnl
            self.positive_trades += int(pnl > 0)
            self.short_trades += 1
            self.positive_short_trades += int(pnl > 0)
            
            # Reset position
            self.btc = 0.0
            self.position = 0
            self.entry_price = 0.0
            self.position_size = 0.0
            self.position_value = 0.0
            position_closed = True

        # --- Calculate reward ---
        reward = self._calculate_reward(action, pnl)

        # AFTER (CORRECT):
        if self.position != 0:
            if self.position == 1:
                self.position_value = self.btc * next_price
            else:  # short - use ABSOLUTE value or just the position size
                self.position_value = self.position_size  # Always positive

        # --- Drawdown check ---
        portfolio = self._get_portfolio_value()
        if portfolio > self.peak_portfolio:
            self.peak_portfolio = portfolio
        
        current_drawdown = (self.peak_portfolio - portfolio) / self.peak_portfolio
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        done = False
        if current_drawdown >= self.drawdown_limit:
            reward -= 50.0
            done = True
            log.warning(f"Drawdown limit: portfolio={portfolio:.2f}, dd={current_drawdown:.1%}")
        
        if self.balance < 0:
            log.error(f"Negative balance: {self.balance:.2f}")
            reward -= 100.0
            done = True

        # --- Episode termination ---
        self.t += 1
        steps_in_episode = self.t - self.episode_start

        episode_ending = (self.t >= self.T - 2 or steps_in_episode >= self.episode_len)
        done = False  # Initialize

        if episode_ending and self.position != 0:
            # Force close position BEFORE final log
            final_price = self.raw[min(self.t, self.T - 1), 0]
            if self.position == 1:
                gross = self.btc * final_price
                fee = gross * self.fee
                pnl = gross - fee - self.position_size
                self.balance += gross - fee
            else:  # short
                cost = self.btc * final_price
                fee = cost * self.fee
                pnl = self.position_size - (cost + fee)
                self.balance += self.position_size + pnl
            
            self.btc = 0.0
            self.position = 0
            self.position_size = 0.0
            self.position_value = 0.0
            
            # Mark as position closed for logging
            position_closed = True
            
            # Recalculate reward after forced close
            reward = self._calculate_reward(2 if pre_position == 1 else 4, pnl)
            done = True
        elif episode_ending:
            done = True

        # --- Log this step ---
        post_portfolio = self._get_portfolio_value()
        self._log_step(action, reward, pre_balance, pre_position, 
                    pre_portfolio, pnl, position_opened=position_opened,
                    position_closed=position_closed,
                    post_portfolio=post_portfolio)

        return self._build_obs(), reward, done, self._get_info()

    # -----------------------------------------------------------------------
    def _log_step(self, action: int, reward: float, pre_balance: float,
                  pre_position: int, pre_portfolio: float, pnl: float,
                  illegal: bool = False, position_opened: bool = False,
                  position_closed: bool = False, post_portfolio: float = None):
        """
        NEW: Log each step with format:
        step | liquid | position | total_portfolio | step_reward | cumul_reward | action | pos_type | pnl
        """
        if post_portfolio is None:
            post_portfolio = self._get_portfolio_value()
        
        cumulative_reward = sum([log['step_reward'] for log in self.step_log])
        
        action_names = ['HOLD', 'OPEN_LONG', 'CLOSE_LONG', 'OPEN_SHORT', 'CLOSE_SHORT']
        pos_names = {0: 'NEUTRAL', 1: 'LONG', -1: 'SHORT'}
        
        log_entry = {
            'step': self.t - self.episode_start,
            'absolute_t': self.t,
            'liquid': self.balance,
            'position_value': self.position_value,
            'total_portfolio': post_portfolio,
            'step_reward': reward,
            'cumulative_reward': cumulative_reward + reward,
            'action': action_names[action],
            'position_before': pos_names[pre_position],
            'position_after': pos_names[self.position],
            'pnl': pnl,
            'illegal': illegal,
            'position_opened': position_opened,
            'position_closed': position_closed,
            'price': self.raw[min(self.t, self.T-1), 0],
        }
        
        self.step_log.append(log_entry)

    # -----------------------------------------------------------------------
    def _get_info(self) -> dict:
        portfolio = self._get_portfolio_value()
        portfolio_return = (portfolio / self.init_balance) - 1.0
        
        if len(self.episode_returns) > 1:
            returns = np.array(self.episode_returns)
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
            "position_value": self.position_value,
        }

    # -----------------------------------------------------------------------
    def get_episode_data(self) -> dict:
        """IMPROVED: Return detailed episode data with step logs."""
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
            "final_position_value": self.position_value,
            "max_drawdown": self.max_drawdown,
            "step_log": self.step_log,  # NEW: Full step-by-step log
            "returns": self.episode_returns,
            "sharpe_ratio": np.mean(self.episode_returns) / (np.std(self.episode_returns) + 1e-8),
        }