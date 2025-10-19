"""
Complete Long/Short Crypto Trading Agent with Advanced Logging

Features:
- 5-action space (HOLD, LONG, CLOSE_LONG, SHORT, CLOSE_SHORT)
- Can flip LONG↔SHORT with penalty
- Detailed per-episode logging for wins/losses
- Step-by-step action logs for extreme episodes
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
import random
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def calculate_historical_volatility(df, window=20):
    """Calculate historical volatility using log returns"""
    log_returns = np.log(df['close'] / df['close'].shift(1))
    hist_vol = log_returns.rolling(window=window).std() * np.sqrt(252)
    return hist_vol


class AdvancedLogger:
    """
    Advanced logging system that tracks detailed episode information
    
    Structure:
    training_logs/
    └── run_YYYYMMDD_HHMMSS/
        ├── main.log (overall training log)
        ├── wins/
        │   ├── ep_100_profit_15.2pct.json
        │   └── ep_250_profit_22.5pct.json
        └── losses/
            ├── ep_050_loss_18.3pct.json
            └── ep_150_loss_25.1pct.json
    """
    
    def __init__(self, base_dir="training_logs"):
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"run_{self.run_timestamp}")
        self.wins_dir = os.path.join(self.run_dir, "wins")
        self.losses_dir = os.path.join(self.run_dir, "losses")
        
        # Create directories
        os.makedirs(self.wins_dir, exist_ok=True)
        os.makedirs(self.losses_dir, exist_ok=True)
        
        # Set up main logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.run_dir, "main.log"))
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Clear existing handlers and add new ones
        self.logger.handlers.clear()
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info(f"Advanced logging initialized: {self.run_dir}")
        
    def log_episode_details(self, episode_num, episode_data, threshold=0.08):
        """
        Log detailed episode information for extreme cases
        
        Args:
            episode_num: Episode number
            episode_data: Dict with episode information
            threshold: Log if |return| > threshold (default 8%)
        """
        return_pct = episode_data['final_return']
        
        # Only log extreme episodes
        if abs(return_pct) < threshold:
            return
        
        # Determine win or loss
        is_win = return_pct > 0
        target_dir = self.wins_dir if is_win else self.losses_dir
        
        # Create filename
        return_str = f"{abs(return_pct)*100:.1f}pct"
        filename = f"ep_{episode_num:04d}_{'profit' if is_win else 'loss'}_{return_str}.json"
        filepath = os.path.join(target_dir, filename)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert all numpy types
        episode_data_clean = convert_to_native(episode_data)
        
        # Save detailed log
        with open(filepath, 'w') as f:
            json.dump(episode_data_clean, f, indent=2)
        
        self.logger.info(f"  📝 Detailed log saved: {filename}")


class CryptoTradingEnvLongShort(gym.Env):
    """
    Enhanced crypto trading environment with LONG and SHORT capabilities
    Includes detailed step-by-step logging for analysis
    """
    
    def __init__(self, df_normalized, df_raw, initial_balance=10000, 
                 transaction_fee_percent=0.1, episode_length=500, 
                 random_start=True, log_steps=False, lookback_window=10):
        super(CryptoTradingEnvLongShort, self).__init__()
        
        # Data setup
        required_columns = ['close', 'high', 'low', 'open', 'EMA_5', 'BBM_5_2.0']
        self.df_normalized = df_normalized[required_columns].copy()
        self.df_normalized['volatility'] = calculate_historical_volatility(df_raw)
        self.df_normalized = self.df_normalized.dropna()
        
        self.df_raw = df_raw[['close', 'high', 'low', 'open']].copy()
        self.df_raw = self.df_raw.loc[self.df_normalized.index]
        
        assert len(self.df_normalized) == len(self.df_raw), "Data must be aligned"
        
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.episode_length = episode_length
        self.random_start = random_start
        self.log_steps = log_steps
        self.lookback_window = lookback_window  # NEW: How many past candles to include
        self.max_start_step = max(0, len(self.df_normalized) - episode_length - 1)
        
        # Action space: 5 actions
        self.action_space = spaces.Discrete(5)
        
        # ENHANCED OBSERVATION SPACE with lookback context
        # Base features: 7 (close, high, low, open, EMA_5, BBM_5_2.0, volatility)
        # Position info: 9
        # Historical context: lookback_window × 4 (OHLC for each past candle)
        # Price momentum: 5 (1h, 3h, 6h, 12h, 24h ago returns)
        total_features = 7 + 9 + (lookback_window * 4) + 10
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_features,),
            dtype=np.float32
        )
        
        logger.info(f"Enhanced observation space: {total_features} features")
        logger.info(f"  - Current candle: 7 features")
        logger.info(f"  - Position state: 9 features")
        logger.info(f"  - Historical OHLC: {lookback_window * 4} features ({lookback_window} candles)")
        logger.info(f"  - Price momentum: 5 features")
        
        self.reset()
        
    def reset(self):
        """Reset with step logging initialization"""
        self.balance = self.initial_balance
        self.btc_held = 0
        self.position_type = 0  # -1=SHORT, 0=FLAT, 1=LONG
        self.done = False
        
        self.entry_price = None
        self.short_value_at_entry = 0  # Track USD value when opening short
        self.position_open_time = None
        self.transaction_count = 0
        self.positive_trades = 0
        self.total_trades = 0
        self.total_profit = 0
        
        self.short_trades = 0
        self.positive_short_trades = 0
        self.long_trades = 0
        self.positive_long_trades = 0
        
        self.max_portfolio_value = self.initial_balance
        self.episode_rewards = []
        
        # Step-by-step logging
        self.step_log = []
        
        if self.random_start and self.max_start_step > 0:
            self.current_step = np.random.randint(0, self.max_start_step)
        else:
            self.current_step = 0
        
        self.episode_start_step = self.current_step
        
        return self._next_observation()
    
    def _get_valid_actions_mask(self):
        """Return binary mask of valid actions"""
        # Always can HOLD (action 0)
        mask = [1, 0, 0, 0, 0]  # [HOLD, LONG, CLOSE_LONG, SHORT, CLOSE_SHORT]
        
        if self.position_type == 0:  # FLAT
            mask[1] = 1  # Can LONG
            mask[3] = 1  # Can SHORT
        elif self.position_type == 1:  # In LONG
            mask[2] = 1  # Can CLOSE_LONG
        elif self.position_type == -1:  # In SHORT
            mask[4] = 1  # Can CLOSE_SHORT
        
        return np.array(mask, dtype=np.float32)
    
    def _next_observation(self):
        """
        Construct observation with HISTORICAL CONTEXT
        
        Now includes:
        - Current market data (normalized)
        - Position state
        - Last N candles (OHLC)
        - Price momentum indicators
        """
        # Get NORMALIZED market data for current candle (7 features)
        market_data = self.df_normalized.iloc[self.current_step].values
        
        # Get RAW price for position calculations
        current_price_raw = self.df_raw.iloc[self.current_step]['close']
        
        # Calculate portfolio value based on position type
        if self.position_type == 1:  # LONG
            portfolio_value = self.balance + self.btc_held * current_price_raw
        elif self.position_type == -1:  # SHORT
            unrealized_cost = self.btc_held * current_price_raw
            unrealized_pnl_dollars = self.short_value_at_entry - unrealized_cost if hasattr(self, 'short_value_at_entry') else 0
            portfolio_value = self.balance + unrealized_pnl_dollars
        else:  # FLAT
            portfolio_value = self.balance
        
        # Position information (9 features)
        position_type_indicator = float(self.position_type)
        position_value = abs(self.btc_held * current_price_raw) / self.initial_balance if self.btc_held > 0 else 0.0
        
        if self.entry_price and self.position_type != 0:
            if self.position_type == 1:
                unrealized_pnl = (current_price_raw - self.entry_price) / self.entry_price
            else:
                unrealized_pnl = (self.entry_price - current_price_raw) / self.entry_price
        else:
            unrealized_pnl = 0.0
        
        total_return = (portfolio_value / self.initial_balance) - 1.0
        time_in_position = (self.current_step - self.position_open_time) / 100.0 if self.position_open_time else 0.0
        current_volatility = self.df_normalized.iloc[self.current_step]['volatility']
        win_rate = self.positive_trades / max(self.total_trades, 1)
        activity = self.transaction_count / 100.0
        short_ratio = self.short_trades / max(self.total_trades, 1)
        
        position_state = np.array([
            position_type_indicator, position_value, unrealized_pnl, total_return,
            time_in_position, current_volatility, win_rate, activity, short_ratio
        ], dtype=np.float32)
        
        # NEW: Historical OHLC context (lookback_window × 4 features = 40)
        historical_ohlc = []
        for i in range(self.lookback_window, 0, -1):
            # Get past candle, with boundary checking
            lookback_idx = max(0, self.current_step - i)
            past_candle = self.df_normalized.iloc[lookback_idx][['open', 'high', 'low', 'close']].values
            historical_ohlc.extend(past_candle)
        
        historical_ohlc = np.array(historical_ohlc, dtype=np.float32)
        
        # NEW: Price momentum indicators (5 features)
        # How much has price changed over different timeframes?
        momentum_features = []
        timeframes = [1, 3, 6, 12, 24]  # 1h, 3h, 6h, 12h, 24h ago
        
        for tf in timeframes:
            lookback_idx = max(0, self.current_step - tf)
            past_price = self.df_raw.iloc[lookback_idx]['close']
            price_change = (current_price_raw - past_price) / past_price
            momentum_features.append(price_change)
        
        momentum_features = np.array(momentum_features, dtype=np.float32)
        
        valid_actions = self._get_valid_actions_mask()
        
        # Concatenate to observation (now 66 features: 61 + 5)
        full_observation = np.concatenate([
            market_data, 
            position_state, 
            historical_ohlc, 
            momentum_features,
            valid_actions
        ])
        
        return full_observation
    
    def step(self, action):
        """Execute action with detailed logging"""
        
        valid_mask = self._get_valid_actions_mask()
        
        if valid_mask[action] == 0:
            # This should NEVER happen with proper masking
            logger.error(f"INVALID ACTION ATTEMPTED: {action} in state {self.position_type}")
            return self._next_observation(), -10.0, False, self._get_info()
        position_before = self.position_type
        balance_before = self.balance
        
        self.current_step += 1
        steps_in_episode = self.current_step - self.episode_start_step
        
        # Bounds checking
        if self.current_step >= len(self.df_raw) or steps_in_episode >= self.episode_length:
            self.done = True
            return self._handle_episode_end()
        
        current_price = self.df_raw.iloc[self.current_step]['close']
        prev_price = self.df_raw.iloc[self.current_step - 1]['close']
        
        reward = 0
        action_description = ""
        
        if not hasattr(self, 'action_counts'):
            self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.action_counts[action] += 1
        
        # ================================================================
        # EXECUTE ACTIONS
        # ================================================================
        
        if action == 0:  # HOLD
            reward = self._calculate_hold_reward(current_price, prev_price)
            action_description = "HOLD"
            
        elif action == 1:  # BUY/LONG
            if self.position_type == 0:  # FLAT → LONG
                success = self._open_long(current_price)
                if success:
                    action_description = f"LONG opened at ${current_price:.2f}"
                else:
                    reward -= 1.0
                    action_description = "LONG failed (insufficient balance)"
            elif self.position_type == -1:  # SHORT → LONG (flip)
                close_reward = self._close_short(current_price)
                success = self._open_long(current_price)
                if success:
                    reward += close_reward - 5.0  # Flip penalty
                    action_description = f"FLIP SHORT→LONG at ${current_price:.2f} (penalty: -5)"
                else:
                    reward += close_reward - 1.0
                    action_description = "FLIP SHORT→LONG failed on reopen"
            elif self.position_type == 1:  # LONG → LONG (invalid)
                reward -= 2.0
                action_description = "Invalid LONG (already in long position)"
                
        elif action == 2:  # SELL/CLOSE_LONG
            if self.position_type == 1:  # LONG → FLAT
                reward += self._close_long(current_price)
                action_description = f"CLOSE LONG at ${current_price:.2f}"
            else:
                reward -= 2.0
                if self.position_type == 0:
                    action_description = "Invalid CLOSE_LONG (no position to close)"
                else:
                    action_description = "Invalid CLOSE_LONG (in short position, not long)"
                
        elif action == 3:  # SHORT
            if self.position_type == 0:  # FLAT → SHORT
                success = self._open_short(current_price)
                if success:
                    action_description = f"SHORT opened at ${current_price:.2f}"
                else:
                    reward -= 1.0
                    action_description = "SHORT failed (insufficient balance)"
            elif self.position_type == 1:  # LONG → SHORT (flip)
                close_reward = self._close_long(current_price)
                success = self._open_short(current_price)
                if success:
                    reward += close_reward - 5.0  # Flip penalty
                    action_description = f"FLIP LONG→SHORT at ${current_price:.2f} (penalty: -5)"
                else:
                    reward += close_reward - 1.0
                    action_description = "FLIP LONG→SHORT failed on reopen"
            elif self.position_type == -1:  # SHORT → SHORT (invalid)
                reward -= 2.0
                action_description = "Invalid SHORT (already in short position)"
                
        elif action == 4:  # COVER/CLOSE_SHORT
            if self.position_type == -1:  # SHORT → FLAT
                reward += self._close_short(current_price)
                action_description = f"COVER SHORT at ${current_price:.2f}"
            else:
                reward -= 2.0
                if self.position_type == 0:
                    action_description = "Invalid CLOSE_SHORT (no position to close)"
                else:
                    action_description = "Invalid CLOSE_SHORT (in long position, not short)"
        
        # Calculate portfolio value
        if self.position_type == 1:
            portfolio_value = self.balance + self.btc_held * current_price
        elif self.position_type == -1:
            unrealized_cost = self.btc_held * current_price
            unrealized_pnl_dollars = self.short_value_at_entry - unrealized_cost
            portfolio_value = self.balance + unrealized_pnl_dollars
        else:
            portfolio_value = self.balance
        
        # Additional rewards
        if self.position_type != 0 and self.entry_price:
            if self.position_type == 1:
                step_return = (current_price - prev_price) / prev_price
            else:
                step_return = (prev_price - current_price) / prev_price
            reward += step_return * 0.5
        
        if self.position_type == 0:
            price_change = abs(current_price - prev_price) / prev_price
            if price_change > 0.02:
                reward -= price_change * 5
        
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        self.episode_rewards.append(reward)
        
        # LOG THIS STEP (with correct position_after based on actual position_type)
        if self.log_steps:
            # Get position names
            position_names = {-1: 'SHORT', 0: 'FLAT', 1: 'LONG'}
            
            step_info = {
                'step': steps_in_episode,
                'global_step': self.current_step,
                'action': action,
                'action_description': action_description,
                'price': float(current_price),
                'position_before': position_names.get(position_before, 'UNKNOWN'),
                'position_after': position_names.get(self.position_type, 'UNKNOWN'),  # Use ACTUAL current state
                'balance_before': float(balance_before),
                'balance_after': float(self.balance),
                'portfolio_value': float(portfolio_value),
                'reward': float(reward),
                'unrealized_pnl': float(self._get_unrealized_pnl(current_price)),
                'total_return': float((portfolio_value / self.initial_balance) - 1)
            }
            self.step_log.append(step_info)
        
        # Termination
        portfolio_return = (portfolio_value / self.initial_balance) - 1
        if portfolio_return <= -0.30:
            self.done = True
            reward -= 100
        
        return self._next_observation(), reward, self.done, self._get_info()
    
    def _get_unrealized_pnl(self, current_price):
        """Get current unrealized PnL"""
        if not self.entry_price or self.position_type == 0:
            return 0.0
        if self.position_type == 1:
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price
    
    def _open_long(self, price):
        """Open long position"""
        buy_amount = self.balance * 0.95
        fee = buy_amount * (self.transaction_fee_percent / 100)
        
        if self.balance >= (buy_amount + fee):
            self.btc_held = (buy_amount - fee) / price
            self.balance -= buy_amount
            self.entry_price = price
            self.position_open_time = self.current_step
            self.position_type = 1
            self.transaction_count += 1
            return True
        return False
    
    def _close_long(self, price):
        """Close long position"""
        sell_amount = self.btc_held * price
        fee = sell_amount * (self.transaction_fee_percent / 100)
        self.balance += (sell_amount - fee)
        
        trade_return = (price - self.entry_price) / self.entry_price
        reward = trade_return * 150
        
        if trade_return > 0.10:
            reward += 20
        elif trade_return > 0.05:
            reward += 10
        elif trade_return > 0.02:
            reward += 5
        
        if trade_return > 0:
            self.positive_trades += 1
            self.positive_long_trades += 1
        
        self.btc_held = 0
        self.transaction_count += 1
        self.total_trades += 1
        self.long_trades += 1
        self.position_type = 0
        self.entry_price = None
        self.position_open_time = None
        
        return reward
    
    def _open_short(self, price):
        """Open short position (simplified model)"""
        position_size = self.balance * 0.95
        fee = position_size * (self.transaction_fee_percent / 100)
        
        if self.balance >= (position_size + fee):
            # "Sell" BTC we don't own (borrow and sell)
            self.btc_held = position_size / price  # Amount of BTC we're short
            self.balance -= fee  # Pay fee upfront
            self.entry_price = price
            self.short_value_at_entry = position_size  # Track the USD value we got from selling
            self.position_open_time = self.current_step
            self.position_type = -1
            self.transaction_count += 1
            return True
        return False
    
    def _close_short(self, price):
        """Close short position (buy back the BTC)"""
        # Cost to buy back the BTC we borrowed
        buyback_cost = self.btc_held * price
        fee = buyback_cost * (self.transaction_fee_percent / 100)
        total_cost = buyback_cost + fee
        
        # P&L calculation:
        # We sold BTC at entry_price (got short_value_at_entry)
        # We buy back BTC at current price (costs buyback_cost + fee)
        # Profit = what we sold for - what we bought back for
        pnl = self.short_value_at_entry - total_cost
        
        # Update balance
        self.balance += pnl
        
        # Calculate return for rewards
        trade_return = (self.entry_price - price) / self.entry_price
        
        # CRITICAL: Ensure balance doesn't go negative
        if self.balance < 0:
            logger.warning(f"Short closed with negative balance: ${self.balance:.2f}, PnL: ${pnl:.2f}")
            self.balance = max(0.01, self.balance)  # Floor at $0.01 to prevent NaN
        
        reward = trade_return * 150
        
        if trade_return > 0.10:
            reward += 20
        elif trade_return > 0.05:
            reward += 10
        elif trade_return > 0.02:
            reward += 5
        
        if trade_return > 0:
            self.positive_trades += 1
            self.positive_short_trades += 1
        
        # Reset position
        self.btc_held = 0
        self.transaction_count += 1
        self.total_trades += 1
        self.short_trades += 1
        self.position_type = 0
        self.entry_price = None
        self.position_open_time = None
        
        return reward
    
    def _calculate_hold_reward(self, current_price, prev_price):
        """Calculate reward for holding"""
        reward = 0
        
        if self.position_type == 0:
            return reward
        
        if self.position_type == 1:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl = (self.entry_price - current_price) / self.entry_price
        
        time_held = self.current_step - self.position_open_time
        
        # Profit-taking penalties
        if unrealized_pnl > 0:
            base_penalty = unrealized_pnl * 0.5
            if unrealized_pnl > 0.15:
                if time_held > 75:
                    reward -= 10.0
                elif time_held > 50:
                    reward -= 3.0
                elif time_held > 25:
                    reward -= 1.0
                reward -= base_penalty
            elif unrealized_pnl > 0.10:
                if time_held > 100:
                    reward -= 5.0
                elif time_held > 60:
                    reward -= 2.0
                elif time_held > 30:
                    reward -= 0.5
                reward -= base_penalty
            elif unrealized_pnl > 0.05:
                if time_held > 150:
                    reward -= 3.0
                elif time_held > 100:
                    reward -= 1.0
                elif time_held > 50:
                    reward -= 0.3
                reward -= base_penalty * 0.5
            elif unrealized_pnl > 0.02:
                if time_held > 200:
                    reward -= 1.0
                elif time_held > 150:
                    reward -= 0.5
        
        # Loss-cutting penalties
        elif unrealized_pnl < -0.02:
            base_penalty = abs(unrealized_pnl) * 5
            reward -= base_penalty
            if unrealized_pnl < -0.20:
                reward -= 3.0
                if time_held > 20:
                    reward -= 5.0
            elif unrealized_pnl < -0.10:
                reward -= 1.0
                if time_held > 30:
                    reward -= 2.0
            elif unrealized_pnl < -0.05:
                reward -= 0.5
                if time_held > 50:
                    reward -= 1.0
            elif unrealized_pnl < -0.03:
                reward -= 0.2
                if time_held > 100:
                    reward -= 0.5
        
        return reward
    
    def _handle_episode_end(self):
        """Handle episode termination"""
        current_price = self.df_raw.iloc[min(self.current_step, len(self.df_raw)-1)]['close']
        
        if self.position_type == 1:
            portfolio_value = self.balance + self.btc_held * current_price
        elif self.position_type == -1:
            unrealized_cost = self.btc_held * current_price
            unrealized_pnl_dollars = self.short_value_at_entry - unrealized_cost
            portfolio_value = self.balance + unrealized_pnl_dollars
        else:
            portfolio_value = self.balance
        
        final_return = (portfolio_value / self.initial_balance) - 1
        reward = final_return * 50
        
        self.current_step = min(self.current_step, len(self.df_raw) - 1)
        
        return self._next_observation(), reward, self.done, self._get_info()
    
    def _get_info(self):
        """Return info"""
        current_price = self.df_raw.iloc[self.current_step]['close']
        
        if self.position_type == 1:
            portfolio_value = self.balance + self.btc_held * current_price
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price if self.entry_price else 0
        elif self.position_type == -1:
            unrealized_cost = self.btc_held * current_price
            unrealized_pnl_dollars = self.short_value_at_entry - unrealized_cost if hasattr(self, 'short_value_at_entry') else 0
            portfolio_value = self.balance + unrealized_pnl_dollars
            unrealized_pnl = (self.entry_price - current_price) / self.entry_price if self.entry_price else 0
        else:
            portfolio_value = self.balance
            unrealized_pnl = 0
        
        return {
            "portfolio_value": portfolio_value,
            "portfolio_return": (portfolio_value / self.initial_balance) - 1,
            "position_type": self.position_type,
            "btc_held": self.btc_held,
            "balance": self.balance,
            "transaction_count": self.transaction_count,
            "positive_trades": self.positive_trades,
            "total_trades": self.total_trades,
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "positive_long_trades": self.positive_long_trades,
            "positive_short_trades": self.positive_short_trades,
            "current_step": self.current_step,
            "entry_price": self.entry_price if self.entry_price else 0,
            "unrealized_pnl": unrealized_pnl
        }
    
    def get_episode_data(self):
        """Return complete episode data for logging"""
        return {
            'step_log': self.step_log,
            'action_counts': self.action_counts if hasattr(self, 'action_counts') else {},
            'final_portfolio_value': float(self._get_info()['portfolio_value']),
            'final_return': float(self._get_info()['portfolio_return']),
            'total_trades': self.total_trades,
            'long_trades': self.long_trades,
            'short_trades': self.short_trades,
            'long_win_rate': self.positive_long_trades / max(self.long_trades, 1),
            'short_win_rate': self.positive_short_trades / max(self.short_trades, 1),
            'episode_start_step': self.episode_start_step,
            'episode_length': self.current_step - self.episode_start_step
        }


class DQNAgentLongShort:
    """DQN Agent for 5-action space"""
    def __init__(self, state_size, action_size=5):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.update_target_every = 10
        
    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='huber',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        valid_actions_mask = state[0, -5:].astype(bool)
        valid_action_indices = np.where(valid_actions_mask)[0]
        
        if len(valid_action_indices) == 0:
            logger.error("No valid actions found! Defaulting to HOLD (0)")
            return 0
        
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_action_indices)
        
        act_values = self.model.predict(state, verbose=0)[0]
        masked_values = act_values.copy()
        masked_values[~valid_actions_mask] = -np.inf
        
        return np.argmax(masked_values)
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch]).reshape(batch_size, self.state_size)
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch]).reshape(batch_size, self.state_size)
        dones = np.array([i[4] for i in minibatch])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        targets = rewards + self.gamma * (np.amax(self.target_model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        
        ind = np.array([i for i in range(batch_size)])
        targets_full[[ind], [actions]] = targets
        
        self.model.fit(states, targets_full, epochs=1, verbose=0, batch_size=batch_size)
        
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, name):
        self.model.save_weights(name)
    
    def load(self, name):
        self.model.load_weights(name)


def train_agent_with_logging(env, agent, episodes, batch_size, advanced_logger):
    performance_history = []
    returns_history = []
    breakthrough_detected = False
    breakthrough_episode = None
    best_avg_50 = -float('inf')
    
    for e in tqdm(range(episodes), desc="Training Progress"):
        env.log_steps = True
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        agent.decay_epsilon()
        
        if e % agent.update_target_every == 0:
            agent.update_target_model()
        
        final_return = info['portfolio_return']
        returns_history.append(final_return)
        performance_history.append(info['portfolio_value'])
        
        episode_data = env.get_episode_data()
        advanced_logger.log_episode_details(e + 1, episode_data, threshold=0.08)
        
        if (e + 1) % 10 == 0:
            avg_return_10 = np.mean(returns_history[-10:])
            avg_return_50 = np.mean(returns_history[-50:]) if len(returns_history) >= 50 else avg_return_10
            
            if len(returns_history) >= 50:
                if avg_return_50 > best_avg_50 * 1.15 and not breakthrough_detected:
                    breakthrough_detected = True
                    breakthrough_episode = e + 1
                    advanced_logger.logger.info("\n" + "="*60)
                    advanced_logger.logger.info("BREAKTHROUGH DETECTED!")
                    advanced_logger.logger.info(f"Episode {e+1}: 50-ep avg jumped from {best_avg_50:.2%} to {avg_return_50:.2%}")
                    advanced_logger.logger.info("="*60 + "\n")
                best_avg_50 = max(best_avg_50, avg_return_50)
            
            advanced_logger.logger.info(f"\nEpisode: {e+1}/{episodes}")
            advanced_logger.logger.info(f"  Steps: {steps}, Total Reward: {total_reward:.2f}")
            advanced_logger.logger.info(f"  Final Value: ${info['portfolio_value']:.2f}, Return: {final_return:.2%}")
            advanced_logger.logger.info(f"  Avg Return (last 10): {avg_return_10:.2%}")
            advanced_logger.logger.info(f"  Overall: {info['positive_trades']}/{info['total_trades']} wins")
            advanced_logger.logger.info(f"  Long: {info['long_trades']} trades, {info['positive_long_trades']} wins")
            advanced_logger.logger.info(f"  Short: {info['short_trades']} trades, {info['positive_short_trades']} wins")
            advanced_logger.logger.info(f"  Epsilon: {agent.epsilon:.3f}")
            
            if hasattr(env, 'action_counts') and env.action_counts:
                total_actions = sum(env.action_counts.values())
                action_names = ['HOLD', 'LONG', 'CLOSE_LONG', 'SHORT', 'CLOSE_SHORT']
                advanced_logger.logger.info(f"  Actions:")
                for action_idx in range(5):
                    count = env.action_counts.get(action_idx, 0)
                    pct = count/total_actions*100 if total_actions > 0 else 0
                    advanced_logger.logger.info(f"    {action_names[action_idx]}: {count} ({pct:.1f}%)")
        
        if (e + 1) in [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            advanced_logger.logger.info("\n" + "="*60)
            advanced_logger.logger.info(f"CHECKPOINT: Episode {e+1}")
            advanced_logger.logger.info("="*60)
            
            avg_50 = np.mean(returns_history[-50:]) if len(returns_history) >= 50 else np.mean(returns_history)
            avg_100 = np.mean(returns_history[-100:]) if len(returns_history) >= 100 else avg_50
            avg_200 = np.mean(returns_history[-200:]) if len(returns_history) >= 200 else avg_100
            win_count = sum(1 for r in returns_history[-50:] if r > 0)
            win_rate = win_count / min(50, len(returns_history))
            
            advanced_logger.logger.info(f"Avg Return (last 50): {avg_50:.2%}")
            if len(returns_history) >= 100:
                advanced_logger.logger.info(f"Avg Return (last 100): {avg_100:.2%}")
            if len(returns_history) >= 200:
                advanced_logger.logger.info(f"Avg Return (last 200): {avg_200:.2%}")
            advanced_logger.logger.info(f"Win Rate (last 50): {win_rate:.2%}")
            advanced_logger.logger.info(f"Best Episode: {max(returns_history):.2%}")
            advanced_logger.logger.info(f"Worst Episode: {min(returns_history):.2%}")
            advanced_logger.logger.info(f"Epsilon: {agent.epsilon:.3f}")
            
            if len(returns_history) >= 100:
                trend_100 = avg_100 - np.mean(returns_history[-200:-100]) if len(returns_history) >= 200 else 0
                if trend_100 > 0:
                    advanced_logger.logger.info(f"Trend (last 100 vs prior 100): +{trend_100:.2%} (IMPROVING!)")
                else:
                    advanced_logger.logger.info(f"Trend (last 100 vs prior 100): {trend_100:.2%}")
            
            if avg_50 > 0.02:
                advanced_logger.logger.info("BREAKTHROUGH: Consistently profitable! (avg > 2%)")
            elif avg_50 > -0.01:
                advanced_logger.logger.info("CLOSE TO BREAKEVEN: Breakthrough likely soon!")
            elif avg_50 > -0.05:
                advanced_logger.logger.info("LEARNING: Clear improvement trend")
            
            wins_count = len([f for f in os.listdir(advanced_logger.wins_dir) if f.endswith('.json')])
            losses_count = len([f for f in os.listdir(advanced_logger.losses_dir) if f.endswith('.json')])
            advanced_logger.logger.info(f"Detailed logs: {wins_count} wins, {losses_count} losses")
            advanced_logger.logger.info("="*60 + "\n")
        
        if (e + 1) % 100 == 0:
            model_path = os.path.join(advanced_logger.run_dir, f'model_ep_{e+1}.weights.h5')
            agent.save(model_path)
            advanced_logger.logger.info(f"Model saved: {model_path}")
    
    return performance_history, returns_history, breakthrough_episode


def plot_results(returns_history, run_dir):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(returns_history)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Return (%)')
    plt.title('Portfolio Return Over Episodes')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    window = 50
    rolling_avg = pd.Series(returns_history).rolling(window=window).mean()
    plt.plot(rolling_avg)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel(f'Return % (MA{window})')
    plt.title(f'Rolling Average Return ({window} episodes)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.hist(returns_history, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Returns')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    cumulative_returns = np.cumsum(returns_history)
    plt.plot(cumulative_returns)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Return (%)')
    plt.title('Cumulative Performance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_results.png'))
    plt.close()


if __name__ == "__main__":
    advanced_logger = AdvancedLogger(base_dir="training_logs")
    logger = advanced_logger.logger
    
    csv_file_path = 'selected_features.csv'
    
    logger.info("="*60)
    logger.info("LONG/SHORT CRYPTO TRADING AGENT WITH ACTION MASKING")
    logger.info("="*60)
    logger.info(f"Run directory: {advanced_logger.run_dir}")
    logger.info("="*60 + "\n")
    
    logger.info("Loading data...")
    df_check = pd.read_csv(csv_file_path, nrows=5)
    df_check['timestamp'] = pd.to_datetime(df_check['timestamp'])
    first_timestamp = df_check['timestamp'].iloc[0]
    
    df_check_tail = pd.read_csv(csv_file_path).tail()
    df_check_tail['timestamp'] = pd.to_datetime(df_check_tail['timestamp'])
    last_timestamp = df_check_tail['timestamp'].iloc[-1]
    
    logger.info(f"First timestamp: {first_timestamp}")
    logger.info(f"Last timestamp: {last_timestamp}")
    
    start_date = first_timestamp.strftime('%Y-%m-%d')
    end_date = (first_timestamp + pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    logger.info(f"Training period: {start_date} to {end_date}")
    
    logger.info("Loading raw data...")
    df_raw = pd.read_csv(csv_file_path, index_col='timestamp', parse_dates=True)
    df_raw = df_raw.loc[start_date:end_date]
    df_raw = df_raw.sort_index()
    raw_prices = df_raw[['close', 'high', 'low', 'open']].copy()
    
    logger.info(f"Raw data: {len(df_raw)} rows")
    logger.info(f"Price range: ${df_raw['close'].min():.2f} - ${df_raw['close'].max():.2f}")
    
    logger.info("Normalizing features...")
    features = df_raw.drop(columns=['next_return', 'target'], errors='ignore')
    scaler = StandardScaler()
    normalized_array = scaler.fit_transform(features)
    
    df_normalized = pd.DataFrame(
        normalized_array,
        columns=features.columns,
        index=features.index
    )
    
    logger.info(f"Normalized data created: {len(df_normalized)} rows")
    logger.info("Data loaded and normalized\n")
    
    logger.info("="*60)
    logger.info("INITIALIZING ENVIRONMENT WITH ACTION MASKING")
    logger.info("="*60)
    logger.info("Key Features:")
    logger.info("  - 5 actions: HOLD, LONG, CLOSE_LONG, SHORT, CLOSE_SHORT")
    logger.info("  - Action masking prevents invalid actions")
    logger.info("  - No more -2.0 penalties for invalid actions")
    logger.info("  - Clean reward signal based on trading performance only")
    logger.info("  - 66-feature observation space (61 + 5 mask)")
    logger.info("="*60 + "\n")
    
    env = CryptoTradingEnvLongShort(
        df_normalized=df_normalized,
        df_raw=raw_prices,
        initial_balance=10000,
        transaction_fee_percent=0.65,
        episode_length=500,
        random_start=True,
        log_steps=True,
        lookback_window=10
    )
    
    state_size = env.observation_space.shape[0]
    action_size = 5
    
    agent = DQNAgentLongShort(state_size, action_size)
    
    logger.info(f"Agent initialized:")
    logger.info(f"  State size: {state_size} features (includes 5-element action mask)")
    logger.info(f"  Action size: {action_size}")
    logger.info(f"  Network architecture:")
    logger.info(f"    Input: {state_size} -> Dense(256) -> Dense(256) -> Dense(128) -> Output(5)")
    
    logger.info("="*60)
    logger.info("SANITY CHECK: Testing Action Masking")
    logger.info("="*60)
    
    test_state = env.reset()
    logger.info(f"Initial observation shape: {test_state.shape}")
    logger.info(f"Expected shape: ({state_size},)")
    
    action_mask = test_state[-5:]
    logger.info(f"Action mask from observation: {action_mask}")
    logger.info(f"Valid actions: {np.where(action_mask == 1)[0]} (should be [0, 1, 3] for FLAT position)")
    
    if test_state.shape[0] != state_size:
        logger.error(f"Observation size mismatch!")
        raise ValueError("Observation size mismatch!")
    
    logger.info("Observation size matches!")
    
    start_price = raw_prices.iloc[env.current_step]['close']
    logger.info(f"Starting price: ${start_price:.2f}")
    logger.info(f"Starting balance: ${env.balance:.2f}\n")
    
    logger.info("1. Testing action selection in FLAT position...")
    test_state_reshaped = np.reshape(test_state, [1, state_size])
    for i in range(5):
        action = agent.act(test_state_reshaped)
        action_names = ['HOLD', 'LONG', 'CLOSE_LONG', 'SHORT', 'CLOSE_SHORT']
        logger.info(f"   Selected action: {action} ({action_names[action]})")
    
    logger.info("\n2. Opening LONG position...")
    _, _, _, _ = env.step(1)
    test_state = env._next_observation()
    action_mask = test_state[-5:]
    logger.info(f"   Action mask after LONG: {action_mask}")
    logger.info(f"   Valid actions: {np.where(action_mask == 1)[0]} (should be [0, 2])")
    
    logger.info("\n3. Testing action selection in LONG position...")
    test_state_reshaped = np.reshape(test_state, [1, state_size])
    for i in range(5):
        action = agent.act(test_state_reshaped)
        action_names = ['HOLD', 'LONG', 'CLOSE_LONG', 'SHORT', 'CLOSE_SHORT']
        logger.info(f"   Selected action: {action} ({action_names[action]})")
    
    logger.info("="*60)
    logger.info("Action masking validation passed!")
    logger.info("="*60 + "\n")
    
    episodes = 1000
    batch_size = 64
    
    logger.info("="*60)
    logger.info("STARTING TRAINING WITH ACTION MASKING")
    logger.info("="*60)
    logger.info(f"Episodes: {episodes}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("")
    logger.info("EXPECTED IMPROVEMENTS:")
    logger.info("  - No more invalid action penalties")
    logger.info("  - Clean reward signal = trading performance")
    logger.info("  - Faster convergence (reward not corrupted)")
    logger.info("  - Agent +33% return should = +150 reward (not -774!)")
    logger.info("="*60 + "\n")
    
    performance_history, returns_history, breakthrough_ep = train_agent_with_logging(
        env, agent, episodes, batch_size, advanced_logger
    )
    
    logger.info("\nGenerating plots...")
    plot_results(returns_history, advanced_logger.run_dir)
    logger.info(f"Plots saved to: {advanced_logger.run_dir}")
    
    final_model_path = os.path.join(advanced_logger.run_dir, 'model_final.weights.h5')
    agent.save(final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    logger.info("\n" + "="*60)
    logger.info("FINAL TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total episodes: {episodes}")
    logger.info(f"Average return (all): {np.mean(returns_history):.2%}")
    logger.info(f"Average return (last 100): {np.mean(returns_history[-100:]):.2%}")
    logger.info(f"Average return (last 50): {np.mean(returns_history[-50:]):.2%}")
    logger.info(f"Best return: {np.max(returns_history):.2%}")
    logger.info(f"Worst return: {np.min(returns_history):.2%}")
    logger.info(f"Win rate (all): {np.sum(np.array(returns_history) > 0) / len(returns_history):.2%}")
    logger.info(f"Win rate (last 100): {np.sum(np.array(returns_history[-100:]) > 0) / 100:.2%}")
    logger.info(f"Final epsilon: {agent.epsilon:.3f}")
    
    if breakthrough_ep:
        logger.info(f"\nBreakthrough detected at episode: {breakthrough_ep}")
        post_breakthrough = returns_history[breakthrough_ep:]
        if len(post_breakthrough) > 0:
            logger.info(f"   Post-breakthrough avg return: {np.mean(post_breakthrough):.2%}")
    
    wins_files = [f for f in os.listdir(advanced_logger.wins_dir) if f.endswith('.json')]
    losses_files = [f for f in os.listdir(advanced_logger.losses_dir) if f.endswith('.json')]
    logger.info(f"\nDetailed episode logs:")
    logger.info(f"  Wins: {len(wins_files)} episodes")
    logger.info(f"  Losses: {len(losses_files)} episodes")
    
    logger.info(f"\nAll files saved to: {advanced_logger.run_dir}")
    logger.info("="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    
    summary_path = os.path.join(advanced_logger.run_dir, 'summary.json')
    summary = {
        'total_episodes': episodes,
        'avg_return_all': float(np.mean(returns_history)),
        'avg_return_last_100': float(np.mean(returns_history[-100:])),
        'avg_return_last_50': float(np.mean(returns_history[-50:])),
        'best_return': float(np.max(returns_history)),
        'worst_return': float(np.min(returns_history)),
        'win_rate_all': float(np.sum(np.array(returns_history) > 0) / len(returns_history)),
        'win_rate_last_100': float(np.sum(np.array(returns_history[-100:]) > 0) / 100),
        'final_epsilon': float(agent.epsilon),
        'detailed_logs_wins': len(wins_files),
        'detailed_logs_losses': len(losses_files),
        'run_directory': advanced_logger.run_dir,
        'timestamp': advanced_logger.run_timestamp,
        'action_masking_enabled': True
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary statistics saved: {summary_path}")