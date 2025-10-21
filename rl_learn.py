"""
Enhanced Long/Short Crypto Trading Agent

Improvements:
- Dueling DQN architecture
- Prioritized Experience Replay (PER)
- Full dataset random sampling (no time filtering)
- Deterministic evaluation every 50 episodes
- Simplified reward structure
- Gradient clipping and LR scheduling
- Improved action masking (separated from observation)
- Better exploration strategy
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# import logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARNING)  # Suppress most logs during testing

def calculate_historical_volatility(df, window=20):
    """Calculate historical volatility using log returns"""
    log_returns = np.log(df['close'] / df['close'].shift(1))
    hist_vol = log_returns.rolling(window=window).std() * np.sqrt(252)
    return hist_vol


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    Samples important transitions more frequently
    """
    
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add experience with maximum priority"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """
        Sample batch with priority bias
        Returns: samples, indices, importance_sampling_weights
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities based on TD error"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small constant to ensure non-zero
    
    def __len__(self):
        return len(self.buffer)


class AdvancedLogger:
    """
    Advanced logging system that tracks detailed episode information
    
    Structure:
    training_logs/
    └── run_YYYYMMDD_HHMMSS/
        ├── main.log (overall training log)
        ├── evaluation_results.json
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
        
        # Evaluation tracking
        self.eval_results = []
        
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
        
        self.logger.info(f"  Detailed log saved: {filename}")
    
    def log_evaluation(self, episode_num, eval_metrics):
        """Log evaluation results"""
        self.eval_results.append({
            'episode': episode_num,
            **eval_metrics
        })
        
        eval_path = os.path.join(self.run_dir, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(self.eval_results, f, indent=2)


class CryptoTradingEnvLongShort(gym.Env):
    """
    Enhanced crypto trading environment with LONG and SHORT capabilities
    Improved reward structure and observation handling
    """
    
    def __init__(self, df_normalized, df_raw, initial_balance=10000, 
                 transaction_fee_percent=0.1, episode_length=500, 
                 random_start=True, log_steps=False, lookback_window=10):
        super(CryptoTradingEnvLongShort, self).__init__()
        
        # Data setup - calculate volatility BEFORE selecting columns
        df_raw_with_vol = df_raw.copy()
        
        # Remove duplicate timestamps (keep first occurrence)
        if df_raw_with_vol.index.duplicated().any():
            logger.warning(f"Found {df_raw_with_vol.index.duplicated().sum()} duplicate timestamps, removing...")
            df_raw_with_vol = df_raw_with_vol[~df_raw_with_vol.index.duplicated(keep='first')]
        
        df_raw_with_vol['volatility'] = calculate_historical_volatility(df_raw_with_vol)
        
        # Drop NaN values from volatility calculation
        df_raw_with_vol = df_raw_with_vol.dropna()
        
        # Now align both dataframes to the same index
        common_index = df_raw_with_vol.index
        
        required_columns = ['close', 'high', 'low', 'open', 'EMA_5', 'BBM_5_2.0']
        
        # Also remove duplicates from normalized data
        df_normalized_clean = df_normalized.copy()
        if df_normalized_clean.index.duplicated().any():
            df_normalized_clean = df_normalized_clean[~df_normalized_clean.index.duplicated(keep='first')]
        
        # Select only rows that exist in both dataframes
        common_index = common_index.intersection(df_normalized_clean.index)
        
        self.df_normalized = df_normalized_clean.loc[common_index, required_columns].copy()
        self.df_normalized['volatility'] = df_raw_with_vol.loc[common_index, 'volatility']
        
        self.df_raw = df_raw_with_vol.loc[common_index, ['close', 'high', 'low', 'open']].copy()
        
        assert len(self.df_normalized) == len(self.df_raw), "Data must be aligned"
        
        logger.info(f"Data cleaned: {len(self.df_normalized)} rows after removing duplicates and NaN values")
        
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.episode_length = episode_length
        self.random_start = random_start
        self.log_steps = log_steps
        self.lookback_window = lookback_window
        self.max_start_step = max(0, len(self.df_normalized) - episode_length - 1)
        
        # Action space: 5 actions
        self.action_space = spaces.Discrete(5)
        
        # OBSERVATION SPACE (61 features, NO action mask in observation)
        # Base features: 7 (close, high, low, open, EMA_5, BBM_5_2.0, volatility)
        # Position info: 9
        # Historical context: lookback_window × 4 (OHLC for each past candle) = 40
        # Price momentum: 5 (1h, 3h, 6h, 12h, 24h ago returns)
        total_features = 7 + 9 + (lookback_window * 4) + 5
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_features,),
            dtype=np.float32
        )
        
        logger.info(f"Observation space: {total_features} features")
        logger.info(f"  - Current candle: 7 features")
        logger.info(f"  - Position state: 9 features")
        logger.info(f"  - Historical OHLC: {lookback_window * 4} features ({lookback_window} candles)")
        logger.info(f"  - Price momentum: 5 features")
        logger.info(f"  - Action mask: SEPARATE (not in observation)")
        
        self.reset()
        
    def reset(self):
        """Reset with step logging initialization"""
        self.balance = self.initial_balance
        self.btc_held = 0
        self.position_type = 0  # -1=SHORT, 0=FLAT, 1=LONG
        self.done = False
        
        self.entry_price = None
        self.short_value_at_entry = 0
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
        self.episode_returns = []
        
        # Step-by-step logging
        self.step_log = []
        
        if self.random_start and self.max_start_step > 0:
            self.current_step = np.random.randint(0, self.max_start_step)
        else:
            self.current_step = 0
        
        self.episode_start_step = self.current_step
        
        return self._next_observation()
    
    def get_valid_actions_mask(self):
        """Return binary mask of valid actions (SEPARATE from observation)"""
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
        Construct observation (61 features, NO action mask)
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
        
        # Historical OHLC context (lookback_window × 4 features = 40)
        historical_ohlc = []
        for i in range(self.lookback_window, 0, -1):
            lookback_idx = max(0, self.current_step - i)
            past_candle = self.df_normalized.iloc[lookback_idx][['open', 'high', 'low', 'close']].values
            historical_ohlc.extend(past_candle)
        
        historical_ohlc = np.array(historical_ohlc, dtype=np.float32)
        
        # Price momentum indicators (5 features)
        momentum_features = []
        timeframes = [1, 3, 6, 12, 24]
        
        for tf in timeframes:
            lookback_idx = max(0, self.current_step - tf)
            past_price = self.df_raw.iloc[lookback_idx]['close']
            price_change = (current_price_raw - past_price) / past_price
            momentum_features.append(price_change)
        
        momentum_features = np.array(momentum_features, dtype=np.float32)
        
        # Concatenate observation (61 features total)
        full_observation = np.concatenate([
            market_data, 
            position_state, 
            historical_ohlc, 
            momentum_features
        ])
        
        return full_observation
    
    def step(self, action):
        """Execute action with simplified reward structure"""
        
        valid_mask = self.get_valid_actions_mask()
        
        if valid_mask[action] == 0:
            logger.error(f"INVALID ACTION: {action} in state {self.position_type}")
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
        
        # Execute actions
        if action == 0:  # HOLD
            reward = self._calculate_hold_reward(current_price, prev_price)
            action_description = "HOLD"
            
        elif action == 1:  # BUY/LONG
            if self.position_type == 0:
                success = self._open_long(current_price)
                if success:
                    action_description = f"LONG opened at ${current_price:.2f}"
                else:
                    reward -= 1.0
                    action_description = "LONG failed (insufficient balance)"
            elif self.position_type == -1:
                close_reward = self._close_short(current_price)
                success = self._open_long(current_price)
                if success:
                    reward += close_reward - 5.0
                    action_description = f"FLIP SHORT->LONG at ${current_price:.2f} (penalty: -5)"
                else:
                    reward += close_reward - 1.0
                    action_description = "FLIP SHORT->LONG failed on reopen"
                    
        elif action == 2:  # SELL/CLOSE_LONG
            if self.position_type == 1:
                reward += self._close_long(current_price)
                action_description = f"CLOSE LONG at ${current_price:.2f}"
            else:
                reward -= 2.0
                action_description = "Invalid CLOSE_LONG"
                
        elif action == 3:  # SHORT
            if self.position_type == 0:
                success = self._open_short(current_price)
                if success:
                    action_description = f"SHORT opened at ${current_price:.2f}"
                else:
                    reward -= 1.0
                    action_description = "SHORT failed (insufficient balance)"
            elif self.position_type == 1:
                close_reward = self._close_long(current_price)
                success = self._open_short(current_price)
                if success:
                    reward += close_reward - 5.0
                    action_description = f"FLIP LONG->SHORT at ${current_price:.2f} (penalty: -5)"
                else:
                    reward += close_reward - 1.0
                    action_description = "FLIP LONG->SHORT failed on reopen"
                    
        elif action == 4:  # COVER/CLOSE_SHORT
            if self.position_type == -1:
                reward += self._close_short(current_price)
                action_description = f"COVER SHORT at ${current_price:.2f}"
            else:
                reward -= 2.0
                action_description = "Invalid CLOSE_SHORT"
                

        
        # Calculate portfolio value
        if self.position_type == 1:
            portfolio_value = self.balance + self.btc_held * current_price
        elif self.position_type == -1:
            unrealized_cost = self.btc_held * current_price
            unrealized_pnl_dollars = self.short_value_at_entry - unrealized_cost
            portfolio_value = self.balance + unrealized_pnl_dollars
        else:
            portfolio_value = self.balance
        
        # In step() function, around line 850:
        if self.position_type != 0 and self.entry_price:
            if self.position_type == 1:
                step_return = (current_price - prev_price) / prev_price
            else:
                step_return = (prev_price - current_price) / prev_price
            reward += step_return * 500  # Increased from 200
        
        # In step() function, after the hold reward calculation:
        if self.position_type == 0:  # FLAT
            price_move_pct = abs(current_price - prev_price) / prev_price
            if price_move_pct > 0.02:  # >2% move
                reward -= price_move_pct * 200  # Strong penalty for missing big moves
        
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        self.episode_rewards.append(reward)
        
        # Track returns for Sharpe calculation
        step_return = (portfolio_value / self.initial_balance) - 1.0
        self.episode_returns.append(step_return)
        
        # Log this step
        if self.log_steps:
            position_names = {-1: 'SHORT', 0: 'FLAT', 1: 'LONG'}
            
            step_info = {
                'step': steps_in_episode,
                'global_step': self.current_step,
                'action': action,
                'action_description': action_description,
                'price': float(current_price),
                'position_before': position_names.get(position_before, 'UNKNOWN'),
                'position_after': position_names.get(self.position_type, 'UNKNOWN'),
                'balance_before': float(balance_before),
                'balance_after': float(self.balance),
                'portfolio_value': float(portfolio_value),
                'reward': float(reward),
                'unrealized_pnl': float(self._get_unrealized_pnl(current_price)),
                'total_return': float((portfolio_value / self.initial_balance) - 1)
            }
            self.step_log.append(step_info)
        
        # Termination conditions
        portfolio_return = (portfolio_value / self.initial_balance) - 1
        
        # More lenient drawdown threshold or remove it entirely during early training
        # Option 1: Increase threshold to -50%
        if portfolio_return <= -0.50:
            self.done = True
            reward -= 50  # Reduced penalty
            logger.warning(f"Episode terminated at step {steps_in_episode}: drawdown {portfolio_return:.2%}")
        
        # Option 2: No early termination, let episodes complete
        # (Commented out - uncomment to disable early termination)
        # pass
        
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
        """Open long position - use ALL capital"""
        # Calculate how much we can buy with all capital after fees
        available = self.balance
        fee_pct = self.transaction_fee_percent / 100
        
        # Amount to spend on BTC (before fee)
        buy_amount = available / (1 + fee_pct)
        fee = available - buy_amount
        
        self.btc_held = buy_amount / price
        self.balance = 0  # Use all capital
        self.entry_price = price
        self.position_open_time = self.current_step
        self.position_type = 1
        self.transaction_count += 1
        return True
    
    def _close_long(self, price):
        """Close long position"""
        # Calculate sell proceeds
        btc_value = self.btc_held * price
        fee = btc_value * (self.transaction_fee_percent / 100)
        proceeds = btc_value - fee
        
        self.balance = proceeds  # Return all to cash
        
        trade_return = (price - self.entry_price) / self.entry_price
        
        # Reward structure (keep your existing rewards)
        reward = trade_return * 2000
        
        if trade_return > 0.10:
            reward += 200
        elif trade_return > 0.05:
            reward += 100
        elif trade_return > 0.02:
            reward += 50
        
        if trade_return < -0.05:
            reward -= 50
        
        # Track stats
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
        """Open short position - use ALL capital"""
        available = self.balance
        fee_pct = self.transaction_fee_percent / 100
        
        # Amount to short (before fee)
        short_amount = available / (1 + fee_pct)
        fee = available - short_amount
        
        self.btc_held = short_amount / price  # BTC owed
        self.short_value_at_entry = short_amount  # USD value at entry
        self.balance = 0  # Use all capital
        self.entry_price = price
        self.position_open_time = self.current_step
        self.position_type = -1
        self.transaction_count += 1
        return True
    
    def _close_short(self, price):
        """Close short position"""
        # Cost to buy back the BTC
        buyback_cost = self.btc_held * price
        fee = buyback_cost * (self.transaction_fee_percent / 100)
        total_cost = buyback_cost + fee
        
        # PnL = what we sold for - what we bought back for
        pnl = self.short_value_at_entry - total_cost
        self.balance = pnl
        
        # Ensure balance doesn't go negative
        if self.balance < 0:
            logger.warning(f"Short closed with negative balance: ${self.balance:.2f}")
            self.balance = max(0.01, self.balance)
        
        trade_return = (self.entry_price - price) / self.entry_price
        
        # Reward structure
        reward = trade_return * 2000
        
        if trade_return > 0.10:
            reward += 200
        elif trade_return > 0.05:
            reward += 100
        elif trade_return > 0.02:
            reward += 50
        
        if trade_return < -0.05:
            reward -= 50
        
        if trade_return > 0:
            self.positive_trades += 1
            self.positive_short_trades += 1
        
        self.btc_held = 0
        self.transaction_count += 1
        self.total_trades += 1
        self.short_trades += 1
        self.position_type = 0
        self.entry_price = None
        self.position_open_time = None
        
        return reward
    
    def _calculate_hold_reward(self, current_price, prev_price):
        # """Simplified hold reward calculation with incentive to hold winners"""
        # reward = 0
        
        # if self.position_type == 0:
        #     # REWARD for not trading during low volatility (avoiding fees)
        #     price_change = abs(current_price - prev_price) / prev_price
        #     if price_change < 0.002:  # Less than 0.2% move
        #         reward += 0.05  # Reward patience
        #     else:
        #         # Small penalty for being flat during movement
        #         reward -= 0.05
        #     return reward
        
        # if self.position_type == 1:
        #     unrealized_pnl = (current_price - self.entry_price) / self.entry_price
        # else:
        #     unrealized_pnl = (self.entry_price - current_price) / self.entry_price
        
        # time_held = self.current_step - self.position_open_time
        
        # # REWARD holding winning positions
        # if unrealized_pnl > 0:
        #     # Small positive reward for holding winners
        #     reward += unrealized_pnl * 0.5
            
        #     # Only penalize if held WAY too long with large profit
        #     if unrealized_pnl > 0.15 and time_held > 100:
        #         reward -= 2.0  # Reduced penalty
        #     elif unrealized_pnl > 0.10 and time_held > 150:
        #         reward -= 1.0
        
        # # PENALIZE holding losing positions
        # elif unrealized_pnl < -0.02:
        #     # Stronger penalty for holding losers
        #     reward -= abs(unrealized_pnl) * 10.0  # Increased from 5.0
            
        #     # Extra penalty for holding losers too long
        #     if unrealized_pnl < -0.10 and time_held > 20:
        #         reward -= 10.0
        #     elif unrealized_pnl < -0.05 and time_held > 30:
        #         reward -= 5.0
        
        # return reward
        # Replace complex hold_reward logic with simple:
        if self.position_type == 0:  # FLAT
            # Stronger penalty for missing moves
            price_move_pct = abs(current_price - prev_price) / prev_price
            if price_move_pct > 0.01:  # >1% move
                return -5.0  # Increased from -0.2
            return -0.5  # Still penalize being flat
        
        # In position: reward/punish based on direction
        if self.position_type == 1:  # LONG
            step_pnl = (current_price - prev_price) / prev_price
        else:  # SHORT
            step_pnl = (prev_price - current_price) / prev_price
        
        return step_pnl * 500  # Increased from 200
    
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
        """Return episode info"""
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
        
        # Calculate Sharpe ratio for this episode
        sharpe_ratio = 0
        if len(self.episode_returns) > 1:
            returns_array = np.array(self.episode_returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            if std_return > 0:
                sharpe_ratio = mean_return / std_return
        
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
            "unrealized_pnl": unrealized_pnl,
            "sharpe_ratio": sharpe_ratio
        }
    
    def get_episode_data(self):
        """Return complete episode data for logging"""
        return {
            'step_log': self.step_log,
            'action_counts': self.action_counts if hasattr(self, 'action_counts') else {},
            'final_portfolio_value': float(self._get_info()['portfolio_value']),
            'final_return': float(self._get_info()['portfolio_return']),
            'sharpe_ratio': float(self._get_info()['sharpe_ratio']),
            'total_trades': self.total_trades,
            'long_trades': self.long_trades,
            'short_trades': self.short_trades,
            'long_win_rate': self.positive_long_trades / max(self.long_trades, 1),
            'short_win_rate': self.positive_short_trades / max(self.short_trades, 1),
            'episode_start_step': self.episode_start_step,
            'episode_length': self.current_step - self.episode_start_step
        }


class DuelingDQNAgent:
    """
    Dueling DQN Agent with Prioritized Experience Replay
    Separates value and advantage streams for better learning
    """
    
    def __init__(self, state_size, action_size=5, learning_rate=0.002):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.beta = 0.4  # Initial beta for importance sampling
        self.beta_increment = 0.001  # Anneal beta to 1.0
        
        self.model = self._build_dueling_model()
        self.target_model = self._build_dueling_model()
        self.update_target_model()
        self.update_target_every = 10
        
    def _build_dueling_model(self):
        """
        Build Dueling DQN architecture
        Separates state value V(s) and advantage A(s,a)
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        input_layer = tf.keras.layers.Input(shape=(self.state_size,))
        
        # Shared feature extraction layers
        shared = tf.keras.layers.Dense(256, activation='relu')(input_layer)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.Dropout(0.2)(shared)
        
        shared = tf.keras.layers.Dense(256, activation='relu')(shared)
        shared = tf.keras.layers.BatchNormalization()(shared)
        shared = tf.keras.layers.Dropout(0.2)(shared)
        
        # Value stream: V(s)
        value_stream = tf.keras.layers.Dense(128, activation='relu')(shared)
        value = tf.keras.layers.Dense(1, activation='linear', name='value')(value_stream)
        
        # Advantage stream: A(s,a)
        advantage_stream = tf.keras.layers.Dense(128, activation='relu')(shared)
        advantage = tf.keras.layers.Dense(self.action_size, activation='linear', name='advantage')(advantage_stream)
        
        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # Subtract mean for identifiability
        advantage_mean = tf.keras.layers.Lambda(
            lambda x: x - tf.reduce_mean(x, axis=1, keepdims=True)
        )(advantage)
        
        q_values = tf.keras.layers.Add(name='q_values')([value, advantage_mean])
        
        model = tf.keras.Model(inputs=input_layer, outputs=q_values)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=1.0  # Gradient clipping
        )
        
        model.compile(
            loss='huber',
            optimizer=optimizer
        )
        
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Add experience to prioritized replay buffer"""
        self.memory.add(state, action, reward, next_state, done)
    
    def act(self, state, valid_actions_mask):
        """
        Select action using epsilon-greedy with action masking
        NEW: Only trade when conviction is high (avoid fee-burning trades)
        """
        valid_action_indices = np.where(valid_actions_mask)[0]
        
        if len(valid_action_indices) == 0:
            logger.error("No valid actions found! Defaulting to HOLD (0)")
            return 0
        
        # Epsilon-greedy exploration
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_action_indices)
        
        # Greedy action selection with masking
        act_values = self.model.predict(state, verbose=0)[0]
        masked_values = np.where(valid_actions_mask, act_values, -np.inf)
            
        # REMOVE THIS ENTIRE BLOCK or lower threshold to 0.1-0.3
        # best_action = np.argmax(masked_values)
        # best_value = masked_values[best_action]
        # hold_value = masked_values[0]
        # 
        # if best_action != 0 and abs(best_value - hold_value) < 0.5:
        #     return 0
        
        return np.argmax(masked_values)  # Just take the best action!
    
    def replay(self, batch_size):
        """
        Train on batch from prioritized replay buffer
        Updates priorities based on TD error
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample from prioritized buffer
        minibatch, indices, weights = self.memory.sample(batch_size, beta=self.beta)
        
        # Extract batch data
        states = np.array([i[0] for i in minibatch]).reshape(batch_size, self.state_size)
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch]).reshape(batch_size, self.state_size)
        dones = np.array([i[4] for i in minibatch])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        # Double DQN: use main network to select action, target network to evaluate
        next_q_values_main = self.model.predict_on_batch(next_states)
        next_actions = np.argmax(next_q_values_main, axis=1)
        
        next_q_values_target = self.target_model.predict_on_batch(next_states)
        next_q_values = next_q_values_target[np.arange(batch_size), next_actions]
        
        # Compute TD targets
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Get current Q values
        targets_full = self.model.predict_on_batch(states)
        
        # Calculate TD errors for priority updates
        td_errors = np.abs(targets - targets_full[np.arange(batch_size), actions])
        
        # Update priorities in buffer
        self.memory.update_priorities(indices, td_errors)
        
        # Update Q values with TD targets
        targets_full[np.arange(batch_size), actions] = targets
        
        # Train with importance sampling weights
        self.model.fit(
            states, 
            targets_full, 
            sample_weight=weights,
            epochs=1, 
            verbose=0, 
            batch_size=batch_size
        )
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, name):
        """Save model weights"""
        self.model.save_weights(name)
    
    def load(self, name):
        """Load model weights"""
        self.model.load_weights(name)


def evaluate_agent(env, agent, n_episodes=10):
    """
    Evaluate agent performance deterministically
    Returns metrics without exploration noise
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during evaluation
    
    eval_returns = []
    eval_sharpes = []
    eval_win_rates = []
    eval_trades = []
    
    for ep in range(n_episodes):
        env.log_steps = False  # Don't log during evaluation
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        done = False
        
        while not done:
            valid_mask = env.get_valid_actions_mask()
            action = agent.act(state, valid_mask)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            state = next_state
        
        eval_returns.append(info['portfolio_return'])
        eval_sharpes.append(info['sharpe_ratio'])
        
        if info['total_trades'] > 0:
            win_rate = info['positive_trades'] / info['total_trades']
            eval_win_rates.append(win_rate)
        
        eval_trades.append(info['total_trades'])
    
    agent.epsilon = original_epsilon  # Restore epsilon
    
    return {
        'mean_return': float(np.mean(eval_returns)),
        'std_return': float(np.std(eval_returns)),
        'mean_sharpe': float(np.mean(eval_sharpes)),
        'mean_win_rate': float(np.mean(eval_win_rates)) if eval_win_rates else 0.0,
        'mean_trades': float(np.mean(eval_trades)),
        'best_return': float(np.max(eval_returns)),
        'worst_return': float(np.min(eval_returns))
    }


def train_agent_with_evaluation(env, agent, episodes, batch_size, advanced_logger, eval_every=50, warmup_episodes=100):
    """
    Training loop with periodic deterministic evaluation
    Includes warmup period to fill replay buffer before training
    """
    performance_history = []
    returns_history = []
    eval_history = []
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
            valid_mask = env.get_valid_actions_mask()
            action = agent.act(state, valid_mask)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state[0], action, reward, next_state[0], done)
            state = next_state
            total_reward += reward
            steps += 1
            
            # Only train after warmup period AND if buffer is large enough
            # Train every 4 steps instead of every step for speed
            if e >= warmup_episodes and len(agent.memory) > batch_size and steps % 4 == 0:
                agent.replay(batch_size)
        
        # Only decay epsilon after warmup
        if e >= warmup_episodes:
            agent.decay_epsilon()
        
        if e % agent.update_target_every == 0:
            agent.update_target_model()
        
        final_return = info['portfolio_return']
        returns_history.append(final_return)
        performance_history.append(info['portfolio_value'])
        
        episode_data = env.get_episode_data()
        advanced_logger.log_episode_details(e + 1, episode_data, threshold=0.08)
        
        # Periodic evaluation
        if (e + 1) % eval_every == 0:
            eval_metrics = evaluate_agent(env, agent, n_episodes=10)
            eval_history.append({
                'episode': e + 1,
                **eval_metrics
            })
            advanced_logger.log_evaluation(e + 1, eval_metrics)
            
            advanced_logger.logger.info("\n" + "="*60)
            advanced_logger.logger.info(f"EVALUATION at Episode {e+1}")
            advanced_logger.logger.info("="*60)
            advanced_logger.logger.info(f"  Deterministic Performance (10 episodes):")
            advanced_logger.logger.info(f"    Mean Return: {eval_metrics['mean_return']:.2%}")
            advanced_logger.logger.info(f"    Std Return: {eval_metrics['std_return']:.2%}")
            advanced_logger.logger.info(f"    Mean Sharpe: {eval_metrics['mean_sharpe']:.3f}")
            advanced_logger.logger.info(f"    Mean Win Rate: {eval_metrics['mean_win_rate']:.2%}")
            advanced_logger.logger.info(f"    Mean Trades: {eval_metrics['mean_trades']:.1f}")
            advanced_logger.logger.info(f"    Best: {eval_metrics['best_return']:.2%} | Worst: {eval_metrics['worst_return']:.2%}")
            advanced_logger.logger.info("="*60 + "\n")
        
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
            advanced_logger.logger.info(f"  Sharpe Ratio: {info['sharpe_ratio']:.3f}")
            advanced_logger.logger.info(f"  Avg Return (last 10): {avg_return_10:.2%}")
            advanced_logger.logger.info(f"  Overall: {info['positive_trades']}/{info['total_trades']} wins")
            advanced_logger.logger.info(f"  Long: {info['long_trades']} trades, {info['positive_long_trades']} wins")
            advanced_logger.logger.info(f"  Short: {info['short_trades']} trades, {info['positive_short_trades']} wins")
            advanced_logger.logger.info(f"  Epsilon: {agent.epsilon:.3f}, Beta: {agent.beta:.3f}")
            
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
    
    return performance_history, returns_history, eval_history, breakthrough_episode


def plot_results(returns_history, eval_history, run_dir):
    """Generate comprehensive training visualizations"""
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Episode returns
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(returns_history, alpha=0.6, linewidth=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Portfolio Return Over Episodes')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rolling average
    ax2 = plt.subplot(2, 3, 2)
    window = 50
    rolling_avg = pd.Series(returns_history).rolling(window=window).mean()
    ax2.plot(rolling_avg, linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel(f'Return % (MA{window})')
    ax2.set_title(f'Rolling Average Return ({window} episodes)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Return distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(returns_history, bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Returns')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative returns
    ax4 = plt.subplot(2, 3, 4)
    cumulative_returns = np.cumsum(returns_history)
    ax4.plot(cumulative_returns, linewidth=2)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Cumulative Return (%)')
    ax4.set_title('Cumulative Performance')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Evaluation metrics
    if eval_history:
        ax5 = plt.subplot(2, 3, 5)
        eval_episodes = [e['episode'] for e in eval_history]
        eval_returns = [e['mean_return'] for e in eval_history]
        eval_sharpes = [e['mean_sharpe'] for e in eval_history]
        
        ax5_twin = ax5.twinx()
        line1 = ax5.plot(eval_episodes, eval_returns, 'b-o', label='Mean Return', linewidth=2)
        line2 = ax5_twin.plot(eval_episodes, eval_sharpes, 'g-s', label='Sharpe Ratio', linewidth=2)
        
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Mean Return (%)', color='b')
        ax5_twin.set_ylabel('Sharpe Ratio', color='g')
        ax5.set_title('Evaluation Performance')
        ax5.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='upper left')
    
    # Plot 6: Win rate over time
    ax6 = plt.subplot(2, 3, 6)
    window_wr = 50
    win_indicators = [1 if r > 0 else 0 for r in returns_history]
    rolling_wr = pd.Series(win_indicators).rolling(window=window_wr).mean()
    ax6.plot(rolling_wr, linewidth=2)
    ax6.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='50% Win Rate')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Win Rate')
    ax6.set_title(f'Rolling Win Rate ({window_wr} episodes)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_results.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    advanced_logger = AdvancedLogger(base_dir="training_logs")
    logger = advanced_logger.logger
    
    csv_file_path = 'selected_features.csv'
    
    logger.info("="*60)
    logger.info("ENHANCED LONG/SHORT CRYPTO TRADING AGENT")
    logger.info("="*60)
    logger.info("Improvements:")
    logger.info("  - Dueling DQN architecture")
    logger.info("  - Prioritized Experience Replay")
    logger.info("  - Full dataset random sampling")
    logger.info("  - Deterministic evaluation every 50 episodes")
    logger.info("  - Gradient clipping & improved stability")
    logger.info(f"Run directory: {advanced_logger.run_dir}")
    logger.info("="*60 + "\n")
    
    logger.info("Loading data for training...")
    df_raw_full = pd.read_csv(csv_file_path, index_col='timestamp', parse_dates=True)
    df_raw_full = df_raw_full.sort_index()
    
    # Check data frequency
    time_diffs = df_raw_full.index.to_series().diff()
    median_interval = time_diffs.median()
    logger.info(f"Median time between candles: {median_interval}")
    logger.info(f"Approximate candles per day: {pd.Timedelta('1D') / median_interval:.1f}")
    
    # FOCUS ON A SPECIFIC TIME PERIOD FOR FASTER LEARNING
    # Option 1: Recent bull market (2020-2021) - easier to learn
    train_start = '2020-01-01'
    train_end = '2021-12-31'
    
    # Option 2: More stable period (2019-2020)
    # train_start = '2019-01-01'
    # train_end = '2020-12-31'
    
    # Option 3: Full dataset (current - very hard!)
    # train_start = df_raw_full.index[0]
    # train_end = df_raw_full.index[-1]
    
    df_raw = df_raw_full.loc[train_start:train_end].copy()
    
    logger.info(f"\nDataset info:")
    logger.info(f"  Training period: {train_start} to {train_end}")
    logger.info(f"  Total candles: {len(df_raw)}")
    logger.info(f"  Date range: {df_raw.index[0]} to {df_raw.index[-1]}")
    logger.info(f"  Total days: {(df_raw.index[-1] - df_raw.index[0]).days}")
    logger.info(f"  Price range: ${df_raw['close'].min():.2f} - ${df_raw['close'].max():.2f}")
    
    # Check if indicators exist, if not calculate them
    required_indicators = ['EMA_5', 'BBM_5_2.0']
    missing_indicators = [ind for ind in required_indicators if ind not in df_raw.columns]
    
    if missing_indicators:
        logger.info(f"\nCalculating missing indicators: {missing_indicators}")
        # Calculate EMA
        if 'EMA_5' not in df_raw.columns:
            df_raw['EMA_5'] = df_raw['close'].ewm(span=5, adjust=False).mean()
        
        # Calculate Bollinger Bands
        if 'BBM_5_2.0' not in df_raw.columns:
            rolling_mean = df_raw['close'].rolling(window=5).mean()
            rolling_std = df_raw['close'].rolling(window=5).std()
            df_raw['BBM_5_2.0'] = rolling_mean
            df_raw['BBU_5_2.0'] = rolling_mean + (rolling_std * 2)
            df_raw['BBL_5_2.0'] = rolling_mean - (rolling_std * 2)
        
        # Drop NaN values from indicator calculation
        df_raw = df_raw.dropna()
        logger.info(f"After indicator calculation: {len(df_raw)} candles")
    
    raw_prices = df_raw[['close', 'high', 'low', 'open']].copy()
    
    logger.info("\nNormalizing features...")
    features = df_raw[['close', 'high', 'low', 'open', 'EMA_5', 'BBM_5_2.0']]
    scaler = StandardScaler()
    normalized_array = scaler.fit_transform(features)
    
    df_normalized = pd.DataFrame(
        normalized_array,
        columns=features.columns,
        index=features.index
    )
    
    logger.info(f"Normalized data created: {len(df_normalized)} rows")
    logger.info(f"Episode length: 500 steps")
    logger.info(f"Max possible starting positions: {len(df_normalized) - 500}")
    logger.info("Each episode will sample a random 500-step window\n")
    
    logger.info("="*60)
    logger.info("INITIALIZING ENVIRONMENT")
    logger.info("="*60)
    
    env = CryptoTradingEnvLongShort(
        df_normalized=df_normalized,
        df_raw=raw_prices,
        initial_balance=10000,
        transaction_fee_percent=0.01,  # Reduced from 0.1% to 0.02% (more realistic maker fees)
        episode_length=500,
        random_start=True,
        log_steps=True,
        lookback_window=10
    )
    
    state_size = env.observation_space.shape[0]
    action_size = 5
    
    logger.info(f"\nAgent configuration:")
    logger.info(f"  State size: {state_size} features")
    logger.info(f"  Action size: {action_size}")
    logger.info(f"  Architecture: Dueling DQN")
    logger.info(f"    Input({state_size}) -> Dense(256) -> Dense(256)")
    logger.info(f"    -> Value Stream: Dense(128) -> V(s)")
    logger.info(f"    -> Advantage Stream: Dense(128) -> A(s,a)")
    logger.info(f"    -> Output: Q(s,a) = V(s) + (A(s,a) - mean(A))")
    logger.info(f"  Memory: Prioritized Experience Replay (10000 capacity)")
    logger.info(f"  Gamma: 0.99, LR: 0.0005, Gradient Clipping: 1.0")
    
    agent = DuelingDQNAgent(state_size, action_size, learning_rate=0.002)  # Increased from 0.0005
    
    logger.info("\n" + "="*60)
    logger.info("SANITY CHECK: Testing System")
    logger.info("="*60)
    
    test_state = env.reset()
    logger.info(f"Initial observation shape: {test_state.shape}")
    logger.info(f"Expected shape: ({state_size},)")
    
    if test_state.shape[0] != state_size:
        logger.error(f"Observation size mismatch!")
        raise ValueError("Observation size mismatch!")
    
    valid_mask = env.get_valid_actions_mask()
    logger.info(f"Valid actions mask: {valid_mask}")
    logger.info(f"Valid actions: {np.where(valid_mask == 1)[0]} (should be [0, 1, 3])")
    
    start_price = raw_prices.iloc[env.current_step]['close']
    logger.info(f"Starting price: ${start_price:.2f}")
    logger.info(f"Starting balance: ${env.balance:.2f}")
    
    logger.info("\nTesting action selection...")
    test_state_reshaped = np.reshape(test_state, [1, state_size])
    for i in range(5):
        action = agent.act(test_state_reshaped, valid_mask)
        action_names = ['HOLD', 'LONG', 'CLOSE_LONG', 'SHORT', 'CLOSE_SHORT']
        logger.info(f"   Selected action: {action} ({action_names[action]})")
    
    logger.info("\nOpening LONG position...")
    _, _, _, _ = env.step(1)
    test_state = env._next_observation()
    valid_mask = env.get_valid_actions_mask()
    logger.info(f"   Action mask after LONG: {valid_mask}")
    logger.info(f"   Valid actions: {np.where(valid_mask == 1)[0]} (should be [0, 2])")
    
    logger.info("\nTesting action selection in LONG position...")
    test_state_reshaped = np.reshape(test_state, [1, state_size])
    for i in range(5):
        action = agent.act(test_state_reshaped, valid_mask)
        action_names = ['HOLD', 'LONG', 'CLOSE_LONG', 'SHORT', 'CLOSE_SHORT']
        logger.info(f"   Selected action: {action} ({action_names[action]})")
    
    logger.info("\n" + "="*60)
    logger.info("System validation passed!")
    logger.info("="*60 + "\n")
    
    episodes = 1000
    batch_size = 32  # Reduced from 64 to 32 for faster training
    warmup_episodes = 100  # Fill replay buffer for 100 episodes before training
    
    logger.info("="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    logger.info(f"Episodes: {episodes}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Warmup episodes: {warmup_episodes} (pure exploration, no training)")
    logger.info(f"Evaluation: Every 50 episodes (10 deterministic episodes)")
    logger.info("")
    logger.info("KEY FEATURES:")
    logger.info("  - Dueling DQN separates value and advantage")
    logger.info("  - Prioritized replay focuses on important transitions")
    logger.info("  - Full dataset ensures diverse market exposure")
    logger.info("  - Action masking prevents invalid actions")
    logger.info("  - Gradient clipping improves stability")
    logger.info("  - Volatility-aware position sizing")
    logger.info("  - Reduced transaction fees (0.1%) for faster learning")
    logger.info("  - More lenient drawdown threshold (-50%)")
    logger.info("="*60 + "\n")
    
    performance_history, returns_history, eval_history, breakthrough_ep = train_agent_with_evaluation(
        env, agent, episodes, batch_size, advanced_logger, eval_every=50, warmup_episodes=warmup_episodes
    )
    
    logger.info("\nGenerating plots...")
    plot_results(returns_history, eval_history, advanced_logger.run_dir)
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
    logger.info(f"Final beta (PER): {agent.beta:.3f}")
    
    if breakthrough_ep:
        logger.info(f"\nBreakthrough detected at episode: {breakthrough_ep}")
        post_breakthrough = returns_history[breakthrough_ep:]
        if len(post_breakthrough) > 0:
            logger.info(f"   Post-breakthrough avg return: {np.mean(post_breakthrough):.2%}")
    
    if eval_history:
        logger.info("\nEvaluation Summary (Deterministic Performance):")
        final_eval = eval_history[-1]
        logger.info(f"  Final Mean Return: {final_eval['mean_return']:.2%}")
        logger.info(f"  Final Mean Sharpe: {final_eval['mean_sharpe']:.3f}")
        logger.info(f"  Final Win Rate: {final_eval['mean_win_rate']:.2%}")
        
        best_eval = max(eval_history, key=lambda x: x['mean_return'])
        logger.info(f"\n  Best Evaluation (Episode {best_eval['episode']}):")
        logger.info(f"    Mean Return: {best_eval['mean_return']:.2%}")
        logger.info(f"    Mean Sharpe: {best_eval['mean_sharpe']:.3f}")
    
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
        'final_beta': float(agent.beta),
        'detailed_logs_wins': len(wins_files),
        'detailed_logs_losses': len(losses_files),
        'run_directory': advanced_logger.run_dir,
        'timestamp': advanced_logger.run_timestamp,
        'architecture': 'Dueling DQN',
        'prioritized_replay': True,
        'action_masking_enabled': True,
        'full_dataset_sampling': True,
        'evaluation_results': eval_history
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary statistics saved: {summary_path}")
    
    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS:")
    logger.info("="*60)
    logger.info("1. Review evaluation results in evaluation_results.json")
    logger.info("2. Analyze extreme episodes in wins/ and losses/ folders")
    logger.info("3. Examine training_results.png for visual insights")
    logger.info("4. If performance is good, consider:")
    logger.info("   - Extending training (more episodes)")
    logger.info("   - Testing on out-of-sample data")
    logger.info("   - Implementing curriculum learning")
    logger.info("   - Adding more features (volume, order book)")
    logger.info("5. If performance needs improvement:")
    logger.info("   - Adjust reward structure")
    logger.info("   - Tune hyperparameters")
    logger.info("   - Increase network capacity")
    logger.info("="*60)