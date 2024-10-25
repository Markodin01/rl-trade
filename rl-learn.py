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
import time
import functools
import matplotlib.pyplot as plt
import warnings

from enum import IntEnum

class Columns(IntEnum):
    CLOSE = 0
    HIGH = 1
    LOW = 2
    OPEN = 3
    EMA_5 = 4
    BBM_5 = 5
    VOLATILITY = 6

# Set up logging
log_dir = "training_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{log_dir}/training_log_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Filter out matplotlib font warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds")
        return result
    return wrapper

def calculate_historical_volatility(df, window=20):
    """
    Calculate historical volatility using log returns
    window: number of periods (default 20 for approximately 1 month of trading days)
    """
    # Calculate log returns
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate historical volatility (annualized)
    hist_vol = log_returns.rolling(window=window).std() * np.sqrt(252)  # 252 trading days in a year
    
    return hist_vol

def prepare_data_with_volatility(df):
    """
    Prepare the dataframe with all required features including volatility
    """
    # Calculate historical volatility
    df['volatility'] = calculate_historical_volatility(df)
    
    # Calculate True Range for additional volatility insight
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Average True Range (ATR)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Volatility adjusted features
    df['vol_adjusted_range'] = (df['high'] - df['low']) / df['volatility']
    
    return df

class CryptoTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee_percent=0.1):
        super(CryptoTradingEnv, self).__init__()
        
        # Prepare dataframe with essential features and volatility
        required_columns = ['close', 'high', 'low', 'open', 'EMA_5', 'BBM_5_2.0']
        self.df = df[required_columns].copy()
        self.df['volatility'] = calculate_historical_volatility(self.df)
        
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        
        # Keep original action space
        self.action_space = spaces.Discrete(22)
        
        # Observation space including all features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(self.df.columns) + 5,),  # +5 for additional state information
            dtype=np.float32
        )
        
        # Initialize all tracking attributes
        self.balance = initial_balance
        self.btc_held = 0
        self.current_step = 0
        self.done = False
        self.hold_count = 0
        self.transaction_count = 0
        self.positive_trades = 0
        self.total_trades = 0
        self.total_profit = 0
        self.max_portfolio_value = initial_balance
        self.min_portfolio_value = initial_balance
        self.last_action = None
        self.last_trade_price = None
        self.position_open_time = None
        self.market_trend = 0
        self.volatility = 0
        self.returns = []
        self.sharpe_ratio = 0
        self.max_drawdown = 0
        
    def _calculate_sharpe_ratio(self):
        if len(self.returns) < 2:
            return 0
        
        # Calculate the average return
        avg_return = np.mean(self.returns)
        
        # Calculate the standard deviation of returns
        std_return = np.std(self.returns)
        
        # Assuming risk-free rate is 0 for simplicity
        # Annualize the Sharpe ratio (assuming daily returns)
        sharpe_ratio = (avg_return / std_return) * np.sqrt(252)
        
        return sharpe_ratio
    
    def _calculate_max_drawdown(self):
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + np.array(self.returns))
        
        # Calculate the running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate the percentage drawdown
        drawdown = (running_max - cumulative_returns) / running_max
        
        # Get the maximum drawdown
        max_drawdown = np.max(drawdown)
        
        return max_drawdown

     
    def reset(self):
        self.balance = self.initial_balance
        self.btc_held = 0
        self.current_step = 0
        self.done = False
        self.hold_count = 0
        self.transaction_count = 0
        self.positive_trades = 0
        self.total_trades = 0
        self.total_profit = 0
        self.max_portfolio_value = self.initial_balance
        self.min_portfolio_value = self.initial_balance
        self.last_action = None
        self.last_trade_price = None
        self.position_open_time = None
        self.market_trend = 0
        self.volatility = 0
        self.returns = []
        self.sharpe_ratio = 0
        self.max_drawdown = 0
        
        return self._next_observation()

     
    def _next_observation(self):
        # Get the market data features
        obs = self.df.iloc[self.current_step].values
        
        # Add additional state information
        portfolio_value = self.balance + self.btc_held * self.df.iloc[self.current_step]['close']
        
        additional_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.btc_held * self.df.iloc[self.current_step]['close'] / self.initial_balance,  # Normalized position value
            self.market_trend,  # Current market trend
            self.df.iloc[self.current_step]['volatility'],  # Current volatility
            self.hold_count / 100  # Normalized hold count
        ])

        return np.concatenate([obs, additional_state])
    

    def _update_action_space(self):
        current_price = self.df['close'].iloc[self.current_step]
        max_buy_units = min(10, int(self.balance // current_price))
        max_sell_units = min(10, int(self.btc_held))
        
        self.valid_actions = [11]  # Hold is always valid
        self.valid_actions.extend(range(11 - max_sell_units, 11))  # Sell actions
        self.valid_actions.extend(range(12, 12 + max_buy_units))  # Buy actions
        
        self.action_space = spaces.Discrete(len(self.valid_actions))

    def _calculate_position_size(self, action):
        """Calculate position size based on action and current volatility"""
        current_volatility = self.df.iloc[self.current_step]['volatility']
        base_size = abs(action - 11) / 1000  # Original calculation
        
        # Adjust size based on volatility
        volatility_factor = 1.0 / (1.0 + current_volatility)
        adjusted_size = base_size * volatility_factor
        
        return adjusted_size
     

    def step(self, action):
        self.current_step += 1
        
        if self.current_step >= len(self.df):
            self.done = True
            return self._next_observation(), 0, self.done, {}

        current_price = self.df.iloc[self.current_step]['close']
        
        # Calculate portfolio value before action
        portfolio_value_before = self.balance + self.btc_held * current_price

        if action > 11:  # Buy action
            position_size = self._calculate_position_size(action)
            buy_amount = position_size * current_price
            fee = buy_amount * (self.transaction_fee_percent / 100)
            
            if self.balance >= (buy_amount + fee):
                self.balance -= (buy_amount + fee)
                self.btc_held += position_size
                self.hold_count = 0
                self.transaction_count += 1
                self.total_trades += 1
                self.last_action = 'buy'
                self.last_trade_price = current_price
                self.position_open_time = self.current_step
                
        elif action < 11:  # Sell action
            position_size = self._calculate_position_size(action)
            if self.btc_held >= position_size:
                sell_amount = position_size * current_price
                fee = sell_amount * (self.transaction_fee_percent / 100)
                self.balance += (sell_amount - fee)
                self.btc_held -= position_size
                self.hold_count = 0
                self.transaction_count += 1
                self.total_trades += 1
                self.last_action = 'sell'
                if self.last_trade_price and current_price > self.last_trade_price:
                    self.positive_trades += 1
                self.total_profit += sell_amount - fee - (self.last_trade_price * position_size if self.last_trade_price else 0)
                
        else:  # Hold action
            self.hold_count += 1
            self.last_action = 'hold'

        # Calculate portfolio value after action
        portfolio_value_after = self.balance + self.btc_held * current_price
        
        # Update portfolio tracking
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value_after)
        self.min_portfolio_value = min(self.min_portfolio_value, portfolio_value_after)
        
        # Calculate reward with volatility adjustment
        raw_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        current_volatility = self.df.iloc[self.current_step]['volatility']
        volatility_adjusted_return = raw_return / (current_volatility if current_volatility > 0 else 1)
        
        # Add holding penalty (adjusted for volatility)
        hold_penalty = -0.01 * (self.hold_count / 100) * (1 - min(current_volatility, 0.5))
        reward = np.clip(volatility_adjusted_return + hold_penalty, -1, 1)

        # Update returns for performance metrics
        self.returns.append(reward)
        
        # Update market trend and volatility
        self.market_trend = np.sign(self.df.iloc[self.current_step]['close'] - 
                                  self.df.iloc[self.current_step-1]['close'])
        self.volatility = current_volatility

        # Check termination conditions
        portfolio_return = (portfolio_value_after / self.initial_balance) - 1

        if portfolio_return <= -0.1:  # Lost 10% or more
            self.done = True
            reward = -10
        elif portfolio_return >= 0.2:  # Gained 20% or more
            self.done = True
            reward = 10

        info = {
            "portfolio_value": portfolio_value_after,
            "portfolio_return": portfolio_return,
            "action": self.last_action,
            "units": abs(action - 11) if action != 11 else 0,
            "btc_held": self.btc_held,
            "balance": self.balance,
            "transaction_count": self.transaction_count,
            "positive_trades": self.positive_trades,
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "max_drawdown": self._calculate_max_drawdown(),
            "current_step": self.current_step,
            "volatility": current_volatility
        }

        return self._next_observation(), reward, self.done, info

class DQNAgent:
    def __init__(self, state_size, action_size=22):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

     
    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        act_values = self.model.predict(state, verbose=0)
        valid_act_values = act_values[0][valid_actions]
        return valid_actions[np.argmax(valid_act_values)]

     
    def replay(self, batch_size):
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

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save(name)

 
def fetch_and_preprocess_data(csv_file_path, start_date, end_date, window_size=10):
    try:
        logger.info(f"Loading data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path, index_col='timestamp', parse_dates=True)
        
        # Filter the dataframe for the specified date range
        df = df.loc[start_date:end_date]
        
        logger.info(f"Data loaded successfully. Processing {len(df)} data points...")

        # Ensure the data is sorted by timestamp
        df = df.sort_index()

        # Separate features and target
        features = df.drop(columns=['next_return'])
        target = df['next_return']

        # Normalize features
        scaler = StandardScaler()
        normalized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)

        # Create lagged features efficiently
        lagged_features = {}
        for col in normalized_features.columns:
            for i in range(1, window_size):
                lagged_features[f'{col}_lag_{i}'] = normalized_features[col].shift(i)
        
        # Combine all features efficiently
        df_normalized = pd.concat([normalized_features, pd.DataFrame(lagged_features), target], axis=1)

        # Drop rows with NaN values resulting from lag creation
        df_normalized = df_normalized.dropna()

        if df_normalized.empty:
            raise ValueError("DataFrame is empty after preprocessing")

        logger.info(f"Data processing complete. Final dataset contains {len(df_normalized)} rows.")
        return df_normalized

    except Exception as e:
        logger.error(f"Error in fetch_and_preprocess_data: {str(e)}")
        raise

def get_timestamped_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_dir = f"imgs/{timestamp}"
    os.makedirs(img_dir, exist_ok=True)
    return img_dir

def plot_net_balance_change(performance_history):
    episodes = len(performance_history)
    final_values = [episode[-1] for episode in performance_history]
    net_changes = [value - 10000 for value in final_values]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), net_changes, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Net Balance Change ($)')
    plt.title('Net Balance Change Over Simulation')
    plt.grid(True)
    
    img_dir = get_timestamped_dir()
    plt.savefig(f'{img_dir}/net_balance_change.png')
    plt.close()

def plot_percent_balance_change(performance_history):
    episodes = len(performance_history)
    final_values = [episode[-1] for episode in performance_history]
    percent_changes = [((value - 10000) / 10000) * 100 for value in final_values]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), percent_changes, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Percent Balance Change (%)')
    plt.title('Percent Balance Change Over Simulation')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)
    
    img_dir = get_timestamped_dir()
    plt.savefig(f'{img_dir}/percent_balance_change.png')
    plt.close()

def plot_successful_trades_percentage(successful_trades_history):
    episodes = len(successful_trades_history)
    successful_percentages = [trades[0] / trades[1] * 100 if trades[1] > 0 else 0 for trades in successful_trades_history]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), successful_percentages, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Successful Trades (%)')
    plt.title('Percentage of Successful Trades Over Simulation')
    plt.ylim(0, 100)
    plt.grid(True)
    
    img_dir = get_timestamped_dir()
    plt.savefig(f'{img_dir}/successful_trades_percentage.png')
    plt.close()

def plot_transaction_count(transaction_count_history):
    episodes = len(transaction_count_history)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), transaction_count_history, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Number of Transactions')
    plt.title('Number of Transactions Over Simulation')
    plt.grid(True)
    
    img_dir = get_timestamped_dir()
    plt.savefig(f'{img_dir}/transaction_count.png')
    plt.close()
    
 
def train_agent(env, agent, episodes, batch_size, max_span, debug=False):
    total_start_time = time.time()
    total_steps = 0
    step_times = []
    performance_history = []
    successful_trades_history = []
    transaction_count_history = []
    
    for e in tqdm(range(episodes), desc="Training Progress"):
        episode_start_time = time.time()
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        total_reward = 0
        step_count = 0
        episode_performance = [env.initial_balance]
        
        done = False
        while not done:
            step_start_time = time.time()
            
            action = agent.act(state, env.valid_actions)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            
            step_end_time = time.time()
            step_times.append(step_end_time - step_start_time)
            
            if debug and step_count % 100 == 0:
                logger.debug(f"Episode {e+1}, Step {step_count}: Action={info['action']} {info['units']} units, Reward={reward:.4f}, Done={done}")
                logger.debug(f"  Portfolio Value: ${info['portfolio_value']:.2f}, Return: {info['portfolio_return']:.2%}")
                logger.debug(f"  State: {state}")
                logger.debug(f"  Next State: {next_state}")
                logger.debug(f"  Valid Actions: {env.valid_actions}")
            
            state = next_state
            total_reward += reward
            step_count += 1
            episode_performance.append(info['portfolio_value'])
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
            if info['current_step'] > max_span:
                done = True
            
        
        agent.update_target_model()  # Update the target network after each episode
        
        total_steps += step_count
        episode_time = time.time() - episode_start_time
        
        logger.info(f"Episode: {e+1}/{episodes}, Steps: {step_count}, Total Reward: {total_reward:.4f}")
        logger.info(f"  Final Portfolio Value: ${info['portfolio_value']:.2f}, Return: {info['portfolio_return']:.2%}")
        logger.info(f"  Time: {episode_time:.2f} seconds")
        
        performance_history.append(episode_performance)
        successful_trades_history.append((info['positive_trades'], info['transaction_count']))
        transaction_count_history.append(info['transaction_count'])
        
        
        # if (e + 1) % 10 == 0:
        #     agent.save(f'crypto_trading_model_episode_{e+1}.h5')
        #     logger.info(f"Model saved at episode {e+1}")
    
    total_time = time.time() - total_start_time
    logger.info(f"Total training time: {total_time:.2f} seconds")
    logger.info(f"Average time per episode: {total_time/episodes:.2f} seconds")
    logger.info(f"Total steps executed: {total_steps}")
    logger.info(f"Average steps per episode: {total_steps/episodes:.2f}")
    logger.info(f"Average time per step: {np.mean(step_times):.6f} seconds")

    # Plot performance metrics
    plot_net_balance_change(performance_history)
    plot_percent_balance_change(performance_history)
    plot_successful_trades_percentage(successful_trades_history)
    plot_transaction_count(transaction_count_history)

def validate_csv_structure(file_path):
    """
    Validates the structure and content of the input CSV file.
    Returns tuple (is_valid, error_message)
    """
    required_columns = {
        'timestamp': str,  # Will be converted to datetime
        'open': float,
        'high': float,
        'low': float,
        'close': float,
        'volume': float,
        'price_change_1m': float,
        'price_change_5m': float,
        'volatility': float,
        'next_return': float
    }
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
            
        # Try reading the first few rows
        try:
            df = pd.read_csv(file_path, nrows=5)
        except pd.errors.EmptyDataError:
            return False, "CSV file is empty"
        except pd.errors.ParserError:
            return False, "Invalid CSV format"
            
        # Check for required columns
        missing_columns = set(required_columns.keys()) - set(df.columns)
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
            
        # Check column data types and values
        for col, expected_type in required_columns.items():
            # Skip timestamp as it needs special handling
            if col == 'timestamp':
                try:
                    pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    return False, f"Invalid timestamp format in column: {col}"
                continue
                
            # Check if column can be converted to expected type
            try:
                df[col].astype(expected_type)
            except (ValueError, TypeError):
                return False, f"Invalid data type in column: {col}. Expected {expected_type.__name__}"
                
            # Check for NaN values
            if df[col].isnull().any():
                return False, f"Found NaN values in column: {col}"
                
            # For numeric columns, check for infinite values
            if expected_type in (float, int):
                if np.isinf(df[col]).any():
                    return False, f"Found infinite values in column: {col}"
                    
        # Check if data is sorted by timestamp
        if not df['timestamp'].is_monotonic_increasing:
            return False, "Data is not sorted by timestamp"
            
        # All validations passed
        return True, "Validation successful"
        
    except Exception as e:
        return False, f"Unexpected error during validation: {str(e)}"

def validate_and_load_data(file_path, start_date, end_date, window_size=10):
    """
    Validates and loads the CSV file with proper error handling.
    Returns preprocessed DataFrame or raises exception with detailed error message.
    """
    logger.info(f"Validating CSV file: {file_path}")
    
    # Run validation
    is_valid, validation_message = validate_csv_structure(file_path)
    if not is_valid:
        logger.error(f"CSV validation failed: {validation_message}")
        raise ValueError(f"CSV validation failed: {validation_message}")
    
    logger.info("CSV validation successful. Loading data...")
    
    try:
        # Load and preprocess data using existing function
        df = fetch_and_preprocess_data(file_path, start_date, end_date, window_size)
        
        # Additional post-load validations
        if len(df) == 0:
            raise ValueError("No data available for the specified date range")
            
        if df.isnull().any().any():
            raise ValueError("Preprocessed data contains NaN values")
            
        logger.info(f"Successfully loaded {len(df)} rows of data")
        return df
        
    except Exception as e:
        logger.error(f"Error during data loading and preprocessing: {str(e)}")
        raise

def load_initial_timestamps(csv_path):
    """Separate function to load and validate timestamps"""
    try:
        df_check = pd.read_csv(csv_path, nrows=5)
        df_check['timestamp'] = pd.to_datetime(df_check['timestamp'], format='%Y-%m-%d %H:%M:%S')
        first_timestamp = df_check['timestamp'].iloc[0]
        
        df_check_tail = pd.read_csv(csv_path).tail()
        df_check_tail['timestamp'] = pd.to_datetime(df_check_tail['timestamp'])
        last_timestamp = df_check_tail['timestamp'].iloc[-1]
        
        return first_timestamp, last_timestamp
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty")
        raise
    except ValueError as e:
        logger.error(f"Invalid timestamp format in CSV: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading timestamps: {e}")
        raise

def initialize_agent_and_env(df):
    """Separate function to initialize the agent and environment"""
    try:
        env = CryptoTradingEnv(df)
        state_size = env.observation_space.shape[0]
        action_size = 22
        agent = DQNAgent(state_size, action_size)
        
        logger.info(f"Initialized environment and agent. State size: {state_size}, Action size: {action_size}")
        return env, agent
    except ValueError as e:
        logger.error(f"Invalid environment or agent parameters: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        raise

if __name__ == "__main__":
    csv_file_path = 'selected_features.csv'
    window_size = 10
    
    # Step 1: Load timestamps
    try:
        first_timestamp, last_timestamp = load_initial_timestamps(csv_file_path)
        logger.info(f"First timestamp: {first_timestamp}")
        logger.info(f"Last timestamp: {last_timestamp}")
    except Exception as e:
        logger.error("Failed to load timestamps")
        raise SystemExit(1)

    # Step 2: Set date range
    start_date = first_timestamp.strftime('%Y-%m-%d')
    end_date = (first_timestamp + pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    
    # Step 3: Load and validate data
    try:
        df = validate_and_load_data(
            file_path=csv_file_path,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size
        )
    except ValueError as e:
        logger.error(f"Data validation failed: {e}")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Unexpected error during data loading: {e}")
        raise SystemExit(1)

    # Step 4: Initialize environment and agent
    try:
        env, agent = initialize_agent_and_env(df)
    except Exception as e:
        logger.error("Failed to initialize environment and agent")
        raise SystemExit(1)

    # Step 5: Training setup
    episodes = 500
    batch_size = 64
    max_span = batch_size * 6

    # Step 6: Run training
    try:
        logger.info("Starting training...")
        train_agent(env, agent, episodes, batch_size, max_span, debug=False)
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise SystemExit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise SystemExit(1)