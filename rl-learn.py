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
    OPEN = 0
    HIGH = 1
    LOW = 2
    CLOSE = 3
    VOLUME = 4
    PRICE_CHANGE_1M = 5
    PRICE_CHANGE_5M = 6
    VOLATILITY = 7

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

class CryptoTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee_percent=0.1):
        super(CryptoTradingEnv, self).__init__()
        self.df = df  # Keep as DataFrame instead of converting to numpy
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        
        # Define action space
        self.action_space = spaces.Discrete(22)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(df.columns) + 5,),  # +5 for additional state information
            dtype=np.float32
        )
        
        # Initialize attributes
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
        self.last_portfolio_value = self.initial_balance
        self.transaction_count = 0
        self.positive_trades = 0
        
        # New fields
        self.total_trades = 0
        self.total_profit = 0
        self.max_portfolio_value = self.initial_balance
        self.min_portfolio_value = self.initial_balance
        self.last_action = None
        self.last_trade_price = None
        self.position_open_time = None
        
        # Market state trackers
        self.market_trend = 0  # 0 for neutral, 1 for uptrend, -1 for downtrend
        self.volatility = 0
        
        # Performance metrics
        self.returns = []
        self.sharpe_ratio = 0
        self.max_drawdown = 0
        
        self._update_action_space()
        return self._next_observation()

     
    def _next_observation(self):
        # Get the market data features
        obs = self.df.iloc[self.current_step].values
        
        # Add additional state information
        additional_state = np.array([
            self.balance / self.initial_balance,
            self.btc_held * self.df.iloc[self.current_step]['close'] / self.initial_balance,
            self.market_trend,
            self.volatility,
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

     
    def step(self, action):
        self.current_step += 1
        
        if self.current_step >= len(self.df):
            self.done = True
            return self._next_observation(), 0, self.done, {}

        current_price = self.df.iloc[self.current_step]['close']
        
        # Calculate portfolio value before action
        portfolio_value_before = self.balance + self.btc_held * current_price

        if action > 11:  # Buy action
            buy_units = action - 11
            buy_amount = buy_units / 1000 * current_price
            fee = buy_amount * (self.transaction_fee_percent / 100)
            if self.balance >= (buy_amount + fee):
                self.balance -= (buy_amount + fee)
                self.btc_held += buy_units / 1000
                self.hold_count = 0
                self.transaction_count += 1
                self.last_action = 'buy'
                self.last_trade_price = current_price
                self.position_open_time = self.current_step
        elif action < 11:  # Sell action
            sell_units = action + 1
            sell_amount = (sell_units / 1000) * current_price
            if self.btc_held >= (sell_units / 1000):
                fee = sell_amount * (self.transaction_fee_percent / 100)
                self.balance += (sell_amount - fee)
                self.btc_held -= sell_units / 1000
                self.hold_count = 0
                self.transaction_count += 1
                self.last_action = 'sell'
                if self.last_trade_price and current_price > self.last_trade_price:
                    self.positive_trades += 1
                self.total_profit += sell_amount - fee - (self.last_trade_price * (sell_units / 1000) if self.last_trade_price else 0)
        else:  # Hold action
            self.hold_count += 1
            self.last_action = 'hold'

        # Calculate portfolio value after action
        portfolio_value_after = self.balance + self.btc_held * current_price
        
        # Update max and min portfolio values
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value_after)
        self.min_portfolio_value = min(self.min_portfolio_value, portfolio_value_after)
        
        # Calculate reward
        action_reward = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        hold_penalty = -0.01 * (self.hold_count / 100)  # Normalized hold penalty
        reward = np.clip(action_reward + hold_penalty, -1, 1)

        # Update returns for performance metrics
        self.returns.append(reward)
        
        # Update market trend and volatility
        self.market_trend = np.sign(self.df.iloc[self.current_step]['close'] - self.df.iloc[self.current_step-1]['close'])
        self.volatility = self.df.iloc[self.current_step]['volatility'] if 'volatility' in self.df.columns else 0

        # Check end game conditions
        portfolio_return = (portfolio_value_after / self.initial_balance) - 1

        if portfolio_return <= -0.1:  # Lost 10% or more
            self.done = True
            reward = -10  # Significant penalty for major loss
        elif portfolio_return >= 0.2:  # Gained 20% or more
            self.done = True
            reward = 10  # Significant reward for major gain

        if self.done:
            logger.info(f"Episode ended. Portfolio value: ${portfolio_value_after:.2f}, Return: {portfolio_return:.2%}")

        self.last_portfolio_value = portfolio_value_after

        # Calculate performance metrics
        self.sharpe_ratio = self._calculate_sharpe_ratio()
        self.max_drawdown = self._calculate_max_drawdown()

        self._update_action_space()

        return self._next_observation(), reward, self.done, {
            "portfolio_value": portfolio_value_after, 
            "portfolio_return": portfolio_return,
            "action": self.last_action,
            "units": abs(action - 11) if action != 11 else 0,
            "btc_held": self.btc_held,
            "balance": self.balance,
            "transaction_count": self.transaction_count,
            "positive_trades": self.positive_trades,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown
        }

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
    
 
def train_agent(env, agent, episodes, batch_size, debug=False):
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

# Update in the main execution block:
if __name__ == "__main__":
    try:
        csv_file_path = 'selected_features.csv'        
        # Read the first few rows to check the date range
        df_check = pd.read_csv(csv_file_path, nrows=5)
        df_check['timestamp'] = pd.to_datetime(df_check['timestamp'], format='%Y-%m-%d %H:%M:%S')
        first_timestamp = df_check['timestamp'].iloc[0]
        logger.info(f"First timestamp in the file: {first_timestamp}")
        
        # Read the last few rows
        df_check_tail = pd.read_csv(csv_file_path).tail()
        last_timestamp = df_check_tail['timestamp'].iloc[-1]
        logger.info(f"Last timestamp in the file: {last_timestamp}")
        
        # Adjust these dates based on the actual data range
        start_date = first_timestamp.strftime('%Y-%m-%d')
        end_date = (first_timestamp + pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        window_size = 10
        
        df = fetch_and_preprocess_data(csv_file_path, start_date, end_date, window_size)

        env = CryptoTradingEnv(df)
        state_size = env.observation_space.shape[0]
        action_size = 22  # Fixed to maximum possible actions
        agent = DQNAgent(state_size, action_size)
        logger.info(f"Initialized environment and agent. State size: {state_size}, Action size: {action_size}")

        episodes = 100
        batch_size = 32

        logger.info("Starting training...")
        train_agent(env, agent, episodes, batch_size, debug=False)  # Changed debug to False
        logger.info("Training completed.")

        # agent.save('final_crypto_trading_model.h5')
        # logger.info("Final model saved as 'final_crypto_trading_model.h5'")
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        raise  # Re-raise the exception to see the full traceback