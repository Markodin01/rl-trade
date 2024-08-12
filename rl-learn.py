import gym
from gym import spaces
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
import random
import ccxt
import logging
from datetime import datetime
import os
from tqdm import tqdm
import time
import functools
import matplotlib.pyplot as plt
import warnings

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
        self.df = df.to_numpy()
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.action_space = spaces.Discrete(22)  # This will be dynamically adjusted
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.hold_count = 0
        self.transaction_count = 0
        self.positive_trades = 0
        self.reset()

     
    def reset(self):
        self.balance = self.initial_balance
        self.btc_held = 0
        self.current_step = 0
        self.done = False
        self.hold_count = 0
        self.last_portfolio_value = self.initial_balance
        self.transaction_count = 0
        self.positive_trades = 0
        self._update_action_space()
        return self._next_observation()

     
    def _next_observation(self):
        return np.array([
            self.balance,
            self.btc_held,
            self.df[self.current_step, 4],  # close price
            self.df[self.current_step, 5],  # price_change_1m
            self.df[self.current_step, 6],  # price_change_5m
            self.df[self.current_step, 5],  # volume
            self.df[self.current_step, 7],  # volatility
        ], dtype=np.float32)

    def _update_action_space(self):
        current_price = self.df[self.current_step, 4]
        max_buy_units = min(10, int(self.balance // current_price))
        max_sell_units = min(10, int(self.btc_held))
        
        self.valid_actions = [11]  # Hold is always valid
        self.valid_actions.extend(range(11 - max_sell_units, 11))  # Sell actions
        self.valid_actions.extend(range(12, 12 + max_buy_units))  # Buy actions
        
        self.action_space = spaces.Discrete(len(self.valid_actions))

     
    def step(self, action):
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            self.done = True

        current_price = self.df[self.current_step, 4]  # close price
        
        # Calculate portfolio value before action
        portfolio_value_before = self.balance + self.btc_held * current_price

        if action > 11:  # Buy action
            buy_units = action - 11
            buy_amount = buy_units * current_price
            fee = buy_amount * (self.transaction_fee_percent / 100)
            self.balance -= (buy_amount + fee)
            self.btc_held += buy_units
            self.hold_count = 0
            self.transaction_count += 1
        elif action < 11:  # Sell action
            sell_units = action + 1
            sell_amount = sell_units * current_price
            fee = sell_amount * (self.transaction_fee_percent / 100)
            self.balance += (sell_amount - fee)
            self.btc_held -= sell_units
            self.hold_count = 0
            self.transaction_count += 1
        else:  # Hold action
            self.hold_count += 1

        # Calculate portfolio value after action
        portfolio_value_after = self.balance + self.btc_held * current_price
        
        # Calculate normalized reward
        action_reward = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        hold_penalty = -0.01 * (self.hold_count / 100)  # Normalized hold penalty
        
        # Normalize reward to be between -1 and 1
        reward = np.clip(action_reward + hold_penalty, -1, 1)

        if action != 11 and portfolio_value_after > portfolio_value_before:
            self.positive_trades += 1

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

        self._update_action_space()
        next_obs = self._next_observation()
        
        self.last_portfolio_value = portfolio_value_after

        return next_obs, reward, self.done, {
            "portfolio_value": portfolio_value_after, 
            "portfolio_return": portfolio_return,
            "action": "Buy" if action > 11 else ("Sell" if action < 11 else "Hold"),
            "units": abs(action - 11) if action != 11 else 0,
            "transaction_count": self.transaction_count,
            "positive_trades": self.positive_trades
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
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
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

 
def fetch_and_preprocess_data(exchange_id, symbol, timeframe='1h', limit=1000):
    try:
        logger.info(f"Fetching data from {exchange_id} for {symbol}...")
        exchange = getattr(ccxt, exchange_id)()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        logger.info(f"Data fetched successfully. Processing {len(ohlcv)} data points...")
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        df['price_change_1m'] = df['close'].pct_change()
        df['price_change_5m'] = df['close'].pct_change(periods=5)
        df['volatility'] = df['close'].rolling(window=20).std()
        df = df.dropna()
        
        if df.empty:
            raise ValueError("DataFrame is empty after preprocessing")
        
        logger.info(f"Data processing complete. Final dataset contains {len(df)} rows.")
        return df
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
        state = np.reshape(state, [1, 7])
        total_reward = 0
        step_count = 0
        episode_performance = [env.initial_balance]
        
        done = False
        while not done:
            step_start_time = time.time()
            
            action = agent.act(state, env.valid_actions)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, 7])
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
        exchange_id = 'binance'
        symbol = 'BTC/USDT'
        df = fetch_and_preprocess_data(exchange_id, symbol)

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