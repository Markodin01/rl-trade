import ccxt
import numpy as np
import pandas as pd
import time
from collections import deque
import logging
from datetime import datetime
import os

class CryptoTradingInterface:
    def __init__(self, exchange_id, symbol, agent, initial_balance=1000, stop_loss_pct=0.05, take_profit_pct=0.1):
        self.exchange = getattr(ccxt, exchange_id)()
        self.symbol = symbol
        self.agent = agent
        self.balance = initial_balance
        self.position = 0
        self.entry_price = 0
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.state_size = 7
        self.price_history = deque(maxlen=60)
        
        # Set up logging
        self.setup_logging()

    def setup_logging(self):
        log_dir = "trading_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{log_dir}/trading_log_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()

    def log_state(self, state):
        self.logger.info(f"Current State: Balance={state[0]:.2f}, Position={state[1]:.4f}, "
                         f"Price={state[2]:.2f}, 1m_Change={state[3]:.2%}, 5m_Change={state[4]:.2%}, "
                         f"Volume={state[5]:.2f}, Volatility={state[6]:.4f}")

    def log_action(self, action, q_values):
        action_names = ["Sell", "Hold", "Buy"]
        self.logger.info(f"Agent Action: {action_names[action]}")
        self.logger.info(f"Q-values: Sell={q_values[0]:.4f}, Hold={q_values[1]:.4f}, Buy={q_values[2]:.4f}")
        
        # Explain the decision
        max_q = np.max(q_values)
        if max_q == q_values[action]:
            explanation = f"The agent chose to {action_names[action]} because it had the highest Q-value of {max_q:.4f}."
            if action == 1:  # Hold
                explanation += " The agent believes holding is currently the best action given the market conditions."
            elif action == 2:  # Buy
                explanation += " The agent sees a potential upward trend or undervaluation in the current market."
            elif action == 0:  # Sell
                explanation += " The agent perceives a potential downward trend or overvaluation in the current market."
        else:
            explanation = f"The agent chose to {action_names[action]} due to exploration (epsilon-greedy policy)."
        
        self.logger.info(f"Decision Explanation: {explanation}")

    def fetch_latest_data(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1m', limit=60)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            self.logger.info(f"Fetched latest data for {self.symbol}")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return None

    def preprocess_data(self, df):
        df['price_change_1m'] = df['close'].pct_change()
        df['price_change_5m'] = df['close'].pct_change(periods=5)
        df['volatility'] = df['close'].rolling(window=60).std()
        df = df.fillna(method='ffill')
        self.logger.info("Preprocessed data successfully")
        return df.iloc[-1]

    def get_state(self, latest_data):
        state = np.array([
            self.balance,
            self.position,
            latest_data['close'],
            latest_data['price_change_1m'],
            latest_data['price_change_5m'],
            latest_data['volume'],
            latest_data['volatility']
        ])
        self.log_state(state)
        return state

    def execute_trade(self, action, current_price):
        if action == 2 and self.position == 0:  # Buy
            amount = min(self.balance, self.balance * 0.1)
            self.position = amount / current_price
            self.balance -= amount
            self.entry_price = current_price
            self.logger.info(f"Executed Buy: {self.position:.4f} {self.symbol} at {current_price:.2f}")
        elif action == 0 and self.position > 0:  # Sell
            sale_amount = self.position * current_price
            self.balance += sale_amount
            self.logger.info(f"Executed Sell: {self.position:.4f} {self.symbol} at {current_price:.2f}. Profit/Loss: {sale_amount - (self.position * self.entry_price):.2f}")
            self.position = 0
            self.entry_price = 0

    def check_stop_loss_take_profit(self, current_price):
        if self.position > 0:
            if current_price <= self.entry_price * (1 - self.stop_loss_pct):
                sale_amount = self.position * current_price
                self.balance += sale_amount
                self.logger.warning(f"Stop Loss Triggered: Sold {self.position:.4f} {self.symbol} at {current_price:.2f}. Loss: {sale_amount - (self.position * self.entry_price):.2f}")
                self.position = 0
                self.entry_price = 0
            elif current_price >= self.entry_price * (1 + self.take_profit_pct):
                sale_amount = self.position * current_price
                self.balance += sale_amount
                self.logger.info(f"Take Profit Triggered: Sold {self.position:.4f} {self.symbol} at {current_price:.2f}. Profit: {sale_amount - (self.position * self.entry_price):.2f}")
                self.position = 0
                self.entry_price = 0

    def check_arbitrage(self):
        try:
            binance_orderbook = ccxt.binance().fetch_order_book(self.symbol)
            kucoin_orderbook = ccxt.kucoin().fetch_order_book(self.symbol)

            binance_bid = binance_orderbook['bids'][0][0] if len(binance_orderbook['bids']) > 0 else None
            binance_ask = binance_orderbook['asks'][0][0] if len(binance_orderbook['asks']) > 0 else None
            kucoin_bid = kucoin_orderbook['bids'][0][0] if len(kucoin_orderbook['bids']) > 0 else None
            kucoin_ask = kucoin_orderbook['asks'][0][0] if len(kucoin_orderbook['asks']) > 0 else None

            if binance_bid and kucoin_ask and binance_bid > kucoin_ask:
                profit = (binance_bid - kucoin_ask) / kucoin_ask
                self.logger.info(f"Arbitrage Opportunity: Buy on KuCoin at {kucoin_ask:.2f}, Sell on Binance at {binance_bid:.2f}. Potential Profit: {profit:.2%}")
            elif kucoin_bid and binance_ask and kucoin_bid > binance_ask:
                profit = (kucoin_bid - binance_ask) / binance_ask
                self.logger.info(f"Arbitrage Opportunity: Buy on Binance at {binance_ask:.2f}, Sell on KuCoin at {kucoin_bid:.2f}. Potential Profit: {profit:.2%}")
            else:
                self.logger.info("No significant arbitrage opportunities found.")

        except Exception as e:
            self.logger.error(f"Error checking arbitrage: {e}")

    def run(self, runtime=3600):
        start_time = time.time()
        self.logger.info(f"Starting trading session for {self.symbol} on {self.exchange.id}")
        while time.time() - start_time < runtime:
            df = self.fetch_latest_data()
            if df is not None:
                latest_data = self.preprocess_data(df)
                current_price = latest_data['close']
                self.price_history.append(current_price)

                state = self.get_state(latest_data)
                action = self.agent.act(state.reshape(1, -1))
                q_values = self.agent.model.predict(state.reshape(1, -1))[0]
                self.log_action(action, q_values)

                self.execute_trade(action, current_price)
                self.check_stop_loss_take_profit(current_price)
                self.check_arbitrage()

                self.logger.info(f"Current Balance: {self.balance:.2f}, Position: {self.position:.4f} {self.symbol}")
                time.sleep(60)  # Wait for 1 minute before next iteration

        self.logger.info(f"Trading session ended. Final Balance: {self.balance:.2f}")

# Usage remains the same as before
exchange_id = 'binance'
symbol = 'BTC/USDT'
agent = DQNAgent(state_size=7, action_size=3)  # Assuming you have this from previous training

try:
    agent.model.load_weights('dqn_model_weights.h5')
    print("Loaded pre-trained model weights.")
except:
    print("No pre-trained model found. Using untrained model.")

trading_interface = CryptoTradingInterface(exchange_id, symbol, agent)
trading_interface.run()