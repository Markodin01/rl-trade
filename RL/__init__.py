"""
Confidence-Gated RL Trading System
Exploiting Power Law Dynamics in Cryptocurrency Markets

A VC-style approach to tail-event capture using
Self-Organized Criticality detection and Reinforcement Learning
"""

__version__ = "0.2.0"
__author__ = "rl-trade"

# Core components
from .env import CryptoTradingEnvLongShort
from .agent import DuelingDQNAgent
from .utils import AdvancedLogger

__all__ = [
    'CryptoTradingEnvLongShort',
    'DuelingDQNAgent',
    'AdvancedLogger',
]
