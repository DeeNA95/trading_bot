"""
Reward functions for the trading environment.

This module contains different reward functions that can be used to train 
reinforcement learning agents with different objectives.
"""

import numpy as np
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """Base class for reward functions."""
    
    @abstractmethod
    def calculate_reward(self, state, action, next_state, info):
        """
        Calculate the reward based on the state transition and action.
        
        Args:
            state: The current state
            action: The action taken
            next_state: The resulting state
            info: Additional information dict
            
        Returns:
            float: The calculated reward
        """
        pass


class SimpleReward(RewardFunction):
    """Simple profit and loss based reward function."""
    
    def calculate_reward(self, state, action, next_state, info):
        """
        Calculate reward based on realized and unrealized PnL.
        
        Args:
            state: The current state
            action: The action taken
            next_state: The resulting state
            info: Additional information dict containing:
                - realized_pnl: PnL from closed positions
                - unrealized_pnl: PnL from open positions
                - transaction_cost: Cost of executing the trade
        
        Returns:
            float: The calculated reward
        """
        # Extract PnL information
        realized_pnl = info.get('realized_pnl', 0)
        unrealized_pnl = info.get('unrealized_pnl', 0)
        transaction_cost = info.get('transaction_cost', 0)
        
        # Reward is the total PnL minus transaction costs
        reward = realized_pnl + 0.1 * unrealized_pnl - transaction_cost
        
        return reward


class SharpeReward(RewardFunction):
    """Sharpe ratio based reward function that balances returns and risk."""
    
    def __init__(self, window_size=20, risk_free_rate=0.0, epsilon=1e-8):
        """
        Initialize the Sharpe reward function.
        
        Args:
            window_size: Number of steps to calculate rolling Sharpe ratio
            risk_free_rate: Risk-free rate for Sharpe calculation
            epsilon: Small value to avoid division by zero
        """
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.epsilon = epsilon
        self.returns_history = []
    
    def calculate_reward(self, state, action, next_state, info):
        """
        Calculate reward based on Sharpe ratio of recent returns.
        
        Args:
            state: The current state
            action: The action taken
            next_state: The resulting state
            info: Additional information dict containing:
                - return: The return for the current step
        
        Returns:
            float: The calculated reward (Sharpe ratio)
        """
        # Get the current return
        current_return = info.get('return', 0)
        
        # Add to returns history
        self.returns_history.append(current_return)
        
        # Keep only the last window_size returns
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)
        
        # If we don't have enough history, return the current return
        if len(self.returns_history) < 2:
            return current_return
        
        # Calculate Sharpe ratio
        returns_array = np.array(self.returns_history)
        returns_mean = np.mean(returns_array)
        returns_std = np.std(returns_array)
        
        # Calculate the Sharpe ratio (add epsilon to avoid division by zero)
        sharpe = (returns_mean - self.risk_free_rate) / (returns_std + self.epsilon)
        
        # Scale the Sharpe ratio to a reasonable reward range
        reward = sharpe
        
        return reward


class RiskAdjustedReward(RewardFunction):
    """
    Advanced reward function that considers both returns and various risk factors.
    Penalizes excessive trading, high leverage, and large drawdowns.
    """
    
    def __init__(self, trade_penalty=0.0005, leverage_penalty=0.01, drawdown_penalty=0.1):
        """
        Initialize the risk-adjusted reward function.
        
        Args:
            trade_penalty: Penalty for making a trade (encourages less frequent trading)
            leverage_penalty: Penalty factor for high leverage
            drawdown_penalty: Penalty factor for drawdowns
        """
        self.trade_penalty = trade_penalty
        self.leverage_penalty = leverage_penalty
        self.drawdown_penalty = drawdown_penalty
    
    def calculate_reward(self, state, action, next_state, info):
        """
        Calculate a risk-adjusted reward.
        
        Args:
            state: The current state
            action: The action taken
            next_state: The resulting state
            info: Additional information dict containing:
                - realized_pnl: PnL from closed positions
                - unrealized_pnl: PnL from open positions
                - transaction_cost: Cost of executing the trade
                - is_trade: Whether a new trade was executed
                - leverage: Current leverage used
                - drawdown: Current drawdown percentage
        
        Returns:
            float: The calculated reward
        """
        # Extract PnL information
        realized_pnl = info.get('realized_pnl', 0)
        unrealized_pnl = info.get('unrealized_pnl', 0)
        transaction_cost = info.get('transaction_cost', 0)
        
        # Extract risk information
        is_trade = info.get('is_trade', False)
        leverage = info.get('leverage', 1.0)
        drawdown = info.get('drawdown', 0.0)
        
        # Base reward is PnL
        reward = realized_pnl + 0.1 * unrealized_pnl - transaction_cost
        
        # Apply trade penalty if a new trade was made
        if is_trade:
            reward -= self.trade_penalty
        
        # Apply leverage penalty (quadratic to penalize high leverage more)
        reward -= self.leverage_penalty * (leverage ** 2)
        
        # Apply drawdown penalty
        reward -= self.drawdown_penalty * drawdown
        
        return reward
