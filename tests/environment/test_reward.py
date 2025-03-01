"""
Unit tests for reward functions.
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests.mocks.environment_mock import MockReward


class TestRewardFunctions(unittest.TestCase):
    """Test cases for reward calculation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reward_fn = MockReward()
        
    def test_reward_calculation(self):
        """Test reward calculation with typical inputs."""
        # Create test inputs
        state = np.array([1.0, 2.0, 3.0])
        action = np.array([0.5, 0.2, 0.1, 0.3])
        next_state = np.array([1.1, 2.2, 3.3])
        info = {
            'realized_pnl': 100.0,
            'unrealized_pnl': 50.0,
            'transaction_cost': 10.0
        }
        
        # Calculate reward
        reward = self.reward_fn.calculate_reward(state, action, next_state, info)
        
        # Verify reward is a number
        self.assertIsInstance(reward, float)
        
    def test_reward_negative_pnl(self):
        """Test reward calculation with negative P&L."""
        # Create test inputs with negative P&L
        state = np.array([1.0, 2.0, 3.0])
        action = np.array([0.5, 0.2, 0.1, 0.3])
        next_state = np.array([0.9, 1.8, 2.7])
        info = {
            'realized_pnl': -50.0,
            'unrealized_pnl': -20.0,
            'transaction_cost': 10.0
        }
        
        # Calculate reward
        reward = self.reward_fn.calculate_reward(state, action, next_state, info)
        
        # Verify reward is a number
        self.assertIsInstance(reward, float)
        
    def test_reward_zero_pnl(self):
        """Test reward calculation with zero P&L."""
        # Create test inputs with zero P&L
        state = np.array([1.0, 2.0, 3.0])
        action = np.array([0.5, 0.2, 0.1, 0.3])
        next_state = np.array([1.0, 2.0, 3.0])
        info = {
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'transaction_cost': 5.0
        }
        
        # Calculate reward
        reward = self.reward_fn.calculate_reward(state, action, next_state, info)
        
        # Verify reward is a number
        self.assertIsInstance(reward, float)


if __name__ == '__main__':
    unittest.main()
