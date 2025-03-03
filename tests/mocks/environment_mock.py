"""
Mock classes for the environment module.
"""

import numpy as np


class MockTradingEnvironment:
    """Mock trading environment for tests."""

    def __init__(self, initial_balance=10000.0, window_size=10):
        """Initialize with balance and window size."""
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.balance = initial_balance
        self.position = 0
        self.current_step = 0
        self.done = False

    def reset(self):
        """Reset the environment."""
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.done = False
        return self._get_observation(), {}

    def step(self, action):
        """Take a step in the environment."""
        # Simulate a step in the environment
        reward = np.random.normal(0, 1)
        self.current_step += 1
        self.done = self.current_step >= 50

        return self._get_observation(), reward, self.done, False, self._get_info()

    def _get_observation(self):
        """Get the current observation."""
        return np.random.random(10)

    def _get_info(self):
        """Get info about the environment."""
        return {
            "balance": self.balance,
            "position": self.position,
            "realized_pnl": 0,
            "unrealized_pnl": 0,
        }


class MockReward:
    """Mock reward function for tests."""

    def calculate_reward(self, state, action, next_state, info):
        """Calculate reward based on state and action.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            info: Additional information

        Returns:
            Calculated reward value
        """
        return np.random.normal(0, 1)
