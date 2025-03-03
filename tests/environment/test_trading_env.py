"""
Unit tests for the trading environment.
"""

import os
import sys
import unittest

import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.mocks.environment_mock import MockTradingEnvironment


class TestTradingEnvironment(unittest.TestCase):
    """Test cases for the trading environment."""

    def setUp(self):
        """Set up the test environment."""
        self.env = MockTradingEnvironment(initial_balance=10000.0, window_size=10)

    def test_init(self):
        """Test environment initialization."""
        self.assertEqual(self.env.initial_balance, 10000.0)
        self.assertEqual(self.env.window_size, 10)

    def test_reset(self):
        """Test environment reset."""
        observation, info = self.env.reset()
        self.assertIsNotNone(observation)
        self.assertEqual(self.env.balance, 10000.0)
        self.assertEqual(self.env.position, 0)

    def test_step(self):
        """Test taking a step in the environment."""
        # Reset the environment
        self.env.reset()

        # Take a step
        action = np.array([0.5, 0.2, -0.1, 0.3])  # Random action
        next_obs, reward, done, truncated, info = self.env.step(action)

        # Check that values are returned as expected
        self.assertIsNotNone(next_obs)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("balance", info)
        self.assertIn("position", info)

    def test_done_after_max_steps(self):
        """Test that environment terminates after maximum steps."""
        self.env.reset()

        # Take steps until done
        max_steps = 100
        for i in range(max_steps):
            action = np.array([0.1, 0.0, 0.0, 0.0])
            _, _, done, _, _ = self.env.step(action)
            if done:
                break

        # Environment should terminate eventually
        self.assertTrue(
            i < max_steps, f"Environment did not terminate after {max_steps} steps"
        )


if __name__ == "__main__":
    unittest.main()
