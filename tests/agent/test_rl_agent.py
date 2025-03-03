"""
Unit tests for the reinforcement learning agent.
"""

import os
import sys
import unittest

import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.mocks.agent_mock import MockAgent
from tests.mocks.environment_mock import MockTradingEnvironment


class TestRLAgent(unittest.TestCase):
    """Tests for the reinforcement learning agent."""

    def setUp(self):
        """Set up test fixtures."""
        self.env = MockTradingEnvironment()
        self.agent = MockAgent(input_dim=10, output_dim=4)

    def test_choose_action(self):
        """Test the agent can choose an action given an observation."""
        # Get an observation from the environment
        observation, _ = self.env.reset()

        # Choose an action
        action, log_probs, value = self.agent.choose_action(observation)

        # Check that the action is of the expected shape and type
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(len(action), 4)

        # Check log probabilities and value
        self.assertIsInstance(log_probs, np.ndarray)
        self.assertIsInstance(value, np.ndarray)

    def test_learn(self):
        """Test the agent can learn from experiences."""
        # Run a few steps to generate experiences
        observation, _ = self.env.reset()

        for _ in range(5):
            action, log_probs, value = self.agent.choose_action(observation)
            next_observation, reward, done, _, info = self.env.step(action)

            # Store experience in agent memory
            self.agent.memory.store_memory(
                observation, action, log_probs, value, reward, done
            )

            observation = next_observation

        # Run learning step
        loss_info = self.agent.learn()

        # Check that loss info is returned
        self.assertIn("actor_loss", loss_info)
        self.assertIn("critic_loss", loss_info)
        self.assertIn("total_loss", loss_info)

    def test_save_load_models(self):
        """Test the agent can save and load models."""
        # Make sure the save method doesn't crash
        try:
            self.agent.save_models("test_model")
        except Exception as e:
            self.fail(f"save_models raised exception: {e}")

        # Make sure the load method doesn't crash
        try:
            self.agent.load_models("test_model")
        except Exception as e:
            self.fail(f"load_models raised exception: {e}")


if __name__ == "__main__":
    unittest.main()
