"""
Integration tests for the agent and environment interaction.
"""

import os
import sys
import unittest

import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.mocks.agent_mock import MockAgent
from tests.mocks.environment_mock import MockTradingEnvironment


class TestAgentEnvironmentIntegration(unittest.TestCase):
    """Tests for the interaction between agent and environment."""

    def setUp(self):
        """Set up test fixtures."""
        self.env = MockTradingEnvironment(initial_balance=10000.0, window_size=10)
        self.agent = MockAgent(input_dim=10, output_dim=4)

    def test_single_step_interaction(self):
        """Test a single step interaction between agent and environment."""
        # Reset environment
        observation, _ = self.env.reset()

        # Agent chooses action
        action, _, _ = self.agent.choose_action(observation)

        # Environment takes step with action
        next_observation, reward, done, _, info = self.env.step(action)

        # Check that we got back valid responses
        self.assertIsNotNone(next_observation)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_episode_interaction(self):
        """Test interaction over a full episode."""
        # Reset environment
        observation, _ = self.env.reset()

        # Run for a few steps
        episode_rewards = []
        for _ in range(10):
            # Agent chooses action
            action, log_probs, value = self.agent.choose_action(observation)

            # Environment takes step
            next_observation, reward, done, _, info = self.env.step(action)

            # Store experience
            self.agent.memory.store_memory(
                observation, action, log_probs, value, reward, done
            )

            # Update observation
            observation = next_observation
            episode_rewards.append(reward)

            # Break if episode is done
            if done:
                break

        # Check that rewards were collected
        self.assertTrue(len(episode_rewards) > 0)

        # Train the agent
        loss_info = self.agent.learn()

        # Check that loss info was returned
        self.assertIn("actor_loss", loss_info)
        self.assertIn("critic_loss", loss_info)

    def test_multiple_episodes(self):
        """Test interaction over multiple episodes."""
        total_rewards = []

        # Run multiple episodes
        for _ in range(2):
            # Reset environment
            observation, _ = self.env.reset()
            episode_rewards = []

            # Run until episode is done
            while True:
                # Agent chooses action
                action, log_probs, value = self.agent.choose_action(observation)

                # Environment takes step
                next_observation, reward, done, _, info = self.env.step(action)

                # Store reward
                episode_rewards.append(reward)

                # Break if episode is done
                if done:
                    break

                # Update observation
                observation = next_observation

            # Store total episode reward
            total_rewards.append(sum(episode_rewards))

        # Check that multiple episodes completed
        self.assertEqual(len(total_rewards), 2)


if __name__ == "__main__":
    unittest.main()
