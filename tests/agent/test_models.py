"""
Unit tests for neural network models used by the agent.
"""

import os
import sys
import unittest

import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.mocks.agent_mock import MockNetwork


class TestNetworks(unittest.TestCase):
    """Tests for neural network models."""

    def setUp(self):
        """Set up test fixtures."""
        self.actor_network = MockNetwork(input_dim=10, output_dim=4)
        self.critic_network = MockNetwork(input_dim=10, output_dim=1)

    def test_actor_forward(self):
        """Test actor network forward pass."""
        # Create mock input
        state = np.random.random(10)

        # Get output from the actor network
        output = self.actor_network.forward(state)

        # Check output shape
        self.assertEqual(len(output), 4)

        # Check that outputs are within expected range
        self.assertTrue(np.all(np.isfinite(output)))

    def test_critic_forward(self):
        """Test critic network forward pass."""
        # Create mock input
        state = np.random.random(10)

        # Get output from the critic network
        output = self.critic_network.forward(state)

        # Check output shape
        self.assertEqual(len(output), 1)

        # Check that output is a finite value
        self.assertTrue(np.all(np.isfinite(output)))

    def test_model_parameters(self):
        """Test that network parameters can be accessed."""
        # Get parameters from the network
        params = list(self.actor_network.parameters())

        # Check that we have parameters
        self.assertTrue(len(params) > 0)

        # Check that parameters are arrays
        for param in params:
            self.assertIsInstance(param, np.ndarray)

    def test_device_placement(self):
        """Test that the network can be placed on a device."""
        # Try placing the network on 'cpu'
        network_on_device = self.actor_network.to("cpu")

        # Check that we got the network back
        self.assertIs(network_on_device, self.actor_network)


if __name__ == "__main__":
    unittest.main()
