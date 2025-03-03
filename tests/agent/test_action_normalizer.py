"""
Tests for the action space normalization.
"""

import unittest
import torch
import numpy as np
from agent.models import ActionNormalizer, ActorNetwork


class TestActionNormalizer(unittest.TestCase):
    """Test cases for the ActionNormalizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.action_bounds = torch.tensor([
            [-1.0, 0.0, 0.5, 0.5],  # lower bounds
            [1.0, 1.0, 5.0, 5.0]    # upper bounds
        ], dtype=torch.float32)
        self.normalizer = ActionNormalizer(self.action_bounds)

    def test_normalize(self):
        """Test normalizing actions from action space to [-1, 1]."""
        # Test with lower bounds
        action = self.action_bounds[0]
        normalized = self.normalizer.normalize(action)
        expected = torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(normalized, expected, atol=1e-6))

        # Test with upper bounds
        action = self.action_bounds[1]
        normalized = self.normalizer.normalize(action)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(normalized, expected, atol=1e-6))

        # Test with middle values
        action = (self.action_bounds[0] + self.action_bounds[1]) / 2
        normalized = self.normalizer.normalize(action)
        expected = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(normalized, expected, atol=1e-6))

    def test_denormalize(self):
        """Test denormalizing actions from [-1, 1] to action space."""
        # Test with -1
        normalized_action = torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32)
        denormalized = self.normalizer.denormalize(normalized_action)
        expected = self.action_bounds[0]
        self.assertTrue(torch.allclose(denormalized, expected, atol=1e-6))

        # Test with 1
        normalized_action = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        denormalized = self.normalizer.denormalize(normalized_action)
        expected = self.action_bounds[1]
        self.assertTrue(torch.allclose(denormalized, expected, atol=1e-6))

        # Test with 0
        normalized_action = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        denormalized = self.normalizer.denormalize(normalized_action)
        expected = (self.action_bounds[0] + self.action_bounds[1]) / 2
        self.assertTrue(torch.allclose(denormalized, expected, atol=1e-6))

    def test_clip_action(self):
        """Test clipping actions to be within bounds."""
        # Test with values below lower bounds
        action = torch.tensor([-2.0, -1.0, 0.0, 0.0], dtype=torch.float32)
        clipped = self.normalizer.clip_action(action)
        expected = torch.tensor([-1.0, 0.0, 0.5, 0.5], dtype=torch.float32)
        self.assertTrue(torch.allclose(clipped, expected, atol=1e-6))

        # Test with values above upper bounds
        action = torch.tensor([2.0, 2.0, 6.0, 6.0], dtype=torch.float32)
        clipped = self.normalizer.clip_action(action)
        expected = torch.tensor([1.0, 1.0, 5.0, 5.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(clipped, expected, atol=1e-6))

        # Test with values within bounds
        action = torch.tensor([0.0, 0.5, 2.0, 3.0], dtype=torch.float32)
        clipped = self.normalizer.clip_action(action)
        self.assertTrue(torch.allclose(clipped, action, atol=1e-6))

    def test_roundtrip(self):
        """Test normalizing and then denormalizing returns the original action."""
        original_action = torch.tensor([0.5, 0.7, 2.0, 3.0], dtype=torch.float32)
        normalized = self.normalizer.normalize(original_action)
        denormalized = self.normalizer.denormalize(normalized)
        self.assertTrue(torch.allclose(denormalized, original_action, atol=1e-6))


class TestActorNetworkWithNormalizer(unittest.TestCase):
    """Test cases for the ActorNetwork with action normalization."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 10
        self.action_dim = 4
        self.actor = ActorNetwork(self.state_dim, self.action_dim)
        # Get the device that the actor is using
        self.device = self.actor.device

    def test_action_bounds(self):
        """Test that actions are properly bounded."""
        # Create a random state and move it to the same device as the actor
        state = torch.randn(1, self.state_dim).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            action_mean, action_log_std, action, log_prob = self.actor(state)
        
        # Check that action is within bounds
        lower_bound, upper_bound = self.actor.action_bounds
        
        for i in range(self.action_dim):
            self.assertGreaterEqual(action[0, i].item(), lower_bound[i].item())
            self.assertLessEqual(action[0, i].item(), upper_bound[i].item())


if __name__ == '__main__':
    unittest.main()
