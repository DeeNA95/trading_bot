"""
Neural network architectures for the reinforcement learning agent.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ActorNetwork(nn.Module):
    """Actor network for PPO policy."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (256, 128)):
        """Initialize the actor network.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: Dimensions of hidden layers
        """
        super(ActorNetwork, self).__init__()

        # Device assignment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU())
            prev_dim = dim

        self.feature_extractor = nn.Sequential(*layers).to(self.device)

        # Action mean
        self.action_mean = nn.Linear(prev_dim, action_dim).to(self.device)

        # Action log std (learned)
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim, device=self.device))

        # Action bounds for our environment
        # [direction, size, take_profit, stop_loss]
        self.action_bounds = torch.tensor([
            [-1.0, 0.0, 0.5, 0.5],  # lower bounds
            [1.0, 1.0, 5.0, 5.0]    # upper bounds
        ], dtype=torch.float32, device=self.device)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            state: Current state

        Returns:
            action_mean: Mean of the action distribution
            action_log_std: Log standard deviation of the action distribution
            action: Sampled action
        """
        x = self.feature_extractor(state)

        # Get mean of action distribution
        action_mean = self.action_mean(x)

        # Bound the action mean to our action space
        action_mean = torch.tanh(action_mean)

        # Scale to the action bounds
        lower_bound, upper_bound = self.action_bounds
        action_mean = lower_bound + (upper_bound - lower_bound) * (action_mean + 1) / 2

        # Get log std and std
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        # Sample action from distribution
        normal = torch.distributions.Normal(action_mean, action_std)

        # Reparameterization trick
        x_t = normal.rsample()

        # Enforce action bounds
        action = torch.max(torch.min(x_t, upper_bound), lower_bound)

        # Calculate log probability of the action
        log_prob = normal.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return action_mean, action_log_std, action, log_prob


class CriticNetwork(nn.Module):
    """Critic network for PPO value function."""

    def __init__(self, state_dim: int, hidden_dims: Tuple[int, ...] = (256, 128)):
        """Initialize the critic network.

        Args:
            state_dim: Dimension of the state space
            hidden_dims: Dimensions of hidden layers
        """
        super(CriticNetwork, self).__init__()

        # Device assignment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU())
            prev_dim = dim

        self.feature_extractor = nn.Sequential(*layers).to(self.device)

        # Value head
        self.value_head = nn.Linear(prev_dim, 1).to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state: Current state

        Returns:
            value: Estimated value of the state
        """
        x = self.feature_extractor(state.to(self.device))
        value = self.value_head(x)

        return value
