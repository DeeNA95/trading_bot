"""
PPO (Proximal Policy Optimization) agent implementation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym

from agent.models import ActorNetwork, CriticNetwork


class PPOMemory:
    """Memory buffer for PPO agent."""

    def __init__(self, batch_size: int):
        """Initialize the memory buffer.

        Args:
            batch_size: Size of mini-batches for training
        """
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def store_memory(self, state, action, probs, vals, reward, done):
        """Store a transition in memory."""
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """Clear all stored memories."""
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self) -> List[Tuple[torch.Tensor, ...]]:
        """Generate mini-batches for training.

        Returns:
            List of mini-batches containing (states, actions, log_probs, values, returns, advantages)
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        states = torch.tensor(np.array(self.states), dtype=torch.float32)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32)
        old_probs = torch.tensor(np.array(self.probs), dtype=torch.float32)
        values = torch.tensor(np.array(self.vals), dtype=torch.float32)

        return states, actions, old_probs, values


class PPOAgent:
    """PPO Agent implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128),
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        batch_size: int = 64,
        n_epochs: int = 10,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ):
        """Initialize the PPO agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: Dimensions of hidden layers
            lr_actor: Learning rate for the actor network
            lr_critic: Learning rate for the critic network
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            policy_clip: Clipping parameter for PPO
            batch_size: Size of mini-batches for training
            n_epochs: Number of epochs to train on each batch of data
            entropy_coef: Entropy coefficient for exploration
            value_coef: Value loss coefficient
            device: Device to use for training (cpu or cuda)
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.device = device

        # Actor network
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic network
        self.critic = CriticNetwork(state_dim, hidden_dims).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Memory buffer
        self.memory = PPOMemory(batch_size)

    def choose_action(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Choose an action based on the current observation.

        Args:
            observation: Current observation

        Returns:
            action: Selected action
            log_prob: Log probability of the selected action
            value: Estimated value of the current state
        """
        state = torch.tensor(np.array([observation]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action_mean, action_log_std, action, log_prob = self.actor(state)
            value = self.critic(state)

        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]

    def store_transition(self, state, action, probs, vals, reward, done):
        """Store a transition in memory."""
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool], next_value: float) -> np.ndarray:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate of the next state

        Returns:
            advantages: Computed advantages
        """
        values = np.append(values, next_value)
        advantages = np.zeros_like(rewards, dtype=np.float32)

        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + np.array(values[:-1])
        return advantages, returns

    def learn(self, next_value: float = 0):
        """Update policy and value networks.

        Args:
            next_value: Value estimate of the next state
        """
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            self.memory.rewards,
            self.memory.vals,
            self.memory.dones,
            next_value
        )

        # Convert to tensors
        states, actions, old_log_probs, _ = self.memory.generate_batches()
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create dataset
        dataset = TensorDataset(
            states.to(self.device),
            actions.to(self.device),
            old_log_probs.to(self.device),
            advantages,
            returns
        )
        data_loader = DataLoader(dataset, batch_size=self.memory.batch_size, shuffle=True)

        # PPO update
        for _ in range(self.n_epochs):
            for batch in data_loader:
                states_batch, actions_batch, old_log_probs_batch, advantages_batch, returns_batch = batch

                # Get current policy and value estimates
                _, _, _, log_probs = self.actor(states_batch)
                values = self.critic(states_batch)
                values = values.squeeze(-1)

                # Calculate the ratio
                ratio = torch.exp(log_probs - old_log_probs_batch)

                # Calculate the surrogate losses
                surrogate1 = ratio * advantages_batch
                surrogate2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages_batch

                # Calculate the actor loss
                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                # Calculate the value loss
                value_loss = nn.MSELoss()(values, returns_batch)

                # Calculate the entropy bonus
                entropy = -log_probs.mean()

                # Calculate the total loss
                total_loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Update the networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # Clear memory
        self.memory.clear_memory()

    def save_models(self, path: str):
        """Save the actor and critic models.

        Args:
            path: Path to save the models
        """
        torch.save(self.actor.state_dict(), f"{path}_actor.pt")
        torch.save(self.critic.state_dict(), f"{path}_critic.pt")

    def load_models(self, path: str):
        """Load the actor and critic models.

        Args:
            path: Path to load the models from
        """
        self.actor.load_state_dict(torch.load(f"{path}_actor.pt"))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pt"))
