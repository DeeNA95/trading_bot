#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) agent implementation for cryptocurrency trading.
This module contains neural network models, a prioritized replay buffer using a SumTree,
and a PPOAgent class that updates actor and critic networks based on sampled on‐policy transitions.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, TensorDataset

# Import actor and critic models (assumed to be defined in agent.models)
from rl_algo.agent.models import ActorNetwork, CriticNetwork

###############################################################################
# SumTree: A binary tree data structure for prioritized sampling.
###############################################################################


class SumTree:
    """Sum tree data structure for efficient sampling based on priorities."""

    def __init__(self, capacity: int):
        """
        Initialize a sum tree with given capacity.
        Args:
            capacity: Maximum number of leaf nodes (transitions)
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.data_pointer = 0
        self.size = 0

    def add(self, priority: float, data: Any):
        """Add new data with priority to the tree."""
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        """Update the priority of a node and propagate the change upward."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v: float) -> Tuple[int, float, Any]:
        """Retrieve the leaf corresponding to value v."""
        parent_idx = 0
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                v -= self.tree[left_idx]
                parent_idx = right_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self) -> float:
        """Return the total priority (root node)."""
        return self.tree[0]


###############################################################################
# PrioritizedReplayBuffer: Buffer for storing transitions with priorities.
###############################################################################


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    Stores transitions in a SumTree to enable efficient prioritized sampling.
    """

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
        self.tree = SumTree(capacity)

    def store_memory(self, state, action, probs, vals, reward, done):
        """
        Store a transition with maximum priority.
        """
        transition = (state, action, probs, vals, reward, done)
        priority = (self.max_priority + self.epsilon) ** self.alpha
        self.tree.add(priority, transition)

    def sample_batch(self, beta: Optional[float] = None) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions according to their priorities.
        Returns:
            Tensors for states, actions, old_probs, values, rewards, dones, weights, indices.
        """
        beta = beta if beta is not None else self.beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        segment = self.tree.total_priority() / self.batch_size

        batch = []
        batch_indices = []
        batch_priorities = []

        for i in range(self.batch_size):
            a, b = segment * i, segment * (i + 1)
            v = np.random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(v)
            batch_indices.append(idx)
            batch_priorities.append(priority)
            batch.append(data)

        sampling_prob = np.array(batch_priorities) / self.tree.total_priority()
        weights = (self.tree.size * sampling_prob) ** (-beta)
        weights = weights / weights.max()

        # Unpack batch
        states, actions, probs, vals, rewards, dones = [], [], [], [], [], []
        for data in batch:
            s, a, p, v, r, d = data
            states.append(s)
            actions.append(a)
            probs.append(p)
            vals.append(v)
            rewards.append(r)
            dones.append(d)

        # Convert to torch tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        probs = torch.tensor(np.array(probs), dtype=torch.float32)
        vals = torch.tensor(np.array(vals), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        indices = torch.tensor(batch_indices, dtype=torch.int32)

        return states, actions, probs, vals, rewards, dones, weights, indices

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update priorities for the sampled transitions.
        """
        for idx, priority in zip(indices, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def clear_memory(self):
        """Clear the replay buffer."""
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0

    def is_sufficient(self) -> bool:
        """Return True if there are at least batch_size transitions stored."""
        return self.tree.size >= self.batch_size


###############################################################################
# PPOAgent: Proximal Policy Optimization agent.
###############################################################################


class PPOAgent:
    """PPO Agent implementation with prioritized replay for off-policy updates (nonstandard for PPO)."""

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
        target_update_freq: int = 10,
        target_update_tau: float = 0.005,
        lr_scheduler_type: str = "step",
        lr_scheduler_step_size: int = 100,
        lr_scheduler_gamma: float = 0.9,
        lr_scheduler_patience: int = 10,
        grad_clip_value: float = 0.5,
        memory_capacity: int = 10000,
        prioritized_replay_alpha: float = 0.6,
        prioritized_replay_beta: float = 0.4,
        prioritized_replay_beta_increment: float = 0.001,
        device: str = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.target_update_freq = target_update_freq
        self.target_update_tau = target_update_tau
        self.episode_count = 0
        self.lr_scheduler_type = lr_scheduler_type
        self.grad_clip_value = grad_clip_value

        self.device = device

        # Actor network
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Target actor network (for stability)
        self.target_actor = ActorNetwork(state_dim, action_dim, hidden_dims).to(
            self.device
        )
        self._hard_update_target_network(self.actor, self.target_actor)

        # Critic network
        self.critic = CriticNetwork(state_dim, hidden_dims).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Target critic network
        self.target_critic = CriticNetwork(state_dim, hidden_dims).to(self.device)
        self._hard_update_target_network(self.critic, self.target_critic)

        # Learning rate schedulers
        if lr_scheduler_type == "step":
            self.actor_scheduler = StepLR(
                self.actor_optimizer,
                step_size=lr_scheduler_step_size,
                gamma=lr_scheduler_gamma,
            )
            self.critic_scheduler = StepLR(
                self.critic_optimizer,
                step_size=lr_scheduler_step_size,
                gamma=lr_scheduler_gamma,
            )
        elif lr_scheduler_type == "exp":
            self.actor_scheduler = ExponentialLR(
                self.actor_optimizer, gamma=lr_scheduler_gamma
            )
            self.critic_scheduler = ExponentialLR(
                self.critic_optimizer, gamma=lr_scheduler_gamma
            )
        elif lr_scheduler_type == "plateau":
            self.actor_scheduler = ReduceLROnPlateau(
                self.actor_optimizer, patience=lr_scheduler_patience
            )
            self.critic_scheduler = ReduceLROnPlateau(
                self.critic_optimizer, patience=lr_scheduler_patience
            )
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None

        # Replay memory (prioritized)
        self.memory = PrioritizedReplayBuffer(
            capacity=memory_capacity,
            batch_size=batch_size,
            alpha=prioritized_replay_alpha,
            beta=prioritized_replay_beta,
            beta_increment=prioritized_replay_beta_increment,
        )

    def _hard_update_target_network(self, source_network, target_network):
        for target_param, source_param in zip(
            target_network.parameters(), source_network.parameters()
        ):
            target_param.data.copy_(source_param.data)

    def _soft_update_target_network(self, source_network, target_network, tau):
        for target_param, source_param in zip(
            target_network.parameters(), source_network.parameters()
        ):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )

    def update_target_networks(self):
        self._soft_update_target_network(
            self.actor, self.target_actor, self.target_update_tau
        )
        self._soft_update_target_network(
            self.critic, self.target_critic, self.target_update_tau
        )

    def choose_action(
        self, observation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select an action given the current observation."""
        state = torch.tensor(np.array([observation]), dtype=torch.float32).to(
            self.device
        )
        with torch.no_grad():
            action_mean, action_log_std, action, log_prob = self.actor(state)
            value = self.critic(state)
        return (
            action.cpu().numpy()[0],
            log_prob.cpu().numpy()[0],
            value.cpu().numpy()[0],
        )

    def store_transition(self, state, action, probs, vals, reward, done):
        """Store transition in the replay buffer."""
        # Convert numpy arrays to lists if needed.
        state = state.tolist() if isinstance(state, np.ndarray) else state
        action = action.tolist() if isinstance(action, np.ndarray) else action
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages and returns using GAE."""
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.gamma
                * (next_value if t == len(rewards) - 1 else values[t + 1])
                * (1 - dones[t])
                - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(values)
        return advantages, returns

    def learn(self, next_value: float = 0, episode_reward: float = None):
        """Update actor and critic networks using PPO updates."""
        self.episode_count += 1

        if not self.memory.is_sufficient():
            print("Not enough samples in memory for learning")
            return

        # Sample a batch of transitions
        states, actions, old_log_probs, vals, rewards, dones, weights, indices = (
            self.memory.sample_batch()
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        vals = vals.to(self.device).squeeze(-1)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # Compute advantages and returns on the batch using a simple loop.
        # (Note: For on-policy PPO, you typically don't use a replay buffer.
        # This implementation is provided for your prioritized buffer design.)
        advantages_list = []
        returns_list = []
        batch_rewards = rewards.cpu().numpy()
        batch_vals = vals.cpu().numpy()
        batch_dones = dones.cpu().numpy()
        gae = 0.0
        for t in reversed(range(len(batch_rewards))):
            delta = (
                batch_rewards[t]
                + self.gamma
                * (next_value if t == len(batch_rewards) - 1 else batch_vals[t + 1])
                * (1 - batch_dones[t])
                - batch_vals[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - batch_dones[t]) * gae
            advantages_list.insert(0, gae)
            returns_list.insert(0, gae + batch_vals[t])
        advantages_batch = torch.tensor(advantages_list, dtype=torch.float32).to(
            self.device
        )
        returns_batch = torch.tensor(returns_list, dtype=torch.float32).to(self.device)

        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
        )

        # PPO update loop
        for _ in range(self.n_epochs):
            _, _, _, log_probs = self.actor(states)
            current_values = self.critic(states).squeeze(-1)
            ratio = torch.exp(log_probs - old_log_probs)
            surrogate1 = ratio * advantages_batch * weights
            surrogate2 = (
                torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                * advantages_batch
                * weights
            )
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = (weights * (current_values - returns_batch).pow(2)).mean()
            entropy = -log_probs.mean()
            total_loss = (
                actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            )

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_clip_value
            )
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.grad_clip_value
            )
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        if self.episode_count % self.target_update_freq == 0:
            self.update_target_networks()

        if self.actor_scheduler is not None:
            if self.lr_scheduler_type == "plateau" and episode_reward is not None:
                self.actor_scheduler.step(episode_reward)
                self.critic_scheduler.step(episode_reward)
            elif self.lr_scheduler_type in ["step", "exp"]:
                self.actor_scheduler.step()
                self.critic_scheduler.step()

        # Update priorities based on TD errors
        with torch.no_grad():
            td_errors = (returns_batch - current_values).abs().cpu().numpy()
            self.memory.update_priorities(indices.cpu().numpy(), td_errors)

        self.memory.clear_memory()

    def get_learning_rates(self) -> Dict[str, float]:
        return {
            "actor": self.actor_optimizer.param_groups[0]["lr"],
            "critic": self.critic_optimizer.param_groups[0]["lr"],
        }

    def save_models(
        self, path: str, save_optimizer: bool = True, save_memory: bool = False
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_actor_state_dict": self.target_actor.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "episode_count": self.episode_count,
            "hyperparameters": {
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "policy_clip": self.policy_clip,
                "n_epochs": self.n_epochs,
                "entropy_coef": self.entropy_coef,
                "value_coef": self.value_coef,
                "target_update_freq": self.target_update_freq,
                "target_update_tau": self.target_update_tau,
                "lr_scheduler_type": self.lr_scheduler_type,
                "grad_clip_value": self.grad_clip_value,
            },
        }
        if save_optimizer:
            checkpoint["actor_optimizer_state_dict"] = self.actor_optimizer.state_dict()
            checkpoint["critic_optimizer_state_dict"] = (
                self.critic_optimizer.state_dict()
            )
            if self.actor_scheduler is not None:
                checkpoint["actor_scheduler_state_dict"] = (
                    self.actor_scheduler.state_dict()
                )
                checkpoint["critic_scheduler_state_dict"] = (
                    self.critic_scheduler.state_dict()
                )
        if save_memory and self.memory.tree.size > 0:
            memory_data = {
                "tree": self.memory.tree.tree,
                "data": self.memory.tree.data,
                "size": self.memory.tree.size,
                "data_pointer": self.memory.tree.data_pointer,
                "max_priority": self.memory.max_priority,
                "alpha": self.memory.alpha,
                "beta": self.memory.beta,
            }
            torch.save(memory_data, f"{path}_memory.pt")
        torch.save(checkpoint, f"{path}_checkpoint.pt")
        print(f"Model saved to {path}_checkpoint.pt")
        if save_memory and self.memory.tree.size > 0:
            print(f"Memory saved to {path}_memory.pt")

    def load_models(
        self, path: str, load_optimizer: bool = True, load_memory: bool = False
    ) -> bool:
        checkpoint_path = f"{path}_checkpoint.pt"
        try:
            map_location = self.device
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.target_actor.load_state_dict(checkpoint["target_actor_state_dict"])
            self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
            self.episode_count = checkpoint.get("episode_count", self.episode_count)
            hp = checkpoint.get("hyperparameters", {})
            self.gamma = hp.get("gamma", self.gamma)
            self.gae_lambda = hp.get("gae_lambda", self.gae_lambda)
            self.policy_clip = hp.get("policy_clip", self.policy_clip)
            self.n_epochs = hp.get("n_epochs", self.n_epochs)
            self.entropy_coef = hp.get("entropy_coef", self.entropy_coef)
            self.value_coef = hp.get("value_coef", self.value_coef)
            self.target_update_freq = hp.get(
                "target_update_freq", self.target_update_freq
            )
            self.target_update_tau = hp.get("target_update_tau", self.target_update_tau)
            self.lr_scheduler_type = hp.get("lr_scheduler_type", self.lr_scheduler_type)
            self.grad_clip_value = hp.get("grad_clip_value", self.grad_clip_value)
            if load_optimizer:
                if "actor_optimizer_state_dict" in checkpoint:
                    self.actor_optimizer.load_state_dict(
                        checkpoint["actor_optimizer_state_dict"]
                    )
                if "critic_optimizer_state_dict" in checkpoint:
                    self.critic_optimizer.load_state_dict(
                        checkpoint["critic_optimizer_state_dict"]
                    )
                if self.actor_scheduler is not None:
                    self.actor_scheduler.load_state_dict(
                        checkpoint["actor_scheduler_state_dict"]
                    )
                    self.critic_scheduler.load_state_dict(
                        checkpoint["critic_scheduler_state_dict"]
                    )
            if load_memory:
                memory_path = f"{path}_memory.pt"
                try:
                    memory_data = torch.load(memory_path, map_location=map_location)
                    self.memory.tree.tree = memory_data["tree"]
                    self.memory.tree.data = memory_data["data"]
                    self.memory.tree.size = memory_data["size"]
                    self.memory.tree.data_pointer = memory_data["data_pointer"]
                    self.memory.max_priority = memory_data["max_priority"]
                    self.memory.alpha = memory_data.get("alpha", self.memory.alpha)
                    self.memory.beta = memory_data.get("beta", self.memory.beta)
                    print(f"Memory loaded from {memory_path}")
                except FileNotFoundError:
                    print(f"Memory file {memory_path} not found, using empty memory")
            print(f"Model loaded from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
