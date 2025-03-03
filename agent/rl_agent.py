"""
PPO (Proximal Policy Optimization) agent implementation.
"""

from typing import Any, Dict, List, Optional, Tuple
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau

from agent.models import ActorNetwork, CriticNetwork


class SumTree:
    """Sum tree data structure for efficient sampling based on priorities."""
    
    def __init__(self, capacity: int):
        """Initialize a sum tree with given capacity.
        
        Args:
            capacity: Maximum number of leaf nodes (transitions)
        """
        self.capacity = capacity  # Number of leaf nodes
        self.tree = np.zeros(2 * capacity - 1)  # Total nodes in the tree
        self.data = [None] * capacity  # Storage for data
        self.data_pointer = 0  # Current position for adding new data
        self.size = 0  # Current size of buffer
        
    def add(self, priority: float, data: Any):
        """Add new data with priority to the tree.
        
        Args:
            priority: Priority value for the data
            data: The data to store
        """
        # Find the leaf index to insert data
        tree_idx = self.data_pointer + self.capacity - 1
        
        # Store the data
        self.data[self.data_pointer] = data
        
        # Update the tree with new priority
        self.update(tree_idx, priority)
        
        # Move to next position
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        # Update size
        if self.size < self.capacity:
            self.size += 1
    
    def update(self, tree_idx: int, priority: float):
        """Update the priority of a node.
        
        Args:
            tree_idx: Index of the node in the tree
            priority: New priority value
        """
        # Change = new priority - old priority
        change = priority - self.tree[tree_idx]
        
        # Update the node
        self.tree[tree_idx] = priority
        
        # Propagate the change through tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get_leaf(self, v: float) -> Tuple[int, float, Any]:
        """Get a leaf node based on a value.
        
        Args:
            v: Value to search for (should be in range [0, total_priority])
            
        Returns:
            Tuple of (leaf_idx, priority, data)
        """
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we reach bottom, end the search
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            # Otherwise, go left or right based on comparison with left child
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    def total_priority(self) -> float:
        """Get the total priority of the tree.
        
        Returns:
            Sum of all priorities
        """
        return self.tree[0]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for more efficient learning."""
    
    def __init__(
        self, 
        capacity: int, 
        batch_size: int,
        alpha: float = 0.6, 
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ):
        """Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            batch_size: Size of mini-batches for training
            alpha: How much prioritization to use (0 = no prioritization, 1 = full prioritization)
            beta: Importance sampling correction factor (0 = no correction, 1 = full correction)
            beta_increment: Amount to increase beta each time get_batch is called
            epsilon: Small value to add to priorities to ensure non-zero probability
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
        
        # Storage for transitions
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
    def store_memory(self, state, action, probs, vals, reward, done):
        """Store a transition in memory with maximum priority.
        
        Args:
            state: Current state
            action: Action taken
            probs: Log probability of the action
            vals: Value estimate
            reward: Reward received
            done: Done flag
        """
        # Create transition data
        transition = (state, action, probs, vals, reward, done)
        
        # Calculate priority (use max priority for new transitions)
        priority = self.max_priority ** self.alpha
        
        # Add to sum tree
        self.tree.add(priority, transition)
        
    def sample_batch(self, beta: Optional[float] = None) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions based on priorities.
        
        Args:
            beta: Importance sampling factor, if None use current beta
            
        Returns:
            Tuple of tensors (states, actions, old_probs, values, rewards, dones, weights, indices)
        """
        beta = beta if beta is not None else self.beta
        
        # Increase beta over time for more accurate importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate segment size
        segment = self.tree.total_priority() / self.batch_size
        
        # Sample from each segment
        batch_indices = []
        batch_priorities = []
        batch = []
        
        for i in range(self.batch_size):
            # Get a value within the segment
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            
            # Get a sample from the tree
            idx, priority, data = self.tree.get_leaf(v)
            
            batch_indices.append(idx)
            batch_priorities.append(priority)
            batch.append(data)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(batch_priorities) / self.tree.total_priority()
        weights = (self.tree.size * sampling_probabilities) ** -beta
        weights = weights / weights.max()  # Normalize weights
        
        # Unpack batch
        states = []
        actions = []
        probs = []
        vals = []
        rewards = []
        dones = []
        
        for transition in batch:
            state, action, prob, val, reward, done = transition
            states.append(state)
            actions.append(action)
            probs.append(prob)
            vals.append(val)
            rewards.append(reward)
            dones.append(done)
        
        # Convert to tensors
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
        """Update priorities of sampled transitions.
        
        Args:
            indices: Indices of the transitions in the tree
            priorities: New priorities for these transitions
        """
        for idx, priority in zip(indices, priorities):
            # Add a small value to avoid zero priority
            priority = (priority + self.epsilon) ** self.alpha
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
            
            # Update tree
            self.tree.update(idx, priority)
    
    def clear_memory(self):
        """Clear all stored memories."""
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        
    def is_sufficient(self) -> bool:
        """Check if there are enough samples for a batch.
        
        Returns:
            True if there are enough samples, False otherwise
        """
        return self.tree.size >= self.batch_size


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
        target_update_freq: int = 10,  # Frequency of target network updates
        target_update_tau: float = 0.005,  # Soft update parameter
        lr_scheduler_type: str = "step",  # Type of learning rate scheduler: "step", "exp", "plateau", or "none"
        lr_scheduler_step_size: int = 100,  # Step size for StepLR scheduler
        lr_scheduler_gamma: float = 0.9,  # Gamma for StepLR and ExponentialLR schedulers
        lr_scheduler_patience: int = 10,  # Patience for ReduceLROnPlateau scheduler
        grad_clip_value: float = 0.5,  # Value for gradient clipping
        memory_capacity: int = 10000,  # Capacity of the replay buffer
        prioritized_replay_alpha: float = 0.6,  # How much prioritization to use
        prioritized_replay_beta: float = 0.4,  # Importance sampling correction factor
        prioritized_replay_beta_increment: float = 0.001,  # Beta increment per batch
        device: str = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
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
            target_update_freq: Frequency of target network updates (in episodes)
            target_update_tau: Soft update parameter for target networks
            lr_scheduler_type: Type of learning rate scheduler to use
            lr_scheduler_step_size: Step size for StepLR scheduler
            lr_scheduler_gamma: Gamma for StepLR and ExponentialLR schedulers
            lr_scheduler_patience: Patience for ReduceLROnPlateau scheduler
            grad_clip_value: Maximum norm of gradients for clipping
            memory_capacity: Capacity of the replay buffer
            prioritized_replay_alpha: Alpha parameter for prioritized replay (0 = uniform, 1 = full prioritization)
            prioritized_replay_beta: Beta parameter for importance sampling correction (0 = no correction, 1 = full correction)
            prioritized_replay_beta_increment: Amount to increase beta each time get_batch is called
            device: Device to use for training (cpu or cuda)
        """
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

        # Target actor network
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
                self.actor_optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma
            )
            self.critic_scheduler = StepLR(
                self.critic_optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma
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

        # Memory buffer
        self.memory = PrioritizedReplayBuffer(
            capacity=memory_capacity, 
            batch_size=batch_size,
            alpha=prioritized_replay_alpha, 
            beta=prioritized_replay_beta,
            beta_increment=prioritized_replay_beta_increment,
        )

    def _hard_update_target_network(self, source_network, target_network):
        """Hard update target network: target_params = source_params."""
        for target_param, source_param in zip(
            target_network.parameters(), source_network.parameters()
        ):
            target_param.data.copy_(source_param.data)

    def _soft_update_target_network(self, source_network, target_network, tau):
        """Soft update target network: target_params = tau * source_params + (1 - tau) * target_params."""
        for target_param, source_param in zip(
            target_network.parameters(), source_network.parameters()
        ):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )

    def update_target_networks(self):
        """Update target networks using soft update."""
        self._soft_update_target_network(
            self.actor, self.target_actor, self.target_update_tau
        )
        self._soft_update_target_network(
            self.critic, self.target_critic, self.target_update_tau
        )

    def choose_action(
        self, observation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Choose an action based on the current observation.

        Args:
            observation: Current observation

        Returns:
            action: Selected action
            log_prob: Log probability of the selected action
            value: Estimated value of the current state
        """
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
        """Store a transition in memory.

        Args:
            state: Current state
            action: Action taken
            probs: Log probability of the action
            vals: Value estimate
            reward: Reward received
            done: Done flag
        """
        # Convert numpy arrays to lists for storage
        if isinstance(state, np.ndarray):
            state = state.tolist()
        if isinstance(action, np.ndarray):
            action = action.tolist()
            
        # Store the transition in memory
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def compute_gae(
        self, rewards: List[float], values: List[float], dones: List[bool], next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards for each step
            values: List of value estimates for each step
            dones: List of done flags for each step
            next_value: Value estimate for the next state

        Returns:
            Tuple of (advantages, returns)
        """
        # If memory is empty, return empty arrays
        if len(rewards) == 0:
            return np.array([]), np.array([])
            
        # Convert to numpy arrays for easier manipulation
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)
        
        # Append next_value to values
        values = np.append(values, next_value)
        
        # Initialize advantages and gae
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        
        # Compute advantages using GAE
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        # Compute returns
        returns = advantages + values[:-1]
        
        return advantages, returns

    def learn(self, next_value: float = 0, episode_reward: float = None):
        """Update policy and value networks.

        Args:
            next_value: Value estimate of the next state
            episode_reward: Total reward for the episode, used for plateau scheduler
        """
        # Increment episode counter
        self.episode_count += 1

        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            self.memory.rewards, self.memory.vals, self.memory.dones, next_value
        )

        # Check if we have enough samples
        if not self.memory.is_sufficient():
            print("Not enough samples in memory for learning")
            return

        # Sample a batch from memory with prioritized experience replay
        states, actions, old_log_probs, vals, rewards, dones, weights, indices = self.memory.sample_batch()
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        vals = vals.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Compute advantages and returns for the sampled transitions
        advantages_batch = []
        returns_batch = []
        
        for i in range(len(rewards)):
            # Get consecutive transitions for GAE calculation
            if i == 0 or dones[i-1].item():
                # Start of episode or after done
                adv = rewards[i] - vals[i]
                ret = rewards[i]
            else:
                # Middle of episode
                # Check if i+1 is within bounds to avoid IndexError
                if i+1 < len(vals):
                    delta = rewards[i] + self.gamma * vals[i+1] * (1 - dones[i]) - vals[i]
                else:
                    # Use next_value if we're at the end of the batch
                    delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - vals[i]
                adv = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * advantages_batch[-1]
                ret = rewards[i] + self.gamma * (1 - dones[i]) * returns_batch[-1]
            
            advantages_batch.append(adv)
            returns_batch.append(ret)
        
        # Convert to tensors
        advantages_batch = torch.tensor(advantages_batch, dtype=torch.float32).to(self.device)
        returns_batch = torch.tensor(returns_batch, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        # PPO update
        for _ in range(self.n_epochs):
            # Get current policy and value estimates
            _, _, _, log_probs = self.actor(states)
            values = self.critic(states)
            values = values.squeeze(-1)

            # Calculate the ratio
            ratio = torch.exp(log_probs - old_log_probs)

            # Calculate the surrogate losses with importance sampling weights
            surrogate1 = ratio * advantages_batch * weights
            surrogate2 = (
                torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                * advantages_batch * weights
            )

            # Calculate the actor loss
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            # Calculate the value loss with importance sampling weights
            value_loss = (weights * (values - returns_batch).pow(2)).mean()

            # Calculate the entropy bonus
            entropy = -log_probs.mean()

            # Calculate the total loss
            total_loss = (
                actor_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
            )

            # Update the networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()

            # Clip the gradients using the configurable value
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_value)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_value)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # Update target networks if it's time
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_networks()

        # Update learning rate schedulers
        if self.actor_scheduler is not None:
            if self.lr_scheduler_type == "plateau" and episode_reward is not None:
                self.actor_scheduler.step(episode_reward)
                self.critic_scheduler.step(episode_reward)
            elif self.lr_scheduler_type in ["step", "exp"]:
                self.actor_scheduler.step()
                self.critic_scheduler.step()

        # Update priorities in the replay buffer
        # Calculate TD errors for prioritization
        with torch.no_grad():
            td_errors = (returns_batch - values).abs().cpu().numpy()
            self.memory.update_priorities(indices.cpu().numpy(), td_errors)

        # Clear memory after learning
        self.memory.clear_memory()

    def get_learning_rates(self) -> Dict[str, float]:
        """Get the current learning rates.
        
        Returns:
            Dict with current learning rates for actor and critic
        """
        return {
            "actor": self.actor_optimizer.param_groups[0]["lr"],
            "critic": self.critic_optimizer.param_groups[0]["lr"]
        }

    def save_models(self, path: str, save_optimizer: bool = True, save_memory: bool = False):
        """Save the agent's models and states.

        Args:
            path: Path to save the models
            save_optimizer: Whether to save optimizer states
            save_memory: Whether to save the replay buffer
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create a checkpoint dictionary with all relevant states
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'episode_count': self.episode_count,
            'hyperparameters': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'policy_clip': self.policy_clip,
                'n_epochs': self.n_epochs,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
                'target_update_freq': self.target_update_freq,
                'target_update_tau': self.target_update_tau,
                'lr_scheduler_type': self.lr_scheduler_type,
                'grad_clip_value': self.grad_clip_value
            }
        }
        
        # Save optimizer states if requested
        if save_optimizer:
            checkpoint['actor_optimizer_state_dict'] = self.actor_optimizer.state_dict()
            checkpoint['critic_optimizer_state_dict'] = self.critic_optimizer.state_dict()
            
            # Save scheduler states if they exist
            if self.actor_scheduler is not None:
                checkpoint['actor_scheduler_state_dict'] = self.actor_scheduler.state_dict()
                checkpoint['critic_scheduler_state_dict'] = self.critic_scheduler.state_dict()
        
        # Save memory buffer if requested
        if save_memory and hasattr(self.memory, 'tree') and self.memory.tree.size > 0:
            # Create a separate file for memory as it can be large
            memory_data = {
                'tree': self.memory.tree.tree,
                'data': self.memory.tree.data,
                'size': self.memory.tree.size,
                'data_pointer': self.memory.tree.data_pointer,
                'max_priority': self.memory.max_priority,
                'alpha': self.memory.alpha,
                'beta': self.memory.beta
            }
            torch.save(memory_data, f"{path}_memory.pt")
            
        # Save the checkpoint
        torch.save(checkpoint, f"{path}_checkpoint.pt")
        
        print(f"Model saved to {path}_checkpoint.pt")
        if save_memory and hasattr(self.memory, 'tree') and self.memory.tree.size > 0:
            print(f"Memory saved to {path}_memory.pt")

    def load_models(self, path: str, load_optimizer: bool = True, load_memory: bool = False):
        """Load the agent's models and states.

        Args:
            path: Path to load the models from
            load_optimizer: Whether to load optimizer states
            load_memory: Whether to load the replay buffer
        """
        checkpoint_path = f"{path}_checkpoint.pt"
        
        try:
            # Determine the appropriate map_location
            map_location = self.device
            
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            
            # Load model states
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            
            # Load episode count
            if 'episode_count' in checkpoint:
                self.episode_count = checkpoint['episode_count']
            
            # Load hyperparameters if they exist
            if 'hyperparameters' in checkpoint:
                hp = checkpoint['hyperparameters']
                self.gamma = hp.get('gamma', self.gamma)
                self.gae_lambda = hp.get('gae_lambda', self.gae_lambda)
                self.policy_clip = hp.get('policy_clip', self.policy_clip)
                self.n_epochs = hp.get('n_epochs', self.n_epochs)
                self.entropy_coef = hp.get('entropy_coef', self.entropy_coef)
                self.value_coef = hp.get('value_coef', self.value_coef)
                self.target_update_freq = hp.get('target_update_freq', self.target_update_freq)
                self.target_update_tau = hp.get('target_update_tau', self.target_update_tau)
                self.lr_scheduler_type = hp.get('lr_scheduler_type', self.lr_scheduler_type)
                self.grad_clip_value = hp.get('grad_clip_value', self.grad_clip_value)
            
            # Load optimizer states if requested
            if load_optimizer:
                if 'actor_optimizer_state_dict' in checkpoint:
                    self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                if 'critic_optimizer_state_dict' in checkpoint:
                    self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                
                # Load scheduler states if they exist
                if self.actor_scheduler is not None:
                    if 'actor_scheduler_state_dict' in checkpoint:
                        self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
                    if 'critic_scheduler_state_dict' in checkpoint:
                        self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
            
            # Load memory buffer if requested
            if load_memory:
                memory_path = f"{path}_memory.pt"
                try:
                    memory_data = torch.load(memory_path, map_location=map_location)
                    
                    # Recreate the SumTree
                    self.memory.tree.tree = memory_data['tree']
                    self.memory.tree.data = memory_data['data']
                    self.memory.tree.size = memory_data['size']
                    self.memory.tree.data_pointer = memory_data['data_pointer']
                    
                    # Restore memory parameters
                    self.memory.max_priority = memory_data['max_priority']
                    self.memory.alpha = memory_data.get('alpha', self.memory.alpha)
                    self.memory.beta = memory_data.get('beta', self.memory.beta)
                    
                    print(f"Memory loaded from {memory_path}")
                except FileNotFoundError:
                    print(f"Memory file {memory_path} not found, using empty memory")
            
            print(f"Model loaded from {checkpoint_path}")
            return True
            
        except FileNotFoundError:
            print(f"Checkpoint file {checkpoint_path} not found")
            # Try to load individual files for backward compatibility
            try:
                actor_path = f"{path}_actor.pt"
                critic_path = f"{path}_critic.pt"
                
                self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                
                # Load target networks if files exist, otherwise copy from main networks
                try:
                    target_actor_path = f"{path}_target_actor.pt"
                    target_critic_path = f"{path}_target_critic.pt"
                    
                    self.target_actor.load_state_dict(torch.load(target_actor_path, map_location=self.device))
                    self.target_critic.load_state_dict(torch.load(target_critic_path, map_location=self.device))
                except FileNotFoundError:
                    self._hard_update_target_network(self.actor, self.target_actor)
                    self._hard_update_target_network(self.critic, self.target_critic)
                
                # Load optimizer states if files exist and requested
                if load_optimizer:
                    try:
                        actor_optimizer_path = f"{path}_actor_optimizer.pt"
                        critic_optimizer_path = f"{path}_critic_optimizer.pt"
                        
                        self.actor_optimizer.load_state_dict(torch.load(actor_optimizer_path, map_location=self.device))
                        self.critic_optimizer.load_state_dict(torch.load(critic_optimizer_path, map_location=self.device))
                    except FileNotFoundError:
                        print("Optimizer states not found, using default initialization")
                
                # Load scheduler states if they exist and schedulers are initialized
                if load_optimizer and self.actor_scheduler is not None:
                    try:
                        actor_scheduler_path = f"{path}_actor_scheduler.pt"
                        critic_scheduler_path = f"{path}_critic_scheduler.pt"
                        
                        self.actor_scheduler.load_state_dict(torch.load(actor_scheduler_path, map_location=self.device))
                        self.critic_scheduler.load_state_dict(torch.load(critic_scheduler_path, map_location=self.device))
                    except FileNotFoundError:
                        print("Scheduler states not found, using default initialization")
                
                print("Models loaded from individual files (legacy format)")
                return True
                
            except FileNotFoundError:
                print(f"No model files found at {path}, initializing new models")
                return False
        
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
