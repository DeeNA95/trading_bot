"""
Proximal Policy Optimization (PPO) agent for reinforcement learning in trading environments.

This module implements a PPO agent that can be used to train a trading policy
using various neural network architectures (CNN, LSTM, Transformer).
"""

import logging
import os
import time
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from rl_agent.environment.trading_env import BinanceFuturesCryptoEnv

from .models import ActorCriticCNN, ActorCriticLSTM, ActorCriticTransformer


class PPOMemory:
    """
    Memory buffer for PPO algorithm to store experiences during training.
    """

    def __init__(self, batch_size: int):
        """
        Initialize PPO memory buffer.

        Args:
            batch_size: Batch size for sampling
        """
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def store(
        self,
        state: np.ndarray,
        action: int,
        probs: float,
        vals: float,
        reward: float,
        done: bool,
    ) -> None:
        """
        Store a transition in the buffer.

        Args:
            state: Current state
            action: Action taken
            probs: Action probabilities
            vals: Value estimate
            reward: Reward received
            done: Whether the episode terminated
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self) -> None:
        """Clear the memory buffer."""
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self) -> Tuple[List[np.ndarray], ...]:
        """
        Generate batches of experiences for training.

        Returns:
            Tuple of lists containing batched states, actions, old_probs, vals, rewards, dones, batches
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for trading.

    This agent implements the PPO algorithm with clipped objective for
    training trading strategies in cryptocurrency markets.
    """

    def __init__(
        self,
        env: BinanceFuturesCryptoEnv,
        model_type: str = "transformer",
        hidden_dim: int = 128,  # size of first hidden dim
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        batch_size: int = 2048,
        n_epochs: int = 500,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "auto",
        save_dir: str = "models",
        use_gae: bool = True,
        normalize_advantage: bool = True,
        weight_decay: float = 0.01,
    ):
        """
        Initialize the PPO agent.

        Args:
            env: custom environment to train on
            model_type: Type of model to use ('cnn', 'lstm', 'transformer')
            hidden_dim: Number of hidden units in the actor and critic networks
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE (Generalized Advantage Estimation) lambda parameter
            policy_clip: PPO clipping parameter
            batch_size: Batch size for training
            n_epochs: Number of epochs to update the policy for each batch of experiences
            entropy_coef: Entropy coefficient for exploration
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for gradient clipping
            device: Device to run on ('cpu', 'cuda', 'mps', or 'auto')
            save_dir: Directory to save models to
            use_gae: Whether to use Generalized Advantage Estimation
            normalize_advantage: Whether to normalize advantages
            weight_decay: L2 regularization parameter
        """
        # Environment info
        self.env = env
        self.input_shape = (
            env.window_size,
            env.observation_space.shape[1],
        )  # df shape (nrow,ncol)
        self.action_dim = env.action_space.n

        # Agent parameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.use_gae = use_gae
        self.normalize_advantage = normalize_advantage
        self.weight_decay = weight_decay

        # Save path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Initialize actor-critic model
        if model_type.lower() == "cnn":
            self.model = ActorCriticCNN(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                device=self.device,
            )
        elif model_type.lower() == "lstm":
            self.model = ActorCriticLSTM(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                device=self.device,
            )
        elif model_type.lower() == "transformer":
            self.model = ActorCriticTransformer(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Optimizer with L2 regularization
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=self.weight_decay
        )

        # Memory buffer
        self.memory = PPOMemory(batch_size)

        # Training metrics
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_history = []
        self.reward_history = []

        # Logger
        self.logger = logging.getLogger("PPOAgent")
        self.logger.setLevel(logging.INFO)

        # Best model tracking
        self.best_reward = -float("inf")

    def choose_action(self, state: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        Choose an action based on the current state.

        Args:
            state: Current environment state

        Returns:
            Tuple of (action, action probabilities, value)
        """
        # Check for NaN values in state and replace with zeros
        if np.isnan(state).any():
            state = np.nan_to_num(state, nan=0.0)

        # Clip extreme values to prevent NaN issues
        # state = np.clip(state, -10.0, 10.0)

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get action probabilities and value from model
        with torch.no_grad():
            try:
                if isinstance(self.model, ActorCriticLSTM):
                    logits, value = self.model(state_tensor, reset_hidden=False)
                else:
                    logits, value = self.model(state_tensor)

                # Check for NaN values in logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    # If we have NaN values, use a uniform distribution
                    print("Warning: NaN values in logits, using uniform distribution")
                    logits = torch.ones((1, self.action_dim), device=self.device)

                # Get action probabilities (with stability fixes)
                logits = torch.clamp(logits, min=-20.0, max=20.0)
                action_probs = torch.softmax(logits, dim=-1)

                # Add small epsilon to prevent zero probabilities
                action_probs = action_probs + 1e-8
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

                dist = Categorical(action_probs)

                # Sample action
                action = dist.sample().item()

                # Get action probability and value
                action_prob = action_probs[0, action].item()
                value = value.item() if not torch.isnan(value).any() else 0.0

            except Exception as e:
                print(f"Error in choose_action: {e}")
                # Default to a random action if something goes wrong
                action = np.random.randint(0, self.action_dim)
                action_prob = 1.0 / self.action_dim
                value = 0.0

        return action, action_prob, value

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the log probabilities, values, and entropy of actions.

        Args:
            states: Batch of states
            actions: Batch of actions

        Returns:
            Tuple of (action log probabilities, values, entropy)
        """
        # Check for NaN values in states
        if torch.isnan(states).any():
            states = torch.nan_to_num(states, nan=0.0)

        # Clip extreme values
        states = torch.clamp(states, min=-10.0, max=10.0)

        try:
            # Forward pass through model
            if isinstance(self.model, ActorCriticLSTM):
                logits, values = self.model(states, reset_hidden=True)
            else:
                logits, values = self.model(states)

            # Check for NaN values
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Warning: NaN values in logits during evaluation")
                logits = torch.ones_like(logits) / self.action_dim

            if torch.isnan(values).any() or torch.isinf(values).any():
                print("Warning: NaN values in values during evaluation")
                values = torch.zeros_like(values)

            # Clip logits to ensure numerical stability
            logits = torch.clamp(logits, min=-20.0, max=20.0)

            # Get action probabilities with stability fix
            action_probs = torch.softmax(logits, dim=-1)

            # Add small epsilon to prevent zero probabilities
            action_probs = action_probs + 1e-8
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

            dist = Categorical(action_probs)

            # Get log probabilities of actions
            action_log_probs = dist.log_prob(actions)

            # Get entropy
            entropy = dist.entropy()

            # Final NaN check
            action_log_probs = torch.nan_to_num(action_log_probs, nan=0.0)
            values = torch.nan_to_num(values, nan=0.0)
            entropy = torch.nan_to_num(entropy, nan=0.0)


        except Exception as e:
            print(f"Error in evaluate_actions: {e}")
            # Return default values to allow training to continue
            action_log_probs = torch.zeros_like(actions, dtype=torch.float32)
            values = torch.zeros((len(actions),), dtype=torch.float32)
            entropy = torch.zeros((len(actions),), dtype=torch.float32)

        return action_log_probs, values.squeeze(), entropy

    def update(self) -> Dict[str, float]:
        """
        Update the policy and value networks using PPO.

        Returns:
            Dictionary of training metrics
        """
        # Generate batches of experiences
        states, actions, old_probs, values, rewards, dones, batches = (
            self.memory.generate_batches()
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_probs = torch.FloatTensor(old_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)

        # Compute advantages using GAE (Generalized Advantage Estimation)
        advantages = np.zeros(len(rewards), dtype=np.float32)

        if self.use_gae:
            # GAE calculation
            last_gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]

                delta = (
                    rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                )
                last_gae = (
                    delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
                )
                advantages[t] = last_gae
        else:
            # Simple advantage calculation
            for t in range(len(rewards)):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards)):
                    a_t += discount * rewards[k]
                    discount *= self.gamma * (1 - dones[k])
                    if dones[k]:
                        break
                advantages[t] = a_t - values[t]

        advantages = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for n_epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        # Create progress bar for update epochs
        epoch_pbar = tqdm(range(self.n_epochs), desc="PPO Update", leave=False)

        for _ in epoch_pbar:
            batch_policy_loss = 0
            batch_value_loss = 0
            batch_entropy = 0

            # Create progress bar for batches
            batch_pbar = tqdm(batches, desc="Batches", leave=False)

            # Iterate over batches
            for batch in batch_pbar:
                # Evaluate actions
                new_log_probs, new_values, entropy = self.evaluate_actions(
                    states[batch], actions[batch]
                )

                # Calculate ratio of new and old probabilities
                prob_ratio = torch.exp(new_log_probs - torch.log(old_probs[batch]))

                # Clipped surrogate objective
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = advantages[batch] * torch.clamp(
                    prob_ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip
                )

                # Policy loss (negative because we want to maximize rewards)
                policy_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Value loss
                returns = advantages[batch] + values[batch]
                value_loss = torch.nn.functional.mse_loss(new_values, returns)

                # Entropy bonus for exploration
                entropy_loss = entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef
                    * entropy_loss  # Minus because we want to maximize entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                self.optimizer.step()

                # Update batch metrics
                batch_policy_loss += policy_loss.item()
                batch_value_loss += value_loss.item()
                batch_entropy += entropy_loss.item()

                # Update batch progress bar
                batch_pbar.set_postfix(
                    {
                        "policy_loss": f"{policy_loss.item():.4f}",
                        "value_loss": f"{value_loss.item():.4f}",
                    }
                )

            # Update total metrics
            total_policy_loss += batch_policy_loss
            total_value_loss += batch_value_loss
            total_entropy += batch_entropy

            # Update epoch progress bar
            avg_policy_loss_epoch = batch_policy_loss / len(batches)
            avg_value_loss_epoch = batch_value_loss / len(batches)
            epoch_pbar.set_postfix(
                {
                    "policy_loss": f"{avg_policy_loss_epoch:.4f}",
                    "value_loss": f"{avg_value_loss_epoch:.4f}",
                }
            )

        # Clear memory
        self.memory.clear()

        # Compute average losses
        avg_policy_loss = total_policy_loss / (self.n_epochs * len(batches))
        avg_value_loss = total_value_loss / (self.n_epochs * len(batches))
        avg_entropy = total_entropy / (self.n_epochs * len(batches))

        # Store metrics
        self.policy_loss_history.append(avg_policy_loss)
        self.value_loss_history.append(avg_value_loss)
        self.entropy_history.append(avg_entropy)

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
        }

    def train(
        self,
        num_episodes: int,
        max_steps: int = 1000,
        update_freq: int = 2048,
        log_freq: int = 10,
        save_freq: int = 100,
        eval_freq: int = 50,
        num_eval_episodes: int = 3,
    ) -> Dict[str, List[float]]:
        """
        Train the agent.

        Args:
            num_episodes: Number of episodes to train for
            max_steps: Maximum number of steps per episode
            update_freq: Frequency of policy updates
            log_freq: Frequency of logging
            save_freq: Frequency of saving the model
            eval_freq: Frequency of evaluating the model
            num_eval_episodes: Number of episodes to evaluate on

        Returns:
            Dictionary of training history
        """
        start_time = time.time()

        # Training metrics
        total_steps = 0
        best_eval_reward = -float("inf")
        episode_rewards = []

        self.logger.info(f"Starting training for {num_episodes} episodes...")
        self.logger.info(f"Device: {self.device}")

        # Create progress bar
        pbar = tqdm(total=num_episodes, desc="Training Progress")

        for episode in range(1, num_episodes + 1):
            # Reset environment
            state, _ = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            step = 0

            # Reset LSTM hidden state if using LSTM
            if isinstance(self.model, ActorCriticLSTM):
                self.model.reset_hidden_state()

            # Create episode step progress bar
            episode_pbar = tqdm(total=max_steps, desc=f"Episode {episode}", leave=False)

            while not (done or truncated) and step < max_steps:
                # Choose action
                action, action_prob, value = self.choose_action(state)

                # Take step in environment
                next_state, reward, done, truncated, _ = self.env.step(action)

                # Store transition in memory
                self.memory.store(state, action, action_prob, value, reward, done)

                # Update state and metrics
                state = next_state
                episode_reward += reward
                total_steps += 1
                step += 1

                # Update episode progress bar
                episode_pbar.update(1)
                episode_pbar.set_postfix(
                    {"reward": f"{episode_reward:.2f}", "action": action}
                )

                # Update policy if memory is full
                if total_steps % update_freq == 0:
                    update_metrics = self.update()
                    self.logger.info(
                        f"Episode {episode}, "
                        f"Update metrics: policy_loss={update_metrics['policy_loss']:.4f}, "
                        f"value_loss={update_metrics['value_loss']:.4f}, "
                        f"entropy={update_metrics['entropy']:.4f}"
                    )
                    # Update main progress bar postfix with update metrics
                    pbar.set_postfix(
                        {
                            "policy_loss": f"{update_metrics['policy_loss']:.4f}",
                            "value_loss": f"{update_metrics['value_loss']:.4f}",
                            "entropy": f"{update_metrics['entropy']:.4f}",
                        }
                    )

            # Close episode progress bar
            episode_pbar.close()

            # Store episode reward
            episode_rewards.append(episode_reward)
            self.reward_history.append(episode_reward)

            # Log episode metrics
            if episode % log_freq == 0:
                avg_reward = np.mean(episode_rewards[-log_freq:])
                elapsed_time = time.time() - start_time
                self.logger.info(
                    f"Episode {episode}/{num_episodes}, "
                    f"Avg Reward: {avg_reward:.2f}, "
                    f"Steps: {total_steps}, "
                    f"Time: {elapsed_time:.2f}s"
                )
                # Update main progress bar with reward info
                pbar.set_postfix(
                    {"avg_reward": f"{avg_reward:.2f}", "steps": total_steps}
                )

            # Evaluate model
            if episode % eval_freq == 0:
                eval_reward = self.evaluate(num_eval_episodes)
                self.logger.info(
                    f"Evaluation at episode {episode}: "
                    f"Mean reward: {eval_reward:.2f}"
                )
                # Update main progress bar with eval info
                pbar.set_postfix({"eval_reward": f"{eval_reward:.2f}"})

                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.save(os.path.join(self.save_dir, "best_model.pt"))
                    self.logger.info(
                        f"New best model with evaluation reward: {best_eval_reward:.2f}"
                    )
                    pbar.set_postfix({"best_reward": f"{best_eval_reward:.2f}"})

            # Save checkpoint
            if episode % save_freq == 0:
                self.save(os.path.join(self.save_dir, f"model_ep{episode}.pt"))

            # Update main progress bar
            pbar.update(1)

        # Close main progress bar
        pbar.close()

        # Save final model
        self.save(os.path.join(self.save_dir, "final_model.pt"))

        # Return training history
        return {
            "rewards": self.reward_history,
            "policy_loss": self.policy_loss_history,
            "value_loss": self.value_loss_history,
            "entropy": self.entropy_history,
        }

    def evaluate(self, num_episodes: int = 10) -> float:
        """
        Evaluate the agent.

        Args:
            num_episodes: Number of episodes to evaluate on

        Returns:
            Mean reward over evaluation episodes
        """
        total_rewards = []

        # Create progress bar for evaluation
        eval_pbar = tqdm(range(num_episodes), desc="Evaluating", leave=False)

        for i in eval_pbar:
            state, _ = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            step = 0

            # Reset LSTM hidden state if using LSTM
            if isinstance(self.model, ActorCriticLSTM):
                self.model.reset_hidden_state()

            while not (done or truncated):
                # Choose action (no exploration)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                    if isinstance(self.model, ActorCriticLSTM):
                        logits, _ = self.model(state_tensor, reset_hidden=False)
                    else:
                        logits, _ = self.model(state_tensor)

                    action_probs = torch.softmax(logits, dim=-1)
                    action = torch.argmax(action_probs, dim=1).item()

                # Take step in environment
                state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                step += 1

                # Update progress bar
                eval_pbar.set_postfix(
                    {"ep": i + 1, "reward": f"{episode_reward:.2f}", "steps": step}
                )

            total_rewards.append(episode_reward)

            # Update progress bar with final episode reward
            eval_pbar.set_postfix(
                {"ep": i + 1, "reward": f"{episode_reward:.2f}", "steps": step}
            )

        mean_reward = np.mean(total_rewards)
        return mean_reward

    def save(self, path: str) -> None:
        """
        Save the agent to disk or cloud storage.

        Args:
            path: Path to save the agent to (local path or gs:// URL)
        """
        # Prepare model data
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_type": self.model.__class__.__name__,
            "architecture": self.model.__class__.__name__,
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "reward_history": self.reward_history,
            "policy_loss_history": self.policy_loss_history,
            "value_loss_history": self.value_loss_history,
            "entropy_history": self.entropy_history,
        }

        if path.startswith("gs://"):
            # Import Google Cloud Storage if needed
            try:
                import io

                from google.cloud import storage
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required to save to GCS. Install with pip install google-cloud-storage"
                )

            # Parse bucket and blob path
            path_parts = path[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1] if len(path_parts) > 1 else ""

            # Save to a temporary buffer first
            buffer = io.BytesIO()
            torch.save(model_data, buffer)
            buffer.seek(0)

            # Upload to GCS
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.upload_from_file(buffer, content_type="application/octet-stream")
                print(f"Model saved to GCS: {path}")
            except Exception as e:
                print(f"Error saving model to GCS: {e}")
                raise
        else:
            # Save locally
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save model
            torch.save(model_data, path)
            print(f"Model saved locally: {path}")

    def load(self, path: str) -> None:
        """
        Load the agent from disk or cloud storage.

        Args:
            path: Path to load the agent from (local path or gs:// URL)
        """
        if path.startswith("gs://"):
            # Import Google Cloud Storage if needed
            try:
                import io

                from google.cloud import storage
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required to load from GCS. Install with pip install google-cloud-storage"
                )

            # Parse bucket and blob path
            path_parts = path[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1] if len(path_parts) > 1 else ""

            try:
                # Download from GCS to a buffer
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)

                buffer = io.BytesIO()
                blob.download_to_file(buffer)
                buffer.seek(0)

                # Load checkpoint from buffer
                try:
                    # First try with weights_only=False
                    checkpoint = torch.load(
                        buffer, map_location=self.device, weights_only=False
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not load with weights_only=False, trying with weights_only=True: {e}"
                    )
                    # Reset buffer and try again with weights_only=True
                    buffer.seek(0)
                    checkpoint = torch.load(
                        buffer, map_location=self.device, weights_only=True
                    )

                print(f"Model loaded from GCS: {path}")
            except Exception as e:
                raise RuntimeError(f"Error loading model from GCS: {e}")
        else:
            # Load from local file
            try:
                # First try with weights_only=False (for backward compatibility)
                checkpoint = torch.load(
                    path, map_location=self.device, weights_only=False
                )
            except Exception as e:
                print(
                    f"Warning: Could not load with weights_only=False, trying with weights_only=True: {e}"
                )
                # If that fails, try with weights_only=True
                checkpoint = torch.load(
                    path, map_location=self.device, weights_only=True
                )

            print(f"Model loaded from local file: {path}")

        # Load model
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer if available
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")

        # Load training history if available
        if "reward_history" in checkpoint:
            self.reward_history = checkpoint["reward_history"]
        if "policy_loss_history" in checkpoint:
            self.policy_loss_history = checkpoint["policy_loss_history"]
        if "value_loss_history" in checkpoint:
            self.value_loss_history = checkpoint["value_loss_history"]
        if "entropy_history" in checkpoint:
            self.entropy_history = checkpoint["entropy_history"]
