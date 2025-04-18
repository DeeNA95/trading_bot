"""
Proximal Policy Optimization (PPO) agent for reinforcement learning in trading environments.

This module implements a PPO agent that can be used to train a trading policy
using various neural network architectures (CNN, LSTM, Transformer).
"""

import logging
import os
import time
from typing import Dict, List, Tuple, Type, Any, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from rl_agent.environment.trading_env import BinanceFuturesCryptoEnv

try:
    import torch_xla.core.xla_model as xm
    _HAS_XLA = True
except ImportError:
    _HAS_XLA = False

from .models import ActorCriticWrapper
from .transformer.encoder_only import EncoderOnlyTransformer
from .transformer.encoder_decoder_mha import EncoderDecoderTransformer


class PPOMemory:
    """
    Memory buffer for PPO using PyTorch tensors.
    """
    def __init__(self, batch_size: int, device: torch.device):
        self.batch_size = batch_size
        self.device = device
        self.clear()  # Initialize lists

    def store(
        self,
        state: torch.Tensor,  # Expect tensor
        action: torch.Tensor,  # Expect tensor
        probs: torch.Tensor,  # Expect tensor
        vals: torch.Tensor,  # Expect tensor
        reward: torch.Tensor,  # Expect tensor
        done: torch.Tensor,  # Expect tensor
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self) -> None:
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self) -> Tuple[torch.Tensor, ...]:
        n_states = len(self.states)
        # Stack tensors along a new dimension (batch dimension)
        states_tensor = torch.stack(self.states).to(self.device)
        actions_tensor = torch.stack(self.actions).to(self.device)
        probs_tensor = torch.stack(self.probs).to(self.device)
        vals_tensor = torch.stack(self.vals).to(self.device)
        rewards_tensor = torch.stack(self.rewards).to(self.device)
        dones_tensor = torch.stack(self.dones).to(self.device)

        # Generate random indices for batches using torch
        indices = torch.randperm(n_states).to(self.device)
        batches = [indices[i: i + self.batch_size] for i in range(0, n_states, self.batch_size)]

        return (
            states_tensor,
            actions_tensor,
            probs_tensor,
            vals_tensor,
            rewards_tensor,
            dones_tensor,
            batches,  # List of index tensors
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
        transformer_arch_class: Type[nn.Module],
        transformer_core_config: Dict[str, Any],
        # --- PPO Hyperparameters ---
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        batch_size: int = 2048,
        n_epochs: int = 10,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_gae: bool = True,
        normalize_advantage: bool = True,
        weight_decay: float = 0.01,
        # --- Core Model Configuration ---

        # --- Wrapper/Head Configuration ---
        actor_critic_wrapper_class: Type[nn.Module] = ActorCriticWrapper,
        wrapper_config: Optional[Dict[str, Any]] = None,
        # --- General ---
        device: str = "auto",
        save_dir: str = "models",
    ):
        """
        Initialize the PPO agent.

        Args:
            env: custom environment to train on
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE (Generalized Advantage Estimation) lambda parameter
            policy_clip: PPO clipping parameter
            batch_size: Batch size for training
            n_epochs: Number of epochs to update the policy for each batch of experiences
            entropy_coef: Entropy coefficient for exploration
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for gradient clipping
            use_gae: Whether to use Generalized Advantage Estimation
            normalize_advantage: Whether to normalize advantages
            weight_decay: L2 regularization parameter
            transformer_arch_class: Class of the core transformer
            transformer_core_config: Configuration dictionary for the core transformer
            actor_critic_wrapper_class: Class of the actor-critic wrapper
            wrapper_config: Configuration dictionary for the wrapper
            device: Device to run on ('cpu', 'cuda', 'mps', or 'auto')
            save_dir: Directory to save models to
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

        # Determine device (including TPU/XLA)
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif _HAS_XLA:
                self.device = xm.xla_device()
                print("Using TPU (XLA) device.")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            if "xla" in device and not _HAS_XLA:
                print("Warning: XLA device specified but torch_xla not found. Falling back to CPU.")
                self.device = torch.device("cpu")

        # --- Instantiate Core Transformer ---
        if 'window_size' not in transformer_core_config:
            transformer_core_config['window_size'] = env.window_size
        if 'n_embd' not in transformer_core_config:
            if wrapper_config and 'embedding_dim' in wrapper_config:
                transformer_core_config['n_embd'] = wrapper_config['embedding_dim']
            else:
                raise ValueError("embedding_dim ('n_embd') must be provided in transformer_core_config or wrapper_config")

        transformer_core = transformer_arch_class(**transformer_core_config)

        # --- Instantiate ActorCriticWrapper ---
        if wrapper_config is None:
            wrapper_config = {}
        if 'embedding_dim' not in wrapper_config:
            wrapper_config['embedding_dim'] = transformer_core_config['n_embd']
        elif wrapper_config['embedding_dim'] != transformer_core_config['n_embd']:
            raise ValueError("Wrapper embedding_dim must match core n_embd")

        self.model = actor_critic_wrapper_class(
            input_shape=self.input_shape,
            action_dim=self.action_dim,
            transformer_core=transformer_core,
            device=self.device,
            **wrapper_config
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=self.weight_decay
        )

        # Memory buffer
        self.memory = PPOMemory(batch_size, self.device)

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

    def choose_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Choose an action based on the current state.

        Args:
            state: Current environment state

        Returns:
            Tuple of (action, action probabilities, value)
        """
        # Check for NaN values in state and replace with zeros
        state = np.nan_to_num(state, nan=0.0)

        # Convert state to tensor ON THE CORRECT DEVICE
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get action probabilities and value from model
        with torch.no_grad():
            try:
                logits, value = self.model(state_tensor)
                logits = torch.clamp(logits, min=-20.0, max=20.0)
                action_probs = torch.softmax(logits, dim=-1)
                action_probs = action_probs + 1e-8
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

                dist = Categorical(action_probs)
                action = dist.sample()

                # Get action probability and value as tensors
                action_prob_tensor = action_probs.gather(1, action.unsqueeze(-1)).squeeze()
                value_tensor = value.squeeze()

                # Store tensors in memory later, return scalar values for env interaction
                action_item = action.item()
                action_prob_item = action_prob_tensor.item()
                value_item = value_tensor.item() if not torch.isnan(value_tensor).any() else 0.0

            except Exception as e:
                print(f"Error in choose_action: {e}")
                action_item = np.random.randint(0, self.action_dim)
                action_prob_item = 1.0 / self.action_dim
                value_item = 0.0

        return action_item, action_prob_item, value_item

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
        states = torch.nan_to_num(states, nan=0.0)
        states = torch.clamp(states, min=-10.0, max=10.0)

        try:
            logits, values = self.model(states)
            logits = torch.clamp(logits, min=-20.0, max=20.0)
            action_probs = torch.softmax(logits, dim=-1)
            action_probs = action_probs + 1e-8
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

            dist = Categorical(action_probs)
            action_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            action_log_probs = torch.nan_to_num(action_log_probs, nan=0.0)
            values = torch.nan_to_num(values, nan=0.0)
            entropy = torch.nan_to_num(entropy, nan=0.0)

        except Exception as e:
            print(f"Error in evaluate_actions: {e}")
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
        states, actions, old_probs, values, rewards, dones, batches = (
            self.memory.generate_batches()
        )

        advantages = torch.zeros_like(rewards).to(self.device)
        if self.use_gae:
            last_gae = 0.0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = 0.0
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]

                delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t].float()) - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t].float()) * last_gae
                advantages[t] = last_gae
        else:
            returns = torch.zeros_like(rewards).to(self.device)
            running_return = 0.0
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + self.gamma * running_return * (1.0 - dones[t].float())
                returns[t] = running_return
            advantages = returns - values

        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        epoch_pbar = tqdm(range(self.n_epochs), desc="PPO Update", leave=False)

        for _ in epoch_pbar:
            batch_policy_loss = 0
            batch_value_loss = 0
            batch_entropy = 0
            batch_pbar = tqdm(batches, desc="Batches", leave=False)

            for batch_indices in batch_pbar:
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_probs = old_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_values = values[batch_indices]

                new_log_probs, new_values, entropy = self.evaluate_actions(
                    batch_states, batch_actions
                )

                prob_ratio = torch.exp(new_log_probs - torch.log(batch_old_probs + 1e-8))
                weighted_probs = batch_advantages * prob_ratio
                weighted_clipped_probs = batch_advantages * torch.clamp(
                    prob_ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip
                )
                policy_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = batch_advantages + batch_values
                value_loss = torch.nn.functional.mse_loss(new_values, returns)

                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                if self.device.type == 'xla':
                    xm.optimizer_step(self.optimizer)
                else:
                    self.optimizer.step()

                batch_policy_loss += policy_loss.item()
                batch_value_loss += value_loss.item()
                batch_entropy += entropy_loss.item()
                batch_pbar.set_postfix({"policy_loss": f"{policy_loss.item():.4f}", "value_loss": f"{value_loss.item():.4f}"})

            avg_policy_loss_epoch = batch_policy_loss / len(batches)
            avg_value_loss_epoch = batch_value_loss / len(batches)
            epoch_pbar.set_postfix({"policy_loss": f"{avg_policy_loss_epoch:.4f}", "value_loss": f"{avg_value_loss_epoch:.4f}"})

        self.memory.clear()
        avg_policy_loss = total_policy_loss / (self.n_epochs * len(batches))
        avg_value_loss = total_value_loss / (self.n_epochs * len(batches))
        avg_entropy = total_entropy / (self.n_epochs * len(batches))
        self.policy_loss_history.append(avg_policy_loss)
        self.value_loss_history.append(avg_value_loss)
        self.entropy_history.append(avg_entropy)

        return {"policy_loss": avg_policy_loss, "value_loss": avg_value_loss, "entropy": avg_entropy}

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

        total_steps = 0
        best_eval_reward = -float("inf")
        episode_rewards = []

        self.logger.info(f"Starting training for {num_episodes} episodes...")
        self.logger.info(f"Device: {self.device}")

        pbar = tqdm(total=num_episodes, desc="Training Progress")

        for episode in range(1, num_episodes + 1):
            state_np, _ = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            step = 0
            episode_pbar = tqdm(total=max_steps, desc=f"Episode {episode}", leave=False)

            while not (done or truncated) and step < max_steps:
                action_item, action_prob_item, value_item = self.choose_action(state_np)

                next_state_np, reward_item, done_item, truncated_item, _ = self.env.step(action_item)

                state_tensor = torch.FloatTensor(state_np).to(self.device)
                action_tensor = torch.tensor(action_item, dtype=torch.long).to(self.device)
                prob_tensor = torch.tensor(action_prob_item, dtype=torch.float).to(self.device)
                val_tensor = torch.tensor(value_item, dtype=torch.float).to(self.device)
                reward_tensor = torch.tensor(reward_item, dtype=torch.float).to(self.device)
                done_tensor = torch.tensor(done_item, dtype=torch.bool).to(self.device)

                self.memory.store(state_tensor, action_tensor, prob_tensor, val_tensor, reward_tensor, done_tensor)

                state_np = next_state_np
                episode_reward += reward_item
                total_steps += 1
                step += 1

                episode_pbar.update(1)
                episode_pbar.set_postfix({"reward": f"{episode_reward:.2f}", "action": action_item})

                if total_steps % update_freq == 0 and len(self.memory.states) >= self.memory.batch_size:
                    update_metrics = self.update()
                    self.logger.info(
                        f"Episode {episode}, "
                        f"Update metrics: policy_loss={update_metrics['policy_loss']:.4f}, "
                        f"value_loss={update_metrics['value_loss']:.4f}, "
                        f"entropy={update_metrics['entropy']:.4f}"
                    )
                    pbar.set_postfix(
                        {
                            "policy_loss": f"{update_metrics['policy_loss']:.4f}",
                            "value_loss": f"{update_metrics['value_loss']:.4f}",
                            "entropy": f"{update_metrics['entropy']:.4f}",
                        }
                    )

            episode_pbar.close()

            episode_rewards.append(episode_reward)
            self.reward_history.append(episode_reward)

            if episode % log_freq == 0:
                avg_reward = np.mean(episode_rewards[-log_freq:])
                elapsed_time = time.time() - start_time
                self.logger.info(
                    f"Episode {episode}/{num_episodes}, "
                    f"Avg Reward: {avg_reward:.2f}, "
                    f"Steps: {total_steps}, "
                    f"Time: {elapsed_time:.2f}s"
                )
                pbar.set_postfix(
                    {"avg_reward": f"{avg_reward:.2f}", "steps": total_steps}
                )

            if episode % eval_freq == 0:
                eval_reward = self.evaluate(num_eval_episodes)
                self.logger.info(
                    f"Evaluation at episode {episode}: "
                    f"Mean reward: {eval_reward:.2f}"
                )
                pbar.set_postfix({"eval_reward": f"{eval_reward:.2f}"})

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.save(os.path.join(self.save_dir, "best_model.pt"))
                    self.logger.info(
                        f"New best model with evaluation reward: {best_eval_reward:.2f}"
                    )
                    pbar.set_postfix({"best_reward": f"{best_eval_reward:.2f}"})

            if episode % save_freq == 0:
                self.save(os.path.join(self.save_dir, f"model_ep{episode}.pt"))

            pbar.update(1)

        pbar.close()

        self.save(os.path.join(self.save_dir, "final_model.pt"))

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

        eval_pbar = tqdm(range(num_episodes), desc="Evaluating", leave=False)

        for i in eval_pbar:
            state_np, _ = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            step = 0

            while not (done or truncated):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
                    logits, _ = self.model(state_tensor)
                    action_probs = torch.softmax(logits, dim=-1)
                    action = torch.argmax(action_probs, dim=1).item()

                state_np, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                step += 1
                eval_pbar.set_postfix(
                    {"ep": i + 1, "reward": f"{episode_reward:.2f}", "steps": step}
                )

            total_rewards.append(episode_reward)
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
            try:
                import io
                from google.cloud import storage
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required to save to GCS. Install with pip install google-cloud-storage"
                )

            path_parts = path[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1] if len(path_parts) > 1 else ""

            buffer = io.BytesIO()
            torch.save(model_data, buffer)
            buffer.seek(0)

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
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model_data, path)
            print(f"Model saved locally: {path}")

    def load(self, path: str) -> None:
        """
        Load the agent from disk or cloud storage.

        Args:
            path: Path to load the agent from (local path or gs:// URL)
        """
        if path.startswith("gs://"):
            try:
                import io
                from google.cloud import storage
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required to load from GCS. Install with pip install google-cloud-storage"
                )

            path_parts = path[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1] if len(path_parts) > 1 else ""

            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)

                buffer = io.BytesIO()
                blob.download_to_file(buffer)
                buffer.seek(0)

                try:
                    checkpoint = torch.load(
                        buffer, map_location=self.device, weights_only=False
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not load with weights_only=False, trying with weights_only=True: {e}"
                    )
                    buffer.seek(0)
                    checkpoint = torch.load(
                        buffer, map_location=self.device, weights_only=True
                    )

                print(f"Model loaded from GCS: {path}")
            except Exception as e:
                raise RuntimeError(f"Error loading model from GCS: {e}")
        else:
            try:
                checkpoint = torch.load(
                    path, map_location=self.device, weights_only=False
                )
            except Exception as e:
                print(
                    f"Warning: Could not load with weights_only=False, trying with weights_only=True: {e}"
                )
                checkpoint = torch.load(
                    path, map_location=self.device, weights_only=True
                )

            print(f"Model loaded from local file: {path}")

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")

        if "reward_history" in checkpoint:
            self.reward_history = checkpoint["reward_history"]
        if "policy_loss_history" in checkpoint:
            self.policy_loss_history = checkpoint["policy_loss_history"]
        if "value_loss_history" in checkpoint:
            self.value_loss_history = checkpoint["value_loss_history"]
        if "entropy_history" in checkpoint:
            self.entropy_history = checkpoint["entropy_history"]
