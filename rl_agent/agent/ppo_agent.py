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
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from rl_agent.environment.trading_env import BinanceFuturesCryptoEnv

try:
    import torch_xla.core.xla_model as xm
    _HAS_XLA = True
except ImportError:
    _HAS_XLA = False

from dataclasses import asdict # For saving ModelConfig

from .models import ActorCriticWrapper
# Removed old transformer imports as model is pre-built
# from .transformer.encoder_only import EncoderOnlyTransformer
# from .transformer.encoder_decoder_mha import EncoderDecoderTransformer

# Import ModelConfig type hint (assuming it's accessible)
# Need to adjust path based on final location of ModelConfig
try:
    # If ModelConfig is in training.model_factory
    from training.model_factory import ModelConfig
except ImportError:
    # Fallback if ModelConfig is moved or circular dependency occurs
    ModelConfig = Any # Use Any as a fallback type hint


class PPOMemory:
    """
    Memory buffer for PPO using PyTorch tensors.
    """
    def __init__(self, batch_size: int, device: torch.device, mini_batch_size: int = None):
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size if mini_batch_size is not None else batch_size
        self.device = device
        self.clear()  # Initialize lists

    def store(
        self,
        state,
        action,
        probs,
        vals,
        reward,
        done,
    ) -> None:
        # state is already a tensor when passed from PPOAgent.train
        self.states.append(state)  # Append the tensor directly
        self.actions.append(action)  # Append the tensor directly
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

        # Convert lists to numpy arrays first
        try:
            states_np = np.array(self.states, dtype=np.float32)
            actions_np = np.array(self.actions, dtype=np.int64)
            probs_np = np.array(self.probs, dtype=np.float32)
            vals_np = np.array(self.vals, dtype=np.float32)
            rewards_np = np.array(self.rewards, dtype=np.float32)
            dones_np = np.array(self.dones, dtype=np.bool_)
        except ValueError as e:
            print(f"Error converting lists to NumPy arrays: {e}")
            # Potentially add more robust error handling or logging here
            # For example, inspect the types within the lists if conversion fails
            print("Types in self.states:", [type(s) for s in self.states[:5]])
            raise

        # Convert numpy arrays to torch tensors
        states_tensor = torch.from_numpy(states_np).to(self.device)
        actions_tensor = torch.from_numpy(actions_np).to(self.device)
        probs_tensor = torch.from_numpy(probs_np).to(self.device)
        vals_tensor = torch.from_numpy(vals_np).to(self.device)
        rewards_tensor = torch.from_numpy(rewards_np).to(self.device)
        dones_tensor = torch.from_numpy(dones_np).to(self.device)

        # Generate random indices for batches using torch
        indices = torch.randperm(n_states).to(self.device)

        # Use mini_batch_size for actual processing to reduce memory usage
        batches = [indices[i: i + self.mini_batch_size] for i in range(0, n_states, self.mini_batch_size)]

        return (
            states_tensor,
            actions_tensor,
            probs_tensor,
            vals_tensor,
            rewards_tensor,
            dones_tensor,
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
        model: ActorCriticWrapper, # Accept pre-built model
        model_config: ModelConfig, # Accept ModelConfig used to build the model
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
        # --- Learning Rate Scheduler ---
        use_lr_scheduler: bool = True,
        min_lr: float = 1e-5,
        lr_scheduler_max_steps: int = 100000,
        # --- Memory Optimization ---
        gradient_accumulation_steps: int = 4,  # Number of steps to accumulate gradients
        mini_batch_size: int = None,  # Size of mini-batches for processing (defaults to batch_size/gradient_accumulation_steps)
        # --- Core Model Configuration (Removed - handled by factory) ---

        # --- Wrapper/Head Configuration (Removed - handled by factory) ---
        # --- General ---
        device: str = "auto", # Device still needed for agent logic & memory
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
            use_lr_scheduler: Whether to use a learning rate scheduler
            min_lr: Minimum learning rate for the scheduler
            lr_scheduler_max_steps: Number of steps for the scheduler to reach min_lr
            model: Pre-instantiated ActorCriticWrapper model instance.
            model_config: The ModelConfig instance used to create the model.
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

        # Memory optimization parameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mini_batch_size = mini_batch_size if mini_batch_size is not None else max(1, batch_size // gradient_accumulation_steps)

        # Learning rate scheduler parameters
        self.use_lr_scheduler = use_lr_scheduler
        self.min_lr = min_lr
        self.lr_scheduler_max_steps = lr_scheduler_max_steps
        self.total_steps = 0  # Track total steps for scheduler

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
            # if "xla" in device and not _HAS_XLA:
            #     print("Warning: XLA device specified but torch_xla not found. Falling back to CPU.")
            #     self.device = torch.device("cpu")

        # --- Use Pre-instantiated Model ---
        self.model = model
        self.model_config = model_config # Store the model config
        # Ensure the passed model is on the correct device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW( # try AdamW
            self.model.parameters(), lr=lr, weight_decay=self.weight_decay
        )

        # Learning rate scheduler
        if self.use_lr_scheduler:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.lr_scheduler_max_steps,
                eta_min=self.min_lr
            )
        else:
            self.scheduler = None

        # Memory buffer
        self.memory = PPOMemory(batch_size, self.device, self.mini_batch_size)

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

    def choose_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Choose an action based on the current state.

        Args:
            state: Current environment state
            deterministic: Whether to select actions deterministically

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

                # Apply temperature scaling to logits
                logits = logits / self.model.temperature

                logits = torch.clamp(logits, min=-20.0, max=20.0)
                action_probs = torch.softmax(logits, dim=-1)
                action_probs = action_probs + 1e-8
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

                # --- DEBUG: Print state stats ---
                # print(f"DEBUG state_tensor mean: {state_tensor.mean():.4f}, std: {state_tensor.std():.4f}")

                dist = Categorical(action_probs)
                action = dist.sample()

                # Get action probability and value as tensors
                # --- DEBUG: Print raw logits ---
                # print(f"DEBUG action_logits: {logits.detach().cpu().numpy()}")

                action_prob_tensor = action_probs.gather(1, action.unsqueeze(-1)).squeeze()
                value_tensor = value.squeeze()

                # Store tensors in memory later, return scalar values for env interaction
                action_item = action.item()
                action_prob_item = action_prob_tensor.item()
                value_item = value_tensor.item() if not torch.isnan(value_tensor).any() else 0.0

            except Exception as e: # Correctly aligned except block
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
            # Create fallback tensors on the correct device
            action_log_probs = torch.zeros_like(actions, dtype=torch.float32, device=self.device)
            values = torch.zeros((len(actions),), dtype=torch.float32, device=self.device)
            entropy = torch.zeros((len(actions),), dtype=torch.float32, device=self.device)

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
                    next_non_terminal = 1.0 - dones[t].float()
                    next_value = 0.0
                else:
                    next_non_terminal = 1.0 - dones[t + 1].float()
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

            # Zero gradients at the beginning of each epoch
            self.optimizer.zero_grad()

            # Track accumulated batches for gradient accumulation
            accumulated_batches = 0

            for batch_idx, batch_indices in enumerate(batch_pbar):
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_probs = old_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_values = values[batch_indices]

                # Use torch.cuda.amp.autocast() to reduce memory usage if available
                try:
                    with torch.amp.autocast('cuda',enabled=self.device.type=='cuda'):
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

                        # Scale the loss by 1/gradient_accumulation_steps for proper scaling
                        loss = (
                            policy_loss
                            + self.value_coef * value_loss
                            - self.entropy_coef * entropy_loss
                        ) / self.gradient_accumulation_steps
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"CUDA OOM in batch {batch_idx}. Skipping batch and clearing memory.")
                        # Try to free memory
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

                # Backward pass
                loss.backward()

                # Track metrics
                batch_policy_loss += policy_loss.item()
                batch_value_loss += value_loss.item()
                batch_entropy += entropy_loss.item()
                batch_pbar.set_postfix({
                    "policy_loss": f"{policy_loss.item():.4f}",
                    "value_loss": f"{value_loss.item():.4f}",
                    "acc_batch": f"{accumulated_batches+1}/{self.gradient_accumulation_steps}"
                })

                # Increment accumulated batches counter
                accumulated_batches += 1

                # Only update weights after accumulating gradients from multiple batches
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(batches) - 1:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    # Update weights
                    if self.device.type == 'xla':
                        xm.optimizer_step(self.optimizer)
                    else:
                        self.optimizer.step()

                    # Zero gradients for next accumulation cycle
                    self.optimizer.zero_grad()
                    accumulated_batches = 0

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
                self.total_steps += 1
                step += 1

                # Step the learning rate scheduler if enabled
                if self.use_lr_scheduler and self.scheduler is not None:
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                    if self.total_steps % 1000 == 0:  # Log LR periodically
                        self.logger.info(f"Current learning rate: {current_lr:.6f}")

                episode_pbar.update(1)
                episode_pbar.set_postfix({"reward": f"{episode_reward:.2f}", "action": action_item})

                if self.total_steps % update_freq == 0 and len(self.memory.states) >= self.memory.batch_size:
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
                    f"Steps: {self.total_steps}, "
                    f"Time: {elapsed_time:.2f}s"
                )
                pbar.set_postfix(
                    {"avg_reward": f"{avg_reward:.2f}", "steps": self.total_steps}
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
        model_state_dict = self.model.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()

        checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'total_steps': self.total_steps,
            'epoch': getattr(self, 'current_epoch', None),
            'best_reward': self.best_reward,
            'model_config': asdict(self.model_config) if self.model_config else None, # Save the model config
            # 'training_args': vars(self.args) if hasattr(self, 'args') else None, # Keep PPO/env args if needed
            'timestamp': time.strftime("%Y%m%d-%H%M%S"),
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
            torch.save(checkpoint, buffer) # Save the checkpoint dictionary
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
            torch.save(checkpoint, path) # Save the checkpoint dictionary
            print(f"Model saved locally: {path}")
        self.logger.info(f"Model checkpoint saved to: {path}")

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

        # Load scheduler state if available
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}")

        # Load total steps for scheduler
        if "total_steps" in checkpoint:
            self.total_steps = checkpoint["total_steps"]

        # Load history if available
        self.reward_history = checkpoint.get("reward_history", [])
        self.policy_loss_history = checkpoint.get("policy_loss_history", [])
        self.value_loss_history = checkpoint.get("value_loss_history", [])
        self.entropy_history = checkpoint.get("entropy_history", [])
        # Note: We don't load model_config here, as the model structure should be
        # recreated using the saved config *before* calling load.
        # self.model_config = ModelConfig(**checkpoint['model_config']) if 'model_config' in checkpoint else None
