import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Type
import logging
import time
import io # For GCS saving
import os
import json
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler

# RL Agent and Environment Imports
from rl_agent.agent.ppo_agent import PPOAgent
from rl_agent.environment.trading_env import BinanceFuturesCryptoEnv

# Model Factory and Config
from .model_factory import create_model, ModelConfig

logger = logging.getLogger(__name__)

class Trainer:
    """
    Handles the training and evaluation process for a single walk-forward fold.
    Creates the model using the factory, then creates the agent and environments.
    """
    def __init__(self, fold_num: int, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, model_config: ModelConfig, trainer_args: Any):
        """
        Initializes the Trainer for a specific fold.

        Args:
            fold_num: The current fold number.
            train_df: DataFrame for training.
            val_df: DataFrame for validation.
            test_df: DataFrame for testing.
            model_config: Configuration object for the model architecture.
            trainer_args: Namespace object containing trainer, environment,
                          and PPO hyperparameters (but not model architecture args).
        """
        self.fold_num = fold_num
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.model_config = model_config # Store model config
        self.args = trainer_args # Store remaining args (trainer, env, ppo)

        # Extract necessary args for convenience (consider passing trainer_config dataclass instead)
        self.symbol = self.args.symbol
        # Get window_size from model_config as it's part of the architecture definition
        self.window_size = self.model_config.window_size
        self.leverage = self.args.leverage
        self.episodes = self.args.episodes
        self.batch_size = self.args.batch_size
        self.update_freq = self.args.update_freq # Note: PPOAgent might handle update logic internally based on batch_size
        self.eval_freq = self.args.eval_freq
        self.dynamic_leverage = self.args.dynamic_leverage
        self.use_risk_adjusted_rewards = self.args.use_risk_adjusted_rewards
        self.max_position = self.args.max_position
        self.initial_balance = self.args.balance
        self.risk_reward_ratio = self.args.risk_reward
        self.stop_loss_percent = self.args.stop_loss
        self.trade_fee_percent = self.args.trade_fee
        self.base_save_path = self.args.save_path
        self.device_setting = self.args.device # Store requested device setting

        # Determine device
        if self.device_setting == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                 self.device = "mps"
            # Add XLA check if needed: elif hasattr(torch, "xla") and torch.xla.is_available(): self.device = "xla"
            else:
                self.device = "cpu"
        else:
            self.device = self.device_setting
        logger.info(f"Fold {self.fold_num}: Using device: {self.device}")


        # Define fold-specific save path
        self.fold_save_path = os.path.join(self.base_save_path, f"fold_{self.fold_num}")
        os.makedirs(self.fold_save_path, exist_ok=True)
        self.best_model_path = os.path.join(self.fold_save_path, "best_model.pt")
        self.scaler_path = os.path.join(self.fold_save_path, f"scaler_fold_{self.fold_num}.joblib") # Define scaler path

    def _create_environment(self, df: pd.DataFrame, mode: str) -> BinanceFuturesCryptoEnv:
        """Helper to create environment instance."""
        env = BinanceFuturesCryptoEnv(
            df=df,
            symbol=self.symbol,
            window_size=self.window_size, # Use window_size from init
            mode=mode,
            leverage=self.leverage,
            max_position=self.max_position,
            initial_balance=self.initial_balance,
            risk_reward_ratio=self.risk_reward_ratio,
            stop_loss_percent=self.stop_loss_percent,
            dynamic_leverage=self.dynamic_leverage,
            use_risk_adjusted_rewards=self.use_risk_adjusted_rewards,
            trade_fee_percent=self.trade_fee_percent,
        )
        return env

    def _create_agent(self, env: BinanceFuturesCryptoEnv, model: nn.Module, model_config: ModelConfig) -> PPOAgent:
         """Helper to create the PPO agent instance using a pre-built model and its config."""
         # PPOAgent now takes the instantiated model and its config
         agent = PPOAgent(
             env=env,
             model=model, # Pass the instantiated model from the factory
             model_config=model_config, # Pass the config used to create the model
             batch_size=self.args.batch_size,
             save_dir=self.fold_save_path,
             # PPO Hyperparameters from args
             lr=self.args.lr,
             gamma=self.args.gamma,
             gae_lambda=self.args.gae_lambda,
             policy_clip=self.args.policy_clip,
             n_epochs=self.args.n_epochs,
             entropy_coef=self.args.entropy_coef,
             value_coef=self.args.value_coef,
             max_grad_norm=self.args.max_grad_norm,
             device=self.device, # Use determined device
             use_gae=self.args.use_gae,
             normalize_advantage=self.args.normalize_advantage,
             weight_decay=self.args.weight_decay,
         )
         return agent

    def train_and_evaluate_fold(self) -> Tuple[Optional[Dict[str, float]], Dict]:
        """
        Performs the training, validation, and testing for the current fold.
        """
        logger.info(f"--- Starting Processing for Fold {self.fold_num} ---")

        if self.train_df.empty or self.val_df.empty or self.test_df.empty:
            logger.error(f"Fold {self.fold_num}: Empty dataframe received. Skipping.")
            return None, {} # Indicate failure

        # --- Environment Creation ---
        logger.info(f"Fold {self.fold_num}: Creating environments...")
        train_env = self._create_environment(self.train_df, mode="train")
        val_env = self._create_environment(self.val_df, mode="test") # Val uses test logic

        # --- Model Creation using Factory ---
        logger.info(f"Fold {self.fold_num}: Creating model using factory...")
        try:
            # Pass the ModelConfig and the determined device
            model = create_model(config=self.model_config, device=self.device)
            logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")
        except Exception as e:
            logger.error(f"Fold {self.fold_num}: Failed to create model using factory - {e}", exc_info=True)
            return None, {}

        # --- Agent Creation ---
        logger.info(f"Fold {self.fold_num}: Creating PPO Agent...")
        try:
            # Pass the created model and its config to the agent
            agent = self._create_agent(train_env, model, self.model_config)
        except Exception as e: # Catch broader exceptions during agent init
             logger.error(f"Fold {self.fold_num}: Failed to create agent - {e}", exc_info=True)
             return None, {}

        # --- Fit and Save Scaler ---
        logger.info(f"Fold {self.fold_num}: Fitting scaler on training data...")
        try:
            # Prepare data for scaler (exclude non-feature columns used by env state)
            # Assuming train_df already has features calculated
            # Need to identify feature columns consistently
            # Let's assume features are all numeric columns excluding potential known non-features
            # This might need refinement based on how train_df is structured
            numeric_cols = self.train_df.select_dtypes(include=np.number).columns.tolist()
            # Exclude columns added later in env state if they exist in train_df (e.g., balance, position - though unlikely)
            cols_to_exclude_from_scaling = ['balance', 'position', 'unrealized_pnl'] # Add others if needed
            feature_cols = [col for col in numeric_cols if col not in cols_to_exclude_from_scaling]

            if not feature_cols:
                 raise ValueError("No numeric feature columns found in train_df for scaling.")

            scaler = StandardScaler() # Assumes StandardScaler is imported
            # Fit only on the identified feature columns of the training data for this fold
            scaler.fit(self.train_df[feature_cols].fillna(0)) # Fill NaNs before fitting

            # --- Save Scaler (handling GCS) ---
            if self.scaler_path.startswith("gs://"):
                try:
                    from google.cloud import storage # Import here to avoid making it a hard dependency if not used
                except ImportError:
                    raise ImportError("google-cloud-storage is required to save scaler to GCS.")

                path_parts = self.scaler_path[5:].split("/", 1)
                bucket_name = path_parts[0]
                blob_path = path_parts[1]

                buffer = io.BytesIO()
                joblib.dump(scaler, buffer)
                buffer.seek(0)

                try:
                    storage_client = storage.Client() # Assumes auth is handled (e.g., VM service account)
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_path)
                    blob.upload_from_file(buffer, content_type="application/octet-stream")
                    logger.info(f"Fold {self.fold_num}: Scaler fitted and saved to GCS: {self.scaler_path}")
                except Exception as gcs_e:
                    logger.error(f"Fold {self.fold_num}: Error saving scaler to GCS: {gcs_e}", exc_info=True)
                    raise # Re-raise GCS error
            else:
                # Save locally
                os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
                joblib.dump(scaler, self.scaler_path) # Assumes joblib is imported
                logger.info(f"Fold {self.fold_num}: Scaler fitted and saved locally: {self.scaler_path}")

        except NameError as e:
             logger.error(f"Fold {self.fold_num}: Failed to fit/save scaler - Missing import? {e}", exc_info=True)
             return None, {} # Cannot proceed without scaler
        except Exception as e:
            logger.error(f"Fold {self.fold_num}: Failed to fit/save scaler - {e}", exc_info=True)
            return None, {} # Cannot proceed without scaler

        # --- Training Loop ---
        logger.info(f"Fold {self.fold_num}: Starting training for {self.episodes} episodes...")
        # Max steps should be based on the length of the training data for this fold
        max_steps_train = len(self.train_df) - self.window_size
        if max_steps_train <= 0:
             logger.error(f"Fold {self.fold_num}: Training data length ({len(self.train_df)}) is not greater than window size ({self.window_size}). Skipping fold.")
             return None, {}

        best_val_score = -float('inf')
        best_val_episode = -1
        training_info = {"train_rewards": [], "val_rewards": [], "policy_loss": [], "value_loss": [], "entropy": []}
        model_saved_for_fold = False

        pbar_episodes = tqdm(range(self.episodes), desc=f"Fold {self.fold_num} Training", leave=False)
        for episode in pbar_episodes:
            state, _ = train_env.reset()
            done = False
            truncated = False
            episode_reward = 0
            steps_in_episode = 0

            while not (done or truncated):
                # Ensure episode doesn't exceed available training steps
                if steps_in_episode >= max_steps_train:
                    # logger.warning(f"Fold {self.fold_num} Ep {episode+1}: Truncating episode at max steps {max_steps_train}")
                    truncated = True # Mark as truncated

                # Break if truncated before taking action
                if truncated:
                    break

                action, prob, val = agent.choose_action(state)
                next_state, reward, done, truncated_env, info = train_env.step(action) # Capture env truncation flag

                # Use env truncation flag if it occurred
                truncated = truncated or truncated_env

                # Store experience
                # Ensure state, action, prob, val, reward, done are tensors before storing if needed by memory
                agent.memory.store(state, action, prob, val, reward, done or truncated) # Store based on combined done/truncated
                state = next_state
                episode_reward += reward
                steps_in_episode += 1

                # PPO Update Logic (check if agent handles frequency internally)
                if len(agent.memory.states) >= agent.batch_size:
                    update_metrics = agent.update()
                    if update_metrics:
                        training_info["policy_loss"].append(update_metrics["policy_loss"])
                        training_info["value_loss"].append(update_metrics["value_loss"])
                        training_info["entropy"].append(update_metrics["entropy"])

                # Check done/truncated again after step
                if done or truncated:
                    break # End episode loop

            training_info["train_rewards"].append(episode_reward)

            # --- Periodic Validation & Conditional Saving ---
            if (episode + 1) % self.eval_freq == 0 or episode == self.episodes - 1:
                original_env = agent.env
                agent.env = val_env
                num_val_episodes = 5 # Hardcoded or make configurable
                val_reward = agent.evaluate(num_episodes=num_val_episodes)
                agent.env = original_env # Restore training env
                training_info["val_rewards"].append(val_reward)

                # --- Model Saving Logic ---
                # TODO: Define a better validation metric if possible (e.g., Sharpe from val run)
                current_val_metric = val_reward
                MIN_ACCEPTABLE_VAL_SCORE = 1000 
                if current_val_metric > best_val_score and current_val_metric >= MIN_ACCEPTABLE_VAL_SCORE:
                    best_val_score = current_val_metric
                    best_val_episode = episode + 1
                    # Agent's save method saves the model state dict internally
                    agent.save(self.best_model_path)
                    logger.info(f"    -> New best model saved! Val Score: {best_val_score:.2f} at Ep {best_val_episode}")
                    model_saved_for_fold = True
                pbar_episodes.set_description(f"Fold {self.fold_num} Training | Best Val R: {best_val_score:.2f} @ Ep {best_val_episode}")

        pbar_episodes.close()
        logger.info(f"--- Finished Training for Fold {self.fold_num} ---")
        if not model_saved_for_fold:
            logger.warning(f"Fold {self.fold_num}: No model met the saving criteria during training.")
            return {"avg_reward": np.nan, "avg_sharpe": np.nan}, training_info

        # --- Testing Phase ---
        logger.info(f"Fold {self.fold_num}: Loading best model from {self.best_model_path} for testing...")
        try:
            # Recreate the model structure using the factory
            logger.info(f"Fold {self.fold_num}: Recreating model structure for loading...")
            model_for_loading = create_model(config=self.model_config, device=self.device)

            # Recreate agent with the fresh model structure and the original model_config
            test_env_for_agent = self._create_environment(self.test_df, mode="test")
            agent = self._create_agent(test_env_for_agent, model_for_loading, self.model_config)

            # Load the saved state dict into the recreated model within the agent
            # PPOAgent.load loads the checkpoint and applies model state dict
            # It also loads optimizer state if available.
            agent.load(self.best_model_path)
            logger.info(f"Fold {self.fold_num}: Successfully loaded best model state into agent.")
        except FileNotFoundError:
            logger.error(f"Fold {self.fold_num}: Best model checkpoint file not found at {self.best_model_path}. Cannot perform testing.")
            return {"avg_reward": np.nan, "avg_sharpe": np.nan}, training_info
        except Exception as e:
             logger.error(f"Fold {self.fold_num}: Error loading model {self.best_model_path}: {e}", exc_info=True)
             return {"avg_reward": np.nan, "avg_sharpe": np.nan}, training_info


        logger.info(f"Fold {self.fold_num}: Evaluating best model on test set...")
        test_env = self._create_environment(self.test_df, mode="test") # Env for evaluation
        original_env = agent.env
        agent.env = test_env
        num_test_episodes = 20 # Hardcoded or make configurable
        test_rewards = []
        test_sharpes = [] # Placeholder for Sharpe calculation

        for i in range(num_test_episodes):
            test_reward = agent.evaluate(num_episodes=1) # Evaluate one full run on test set
            test_rewards.append(test_reward)
            # TODO: Calculate Sharpe ratio from the test run if possible
            # Accessing env.trade_history or similar might be needed
            # test_sharpes.append(test_env.calculate_sharpe()) # Example
            test_sharpes.append(np.random.rand()) # Placeholder

        agent.env = original_env # Restore original env if needed

        avg_test_reward = np.mean(test_rewards) if test_rewards else np.nan
        avg_test_sharpe = np.mean(test_sharpes) if test_sharpes else np.nan # Placeholder
        logger.info(f"Fold {self.fold_num}: Test Set Avg Reward over {num_test_episodes} runs: {avg_test_reward:.2f}")
        logger.info(f"Fold {self.fold_num}: Test Set Avg Sharpe over {num_test_episodes} runs: {avg_test_sharpe:.2f}") # Placeholder

        test_metrics = {
            "avg_reward": avg_test_reward,
            "avg_sharpe": avg_test_sharpe, # Placeholder
        }

        # Save training info for this fold
        fold_results_path = os.path.join(self.fold_save_path, "training_info.json")
        try:
            with open(fold_results_path, 'w') as f:
                serializable_info = {}
                for key, value in training_info.items():
                    # Improved serialization for numpy types
                    if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                        serializable_info[key] = int(value)
                    elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                        serializable_info[key] = float(value)
                    elif isinstance(value, (np.ndarray,)):
                        serializable_info[key] = value.tolist()
                    elif isinstance(value, list) and value and isinstance(value[0], (np.generic, np.ndarray)):
                         # Handle lists containing numpy types (might need refinement)
                         serializable_info[key] = [item.item() if hasattr(item, 'item') else item for item in value]
                    else:
                         serializable_info[key] = value
                json.dump(serializable_info, f, indent=4)
            logger.info(f"Fold {self.fold_num}: Saved training info to {fold_results_path}")
        except Exception as e:
            logger.error(f"Fold {self.fold_num}: Failed to save training info: {e}", exc_info=True)


        return test_metrics, training_info # Return metrics and info dict
