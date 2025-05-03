#!/usr/bin/env python3
"""
Training script for the Binance Futures RL trading agent.
"""
import argparse
import logging
import os
import random
import json  # Added for parsing JSON args
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
import tqdm
from tqdm import tqdm

# Agent, Env, Data Imports (Keep)
# from rl_agent.agent.ppo_agent import PPOAgent # PPOAgent is used within Trainer now
from rl_agent.environment.trading_env import BinanceFuturesCryptoEnv # Env creation still happens here conceptually before Trainer
# from data import DataHandler # Assuming DataHandler is used in load_and_preprocess

# --- New Imports ---
from training.trainer import Trainer # Import the new Trainer class
from training.model_factory import ModelConfig # Import ModelConfig
from training.load_preprocess_data import load_and_preprocess # Import the new data loading function

# --- Removed old model component/architecture imports and mappings ---
# (These are now handled by the model_factory and PPOAgent internally)

logger = logging.getLogger(__name__)


def set_seeds(seed_value=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
        # Optional: uncomment for full determinism (may impact performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False





# --- Removed create_environment function (logic moved to Trainer) ---
# --- Removed train_evaluate_fold function (logic moved to Trainer) ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for trading")
    # --- Data and Env Args ---
    parser.add_argument("--train_data", type=str, required=True, help="Path to the full historical data file (CSV or Parquet)")
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading pair symbol")
    parser.add_argument("--interval", type=str, default="1m", help="Data interval (e.g., '1m', '15m', '1h')")
    parser.add_argument("--window", type=int, default=60, help="Observation window size")
    parser.add_argument("--leverage", type=int, default=20, help="Trading leverage")
    parser.add_argument("--max_position", type=float, default=1.0, help="Maximum position size as fraction of balance")
    parser.add_argument("--balance", type=float, default=5, help="Initial balance")
    parser.add_argument("--risk_reward", type=float, default=1.5, help="Risk-reward ratio")
    parser.add_argument("--stop_loss", type=float, default=0.005, help="Stop loss percentage")
    parser.add_argument("--trade_fee", type=float, default=0.0004, help="Trade fee percentage (e.g., 0.0004 for 0.04%)")
    parser.add_argument("--static_leverage", action="store_false", dest="dynamic_leverage", help="Use static leverage")
    parser.add_argument("--simple_rewards", action="store_false", dest="use_risk_adjusted_rewards", help="Use simple rewards")

    # --- Training Loop Args ---
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes per fold")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for TimeSeriesSplit walk-forward validation")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction of training data per fold to use for validation")
    parser.add_argument("--eval_freq", type=int, default=20, help="Episodes between evaluations")
    parser.add_argument("--save_path", type=str, default="models", help="Base path to save trained models and results")

    # --- PPO Agent Hyperparameters ---
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")  # Adjusted default
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--policy_clip", type=float, default=0.2, help="PPO policy clipping parameter")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for PPO updates")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs per PPO update")  # Adjusted default
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping")
    parser.add_argument("--update_freq", type=int, default=2048, help="Steps between policy updates (should be multiple of batch_size)")  # Adjusted default
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cpu', 'cuda', 'mps', 'xla')")
    parser.add_argument("--use_gae", type=bool, default=True, help="Use Generalized Advantage Estimation")
    parser.add_argument("--normalize_advantage", type=bool, default=True, help="Normalize advantages")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Adam weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients (reduces memory usage)")

    # --- ModelConfig Arguments (used by model_factory) ---
    model_group = parser.add_argument_group('Model Architecture Configuration (via ModelConfig)')
    model_group.add_argument("--architecture", type=str, default="encoder_only", choices=["encoder_only", "decoder_only", "encoder_decoder"], help="Core transformer architecture type")
    model_group.add_argument("--embedding_dim", type=int, default=256, help="Embedding dimension size")
    model_group.add_argument("--n_encoder_layers", type=int, default=4, help="Number of encoder layers (if applicable)")
    model_group.add_argument("--n_decoder_layers", type=int, default=4, help="Number of decoder layers (if applicable)")
    # -- window_size is already defined under Data/Env args --
    model_group.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for model components")
    model_group.add_argument("--attention_type", type=str, default="mha", choices=["mha", "mla", "pyr", "mqn", "gqa"], help="Attention mechanism type")
    model_group.add_argument("--n_heads", type=int, default=8, help="Number of attention heads (used by some attention types)")
    model_group.add_argument("--n_latents", type=int, default=None, help="Number of latents (for MLA attention)")
    model_group.add_argument("--n_groups", type=int, default=None, help="Number of groups (for GQA attention)")
    model_group.add_argument("--ffn_type", type=str, default="standard", choices=["standard", "moe"], help="Feed-forward network type")
    model_group.add_argument("--ffn_dim", type=int, default=None, help="Feed-forward hidden dimension (defaults based on embedding_dim)")
    model_group.add_argument("--n_experts", type=int, default=None, help="Number of experts (for MoE FFN)")
    model_group.add_argument("--top_k", type=int, default=None, help="Top K experts to use (for MoE FFN)")
    model_group.add_argument("--norm_type", type=str, default="layer_norm", choices=["layer_norm"], help="Normalization layer type")

    # Residual Connection Configuration
    model_group.add_argument("--residual_scale", type=float, default=1.0,
                            help="Scaling factor for residual connections")
    model_group.add_argument("--use_gated_residual", action="store_true",
                            help="Use learnable gates for residual connections")
    model_group.add_argument("--use_final_norm", action="store_true",
                            help="Apply a final layer normalization after all residual connections")

    # Feature Extractor Configuration
    model_group.add_argument("--feature_extractor_type", type=str, default="basic", choices=["basic", "resnet", "inception"],
                            help="Type of feature extractor architecture")
    model_group.add_argument("--feature_extractor_dim", type=int, default=128,
                            help="Hidden dimension for feature extractor")
    model_group.add_argument("--feature_extractor_layers", type=int, default=2,
                            help="Number of layers in feature extractor")
    model_group.add_argument("--use_skip_connections", action="store_true",
                            help="Use skip connections in feature extractor")
    model_group.add_argument("--use_layer_norm", action="store_true",
                            help="Use layer normalization instead of batch norm in feature extractor")
    model_group.add_argument("--use_instance_norm", action="store_true",
                            help="Use instance normalization in feature extractor")
    model_group.add_argument("--feature_dropout", type=float, default=0.0,
                            help="Dropout rate for feature extractor")

    # Actor-Critic Head Configuration
    model_group.add_argument("--head_hidden_dim", type=int, default=128,
                            help="Hidden dimension for actor-critic heads")
    model_group.add_argument("--head_n_layers", type=int, default=2,
                            help="Number of layers in actor-critic heads")
    model_group.add_argument("--head_use_layer_norm", action="store_true",
                            help="Use layer normalization in actor-critic heads")
    model_group.add_argument("--head_use_residual", action="store_true",
                            help="Use residual connections in actor-critic heads")
    model_group.add_argument("--head_dropout", type=float, default=None,
                            help="Dropout rate for actor-critic heads (None = use model dropout)")
    model_group.add_argument("--temperature", type=float, default=0.5,
                            help="Temperature for action selection (lower = more deterministic)")
    # -- action_dim is determined by the environment --

    args = parser.parse_args()

    set_seeds()
    logger.info("Random seeds set.")

    logger.info("Starting Walk-Forward Validation...")

    full_df = load_and_preprocess(args.train_data)
    if full_df.empty or len(full_df) < args.window * (args.n_splits + 1):
        logger.error(f"Not enough data for {args.n_splits} splits with window {args.window}. Exiting.")
        # exit()

    tscv = TimeSeriesSplit(n_splits=args.n_splits)

    all_fold_results = []
    fold_num = 0

    for train_val_indices, test_indices in tscv.split(full_df):
        fold_num += 1
        logger.info(f"\n===== Starting Walk-Forward Fold {fold_num}/{args.n_splits} =====")

        if len(test_indices) < args.window:
            logger.warning(f"Fold {fold_num}: Skipping test set, not enough data ({len(test_indices)} < {args.window}).")
            continue

        val_size = max(args.window, int(len(train_val_indices) * args.val_ratio))
        train_size = len(train_val_indices) - val_size

        if train_size < args.window:
            logger.warning(f"Fold {fold_num}: Skipping, not enough training data ({train_size} < {args.window}) after validation split.")
            continue

        train_indices = train_val_indices[:train_size]
        val_indices = train_val_indices[train_size:]

        train_df = full_df.iloc[train_indices]
        val_df = full_df.iloc[val_indices]
        test_df = full_df.iloc[test_indices]

        logger.info(f"Fold {fold_num}: Train size={len(train_df)}, Val size={len(val_df)}, Test size={len(test_df)}")
        if isinstance(train_df.index, pd.DatetimeIndex):
            logger.info(f"Train period: {train_df.index.min()} to {train_df.index.max()}")
            logger.info(f"Val period:   {val_df.index.min()} to {val_df.index.max()}")
            logger.info(f"Test period:  {test_df.index.min()} to {test_df.index.max()}")

        # --- Create ModelConfig ---
        # Note: action_dim is set by env, not needed in ModelConfig here
        model_config = ModelConfig(
            # Core architecture
            architecture=args.architecture,
            embedding_dim=args.embedding_dim,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            window_size=args.window, # Get window size from env args
            dropout=args.dropout,

            # Attention configuration
            attention_type=args.attention_type,
            n_heads=args.n_heads,
            n_latents=args.n_latents,
            n_groups=args.n_groups,

            # Feed-forward configuration
            ffn_type=args.ffn_type,
            ffn_dim=args.ffn_dim,
            n_experts=args.n_experts,
            top_k=args.top_k,

            # Normalization
            norm_type=args.norm_type,

            # Residual connections
            residual_scale=args.residual_scale,
            use_gated_residual=args.use_gated_residual,
            use_final_norm=args.use_final_norm,

            # Feature extraction configuration
            feature_extractor_type=args.feature_extractor_type,
            feature_extractor_dim=args.feature_extractor_dim,
            feature_extractor_layers=args.feature_extractor_layers,
            use_skip_connections=args.use_skip_connections,
            use_layer_norm=args.use_layer_norm,
            use_instance_norm=args.use_instance_norm,
            feature_dropout=args.feature_dropout,

            # Actor-Critic head configuration
            head_hidden_dim=args.head_hidden_dim,
            head_n_layers=args.head_n_layers,
            head_use_layer_norm=args.head_use_layer_norm,
            head_use_residual=args.head_use_residual,
            head_dropout=args.head_dropout,

            # Temperature for action selection
            temperature=args.temperature,

            # Data-specific
            n_features=train_df.shape[1]+3, # Get actual feature count from data
            # action_dim is determined by env, set within ActorCriticWrapper
        )

        # --- Instantiate and Run Trainer for this Fold ---
        # Pass model_config and the remaining args (for trainer/env/ppo)
        trainer = Trainer(
            fold_num=fold_num,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_config=model_config,
            trainer_args=args # Pass the full args namespace for now
        )
        test_metrics, training_info = trainer.train_and_evaluate_fold()

        if test_metrics is None:
            logger.warning(f"Fold {fold_num}: Training/evaluation failed or was skipped. Continuing to next fold.")
            continue

        all_fold_results.append(test_metrics)
        logger.info(f"Fold {fold_num} Test Results: {test_metrics}")

        # Optional: Save training info per fold
        fold_results_path = os.path.join(args.save_path, f"fold_{fold_num}", "training_info.json")
        try:
            with open(fold_results_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_info = {}
                for key, value in training_info.items():
                    if isinstance(value, list) and value and isinstance(value[0], np.generic):
                         serializable_info[key] = [item.item() for item in value] # Convert numpy types
                    elif isinstance(value, np.ndarray):
                         serializable_info[key] = value.tolist()
                    else:
                         serializable_info[key] = value
                json.dump(serializable_info, f, indent=4)
            logger.info(f"Fold {fold_num}: Saved training info to {fold_results_path}")
        except Exception as e:
            logger.error(f"Fold {fold_num}: Failed to save training info: {e}")


    logger.info("\n===== Walk-Forward Validation Finished =====")

    if not all_fold_results:
        logger.error("No folds completed successfully. Exiting.")
        # exit() # Consider exiting if no folds ran

    # --- Aggregate and Print Final Results ---
    avg_rewards = [r['avg_reward'] for r in all_fold_results if not np.isnan(r['avg_reward'])]
    avg_sharpes = [r['avg_sharpe'] for r in all_fold_results if not np.isnan(r['avg_sharpe'])] # Assuming Sharpe is calculated

    final_avg_reward = np.mean(avg_rewards) if avg_rewards else np.nan
    final_avg_sharpe = np.mean(avg_sharpes) if avg_sharpes else np.nan # Adjust if Sharpe isn't always present

    logger.info(f"Overall Average Test Reward across {len(avg_rewards)} successful folds: {final_avg_reward:.4f}")
    logger.info(f"Overall Average Test Sharpe across {len(avg_sharpes)} successful folds: {final_avg_sharpe:.4f}") # Adjust log message

    # Save overall results
    overall_results = {
        "num_successful_folds": len(avg_rewards),
        "average_test_reward": final_avg_reward,
        "average_test_sharpe": final_avg_sharpe, # Add Sharpe here
        "fold_results": all_fold_results,
        "args": vars(args) # Save arguments used for the run
    }
    results_file = os.path.join(args.save_path, "overall_results.json")
    try:
        with open(results_file, 'w') as f:
            json.dump(overall_results, f, indent=4, default=lambda x: str(x) if isinstance(x, (np.generic, np.ndarray)) else x) # Handle numpy types
        logger.info(f"Saved overall results to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save overall results: {e}")
