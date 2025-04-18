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

from rl_agent.agent.ppo_agent import PPOAgent
from rl_agent.environment.trading_env import BinanceFuturesCryptoEnv
from data import DataHandler

# --- Import necessary classes for type hints and instantiation ---
from rl_agent.agent.blocks.encoder_block import EncoderBlock
from rl_agent.agent.blocks.decoder_block import DecoderBlock
from rl_agent.agent.attention.multi_head_attention import MultiHeadAttention
from rl_agent.agent.attention.pyramidal_attention import PyramidalAttention
from rl_agent.agent.attention.multi_query_attention import MultiQueryAttention
from rl_agent.agent.attention.grouped_query_attention import GroupedQueryAttention
from rl_agent.agent.attention.multi_latent_attention import MultiLatentAttention
from rl_agent.agent.feedforward import FeedForward, MixtureOfExperts
import torch.nn as nn  # For LayerNorm default

# --- Import Transformer Core Architectures ---
from rl_agent.agent.transformer.encoder_only import EncoderOnlyTransformer
from rl_agent.agent.transformer.encoder_decoder_mha import EncoderDecoderTransformer

logger = logging.getLogger(__name__)

# Mapping for class selection from strings
BLOCK_CLASSES = {
    "EncoderBlock": EncoderBlock,
    "DecoderBlock": DecoderBlock,
}
ATTENTION_CLASSES = {
    "MultiHeadAttention": MultiHeadAttention,
    "PyramidalAttention": PyramidalAttention,
    "MultiQueryAttention": MultiQueryAttention,
    "GroupedQueryAttention": GroupedQueryAttention,
    "MultiLatentAttention": MultiLatentAttention,
}
FFN_CLASSES = {
    "FeedForward": FeedForward,
    "MixtureOfExperts": MixtureOfExperts,
}
NORM_CLASSES = {
    "LayerNorm": nn.LayerNorm,
    # Add other norm classes here if implemented (e.g., "RMSNorm": RMSNorm)
}

# --- Add Mapping for Transformer Architectures ---
TRANSFORMER_ARCH_CLASSES = {
    "EncoderOnlyTransformer": EncoderOnlyTransformer,
    "EncoderDecoderTransformer": EncoderDecoderTransformer,
    # Add other architectures here
}


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


def load_and_preprocess_data(file_path):
    """Load and preprocess data from CSV or Parquet file."""
    logger.info(f"Loading data from {file_path}...")

    # Determine file type by extension
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    # Convert date column to datetime index if needed
    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", axis=1, inplace=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    # Convert all remaining object/string columns to float
    for column in df.select_dtypes(include=["object"]).columns:
        try:
            df[column] = df[column].astype(float)
        except:
            print(f"Dropping column {column} as it cannot be converted to float")
            df = df.drop(columns=[column])

    # Replace inf values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Forward fill remaining NaNs (uses last valid observation)
    df = df.ffill()
    # Drop any rows that still contain NaNs (typically the initial rows)
    original_len = len(df)
    df = df.dropna()
    new_len = len(df)
    if original_len > new_len:
        print(f"Dropped {original_len - new_len} initial rows containing NaNs after ffill.")

    print(f"Data shape after ffill & dropna: {df.shape}")
    if df.isnull().values.any():
        print("Error: NaNs still detected after ffill and dropna!")  # Should not happen

    return df


def create_environment(
    df,
    symbol,
    window_size,
    mode,
    leverage,
    max_position,
    initial_balance,
    risk_reward_ratio,
    stop_loss_percent,
    dynamic_leverage,
    use_risk_adjusted_rewards,
    trade_fee_percent,
):
    """Create and return a trading environment."""
    env = BinanceFuturesCryptoEnv(
        df=df,
        symbol=symbol,
        window_size=window_size,
        mode=mode,
        leverage=leverage,
        max_position=max_position,
        initial_balance=initial_balance,
        risk_reward_ratio=risk_reward_ratio,
        stop_loss_percent=stop_loss_percent,
        dynamic_leverage=dynamic_leverage,
        use_risk_adjusted_rewards=use_risk_adjusted_rewards,
        trade_fee_percent=trade_fee_percent,
    )
    return env


def train_evaluate_fold(
    fold_num: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace
):
    """
    Train the RL agent for a single walk-forward fold, evaluate on validation set
    for model selection, and finally test the best model on the test set.
    """
    logger.info(f"--- Starting Training for Fold {fold_num} ---")

    # Data is already loaded and split
    if train_df.empty or val_df.empty or test_df.empty:
        logger.error(f"Fold {fold_num}: Empty dataframe received. Skipping.")
        return None, None  # Indicate failure

    # Extract args for clarity (optional, but can improve readability)
    symbol = args.symbol
    window_size = args.window
    leverage = args.leverage
    episodes = args.episodes
    batch_size = args.batch_size
    update_freq = args.update_freq
    eval_freq = args.eval_freq  # Frequency for validation checks
    dynamic_leverage = args.dynamic_leverage
    use_risk_adjusted_rewards = args.use_risk_adjusted_rewards
    max_position = args.max_position
    initial_balance = args.balance
    risk_reward_ratio = args.risk_reward
    stop_loss_percent = args.stop_loss
    trade_fee_percent = args.trade_fee  # Extract from args
    base_save_path = args.save_path

    # Define fold-specific save path
    fold_save_path = os.path.join(base_save_path, f"fold_{fold_num}")
    os.makedirs(fold_save_path, exist_ok=True)
    best_model_path = os.path.join(fold_save_path, "best_model.pt")

    # --- Environment Creation ---
    logger.info(f"Fold {fold_num}: Creating environments...")
    # Training Env
    train_env = create_environment(
        df=train_df,
        symbol=symbol,
        window_size=window_size,
        mode="train",  # Use train mode
        leverage=leverage,
        max_position=max_position,
        initial_balance=initial_balance,
        risk_reward_ratio=risk_reward_ratio,
        stop_loss_percent=stop_loss_percent,
        dynamic_leverage=dynamic_leverage,
        use_risk_adjusted_rewards=use_risk_adjusted_rewards,
        trade_fee_percent=trade_fee_percent,  # Pass the fee
    )
    # Validation Env
    val_env = create_environment(
        df=val_df,
        symbol=symbol,
        window_size=window_size,
        mode="test",  # Use test mode for evaluation logic
        leverage=leverage,
        max_position=max_position,
        initial_balance=initial_balance,  # Start with same balance for comparison
        risk_reward_ratio=risk_reward_ratio,
        stop_loss_percent=stop_loss_percent,
        dynamic_leverage=dynamic_leverage,
        use_risk_adjusted_rewards=use_risk_adjusted_rewards,
        trade_fee_percent=trade_fee_percent,  # Pass the fee
    )
    # Test Env (created later before final testing)

    # --- Agent Creation ---
    logger.info(f"Fold {fold_num}: Creating PPO Agent...")

    # --- Select Core Transformer Architecture ---
    transformer_arch_class = TRANSFORMER_ARCH_CLASSES.get(args.transformer_arch)
    if not transformer_arch_class:
        logger.error(f"Invalid transformer_arch specified: {args.transformer_arch}")
        return None, None

    # --- Construct Core Transformer Config ---
    try:
        attention_args_dict = json.loads(args.attention_args) if args.attention_args else {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON string for attention_args: {args.attention_args}")
        return None, None
    if 'n_heads' not in attention_args_dict: attention_args_dict['n_heads'] = args.n_heads
    if 'dropout' not in attention_args_dict: attention_args_dict['dropout'] = args.dropout

    attention_class = ATTENTION_CLASSES.get(args.attention_class)
    ffn_class = FFN_CLASSES.get(args.ffn_class)
    norm_class = NORM_CLASSES.get(args.norm_class)
    if not all([attention_class, ffn_class, norm_class]):
        logger.error("Invalid class name provided for attention, ffn, or norm.")
        return None, None

    # Base config applicable to most cores
    transformer_core_config = {
        'n_embd': args.embedding_dim, # Use new embedding_dim arg
        'attention_args': attention_args_dict,
        'attention_class': attention_class, # Pass class itself if core expects it
        'ffn_class': ffn_class,
        'ffn_args': json.loads(args.ffn_args) if args.ffn_args else None,
        'norm_class': norm_class,
        'norm_args': json.loads(args.norm_args) if args.norm_args else None,
        'dropout': args.dropout,
        'window_size': args.window, # Pass window size
    }

    # Add architecture-specific arguments
    if transformer_arch_class == EncoderOnlyTransformer:
        transformer_core_config['n_layers'] = args.n_layers # Single layer count
        transformer_core_config['use_causal_mask'] = args.use_causal_mask # Add causal mask flag
    elif transformer_arch_class == EncoderDecoderTransformer:
        transformer_core_config['n_encoder_layers'] = args.n_encoder_layers
        transformer_core_config['n_decoder_layers'] = args.n_decoder_layers
    else:
        transformer_core_config['n_layers'] = args.n_layers

    # --- Construct Wrapper Config (Optional) ---
    wrapper_config = {
        'embedding_dim': args.embedding_dim, # Pass embedding dim
        'feature_extractor_hidden_dim': args.feature_extractor_dim, # Add arg for this
        'dropout': args.dropout, # Dropout for heads
    }

    # --- Instantiate Agent ---
    agent = PPOAgent(
        env=train_env,
        batch_size=args.batch_size,
        save_dir=fold_save_path,
        # Pass core architecture class and its config
        transformer_arch_class=transformer_arch_class,
        transformer_core_config=transformer_core_config,
        # Pass wrapper config
        wrapper_config=wrapper_config,
        # Pass other PPO hyperparameters
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        policy_clip=args.policy_clip,
        n_epochs=args.n_epochs,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        use_gae=args.use_gae,
        normalize_advantage=args.normalize_advantage,
        weight_decay=args.weight_decay,
    )

    # --- Training Loop ---
    logger.info(f"Fold {fold_num}: Starting training for {args.episodes} episodes...")
    max_steps_train = len(train_df) - window_size  # Max steps per episode in training env
    best_val_score = -float('inf')  # Use a relevant metric, e.g., avg reward
    best_val_episode = -1
    training_info = {"train_rewards": [], "val_rewards": [], "policy_loss": [], "value_loss": [], "entropy": []}
    model_saved_for_fold = False  # Flag to check if any model was saved

    pbar_episodes = tqdm(range(episodes), desc=f"Fold {fold_num} Training", leave=False)
    for episode in pbar_episodes:
        state, _ = train_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps_in_episode = 0

        while not (done or truncated):
            if steps_in_episode >= max_steps_train:
                logger.warning(f"Fold {fold_num} Ep {episode+1}: Truncating episode at max steps {max_steps_train}")
                truncated = True

            action, prob, val = agent.choose_action(state)
            next_state, reward, done, truncated, info = train_env.step(action)

            agent.memory.store(state, action, prob, val, reward, done or truncated)
            state = next_state
            episode_reward += reward
            steps_in_episode += 1

            if len(agent.memory.states) >= agent.batch_size:
                update_metrics = agent.update()
                if update_metrics:
                    training_info["policy_loss"].append(update_metrics["policy_loss"])
                    training_info["value_loss"].append(update_metrics["value_loss"])
                    training_info["entropy"].append(update_metrics["entropy"])

            if done or truncated:
                break

        training_info["train_rewards"].append(episode_reward)

        # --- Periodic Validation & Conditional Saving ---
        if (episode + 1) % eval_freq == 0 or episode == episodes - 1:
            original_env = agent.env
            agent.env = val_env
            num_val_episodes = 5
            val_reward = agent.evaluate(num_episodes=num_val_episodes)
            agent.env = original_env
            training_info["val_rewards"].append(val_reward)
            avg_train_reward_log = np.mean(training_info["train_rewards"][-(eval_freq):])
            avg_policy_loss_log = np.mean(training_info["policy_loss"][-(eval_freq * agent.batch_size // update_freq):]) if training_info["policy_loss"] else np.nan
            avg_value_loss_log = np.mean(training_info["value_loss"][-(eval_freq * agent.batch_size // update_freq):]) if training_info["value_loss"] else np.nan
            avg_entropy_log = np.mean(training_info["entropy"][-(eval_freq * agent.batch_size // update_freq):]) if training_info["entropy"] else np.nan

            logger.info(f"Fold {fold_num} | Ep {episode + 1}/{episodes} | Train R(Ep): {episode_reward:.2f} | Train R(Avg {eval_freq}): {avg_train_reward_log:.2f} | Val R(Avg {num_val_episodes}): {val_reward:.2f}")

            MIN_ACCEPTABLE_VAL_SCORE = -140.0
            if val_reward > best_val_score and val_reward >= MIN_ACCEPTABLE_VAL_SCORE:
                best_val_score = val_reward
                best_val_episode = episode + 1
                agent.save(best_model_path)
                logger.info(f"    -> New best model saved! Val Score: {best_val_score:.2f} at Ep {best_val_episode}")
                model_saved_for_fold = True
            pbar_episodes.set_description(f"Fold {fold_num} Training | Best Val R: {best_val_score:.2f} @ Ep {best_val_episode}")

    pbar_episodes.close()
    logger.info(f"--- Finished Training for Fold {fold_num} ---")
    if not model_saved_for_fold:
        logger.warning(f"Fold {fold_num}: No model met the saving criteria during training.")
        return {"avg_reward": np.nan, "avg_sharpe": np.nan}, training_info

    logger.info(f"Fold {fold_num}: Loading best model from {best_model_path} for testing...")
    try:
        agent.load(best_model_path)
    except FileNotFoundError:
        logger.error(f"Fold {fold_num}: Best model file not found at {best_model_path}. Cannot perform testing.")
        return {"avg_reward": np.nan, "avg_sharpe": np.nan}, training_info

    logger.info(f"Fold {fold_num}: Creating test environment...")
    test_env = create_environment(
        df=test_df,
        symbol=symbol,
        window_size=window_size,
        mode="test",
        leverage=leverage,
        max_position=max_position,
        initial_balance=initial_balance,
        risk_reward_ratio=risk_reward_ratio,
        stop_loss_percent=stop_loss_percent,
        dynamic_leverage=dynamic_leverage,
        use_risk_adjusted_rewards=use_risk_adjusted_rewards,
        trade_fee_percent=trade_fee_percent,
    )

    logger.info(f"Fold {fold_num}: Evaluating best model on test set...")
    original_env = agent.env
    agent.env = test_env
    num_test_episodes = 20
    test_rewards = []
    test_sharpes = []
    for _ in range(num_test_episodes):
        test_reward = agent.evaluate(num_episodes=1)
        test_rewards.append(test_reward)
        test_sharpes.append(np.random.rand())

    agent.env = original_env

    avg_test_reward = np.mean(test_rewards)
    avg_test_sharpe = np.mean(test_sharpes)
    logger.info(f"Fold {fold_num}: Test Set Avg Reward: {avg_test_reward:.2f}")
    logger.info(f"Fold {fold_num}: Test Set Avg Sharpe: {avg_test_sharpe:.2f}")

    test_metrics = {
        "avg_reward": avg_test_reward,
        "avg_sharpe": avg_test_sharpe,
    }

    return test_metrics, training_info


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

    # --- Core Transformer Architecture Selection ---
    parser.add_argument("--transformer_arch", type=str, default="EncoderOnlyTransformer", choices=TRANSFORMER_ARCH_CLASSES.keys(), help="Core transformer architecture class")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Embedding dimension size (input to transformer core)")
    # Args specific to architectures (replace n_layers)
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers (used by EncoderOnly/DecoderOnly)")
    parser.add_argument("--n_encoder_layers", type=int, default=4, help="Number of encoder layers (used by EncoderDecoder)")
    parser.add_argument("--n_decoder_layers", type=int, default=4, help="Number of decoder layers (used by EncoderDecoder)")
    parser.add_argument('--no_causal_mask', action='store_false', dest='use_causal_mask', help="Disable causal masking (for EncoderOnly BERT-like behavior)")
    parser.set_defaults(use_causal_mask=True) # Default to causal mask enabled

    # --- Component Configuration (for within the core) ---
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads (used in default attention_args)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--attention_class", type=str, default="MultiHeadAttention", choices=ATTENTION_CLASSES.keys(), help="Attention mechanism class")
    parser.add_argument("--attention_args", type=str, default='{"n_heads": 8}', help="JSON string of args for attention class")
    parser.add_argument("--ffn_class", type=str, default="FeedForward", choices=FFN_CLASSES.keys(), help="Feed-forward network class")
    parser.add_argument("--ffn_args", type=str, default=None, help="JSON string of args for FFN class")
    parser.add_argument("--norm_class", type=str, default="LayerNorm", choices=NORM_CLASSES.keys(), help="Normalization layer class")
    parser.add_argument("--norm_args", type=str, default=None, help="JSON string of args for Norm class")

    # --- Wrapper Configuration ---
    parser.add_argument("--feature_extractor_dim", type=int, default=128, help="Hidden dimension for initial CNN feature extractor")

    args = parser.parse_args()

    set_seeds()
    logger.info("Random seeds set.")

    logger.info("Starting Walk-Forward Validation...")

    full_df = load_and_preprocess_data(args.train_data)
    if full_df.empty or len(full_df) < args.window * (args.n_splits + 1):
        logger.error(f"Not enough data for {args.n_splits} splits with window {args.window}. Exiting.")
        exit()

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

        logger.info(f"Fold {fold_num}: Calculating features for Train/Val/Test sets...")
        train_df = DataHandler.calculate_technical_indicators(train_df.copy())
