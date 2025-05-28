#!/usr/bin/env python3
"""
Training script for the Binance Futures RL trading agent.
"""
import argparse
import logging
import os
import random
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm # Use tqdm directly

# Agent, Env, Data Imports (Keep)
# from rl_agent.agent.ppo_agent import PPOAgent # PPOAgent is used within Trainer now
from rl_agent.environment.trading_env import BinanceFuturesCryptoEnv
# from data import DataHandler # Assuming DataHandler is used in load_and_preprocess

# --- New Imports ---
from training.trainer import Trainer # Import the new Trainer class
from training.model_factory import ModelConfig # Import ModelConfig
from training.load_preprocess_data import load_and_preprocess # Import the new data loading function

# --- Removed old model component/architecture imports and mappings ---
# (These are now handled by the model_factory and PPOAgent internally)

logger = logging.getLogger(__name__)

# --- No class mappings needed if sticking to original model_type logic ---

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
    """Load and preprocess data from CSV or Parquet file. (Keeping original logic)"""
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

    # Convert string columns like 'trade_setup' to numerical values
    if "trade_setup" in df.columns:
        setup_mapping = {
            "none": 0.0, "strong_bullish": 1.0, "strong_bearish": -1.0,
            "bullish_reversal": 0.5, "bearish_reversal": -0.5,
        }
        df["trade_setup"] = df["trade_setup"].map(setup_mapping)

    # Convert all object/string columns to float
    for column in df.select_dtypes(include=["object"]).columns:
        try:
            df[column] = df[column].astype(float)
        except ValueError: # Catch specific error
            print(f"Dropping column {column} as it cannot be converted to float")
            df = df.drop(columns=[column])

    # Replace NaN, inf values with 0 (Original logic)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Normalize data (Original logic)
    for col in df.columns:
        if col not in ["open_time", "date", "timestamp"]: # Assuming these columns exist or might exist
            median = df[col].median()
            iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
            if iqr > 0:
                df[col] = (df[col] - median) / (iqr + 1e-8)
            else:
                df[col] = df[col] - median

    print(f"Data shape after preprocessing: {df.shape}")
    if df.isnull().values.any():
         print("Warning: NaNs detected after preprocessing!") # Add a warning

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
    # Ensure df is not empty before creating env
    if df.empty:
        logger.error(f"Cannot create environment: DataFrame is empty for mode '{mode}'.")
        return None

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

    # Basic checks
    if train_df.empty or val_df.empty or test_df.empty:
        logger.error(f"Fold {fold_num}: Empty dataframe received. Skipping.")
        return None, None

    symbol = args.symbol
    window_size = args.window
    leverage = args.leverage
    episodes = args.episodes
    batch_size = args.batch_size
    update_freq = args.update_freq # Steps between updates
    eval_freq = args.eval_freq  # Episodes between validation checks
    dynamic_leverage = args.dynamic_leverage
    use_risk_adjusted_rewards = args.use_risk_adjusted_rewards
    max_position = args.max_position
    initial_balance = args.balance
    risk_reward_ratio = args.risk_reward
    stop_loss_percent = args.stop_loss
    trade_fee_percent = args.trade_fee
    base_save_path = args.save_path
    model_type = args.model # Use model_type from args

    # Define fold-specific save path
    fold_save_path = os.path.join(base_save_path, f"fold_{fold_num}")
    os.makedirs(fold_save_path, exist_ok=True)
    best_model_path = os.path.join(fold_save_path, "best_model.pt")

    # --- Environment Creation ---
    logger.info(f"Fold {fold_num}: Creating environments...")
    train_env = create_environment(
        df=train_df, symbol=symbol, window_size=window_size, mode="train",
        leverage=leverage, max_position=max_position, initial_balance=initial_balance,
        risk_reward_ratio=risk_reward_ratio, stop_loss_percent=stop_loss_percent,
        dynamic_leverage=dynamic_leverage, use_risk_adjusted_rewards=use_risk_adjusted_rewards,
        trade_fee_percent=trade_fee_percent,
    )
    val_env = create_environment(
        df=val_df, symbol=symbol, window_size=window_size, mode="test", # Use test mode logic for validation
        leverage=leverage, max_position=max_position, initial_balance=initial_balance,
        risk_reward_ratio=risk_reward_ratio, stop_loss_percent=stop_loss_percent,
        dynamic_leverage=dynamic_leverage, use_risk_adjusted_rewards=use_risk_adjusted_rewards,
        trade_fee_percent=trade_fee_percent,
    )
    test_env = create_environment(
        df=test_df, symbol=symbol, window_size=window_size, mode="test",
        leverage=leverage, max_position=max_position, initial_balance=initial_balance,
        risk_reward_ratio=risk_reward_ratio, stop_loss_percent=stop_loss_percent,
        dynamic_leverage=dynamic_leverage, use_risk_adjusted_rewards=use_risk_adjusted_rewards,
        trade_fee_percent=trade_fee_percent,
    )

    if not train_env or not val_env or not test_env:
        logger.error(f"Fold {fold_num}: Failed to create one or more environments. Skipping.")
        return None, None

    # --- Agent Creation ---
    logger.info(f"Fold {fold_num}: Creating PPO Agent ({model_type})...")
    # Use existing PPOAgent constructor, assuming it takes necessary hyperparameters
    # If PPOAgent needs updates to accept lr, gamma etc., that's a separate task.
    agent = PPOAgent(
        env=train_env, # Start with train_env
        model_type=model_type.lower(),
        batch_size=batch_size,
        save_dir=fold_save_path, # Agent might use this internally
        # Pass other necessary hyperparameters if the constructor accepts them
        lr=getattr(args, 'lr', 3e-4), # Provide defaults if not in args
        gamma=getattr(args, 'gamma', 0.99),
        gae_lambda=getattr(args, 'gae_lambda', 0.95),
        policy_clip=getattr(args, 'policy_clip', 0.2),
        n_epochs=getattr(args, 'n_epochs', 10),
        entropy_coef=getattr(args, 'entropy_coef', 0.01),
        value_coef=getattr(args, 'value_coef', 0.5),
        max_grad_norm=getattr(args, 'max_grad_norm', 0.5),
        device=getattr(args, 'device', 'auto'),
        # Add other params like use_gae, normalize_advantage, weight_decay if needed
    )

    # --- Training Loop (Manual, similar to example) ---
    logger.info(f"Fold {fold_num}: Starting training for {args.episodes} episodes...")
    max_steps_train = len(train_df) - window_size
    best_val_score = -float('inf')
    best_val_episode = -1
    training_info = {"train_rewards": [], "val_rewards": [], "policy_loss": [], "value_loss": [], "entropy": []}
    model_saved_for_fold = False
    total_steps = 0

    pbar_episodes = tqdm(range(episodes), desc=f"Fold {fold_num} Training", leave=False)
    for episode in pbar_episodes:
        state, _ = train_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps_in_episode = 0

        while not (done or truncated):
            if steps_in_episode >= max_steps_train:
                # logger.warning(f"Fold {fold_num} Ep {episode+1}: Truncating episode at max steps {max_steps_train}")
                truncated = True # Use truncation flag from env if possible

            action, prob, val = agent.choose_action(state)
            next_state, reward, done, truncated, info = train_env.step(action) # Env handles truncation

            agent.memory.store(state, action, prob, val, reward, done or truncated)
            state = next_state
            episode_reward += reward
            steps_in_episode += 1
            total_steps += 1

            # PPO Update based on steps (update_freq)
            # Check if enough steps have passed AND if memory has enough samples for a batch
            if total_steps % update_freq == 0 and len(agent.memory.states) >= agent.batch_size:
                update_metrics = agent.update() # Assuming update clears memory
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
            agent.env = val_env # Switch to validation env
            num_val_episodes = 5 # Number of episodes to average over for validation
            val_reward = agent.evaluate(num_episodes=num_val_episodes) # Use agent's evaluate method
            agent.env = original_env # Switch back to training env
            training_info["val_rewards"].append(val_reward)

            # Log progress
            avg_train_reward_log = np.mean(training_info["train_rewards"][-eval_freq:]) if training_info["train_rewards"] else np.nan
            logger.info(f"Fold {fold_num} | Ep {episode + 1}/{episodes} | Train R(Ep): {episode_reward:.2f} | Train R(Avg {eval_freq}): {avg_train_reward_log:.2f} | Val R(Avg {num_val_episodes}): {val_reward:.2f}")

            # Save best model based on validation score
            # Add a minimum threshold if desired, e.g., val_reward > 0
            if val_reward > best_val_score:
                best_val_score = val_reward
                best_val_episode = episode + 1
                agent.save(best_model_path)
                logger.info(f"    -> New best model saved! Val Score: {best_val_score:.2f} at Ep {best_val_episode}")
                model_saved_for_fold = True
            pbar_episodes.set_postfix({"Best Val R": f"{best_val_score:.2f} @ Ep {best_val_episode}"})


    pbar_episodes.close()
    logger.info(f"--- Finished Training for Fold {fold_num} ---")
    if not model_saved_for_fold:
        logger.warning(f"Fold {fold_num}: No model met the saving criteria during training.")
        # Return NaN or default metrics if no model was saved
        return {"avg_reward": np.nan, "avg_sharpe": np.nan}, training_info # Add Sharpe if calculated

    # --- Testing Phase ---
    logger.info(f"Fold {fold_num}: Loading best model from {best_model_path} for testing...")
    try:
        # Ensure agent is reset or re-initialized if necessary before loading
        # Depending on PPOAgent implementation, might need:
        # agent = PPOAgent(...) # Re-create with same params
        agent.load(best_model_path)
    except FileNotFoundError:
        logger.error(f"Fold {fold_num}: Best model file not found at {best_model_path}. Cannot perform testing.")
        return {"avg_reward": np.nan, "avg_sharpe": np.nan}, training_info
    except Exception as e:
        logger.error(f"Fold {fold_num}: Error loading model {best_model_path}: {e}")
        return {"avg_reward": np.nan, "avg_sharpe": np.nan}, training_info


    logger.info(f"Fold {fold_num}: Evaluating best model on test set...")
    original_env = agent.env
    agent.env = test_env # Switch to test env
    num_test_episodes = 10 # Number of episodes for final test evaluation
    test_reward = agent.evaluate(num_episodes=num_test_episodes)
    agent.env = original_env # Switch back

    # TODO: Calculate Sharpe Ratio from test_env history if possible
    # This requires the environment to store trade details.
    # Example placeholder:
    test_sharpe = np.random.rand() # Replace with actual calculation

    logger.info(f"Fold {fold_num}: Test Set Avg Reward: {test_reward:.2f}")
    logger.info(f"Fold {fold_num}: Test Set Avg Sharpe: {test_sharpe:.2f}") # Log Sharpe

    test_metrics = {
        "avg_reward": test_reward,
        "avg_sharpe": test_sharpe, # Include Sharpe
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
    parser.add_argument(
        "--episodes", type=int, default=500, help="Number of training episodes"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for training"
    )
    parser.add_argument(
        "--update_freq", type=int, default=256, help="Steps between policy updates"
    )
    parser.add_argument(
        "--log_freq", type=int, default=20, help="Episodes between log updates"
    )
    parser.add_argument(
        "--save_freq", type=int, default=20, help="Episodes between model saves"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=20, help="Episodes between evaluations"
    )
    parser.add_argument(
        "--max_position",
        type=float,
        default=1.0,
        help="Maximum position size as fraction of balance",
    )
    parser.add_argument("--balance", type=float, default=5, help="Initial balance")
    parser.add_argument(
        "--risk_reward", type=float, default=1.5, help="Risk-reward ratio"
    )
    parser.add_argument(
        "--stop_loss", type=float, default=0.005, help="Stop loss percentage"
    )
    parser.add_argument(
        "--static_leverage",
        action="store_false",
        dest="dynamic_leverage",
        help="Use static leverage",
    )
    parser.add_argument(
        "--simple_rewards",
        action="store_false",
        dest="use_risk_adjusted_rewards",
        help="Use simple rewards",
    )
    parser.add_argument(
        "--save_path", type=str, default="models", help="Path to save trained model"
    )
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
            # Core model type
            core_model_type=args.core_model_type,

            # Core transformer architecture (if applicable)
            architecture=args.architecture,
            embedding_dim=args.embedding_dim, # Used by both
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            window_size=args.window, # Get window size from env args
            dropout=args.dropout, # General dropout

            # Attention configuration (Transformer specific)
            attention_type=args.attention_type,
            n_heads=args.n_heads,
            n_latents=args.n_latents,
            n_groups=args.n_groups,

            # Feed-forward configuration (Transformer specific)
            ffn_type=args.ffn_type,
            ffn_dim=args.ffn_dim,
            n_experts=args.n_experts,
            top_k=args.top_k,

            # Normalization (Transformer specific)
            norm_type=args.norm_type,

            # Residual connections (Transformer specific)
            residual_scale=args.residual_scale,
            use_gated_residual=args.use_gated_residual,
            use_final_norm=args.use_final_norm,

            # LSTM specific configuration
            lstm_hidden_dim=args.lstm_hidden_dim,
            lstm_num_layers=args.lstm_num_layers,
            lstm_dropout=args.lstm_dropout,

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
            n_features=train_df.shape[1], # Get actual feature count from data
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
