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

from rl_agent.agent.ppo_agent import PPOAgent
from rl_agent.environment.trading_env import BinanceFuturesCryptoEnv
# Assuming DataHandler is not used here based on original train.py
# from data import DataHandler

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
    parser = argparse.ArgumentParser(description="Train RL agent for trading with Walk-Forward Validation")

    # --- Data and Env Args ---
    parser.add_argument("--train_data", type=str, required=True, help="Path to the FULL historical data file (CSV or Parquet)")
    # --test_data is removed, testing happens within folds
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol")
    # Add interval if needed by data loading/preprocessing, else remove
    # parser.add_argument("--interval", type=str, default="1m", help="Data interval (e.g., '1m', '15m', '1h')")
    parser.add_argument("--window", type=int, default=60, help="Observation window size") # Increased default
    parser.add_argument("--leverage", type=int, default=20, help="Trading leverage")
    parser.add_argument("--max_position", type=float, default=1.0, help="Maximum position size as fraction of balance")
    parser.add_argument("--balance", type=float, default=1000, help="Initial balance") # Adjusted default
    parser.add_argument("--risk_reward", type=float, default=1.5, help="Risk-reward ratio")
    parser.add_argument("--stop_loss", type=float, default=0.01, help="Stop loss percentage") # Adjusted default
    parser.add_argument("--trade_fee", type=float, default=0.0004, help="Trade fee percentage (e.g., 0.0004 for 0.04%)")
    parser.add_argument("--static_leverage", action="store_false", dest="dynamic_leverage", help="Use static leverage")
    parser.add_argument("--simple_rewards", action="store_false", dest="use_risk_adjusted_rewards", help="Use simple rewards")
    parser.set_defaults(dynamic_leverage=True, use_risk_adjusted_rewards=True) # Set defaults

    # --- Model Architecture ---
    parser.add_argument("--model", type=str, default="lstm", choices=["cnn", "lstm", "transformer"], help="Model architecture type for PPOAgent")

    # --- Walk-Forward Validation Args ---
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for TimeSeriesSplit walk-forward validation")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction of training data per fold to use for validation (e.g., 0.15 for 15%)")
    parser.add_argument("--save_path", type=str, default="models_walk_forward", help="Base path to save trained models and results per fold")

    # --- Training Loop Args ---
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes PER FOLD") # Reduced default for faster runs
    parser.add_argument("--eval_freq", type=int, default=10, help="Episodes between validation evaluations within a fold") # Reduced default

    # --- PPO Agent Hyperparameters (Add if PPOAgent constructor needs them) ---
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--policy_clip", type=float, default=0.2, help="PPO policy clipping parameter")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for PPO updates") # Increased default
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs per PPO update")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping")
    parser.add_argument("--update_freq", type=int, default=2048, help="Steps between policy updates (should ideally be multiple of batch_size)")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cpu', 'cuda', 'mps')") # Removed 'xla' unless supported
    # Add other PPO params like use_gae, normalize_advantage, weight_decay if needed by agent

    args = parser.parse_args()

    # --- Setup ---
    set_seeds()
    logger.info("Random seeds set.")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(args.save_path, exist_ok=True) # Create base save directory

    # --- Load Full Data ---
    logger.info("Loading and preprocessing full dataset...")
    full_df = load_and_preprocess_data(args.train_data)

    if full_df.empty or len(full_df) < args.window * (args.n_splits + 1): # Basic check
        logger.error(f"Not enough data ({len(full_df)}) for {args.n_splits} splits with window {args.window}. Exiting.")
        exit(1)

    # --- Walk-Forward Validation Setup ---
    logger.info(f"Starting Walk-Forward Validation with {args.n_splits} splits...")
    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    all_fold_results = []
    fold_num = 0

    # --- Walk-Forward Loop ---
    for train_val_indices, test_indices in tscv.split(full_df):
        fold_num += 1
        logger.info(f"\n===== Starting Walk-Forward Fold {fold_num}/{args.n_splits} =====")

        # Validate indices
        if len(test_indices) < args.window:
            logger.warning(f"Fold {fold_num}: Skipping test set, not enough data ({len(test_indices)} < {args.window}).")
            continue
        if len(train_val_indices) < 2 * args.window: # Need enough for train + val
             logger.warning(f"Fold {fold_num}: Skipping fold, not enough data in train/val split ({len(train_val_indices)}).")
             continue

        # Split train_val into actual train and validation sets
        val_size = max(args.window, int(len(train_val_indices) * args.val_ratio))
        train_size = len(train_val_indices) - val_size

        if train_size < args.window:
            logger.warning(f"Fold {fold_num}: Skipping, not enough training data ({train_size} < {args.window}) after validation split.")
            continue

        train_indices = train_val_indices[:train_size]
        val_indices = train_val_indices[train_size:]

        # Create dataframes for the fold
        train_df = full_df.iloc[train_indices]
        val_df = full_df.iloc[val_indices]
        test_df = full_df.iloc[test_indices]

        logger.info(f"Fold {fold_num}: Train size={len(train_df)}, Val size={len(val_df)}, Test size={len(test_df)}")
        if isinstance(train_df.index, pd.DatetimeIndex):
            logger.info(f"  Train period: {train_df.index.min()} to {train_df.index.max()}")
            logger.info(f"  Val period:   {val_df.index.min()} to {val_df.index.max()}")
            logger.info(f"  Test period:  {test_df.index.min()} to {test_df.index.max()}")

        # --- Train and Evaluate on this Fold ---
        test_metrics, training_info = train_evaluate_fold(
            fold_num=fold_num,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            args=args
        )

        if test_metrics is None:
            logger.warning(f"Fold {fold_num}: Training/evaluation failed or was skipped. Continuing to next fold.")
            continue

        all_fold_results.append(test_metrics)
        logger.info(f"Fold {fold_num} Test Results: {test_metrics}")

        # --- Save Fold Training Info ---
        fold_results_path = os.path.join(args.save_path, f"fold_{fold_num}", "training_info.json")
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(fold_results_path), exist_ok=True)
            with open(fold_results_path, 'w') as f:
                # Convert numpy types for JSON serialization
                serializable_info = {}
                for key, value in training_info.items():
                    if isinstance(value, list) and value and isinstance(value[0], (np.generic, np.ndarray)):
                         serializable_info[key] = [item.item() if hasattr(item, 'item') else item for item in value]
                    elif isinstance(value, np.ndarray):
                         serializable_info[key] = value.tolist()
                    elif isinstance(value, (np.generic)):
                         serializable_info[key] = value.item()
                    else:
                         serializable_info[key] = value
                json.dump(serializable_info, f, indent=4)
            logger.info(f"Fold {fold_num}: Saved training info to {fold_results_path}")
        except Exception as e:
            logger.error(f"Fold {fold_num}: Failed to save training info to {fold_results_path}: {e}")


    # --- Final Aggregation and Reporting ---
    logger.info("\n===== Walk-Forward Validation Finished =====")

    if not all_fold_results:
        logger.error("No folds completed successfully. Exiting.")
        exit(1)

    # Aggregate results (handle potential NaNs)
    avg_rewards = [r['avg_reward'] for r in all_fold_results if r and 'avg_reward' in r and not np.isnan(r['avg_reward'])]
    avg_sharpes = [r['avg_sharpe'] for r in all_fold_results if r and 'avg_sharpe' in r and not np.isnan(r['avg_sharpe'])]

    final_avg_reward = np.mean(avg_rewards) if avg_rewards else np.nan
    final_std_reward = np.std(avg_rewards) if avg_rewards else np.nan
    final_avg_sharpe = np.mean(avg_sharpes) if avg_sharpes else np.nan
    final_std_sharpe = np.std(avg_sharpes) if avg_sharpes else np.nan

    logger.info(f"Overall Average Test Reward across {len(avg_rewards)} successful folds: {final_avg_reward:.4f} (Std: {final_std_reward:.4f})")
    logger.info(f"Overall Average Test Sharpe across {len(avg_sharpes)} successful folds: {final_avg_sharpe:.4f} (Std: {final_std_sharpe:.4f})")

    # --- Save Overall Results ---
    overall_results = {
        "num_successful_folds": len(avg_rewards),
        "average_test_reward": final_avg_reward,
        "std_test_reward": final_std_reward,
        "average_test_sharpe": final_avg_sharpe,
        "std_test_sharpe": final_std_sharpe,
        "fold_results": all_fold_results,
        "args": vars(args) # Save arguments used for the run
    }
    results_file = os.path.join(args.save_path, "overall_results.json")
    try:
        with open(results_file, 'w') as f:
            # Handle numpy types during saving
             def default_serializer(obj):
                if isinstance(obj, (np.generic, np.ndarray)):
                    # Check if it's a scalar numpy type or an array
                    return obj.item() if hasattr(obj, 'item') and obj.ndim == 0 else obj.tolist()
                # Add handling for other non-serializable types if necessary
                # elif isinstance(obj, datetime): return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")

             json.dump(overall_results, f, indent=4, default=default_serializer)
        logger.info(f"Saved overall results to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save overall results to {results_file}: {e}")
