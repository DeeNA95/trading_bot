#!/usr/bin/env python3
"""
Training script for the Binance Futures RL trading agent.
"""
import argparse
import logging
import os
import random # Added import
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit # Added import
import mlflow # Added import
import mlflow.pytorch # Added import for model logging
import tqdm
from tqdm import tqdm # Explicit import for progress bar usage

from rl_agent.agent.ppo_agent import PPOAgent
from rl_agent.environment.trading_env import BinanceFuturesCryptoEnv
from data import DataHandler # Added import

logger = logging.getLogger(__name__)


def set_seeds(seed_value=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # if using multi-GPU
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

    # Convert string columns like 'trade_setup' to numerical values
    if "trade_setup" in df.columns:
        # Map trade setup values to numerical equivalents
        setup_mapping = {
            "none": 0.0,
            "strong_bullish": 1.0,
            "strong_bearish": -1.0,
            "bullish_reversal": 0.5,
            "bearish_reversal": -0.5,
        }
        df["trade_setup"] = df["trade_setup"].map(setup_mapping)

    # Convert all object/string columns to float
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
         print("Error: NaNs still detected after ffill and dropna!") # Should not happen

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
    args: argparse.Namespace # Pass all args
    # symbol="ETHUSDT", # Get from args
    # window_size=60, # Get from args
    # leverage=20, # Get from args
    # episodes=500, # Get from args
    # batch_size=256, # Get from args
    # update_freq=2048, # Get from args
    # log_freq=10, # Get from args
    # save_freq=20, # Get from args - Will be replaced by validation logic
    # eval_freq=20, # Get from args - Will be used for validation frequency
    # dynamic_leverage=True, # Get from args
    # use_risk_adjusted_rewards=True, # Get from args
    # max_position=1.0, # Get from args
    # initial_balance=10000, # Get from args
    # risk_reward_ratio=1.5, # Get from args
    # stop_loss_percent=0.05, # Get from args
    # model_save_path=None, # Get from args - Base path
    # trade_fee_percent=0.004, # Get from args - TODO: Add as arg? Currently hardcoded in env default
):
    """
    Train the RL agent for a single walk-forward fold, evaluate on validation set
    for model selection, and finally test the best model on the test set.
    """
    logger.info(f"--- Starting Training for Fold {fold_num} ---")

    # Data is already loaded and split
    if train_df.empty or val_df.empty or test_df.empty:
        logger.error(f"Fold {fold_num}: Empty dataframe received. Skipping.")
        return None, None # Indicate failure

    # Extract args for clarity (optional, but can improve readability)
    symbol = args.symbol
    window_size = args.window
    leverage = args.leverage
    episodes = args.episodes
    batch_size = args.batch_size
    update_freq = args.update_freq
    log_freq = args.log_freq
    eval_freq = args.eval_freq # Frequency for validation checks
    dynamic_leverage = args.dynamic_leverage
    use_risk_adjusted_rewards = args.use_risk_adjusted_rewards
    max_position = args.max_position
    initial_balance = args.balance
    risk_reward_ratio = args.risk_reward
    stop_loss_percent = args.stop_loss
    trade_fee_percent = args.trade_fee # Extract from args
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
        mode="train", # Use train mode
        leverage=leverage,
        max_position=max_position,
        initial_balance=initial_balance,
        risk_reward_ratio=risk_reward_ratio,
        stop_loss_percent=stop_loss_percent,
        dynamic_leverage=dynamic_leverage,
        use_risk_adjusted_rewards=use_risk_adjusted_rewards,
        trade_fee_percent=trade_fee_percent, # Pass the fee
    )
    # Validation Env
    val_env = create_environment(
        df=val_df,
        symbol=symbol,
        window_size=window_size,
        mode="test", # Use test mode for evaluation logic
        leverage=leverage,
        max_position=max_position,
        initial_balance=initial_balance, # Start with same balance for comparison
        risk_reward_ratio=risk_reward_ratio,
        stop_loss_percent=stop_loss_percent,
        dynamic_leverage=dynamic_leverage,
        use_risk_adjusted_rewards=use_risk_adjusted_rewards,
        trade_fee_percent=trade_fee_percent, # Pass the fee
    )
    # Test Env (created later before final testing)

    # --- Agent Creation ---
    logger.info(f"Fold {fold_num}: Creating PPO Agent (Transformer)...")
    agent = PPOAgent(
        env=train_env, # Agent holds the training env by default
        # model_type="transformer", # Hardcoded in PPOAgent now
        batch_size=batch_size,
        save_dir=fold_save_path, # Use fold-specific path
        # Pass other relevant hyperparameters from args if needed
        lr=args.lr, # Example if lr was an arg
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        policy_clip=args.policy_clip,
        n_epochs=args.n_epochs,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        # device=args.device, # Using auto device
        use_gae=args.use_gae,
        normalize_advantage=args.normalize_advantage,
        weight_decay=args.weight_decay,
    )

    # --- Training Loop with Validation ---
    logger.info(f"Fold {fold_num}: Starting training for {episodes} episodes...")
    max_steps_train = len(train_df) - window_size # Max steps per episode in training env
    best_val_score = -float('inf') # Use a relevant metric, e.g., avg reward
    best_val_episode = -1
    training_info = {"train_rewards": [], "val_rewards": [], "policy_loss": [], "value_loss": [], "entropy": []}
    model_saved_for_fold = False # Flag to check if any model was saved

    pbar_episodes = tqdm(range(episodes), desc=f"Fold {fold_num} Training", leave=False)
    for episode in pbar_episodes:
        state, _ = train_env.reset()
        # Reset LSTM state at the beginning of each episode if applicable
        # if isinstance(agent.model, ActorCriticLSTM): # Not needed as we only use Transformer
        #     agent.model.reset_hidden_state()

        done = False
        truncated = False
        episode_reward = 0
        steps_in_episode = 0
        # steps_since_update = 0 # Replaced by memory check

        while not (done or truncated):
            # Check max steps for safety, though env should handle 'done' based on data length
            if steps_in_episode >= max_steps_train:
                logger.warning(f"Fold {fold_num} Ep {episode+1}: Truncating episode at max steps {max_steps_train}")
                truncated = True

            action, prob, val = agent.choose_action(state)
            next_state, reward, done, truncated, info = train_env.step(action)

            # Store experience
            # Use done or truncated for the done flag in memory
            agent.memory.store(state, action, prob, val, reward, done or truncated)
            state = next_state
            episode_reward += reward
            # steps_since_update += 1 # Replaced by memory check
            steps_in_episode += 1

            # Perform PPO update when enough experience is collected
            # Check memory size instead of steps_since_update for more standard PPO buffer handling
            if len(agent.memory.states) >= agent.batch_size:
                    update_metrics = agent.update()
                    # Store metrics from the update
                    if update_metrics:
                        training_info["policy_loss"].append(update_metrics["policy_loss"])
                        training_info["value_loss"].append(update_metrics["value_loss"])
                        training_info["entropy"].append(update_metrics["entropy"])
                    # Memory is cleared inside agent.update() now

            # Check termination conditions for training episode
            if done or truncated:
                break # Exit inner while loop

        training_info["train_rewards"].append(episode_reward)

        # --- Periodic Validation & Conditional Saving ---
        if (episode + 1) % eval_freq == 0 or episode == episodes - 1: # Also validate at the end
            # Evaluate on validation set
            original_env = agent.env # Store original training env
            agent.env = val_env # Temporarily switch agent's env for evaluation
            # Use a reasonable number of episodes for validation
            num_val_episodes = 5 # Make this configurable?
            val_reward = agent.evaluate(num_episodes=num_val_episodes)
            agent.env = original_env # Switch back to training env
            training_info["val_rewards"].append(val_reward)
            avg_train_reward_log = np.mean(training_info["train_rewards"][-(eval_freq):]) # Avg reward since last eval
            avg_policy_loss_log = np.mean(training_info["policy_loss"][-(eval_freq*agent.batch_size//update_freq):]) if training_info["policy_loss"] else np.nan # Approx avg loss since last eval
            avg_value_loss_log = np.mean(training_info["value_loss"][-(eval_freq*agent.batch_size//update_freq):]) if training_info["value_loss"] else np.nan
            avg_entropy_log = np.mean(training_info["entropy"][-(eval_freq*agent.batch_size//update_freq):]) if training_info["entropy"] else np.nan

            logger.info(f"Fold {fold_num} | Ep {episode + 1}/{episodes} | Train R(Ep): {episode_reward:.2f} | Train R(Avg {eval_freq}): {avg_train_reward_log:.2f} | Val R(Avg {num_val_episodes}): {val_reward:.2f}")
            # Log metrics to MLflow (nested run for the fold)
            # Use episode number as the step
            mlflow.log_metric("train_reward_episode", episode_reward, step=episode + 1)
            mlflow.log_metric("train_reward_avg_interval", avg_train_reward_log if pd.notna(avg_train_reward_log) else 0.0, step=episode + 1)
            mlflow.log_metric("val_reward_avg", val_reward if pd.notna(val_reward) else 0.0, step=episode + 1)
            mlflow.log_metric("policy_loss_avg_interval", avg_policy_loss_log if pd.notna(avg_policy_loss_log) else 0.0, step=episode + 1)
            mlflow.log_metric("value_loss_avg_interval", avg_value_loss_log if pd.notna(avg_value_loss_log) else 0.0, step=episode + 1)
            mlflow.log_metric("entropy_avg_interval", avg_entropy_log if pd.notna(avg_entropy_log) else 0.0, step=episode + 1)


            # Conditional Saving Logic
            # Define 'better' score (e.g., higher reward)
            # Define minimum acceptable score (e.g., positive reward)
            MIN_ACCEPTABLE_VAL_SCORE = 0.0 # Example threshold - Make configurable?
            if val_reward > best_val_score and val_reward >= MIN_ACCEPTABLE_VAL_SCORE:
                best_val_score = val_reward
                best_val_episode = episode + 1
                agent.save(best_model_path)
                logger.info(f"    -> New best model saved! Val Score: {best_val_score:.2f} at Ep {best_val_episode}")
                model_saved_for_fold = True
            # Update progress bar description
            pbar_episodes.set_description(f"Fold {fold_num} Training | Best Val R: {best_val_score:.2f} @ Ep {best_val_episode}")

        # Log training progress periodically (independent of validation)
        # if (episode + 1) % log_freq == 0:
                # Logging now happens within validation block for combined info

    pbar_episodes.close() # Close the episode progress bar
    logger.info(f"--- Finished Training for Fold {fold_num} ---")
    if not model_saved_for_fold:
        logger.warning(f"Fold {fold_num}: No model met the saving criteria during training.")
        # Handle case where no model was saved (e.g., return failure)
        return {"avg_reward": np.nan, "avg_sharpe": np.nan}, training_info

    # --- Final Testing ---
    logger.info(f"Fold {fold_num}: Loading best model from {best_model_path} for testing...")
    try:
        # Create a new agent instance for testing to ensure clean state? Or just load into existing.
        # Let's load into existing agent for now.
        agent.load(best_model_path)
    except FileNotFoundError:
        logger.error(f"Fold {fold_num}: Best model file not found at {best_model_path}. Cannot perform testing.")
        # Return dummy/failure metrics
        return {"avg_reward": np.nan, "avg_sharpe": np.nan}, training_info # Or raise error

    logger.info(f"Fold {fold_num}: Creating test environment...")
    test_env = create_environment(
        df=test_df,
        symbol=symbol,
        window_size=window_size,
        mode="test", # Use test mode
        leverage=leverage,
        max_position=max_position,
        initial_balance=initial_balance,
        risk_reward_ratio=risk_reward_ratio,
        stop_loss_percent=stop_loss_percent,
        dynamic_leverage=dynamic_leverage,
        use_risk_adjusted_rewards=use_risk_adjusted_rewards,
        trade_fee_percent=trade_fee_percent, # Pass the fee
    )

    logger.info(f"Fold {fold_num}: Evaluating best model on test set...")
    original_env = agent.env
    agent.env = test_env
    # Use a higher number of episodes for more stable test evaluation
    num_test_episodes = 20
    test_rewards = []
    test_sharpes = [] # Add other metrics as needed (drawdown, win rate etc.)
    # TODO: Enhance agent.evaluate or create a dedicated test function
    # to return more detailed metrics like in the original train_agent.
    for _ in range(num_test_episodes):
        test_reward = agent.evaluate(num_episodes=1) # Evaluate one episode at a time
        test_rewards.append(test_reward)
        # Placeholder for Sharpe calculation - needs access to balance history from evaluate
        test_sharpes.append(np.random.rand()) # Dummy value

    agent.env = original_env # Switch back

    avg_test_reward = np.mean(test_rewards)
    avg_test_sharpe = np.mean(test_sharpes) # Dummy value
    logger.info(f"Fold {fold_num}: Test Set Avg Reward: {avg_test_reward:.2f}")
    logger.info(f"Fold {fold_num}: Test Set Avg Sharpe: {avg_test_sharpe:.2f}") # Dummy value

    test_metrics = {
        "avg_reward": avg_test_reward,
        "avg_sharpe": avg_test_sharpe, # Dummy value
        # Add other aggregated test metrics here
    }

    # TODO: Log fold results (training_info, test_metrics) to MLflow

    return test_metrics, training_info



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for trading")
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to the full historical data file (CSV or Parquet)",
    )
    # parser.add_argument( # Removed: Test data generated by walk-forward
    #     "--test_data",
    #     type=str,
    #     help="Path to test data file for evaluation (CSV or Parquet)",
    # )
    parser.add_argument(
        "--symbol", type=str, default="ETHUSDT", help="Trading pair symbol"
    )
    parser.add_argument(
        "--interval", type=str, default="1m", help="Data interval (e.g., '1m', '15m', '1h')"
    )
    parser.add_argument(
        "--window", type=int, default=60, help="Observation window size"
    )
    # parser.add_argument( # Removed: Model type is hardcoded to Transformer in PPOAgent
    #     "--model",
    #     type=str,
    #     default="transformer",
    #     choices=["cnn", "lstm", "transformer"],
    #     help="Model architecture",
    # )
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
        "--trade_fee", type=float, default=0.0004, help="Trade fee percentage (e.g., 0.0004 for 0.04%)"
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
        "--save_path", type=str, default="models", help="Base path to save trained models and results"
    )
    # Walk-forward validation arguments
    parser.add_argument(
        "--n_splits", type=int, default=5, help="Number of splits for TimeSeriesSplit walk-forward validation"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="Fraction of training data per fold to use for validation"
    )
    # Note: test_size is implicitly defined by the gap between folds in TimeSeriesSplit or can be set explicitly.
    # For now, we'll use the default TimeSeriesSplit behavior where test set follows train/val.

    args = parser.parse_args()

    # Set random seeds for reproducibility (using default seed 42 for now)
    # TODO: Optionally add a --seed argument to parser and use args.seed
    set_seeds()
    logger.info("Random seeds set.")

    # --- MLflow Setup ---
    # Set the Tracking URI to the server we started
    # Ensure the server is running before executing this script
    mlflow_tracking_uri = "http://127.0.0.1:5001" # Default if server is local
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    # Optionally set experiment name
    experiment_name = f"{args.symbol}_WalkForward_Training"
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
        logger.info(f"MLflow experiment name set to: {experiment_name}")
    except Exception as e:
        logger.error(f"Could not set MLflow tracking URI or experiment: {e}")
        logger.warning("Proceeding without MLflow tracking.")
        # Optionally exit if MLflow is critical
        # exit()

    # Start the main MLflow run for the entire walk-forward process
    with mlflow.start_run() as parent_run:
        logger.info(f"Started MLflow parent run: {parent_run.info.run_id}")
        # Log hyperparameters from args
        # Convert Namespace to dict, handle potential non-loggable types if necessary
        try:
            mlflow.log_params(vars(args))
        except Exception as e:
            logger.warning(f"Could not log all args to MLflow: {e}")

        # --- Walk-Forward Validation Setup ---
        logger.info("Starting Walk-Forward Validation...")

        # 1. Load the full dataset
    full_df = load_and_preprocess_data(args.train_data)
    if full_df.empty or len(full_df) < args.window * (args.n_splits + 1): # Basic check for enough data
        logger.error(f"Not enough data for {args.n_splits} splits with window {args.window}. Exiting.")
        exit()

    # 2. Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=args.n_splits)

    all_fold_results = []
    fold_num = 0

    # 3. Loop through walk-forward splits
    # Note: tscv yields (train_indices, test_indices). We carve out validation from train.
    for train_val_indices, test_indices in tscv.split(full_df):
        fold_num += 1
        logger.info(f"\n===== Starting Walk-Forward Fold {fold_num}/{args.n_splits} =====")

        # Start nested MLflow run for this fold
        with mlflow.start_run(nested=True, run_name=f"Fold_{fold_num}") as fold_run:
            mlflow.log_param("fold_number", fold_num)
            logger.info(f"Started MLflow nested run for Fold {fold_num}: {fold_run.info.run_id}")

            # Ensure we have enough data for the initial window in the test set
        if len(test_indices) < args.window:
            logger.warning(f"Fold {fold_num}: Skipping test set, not enough data ({len(test_indices)} < {args.window}).")
            continue

        # 4. Split train_val further into train and validation
        # Ensure validation set is reasonably sized, e.g., at least window_size
        val_size = max(args.window, int(len(train_val_indices) * args.val_ratio))
        train_size = len(train_val_indices) - val_size

        if train_size < args.window: # Need at least one window for training
            logger.warning(f"Fold {fold_num}: Skipping, not enough training data ({train_size} < {args.window}) after validation split.")
            continue

        train_indices = train_val_indices[:train_size]
        val_indices = train_val_indices[train_size:]

        train_df = full_df.iloc[train_indices]
        val_df = full_df.iloc[val_indices]
        test_df = full_df.iloc[test_indices]

        logger.info(f"Fold {fold_num}: Train size={len(train_df)}, Val size={len(val_df)}, Test size={len(test_df)}")
        # Log date ranges if index is datetime
        if isinstance(train_df.index, pd.DatetimeIndex):
            logger.info(f"Train period: {train_df.index.min()} to {train_df.index.max()}")
            logger.info(f"Val period:   {val_df.index.min()} to {val_df.index.max()}")
            logger.info(f"Test period:  {test_df.index.min()} to {test_df.index.max()}")
            # Log data params to nested run
            mlflow.log_param("fold_train_size", len(train_df))
            mlflow.log_param("fold_val_size", len(val_df))
            mlflow.log_param("fold_test_size", len(test_df))


        # --- Feature Calculation (Post-Split) ---
        logger.info(f"Fold {fold_num}: Calculating features for Train/Val/Test sets...")
        # Apply feature calculations to each slice independently
        # TODO: Add error handling for feature calculation
        # TODO: Externalize parameters for these functions (window sizes etc.)
        # TODO: Decide if futures metrics (add_futures_metrics) are needed and how to handle them post-split
        #       (requires DataHandler instance and careful time alignment - potentially complex)
        train_df = DataHandler.calculate_technical_indicators(train_df.copy())
        train_df = DataHandler.calculate_risk_metrics(train_df.copy(), interval=args.interval, window_size=args.window) # Pass interval/window?
        train_df = DataHandler.identify_trade_setups(train_df.copy())
        train_df = DataHandler.calculate_price_density(train_df.copy(), window=args.window) # Pass window?
        train_df = DataHandler.normalise_ohlc(train_df.copy(), window=args.window) # Pass window?

        val_df = DataHandler.calculate_technical_indicators(val_df.copy())
        val_df = DataHandler.calculate_risk_metrics(val_df.copy(), interval=args.interval, window_size=args.window)
        val_df = DataHandler.identify_trade_setups(val_df.copy())
        val_df = DataHandler.calculate_price_density(val_df.copy(), window=args.window)
        val_df = DataHandler.normalise_ohlc(val_df.copy(), window=args.window)

        test_df = DataHandler.calculate_technical_indicators(test_df.copy())
        test_df = DataHandler.calculate_risk_metrics(test_df.copy(), interval=args.interval, window_size=args.window)
        test_df = DataHandler.identify_trade_setups(test_df.copy())
        test_df = DataHandler.calculate_price_density(test_df.copy(), window=args.window)
        test_df = DataHandler.normalise_ohlc(test_df.copy(), window=args.window)

        # Important: Re-apply ffill().dropna() after feature calculation as indicators introduce NaNs
        logger.info(f"Fold {fold_num}: Applying final ffill/dropna after feature calculation...")
        train_df = train_df.ffill().dropna()
        val_df = val_df.ffill().dropna()
        test_df = test_df.ffill().dropna()

        # Check if data remains after processing
        if train_df.empty or val_df.empty or test_df.empty:
            logger.error(f"Fold {fold_num}: Dataframe became empty after feature calculation/dropna. Skipping fold.")
            mlflow.log_metric("fold_skipped_post_feature_calc", 1.0)
            continue # Skip to next fold

        logger.info(f"Fold {fold_num}: Final Train size={len(train_df)}, Val size={len(val_df)}, Test size={len(test_df)}")
        # --- End Feature Calculation ---


        # Call the refactored training and evaluation function for this fold
        fold_test_metrics, fold_training_info = train_evaluate_fold(
            fold_num=fold_num,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            args=args
        )

        # Store results if training/evaluation was successful
        if fold_test_metrics:
            all_fold_results.append(fold_test_metrics)
            # Log test metrics for this fold to the nested run
            # Ensure metrics are float for MLflow
            loggable_metrics = {f"fold_test_{k}": float(v) for k, v in fold_test_metrics.items() if pd.notna(v)}
            mlflow.log_metrics(loggable_metrics)
            # Log data ranges for this fold
            if isinstance(train_df.index, pd.DatetimeIndex):
                mlflow.log_param("fold_train_start", str(train_df.index.min()))
                mlflow.log_param("fold_train_end", str(train_df.index.max()))
                mlflow.log_param("fold_val_start", str(val_df.index.min()))
                mlflow.log_param("fold_val_end", str(val_df.index.max()))
                mlflow.log_param("fold_test_start", str(test_df.index.min()))
                mlflow.log_param("fold_test_end", str(test_df.index.max()))
            # Log path to the best model saved for this fold
            # Construct path relative to project root or use absolute path if needed
            best_model_path_fold = os.path.join(args.save_path, f"fold_{fold_num}", "best_model.pt")
            if os.path.exists(best_model_path_fold):
                 # Log artifact - this uploads the model file to MLflow artifact store (GCS in this case)
                 try:
                     mlflow.log_artifact(best_model_path_fold, artifact_path=f"fold_{fold_num}_model")
                     logger.info(f"Logged best model artifact for fold {fold_num}")
                 except Exception as e:
                     logger.warning(f"Could not log model artifact for fold {fold_num}: {e}")
            else:
                 logger.warning(f"Best model artifact not found for fold {fold_num} at {best_model_path_fold}")

        else:
            logger.error(f"Fold {fold_num} failed, skipping results.")
            # Log failure explicitly in MLflow for this fold
            mlflow.log_metric("fold_test_avg_reward", 0.0) # Use 0 or NaN? MLflow prefers numbers.
            mlflow.log_metric("fold_test_avg_sharpe", 0.0) # Match dummy metric name

        logger.info(f"===== Finished Walk-Forward Fold {fold_num}/{args.n_splits} =====")

    # 5. Aggregate and report results (outside the fold loop, inside parent run)
    logger.info("\n===== Walk-Forward Validation Summary =====")
    if all_fold_results:
        # Example aggregation (needs refinement based on actual metrics returned)
        avg_rewards = [r.get("avg_reward", np.nan) for r in all_fold_results]
        avg_sharpes = [r.get("avg_sharpe", np.nan) for r in all_fold_results] # Still dummy values
        mean_reward = np.nanmean(avg_rewards)
        std_reward = np.nanstd(avg_rewards)
        mean_sharpe = np.nanmean(avg_sharpes) # Dummy value
        std_sharpe = np.nanstd(avg_sharpes) # Dummy value

        logger.info(f"Aggregated Test Reward across {len(avg_rewards)} folds: {mean_reward:.4f} (Std: {std_reward:.4f})")
        logger.info(f"Aggregated Test Sharpe across {len(avg_sharpes)} folds: {mean_sharpe:.4f} (Std: {std_sharpe:.4f})") # Dummy value

        # Log aggregated metrics to parent run
        mlflow.log_metric("agg_test_reward_mean", mean_reward if pd.notna(mean_reward) else 0.0)
        mlflow.log_metric("agg_test_reward_std", std_reward if pd.notna(std_reward) else 0.0)
        mlflow.log_metric("agg_test_sharpe_mean", mean_sharpe if pd.notna(mean_sharpe) else 0.0) # Dummy value
        mlflow.log_metric("agg_test_sharpe_std", std_sharpe if pd.notna(std_sharpe) else 0.0) # Dummy value
        mlflow.log_metric("num_successful_folds", len(all_fold_results))

        # TODO: Add more detailed aggregation (drawdown, win rate, plots, etc.) and log them
    else:
        logger.warning("No fold results to aggregate.")
        # Log failure indication to parent run
        mlflow.log_metric("num_successful_folds", 0)


    logger.info("Walk-Forward Validation process completed.")
    # End of parent mlflow run (implicitly happens here)

