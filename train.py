#!/usr/bin/env python3
"""
Training script for the Binance Futures RL trading agent.
"""
import argparse
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from rl_agent.agent.ppo_agent import PPOAgent
from rl_agent.environment.trading_env import BinanceFuturesCryptoEnv

logger = logging.getLogger(__name__)


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
    # Replace NaN, inf values with 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Normalize data to prevent NaN in model outputs
    for col in df.columns:
        if col not in ["open_time", "date", "timestamp"]:
            # Use robust normalization to handle outliers
            median = df[col].median()
            iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
            if iqr > 0:
                df[col] = (df[col] - median) / (iqr + 1e-8)
            else:
                # If IQR is 0, just center the data
                df[col] = df[col] - median

    print(f"Data shape: {df.shape}")
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


def train_agent(
    train_data_path,
    test_data_path=None,
    symbol="BTCUSDT",
    window_size=24,
    mode="train",
    model_type="lstm",
    leverage=20,
    episodes=500,
    batch_size=256,
    update_freq=2048,
    log_freq=10,
    save_freq=20,
    eval_freq=20,
    dynamic_leverage=True,
    use_risk_adjusted_rewards=True,
    max_position=1.0,
    initial_balance=10000,
    risk_reward_ratio=1.5,
    stop_loss_percent=0.05,
    model_save_path=None,
    trade_fee_percent=0.004,
):
    """Train the RL agent on historical data with separate test dataset."""

    # Load and preprocess training data
    train_df = load_and_preprocess_data(train_data_path)

    # Create training environment
    print(f"Creating training environment with {model_type} model...")
    train_env = create_environment(
        df=train_df,
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

    # Create agent with model_save_path
    agent = PPOAgent(
        env=train_env,
        model_type=model_type.lower(),
        batch_size=batch_size,
        save_dir=model_save_path or "models",
    )

    # Train the agent
    print(f"Starting training for {episodes} episodes...")
    max_steps = len(train_df) - window_size  # Maximum steps per episode

    training_info = agent.train(
        num_episodes=episodes,
        max_steps=max_steps,
        update_freq=update_freq,
        log_freq=log_freq,
        save_freq=save_freq,
        eval_freq=eval_freq,
        num_eval_episodes=10,
    )

    # Save final model with timestamp if path provided
    if model_save_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{symbol}_{model_type}_{timestamp}.pt"
        full_path = os.path.join(model_save_path, model_name)
        os.makedirs(model_save_path, exist_ok=True)
        agent.save(full_path)
        logger.info(f"Final model saved to {full_path}")

    # Final evaluation on training data
    train_reward = agent.evaluate(num_episodes=10)
    logger.info(
        f"Final model evaluation on training data: Mean reward={train_reward:.2f}"
    )

    # If test data is provided, evaluate on it
    if test_data_path:
        # Load and preprocess test data
        test_df = load_and_preprocess_data(test_data_path)

        # Create test environment
        print(f"Creating test environment for evaluation...")
        test_env = create_environment(
            df=test_df,
            symbol=symbol,
            window_size=window_size,
            mode="test",  # Always use test mode for evaluation
            leverage=leverage,
            max_position=max_position,
            initial_balance=initial_balance,
            risk_reward_ratio=risk_reward_ratio,
            stop_loss_percent=stop_loss_percent,
            dynamic_leverage=dynamic_leverage,
            use_risk_adjusted_rewards=use_risk_adjusted_rewards,
            trade_fee_percent=trade_fee_percent,
        )

        # Update agent's environment to test environment
        original_env = agent.env
        agent.env = test_env

        # Evaluate on test data
        test_reward = agent.evaluate(num_episodes=20)
        logger.info(
            f"Final model evaluation on test data: Mean reward={test_reward:.2f}"
        )

        # Restore original environment
        agent.env = original_env

    logger.info("Training completed!")
    return agent, training_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for trading")
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data file (CSV or Parquet)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        help="Path to test data file for evaluation (CSV or Parquet)",
    )
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Trading pair symbol"
    )
    parser.add_argument(
        "--window", type=int, default=24, help="Observation window size"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["cnn", "lstm", "transformer"],
        help="Model architecture",
    )
    parser.add_argument("--leverage", type=int, default=20, help="Trading leverage")
    parser.add_argument(
        "--episodes", type=int, default=500, help="Number of training episodes"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--update_freq", type=int, default=2048, help="Steps between policy updates"
    )
    parser.add_argument(
        "--log_freq", type=int, default=50, help="Episodes between log updates"
    )
    parser.add_argument(
        "--save_freq", type=int, default=50, help="Episodes between model saves"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=50, help="Episodes between evaluations"
    )
    parser.add_argument(
        "--max_position",
        type=float,
        default=0.5,
        help="Maximum position size as fraction of balance",
    )
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument(
        "--risk_reward", type=float, default=1.5, help="Risk-reward ratio"
    )
    parser.add_argument(
        "--stop_loss", type=float, default=0.01, help="Stop loss percentage"
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

    train_agent(
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        symbol=args.symbol,
        window_size=args.window,
        model_type=args.model,
        leverage=args.leverage,
        episodes=args.episodes,
        batch_size=args.batch_size,
        update_freq=args.update_freq,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        dynamic_leverage=args.dynamic_leverage,
        use_risk_adjusted_rewards=args.use_risk_adjusted_rewards,
        max_position=args.max_position,
        initial_balance=args.balance,
        risk_reward_ratio=args.risk_reward,
        stop_loss_percent=args.stop_loss,
        model_save_path=args.save_path,
    )
