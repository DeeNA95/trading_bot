#!/usr/bin/env python3
"""
Training script for the Binance Futures RL trading agent.
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import ta
import torch

from rl_agent.agent.models import (
    ActorCriticCNN,
    ActorCriticLSTM,
    ActorCriticTransformer,
)
from rl_agent.agent.ppo_agent import PPOAgent
from rl_agent.environment.trading_env import BinanceFuturesCryptoEnv


def prepare_data(df):
    """
    Add technical indicators to the dataframe for the training data.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original
    data = df.copy()

    # Ensure numeric columns are float
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = data[col].astype(float)

    # If any required columns are missing, skip indicators that need them
    if all(col in data.columns for col in ["close"]):
        # Momentum indicators
        data["rsi"] = ta.momentum.RSIIndicator(data["close"]).rsi()
        data["macd"] = ta.trend.MACD(data["close"]).macd()
        data["macd_signal"] = ta.trend.MACD(data["close"]).macd_signal()
        data["macd_diff"] = ta.trend.MACD(data["close"]).macd_diff()

        # Trend indicators
        data["sma_20"] = ta.trend.SMAIndicator(data["close"], window=20).sma_indicator()
        data["sma_50"] = ta.trend.SMAIndicator(data["close"], window=50).sma_indicator()
        data["ema_12"] = ta.trend.EMAIndicator(data["close"], window=12).ema_indicator()
        data["ema_26"] = ta.trend.EMAIndicator(data["close"], window=26).ema_indicator()

        # Volatility indicators
        if all(col in data.columns for col in ["high", "low"]):
            bb = ta.volatility.BollingerBands(data["close"])
            data["bb_upper"] = bb.bollinger_hband()
            data["bb_lower"] = bb.bollinger_lband()
            data["bb_mid"] = bb.bollinger_mavg()
            data["atr"] = ta.volatility.AverageTrueRange(
                data["high"], data["low"], data["close"]
            ).average_true_range()

            # Volume indicators
            if "volume" in data.columns:
                data["mfi"] = ta.volume.MFIIndicator(
                    data["high"], data["low"], data["close"], data["volume"]
                ).money_flow_index()

    # Fill NaN values that may result from the calculations
    data.fillna(method="bfill", inplace=True)
    data.fillna(0, inplace=True)

    # Convert all columns to float to ensure consistency
    for col in data.columns:
        if data[col].dtype != float:
            try:
                data[col] = data[col].astype(float)
            except Exception:
                # If conversion fails, drop the column
                print(f"Dropping non-numeric column: {col}")
                data.drop(columns=[col], inplace=True)

    return data


def train_agent(
    data_path,
    symbol="BTCUSDT",
    window_size=30,
    mode="train",
    model_type="lstm",
    leverage=2,
    episodes=100,
    batch_size=64,
    update_freq=2048,
    log_freq=10,
    save_freq=20,
    eval_freq=20,
    dynamic_leverage=True,
    use_risk_adjusted_rewards=True,
    max_position=0.2,
    initial_balance=10000,
    risk_reward_ratio=1.5,
    stop_loss_percent=0.01,
    model_save_path=None,
):
    """Train the RL agent on historical data."""
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Convert date column to datetime index if needed
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "date"}, inplace=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    # Add technical indicators
    print("Adding technical indicators...")
    df = prepare_data(df)

    # Create environment
    print(f"Creating environment with {model_type} model...")
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
    )

    # Get state shape from environment
    state_shape = env.observation_space.shape
    action_dim = env.action_space.n

    # The actual model initialization is handled by the PPOAgent class,
    # so we don't need to instantiate the model here. Just pass the
    # model type to the agent.

    # Create agent with model_save_path
    agent = PPOAgent(
        env=env,
        model_type=model_type.lower(),
        batch_size=batch_size,
        save_dir=model_save_path or "models",
    )

    # Train the agent
    print(f"Starting training for {episodes} episodes...")
    max_steps = len(df) - window_size  # Maximum steps per episode

    training_info = agent.train(
        num_episodes=episodes,
        max_steps=max_steps,
        update_freq=update_freq,
        log_freq=log_freq,
        save_freq=save_freq,
        eval_freq=eval_freq,
        num_eval_episodes=3,
    )

    # Save final model with timestamp if path provided
    if model_save_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{symbol}_{model_type}_{timestamp}.pt"
        full_path = os.path.join(model_save_path, model_name)
        os.makedirs(model_save_path, exist_ok=True)
        agent.save(full_path)
        print(f"Final model saved to {full_path}")

    # Final evaluation
    final_reward = agent.evaluate(num_episodes=5)
    print(f"Final model evaluation: Mean reward={final_reward:.2f}")

    print("Training completed!")
    return agent, training_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for trading")
    parser.add_argument("--data", type=str, required=True, help="Path to data CSV file")
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Trading pair symbol"
    )
    parser.add_argument(
        "--window", type=int, default=30, help="Observation window size"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["cnn", "lstm", "transformer"],
        help="Model architecture",
    )
    parser.add_argument("--leverage", type=int, default=2, help="Trading leverage")
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of training episodes"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--update_freq", type=int, default=2048, help="Steps between policy updates"
    )
    parser.add_argument(
        "--log_freq", type=int, default=10, help="Episodes between log updates"
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
        default=0.2,
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
        data_path=args.data,
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
