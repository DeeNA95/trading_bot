#!/usr/bin/env python3
"""
Inference script for the trading bot.
Uses trained RL models or Moving Average strategies to make trading decisions via Binance Futures API.
"""

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from rl_algo.agent.rl_agent import PPOAgent  # Your PPOAgent implementation
from data import DataHandler  # Data processing and indicator calculation
from rl_algo.environment.reward import RiskAdjustedReward
from rl_algo.environment.trading_env import TradingEnvironment
from execution.binance_futures_orders import BinanceFuturesExecutor
# Import MA algorithm components
from ma_algo.strategies.ma_crossover import MACrossoverStrategy
from ma_algo.config import MAConfig

# --- Helper functions for precision rounding --- #


def get_price_precision(executor, symbol):
    """Get the price precision for a symbol from exchange info."""
    try:
        exchange_info = executor.client.exchange_info()
        for symbol_info in exchange_info["symbols"]:
            if symbol_info["symbol"] == symbol:
                for filter_info in symbol_info["filters"]:
                    if filter_info["filterType"] == "PRICE_FILTER":
                        tick_size = float(filter_info["tickSize"])
                        return tick_size
        return 0.1  # Fallback precision
    except Exception as e:
        print(f"Error getting price precision: {e}")
        return 0.1


def get_quantity_precision(executor, symbol):
    """Get the quantity precision for a symbol from exchange info."""
    try:
        exchange_info = executor.client.exchange_info()
        for symbol_info in exchange_info["symbols"]:
            if symbol_info["symbol"] == symbol:
                for filter_info in symbol_info["filters"]:
                    if filter_info["filterType"] == "LOT_SIZE":
                        step_size = float(filter_info["stepSize"])
                        return step_size
        return 0.001  # Fallback precision
    except Exception as e:
        print(f"Error getting quantity precision: {e}")
        return 0.001


def round_to_tick_size(price, tick_size):
    """Round a price to the nearest valid tick size."""
    inverse = 1.0 / tick_size
    return math.floor(price * inverse) / inverse


def round_to_step_size(quantity, step_size):
    """Round a quantity to the nearest valid step size."""
    inverse = 1.0 / step_size
    return math.floor(quantity * inverse) / inverse


# --- Agent creation helper --- #


def create_agent(model_path=None, state_dim=None, action_dim=None):
    """
    Create and load a trained PPO agent.
    Args:
        model_path: Path to the checkpoint (or folder with saved models)
                   If None, will use the default 'saved_models' directory
        state_dim: State dimension from environment
        action_dim: Action dimension (typically 4)
    Returns:
        Loaded PPOAgent instance.
    """
    # Determine best device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Use saved_models directory by default
    if model_path is None:
        model_path = os.path.join("saved_models", "model")

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=(256, 128),
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        entropy_coef=0.01,
        value_coef=0.5,
        target_update_freq=10,
        target_update_tau=0.005,
        lr_scheduler_type="step",
        lr_scheduler_step_size=100,
        lr_scheduler_gamma=0.9,
        grad_clip_value=0.5,
        memory_capacity=10000,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta=0.4,
        prioritized_replay_beta_increment=0.001,
        device=device,
    )

    # Try to load saved models from checkpoint first
    if agent.load_models(model_path):
        print(f"Model loaded from {model_path}")
    else:
        # If checkpoint loading fails, try to load individual model files
        try:
            save_dir = "saved_models"
            agent.actor.load_state_dict(torch.load(os.path.join(save_dir, "actor_model.pt"), map_location=device))
            agent.critic.load_state_dict(torch.load(os.path.join(save_dir, "critic_model.pt"), map_location=device))
            agent.target_actor.load_state_dict(torch.load(os.path.join(save_dir, "target_actor_model.pt"), map_location=device))
            agent.target_critic.load_state_dict(torch.load(os.path.join(save_dir, "target_critic_model.pt"), map_location=device))
            print(f"Models loaded from individual files in {save_dir}")
        except Exception as e:
            print(f"Warning: Could not load models: {e}")
            print("Running with untrained agent.")

    return agent


# --- Binance Futures API initialization --- #


def init_binance_futures_api(use_testnet=True, leverage=5):
    """Initialize Binance Futures API connection."""
    try:
        executor = BinanceFuturesExecutor(
            use_testnet=use_testnet, trading_symbol="BTCUSDT", leverage=leverage
        )
        account_info = executor.get_account_info()
        available_balance = account_info.get("availableBalance", "N/A")
        print(
            f"Using Binance Futures {'testnet ' if use_testnet else ''}with {available_balance} USDT available"
        )
        return executor
    except Exception as e:
        raise ValueError(f"Failed to initialize Binance Futures API: {e}")


# --- Data fetching --- #


def fetch_binance_futures_data(executor, timeframe="1m", limit=500):
    """
    Fetch and process market data from Binance Futures.
    Returns a DataFrame with OHLCV data and calculated technical and risk metrics.
    """
    try:
        klines = executor.get_btc_klines(interval=timeframe, limit=limit)
        if not klines:
            print("Error: No klines data returned from Binance Futures")
            return None

        df = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df[["time", "open", "high", "low", "close", "volume"]]
        # Process metrics via DataHandler
        dh = DataHandler()
        df_metric = df.copy().set_index("time")
        df_metric = dh.calculate_technical_indicators(df_metric)
        df_metric = dh.calculate_risk_metrics(df_metric)
        df_metric = dh.identify_trade_setups(df_metric)
        df_metric = df_metric.ffill().bfill().fillna(0).reset_index()
        print("Market data processed successfully.")
        return df_metric
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return None


# --- Trade execution --- #


def execute_trade(
    executor, action_value, quantity=None, take_profit=None, stop_loss=None
):
    """
    Execute a trade on Binance Futures based on the agent's action.
    Args:
        executor: BinanceFuturesExecutor instance.
        action_value: The primary action value from the agent.
        quantity: Optional trade quantity.
        take_profit: Optional take profit percentage.
        stop_loss: Optional stop loss percentage.
    Returns:
        Order information dictionary.
    """
    # Get current position information
    position_info = executor.get_position_info()
    position_amt = float(position_info.get("positionAmt", 0))
    side = "BUY" if action_value > 0 else "SELL"

    if quantity is None:
        account_info = executor.get_account_info()
        available_balance = float(account_info.get("availableBalance", 0))
        ticker_info = executor.get_ticker_info()
        current_price = float(ticker_info.get("price", 0))
        risk_pct = min(abs(action_value) * 0.1, 0.05)
        trade_amount = available_balance * risk_pct
        quantity = trade_amount / current_price
        tick_size = get_quantity_precision(executor, "BTCUSDT")
        quantity = round_to_step_size(quantity, tick_size)

    if (position_amt > 0 and side == "SELL") or (position_amt < 0 and side == "BUY"):
        close_qty = abs(position_amt)
        print(f"Closing existing position of {close_qty} BTC")
        close_order = executor.execute_market_order(
            side=side, quantity=close_qty, close_position=True
        )
        if abs(action_value) < 0.1:
            return close_order
        quantity = quantity - close_qty
        if quantity <= 0:
            print("No remaining quantity for new position after closing.")
            return close_order

    print(f"Executing {side} order for {quantity:.3f} BTC")
    main_order = executor.execute_market_order(side=side, quantity=quantity)
    if main_order and "orderId" in main_order:
        print(f"Main order executed: {main_order['orderId']}")
        if take_profit is not None or stop_loss is not None:
            updated_pos = executor.get_position_info()
            pos_amt = float(updated_pos.get("positionAmt", 0))
            entry_price = float(updated_pos.get("entryPrice", 0))
            if abs(pos_amt) > 0:
                if side == "BUY":
                    tp_price = entry_price * (1 + take_profit) if take_profit else None
                    sl_price = entry_price * (1 - stop_loss) if stop_loss else None
                    exit_side = "SELL"
                else:
                    tp_price = entry_price * (1 - take_profit) if take_profit else None
                    sl_price = entry_price * (1 + stop_loss) if stop_loss else None
                    exit_side = "BUY"
                print(f"Placing OCO orders: TP at {tp_price:.2f}, SL at {sl_price:.2f}")
                sl_tp = executor.execute_stop_loss_take_profit_order(
                    side=exit_side,
                    quantity=abs(pos_amt),
                    stop_loss_price=sl_price,
                    take_profit_price=tp_price,
                    close_position=True,
                )
                if "orders" in sl_tp:
                    main_order["sl_tp_orders"] = sl_tp["orders"]
                    main_order["sl_client_order_id"] = sl_tp.get("sl_client_order_id")
                    main_order["tp_client_order_id"] = sl_tp.get("tp_client_order_id")
                    print("OCO orders placed successfully.")
        return main_order
    else:
        print("Failed to execute main order.")
        return None


# --- Main Inference Function --- #


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Inference for trading bot")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the saved model checkpoint (default: saved_models directory)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["rl", "ma"],
        default="rl",
        help="Trading algorithm to use: rl (Reinforcement Learning) or ma (Moving Average)",
    )
    parser.add_argument(
        "--binance_futures",
        action="store_true",
        help="Enable live trading with Binance Futures",
    )
    parser.add_argument(
        "--binance_testnet",
        action="store_true",
        default=True,
        help="Use Binance Futures testnet",
    )
    parser.add_argument(
        "--binance_leverage", type=int, default=2, help="Leverage for Binance Futures"
    )
    parser.add_argument(
        "--test_data", type=str, default=None, help="Path to test data file (parquet)"
    )
    parser.add_argument(
        "--trade_interval",
        type=int,
        default=3600,
        help="Seconds between trades for live trading",
    )
    # MA Strategy specific parameters
    parser.add_argument(
        "--short_window", type=int, default=7, help="Short-term MA window"
    )
    parser.add_argument(
        "--long_window", type=int, default=25, help="Long-term MA window"
    )
    parser.add_argument(
        "--risk_per_trade", type=float, default=0.01, help="Risk per trade as % of account"
    )
    parser.add_argument(
        "--atr_multiplier", type=float, default=2.0, help="ATR multiplier for stop loss"
    )
    args = parser.parse_args()

    # Initialize executor if live trading is enabled
    executor = None
    if args.binance_futures:
        executor = init_binance_futures_api(
            use_testnet=args.binance_testnet, leverage=args.binance_leverage
        )

    # Load or fetch market data for inference
    if args.test_data:
        data = pd.read_parquet(args.test_data)
        print(f"Loaded test data from {args.test_data} with {len(data)} rows")
    elif args.binance_futures:
        data = fetch_binance_futures_data(executor)
        if data is None or data.empty:
            print("Error: No market data available for inference")
            return
        print(f"Fetched live market data with {len(data)} rows")
    else:
        print("Error: Provide test_data path or enable binance_futures for live data")
        return

    # Execute trading using selected algorithm
    if args.algorithm == "rl":
        run_rl_inference(args, data, executor)
    else:
        run_ma_inference(args, data, executor)


def run_rl_inference(args, data, executor=None):
    """
    Run inference using the Reinforcement Learning agent.

    Args:
        args: Command line arguments
        data: Market data DataFrame
        executor: Optional BinanceFuturesExecutor instance for live trading
    """
    # Create Trading Environment
    env = TradingEnvironment(
        data=data,
        reward_function=RiskAdjustedReward(),
        initial_balance=10000.0,
        commission=0.001,
        slippage=0.0005,
        max_leverage=args.binance_leverage,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create and load the trained agent
    agent = create_agent(args.model_path, state_dim, action_dim)

    observation, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    # Trading/inference loop (for live mode, wait between trades)
    try:
        while not done and env.current_step < len(data) - 1:
            action, _, _ = agent.choose_action(observation)
            # Scale action outputs if needed (for example, for take profit and stop loss)
            take_profit = None
            stop_loss = None
            if len(action) >= 4:
                raw_tp = action[2]
                raw_sl = action[3]
                # Map raw outputs to realistic percentages (max 5% either way)
                # Normalize raw values from [-1,1] to [0.5%, 5%] range
                take_profit = 0.005 + (raw_tp + 1) * 0.0225  # 0.5% to 5%
                stop_loss = 0.005 + (raw_sl + 1) * 0.0225    # 0.5% to 5%
                print(
                    f"Raw TP: {raw_tp:.4f}, Raw SL: {raw_sl:.4f} -> Scaled TP: {take_profit*100:.2f}%, SL: {stop_loss*100:.2f}%"
                )
            if args.binance_futures and executor is not None:
                # Execute trade via Binance Futures
                order = execute_trade(executor, action[0], None, take_profit, stop_loss)
                if order:
                    print(f"Order details: {json.dumps(order, indent=2)}")
            else:
                # For simulation, simply step the environment
                pass

            next_observation, reward, done, _, info = env.step(action)
            total_reward += reward
            step += 1
            print(
                f"Step: {step}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Balance: {env.balance:.2f}"
            )
            observation = next_observation

            # For live trading, wait for the next trade interval
            if args.binance_futures and not done:
                print(f"Waiting {args.trade_interval} seconds for next trade...")
                time.sleep(args.trade_interval)

    except KeyboardInterrupt:
        print("Inference stopped by user.")

    print(f"RL inference completed after {step} steps. Final balance: {env.balance:.2f}")

    if args.binance_futures and executor is not None:
        # Optionally, print final account info
        account_info = executor.get_account_info()
        print("Final Binance Futures account info:")
        print(json.dumps(account_info, indent=2))


def run_ma_inference(args, data, executor=None):
    """
    Run inference using the Moving Average crossover algorithm.

    Args:
        args: Command line arguments
        data: Market data DataFrame
        executor: Optional BinanceFuturesExecutor instance for live trading
    """
    # Configure MA strategy
    ma_config = MAConfig(
        short_window=args.short_window,
        long_window=args.long_window,
        risk_per_trade=args.risk_per_trade,
        atr_multiplier=args.atr_multiplier,
        max_leverage=args.binance_leverage,
    )

    # Create MA strategy instance
    strategy = MACrossoverStrategy(
        short_window=ma_config.short_window,
        long_window=ma_config.long_window,
        risk_per_trade=ma_config.risk_per_trade,
        atr_multiplier=ma_config.atr_multiplier
    )
    
    # Debug: Print data columns and first few rows
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Data shape: {data.shape}")
    print("First few rows of data:")
    print(data.head(3))
    
    # Ensure data has the right format for MA strategy
    if 'time' in data.columns:
        # Set time as index if it's not already
        if data.index.name != 'time':
            data = data.set_index('time')
    
    # Calculate features
    df = strategy.calculate_features(data)
    
    # Debug: Print calculated features
    print(f"Features calculated. Columns: {df.columns.tolist()}")
    print(f"MA columns present: {'ma_short' in df.columns and 'ma_long' in df.columns}")
    print(f"Non-NaN values in ma_short: {df['ma_short'].count()} out of {len(df)}")
    print(f"Non-NaN values in ma_long: {df['ma_long'].count()} out of {len(df)}")
    
    # Debug: Print a few rows with MA values
    print("Sample rows with MA values:")
    sample_rows = df[['close', 'ma_short', 'ma_long']].dropna().head(3)
    print(sample_rows)

    # Initial account info
    account_balance = 10000.0
    if args.binance_futures and executor is not None:
        account_info = executor.get_account_info()
        account_balance = float(account_info.get("availableBalance", 10000.0))
    
    # Get quantity precision for proper position sizing
    step_size = 0.001  # Default fallback
    if executor is not None:
        step_size = get_quantity_precision(executor, "BTCUSDT")
        print(f"Using quantity step size: {step_size}")
    
    # Add step_size to each row for proper position sizing
    df['step_size'] = step_size

    # Trading loop
    step = 0
    signals_generated = 0
    try:
        for i, row in df.iterrows():
            if pd.isna(row['ma_short']) or pd.isna(row['ma_long']):
                continue

            # Generate trading signal
            direction, quantity = strategy.generate_signal(row, account_balance)
            
            # Limit position size to avoid margin insufficient errors
            # For testnet, use a much smaller position size
            if executor is not None:
                # Get current BTC price
                btc_price = float(executor.get_ticker_info().get("price", 0))
                if btc_price > 0:
                    # Calculate max affordable position based on available balance and leverage
                    max_position = (account_balance * 0.95 * args.binance_leverage) / btc_price
                    # Limit to 1% of max position for safety in testnet
                    safe_max = max_position * 0.01
                    if quantity > safe_max:
                        print(f"Limiting position size from {quantity:.4f} to {safe_max:.4f} BTC")
                        quantity = round_to_step_size(safe_max, step_size)
            
            # Debug: Print signal info for each row
            if step % 10 == 0:  # Print every 10th row to avoid too much output
                print(f"Row {step}: MA short={row['ma_short']:.2f}, MA long={row['ma_long']:.2f}, Direction={direction}, Quantity={quantity:.4f}")

            if direction != 0 and quantity > 0:
                signals_generated += 1
                print(f"Signal generated: Direction={direction}, Quantity={quantity:.4f}")

                if args.binance_futures and executor is not None:
                    # Execute trade via Binance Futures
                    action_value = 1.0 if direction > 0 else -1.0
                    order = execute_trade(executor, action_value, quantity)
                    if order:
                        print(f"Order details: {json.dumps(order, indent=2)}")
                        # Update account balance
                        account_info = executor.get_account_info()
                        account_balance = float(account_info.get("availableBalance", account_balance))

            step += 1

            # For live trading, wait for the next trade interval
            if args.binance_futures and executor is not None:
                print(f"Waiting {args.trade_interval} seconds for next MA check...")
                time.sleep(args.trade_interval)

                # Fetch new data for the next iteration
                new_data = fetch_binance_futures_data(executor)
                if new_data is not None and not new_data.empty:
                    # Ensure data has the right format for MA strategy
                    if 'time' in new_data.columns:
                        if new_data.index.name != 'time':
                            new_data = new_data.set_index('time')
                    
                    # Calculate features on new data
                    df = strategy.calculate_features(new_data)
                    
                    # Add step_size to each row
                    df['step_size'] = step_size
                    
                    print(f"Updated data with {len(df)} rows. MA columns present: {'ma_short' in df.columns and 'ma_long' in df.columns}")
                    print(f"Non-NaN values - ma_short: {df['ma_short'].count()}, ma_long: {df['ma_long'].count()}")

    except KeyboardInterrupt:
        print("MA inference stopped by user.")

    print(f"MA inference completed after {step} steps. Total signals generated: {signals_generated}. Final balance: {account_balance:.2f}")

    if args.binance_futures and executor is not None:
        # Optionally, print final account info
        account_info = executor.get_account_info()
        print("Final Binance Futures account info:")
        print(json.dumps(account_info, indent=2))


if __name__ == "__main__":
    main()
