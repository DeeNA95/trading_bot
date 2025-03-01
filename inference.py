"""
Inference script for the reinforcement learning trading bot.
Uses trained models to make trading decisions through Alpaca's paper trading API.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import time
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from dotenv import load_dotenv

from agent.rl_agent import PPOAgent
from environment.trading_env import TradingEnvironment
from environment.reward import RiskAdjustedReward
from train import get_best_device, prepare_data
from data import DataHandler  # Import DataHandler for calculating metrics

# Load environment variables
load_dotenv()

# Initialize Alpaca API
def init_alpaca_api():
    """Initialize Alpaca API connection."""
    api_key = os.getenv("ALPACA_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    base_url = 'https://paper-api.alpaca.markets'  # Paper trading URL

    if not api_key or not api_secret:
        raise ValueError("Alpaca API key and secret must be set in .env file")

    return tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

# Fetch latest data for a symbol from Alpaca
def fetch_market_data(api, symbol, timeframe='1D', limit=100):
    """Fetch market data from Alpaca API.

    Args:
        api: Alpaca API instance
        symbol: Trading symbol (e.g., 'BTC/USD')
        timeframe: Timeframe for bars ('1D', '1H', etc.)
        limit: Number of bars to fetch
        
    Returns:
        DataFrame with OHLCV data and calculated metrics
    """
    # Convert crypto symbol format if needed (Alpaca now requires symbol in 'BTC/USD' format)
    if '/' not in symbol:
        # If symbol doesn't have a slash, add it (e.g., 'BTCUSD' -> 'BTC/USD')
        if len(symbol) > 3:
            # Assuming format like BTCUSD needs to be BTC/USD
            base = symbol[:-3]
            quote = symbol[-3:]
            alpaca_symbol = f"{base}/{quote}"
        else:
            # If we can't parse it, keep as is
            alpaca_symbol = symbol
    else:
        # Already in correct format
        alpaca_symbol = symbol
    
    print(f"Using symbol format for Alpaca API: {alpaca_symbol}")
    
    # Calculate time window
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=limit)
    
    # Format dates in YYYY-MM-DD format that Alpaca accepts
    start_str = start_dt.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')
    
    try:
        # Fetch bars from Alpaca
        bars = api.get_crypto_bars(
            alpaca_symbol,
            timeframe,
            start=start_str,
            end=end_str
        ).df
        
        # Format the data to match our expected format
        bars = bars.reset_index()
        
        # Ensure column names match what our model expects
        bars.rename(columns={
            'timestamp': 'time',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }, inplace=True)
        
        # Ensure the dataframe has all the required columns
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in bars.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Initialize DataHandler and calculate all metrics
        dh = DataHandler()
        
        # Convert to the format expected by DataHandler
        df_for_metrics = bars.copy()
        if 'time' in df_for_metrics.columns:
            df_for_metrics = df_for_metrics.set_index('time')
        
        # Calculate technical indicators
        df_with_indicators = dh.calculate_technical_indicators(df_for_metrics)
        
        # Calculate risk metrics
        df_with_risk = dh.calculate_risk_metrics(df_with_indicators)
        
        # Identify trade setups
        df_with_setups = dh.identify_trade_setups(df_with_risk)
        
        # Fill NaN values
        df_final = df_with_setups.fillna(method='ffill')
        df_final = df_final.fillna(method='bfill')
        
        # Any remaining NaNs should be filled with zeros
        df_final = df_final.fillna(0)
        
        # Reset index to have time as a column
        if df_final.index.name == 'timestamp':
            df_final = df_final.reset_index()
        
        print(f"Data processed successfully with all required metrics added")
        return df_final
        
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return None

# Create trading agent with trained model
def create_agent(model_path, state_dim, action_dim):
    """Create and load a trained PPO agent.

    Args:
        model_path: Path to saved model
        state_dim: State dimension
        action_dim: Action dimension

    Returns:
        Loaded PPO agent
    """
    device = get_best_device()

    # Create agent with the same architecture as during training
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
        device=device
    )

    # Load trained models
    agent.load_models(model_path)

    return agent

# Execute trade on Alpaca
def execute_trade(api, symbol, action, quantity):
    """Execute a trade using Alpaca API.

    Args:
        api: Alpaca API instance
        symbol: Trading symbol
        action: Action value from agent (normalized between -1 and 1)
        quantity: Quantity to trade

    Returns:
        Order information
    """
    # Convert crypto symbol format if needed (Alpaca now requires symbol in 'BTC/USD' format)
    if '/' not in symbol:
        # If symbol doesn't have a slash, add it (e.g., 'BTCUSD' -> 'BTC/USD')
        if len(symbol) > 3:
            # Assuming format like BTCUSD needs to be BTC/USD
            base = symbol[:-3]
            quote = symbol[-3:]
            alpaca_symbol = f"{base}/{quote}"
        else:
            # If we can't parse it, keep as is
            alpaca_symbol = symbol
    else:
        # Already in correct format
        alpaca_symbol = symbol

    try:
        # Use a fixed small quantity to stay under Alpaca's limits
        # Instead of trying to calculate based on price which requires additional API calls
        # BTC is around $50,000-60,000, so 0.001 BTC is about $50-60
        safe_quantity = 0.001
        
        # Determine if we should buy or sell
        side = 'buy' if action > 0 else 'sell'
        
        print(f"Executing {side} order for {safe_quantity} {alpaca_symbol}")
        
        # Submit order
        order = api.submit_order(
            symbol=alpaca_symbol,
            qty=safe_quantity,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        
        return order

    except Exception as e:
        print(f"Error executing trade: {e}")
        return None

# Main inference function
def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run trading bot inference')
    parser.add_argument('--model_path', type=str, default='models/best',
                        help='Path to trained model')
    parser.add_argument('--symbol', type=str, default='BTC/USD',
                        help='Trading symbol')
    parser.add_argument('--initial_balance', type=float, default=10000,
                        help='Initial balance for simulation')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate')
    parser.add_argument('--max_leverage', type=float, default=3.0,
                        help='Maximum leverage')
    parser.add_argument('--trade_interval', type=int, default=3600,
                        help='Seconds between trades')
    parser.add_argument('--paper_trading', action='store_true',
                        help='Enable paper trading with Alpaca')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data (if not using live data)')

    args = parser.parse_args()

    # Initialize Alpaca API for paper trading
    if args.paper_trading:
        api = init_alpaca_api()
        account = api.get_account()
        print(f"Trading with Alpaca paper account: ${float(account.cash)} available")

    # Load data for inference
    if args.test_data:
        # Use test data file
        data = prepare_data(args.test_data)
        print(f"Using test data from {args.test_data}, {len(data)} rows")
    elif args.paper_trading:
        # Fetch live data from Alpaca
        data = fetch_market_data(api, args.symbol)
        if data is None or len(data) == 0:
            print("Error: No data available for inference")
            return
        print(f"Using live data from Alpaca, {len(data)} rows")
    else:
        print("Error: Either test_data or paper_trading must be specified")
        return

    # Create environment
    env = TradingEnvironment(
        data=data,
        reward_function=RiskAdjustedReward(),
        initial_balance=args.initial_balance,
        commission=args.commission,
        max_leverage=args.max_leverage
    )

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create and load agent
    agent = create_agent(args.model_path, state_dim, action_dim)
    print(f"Model loaded from {args.model_path}")

    # Trading loop
    observation, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    position = 0  # Current position

    try:
        while not done:
            # Get action from agent
            action, _, _ = agent.choose_action(observation)
            action_value = action[0]  # Extract scalar value

            # Execute trade in paper trading mode
            if args.paper_trading:
                # Only execute trade if action is significant
                if abs(action_value) > 0.1:
                    execute_trade(api, args.symbol, action_value, None)
                else:
                    print(f"Action value {action_value} too small, skipping trade")

            # Update environment
            next_observation, reward, done, truncated, info = env.step(action)

            # Update tracking variables
            observation = next_observation
            total_reward += reward
            step += 1

            # Print current status
            print(f"Step: {step}, Action: {action_value:.4f}, " +
                  f"Reward: {reward:.2f}, Total: {total_reward:.2f}, " +
                  f"Balance: ${env.balance:.2f}")

            # In paper trading mode, wait for next interval
            if args.paper_trading and not done:
                print(f"Waiting {args.trade_interval} seconds until next trade...")
                time.sleep(args.trade_interval)

            # End if we're done or truncated
            if done or truncated:
                break

    except KeyboardInterrupt:
        print("Trading stopped by user")

    # Print final results
    print(f"\nTrading completed after {step} steps")
    print(f"Final balance: ${env.balance:.2f}")
    print(f"Total reward: {total_reward:.2f}")

    # In paper trading mode, show final account status
    if args.paper_trading:
        try:
            account = api.get_account()
            positions = api.list_positions()

            print("\nFinal Account Status:")
            print(f"Cash: ${float(account.cash):.2f}")
            print(f"Portfolio Value: ${float(account.portfolio_value):.2f}")

            print("\nOpen Positions:")
            for position in positions:
                print(f"{position.symbol}: {position.qty} @ ${float(position.avg_entry_price):.2f} " +
                      f"(Current: ${float(position.current_price):.2f}, " +
                      f"P&L: ${float(position.unrealized_pl):.2f})")
        except Exception as e:
            print(f"Error retrieving account status: {e}")


if __name__ == "__main__":
    main()
