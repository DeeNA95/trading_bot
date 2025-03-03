"""
Inference script for the reinforcement learning trading bot.
Uses trained models to make trading decisions through Alpaca's paper trading API
or Binance Futures API.
"""

import argparse
import os
import time
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

# Try to import Alpaca API, but make it optional
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: alpaca_trade_api module not found. Alpaca paper trading will not be available.")

from agent.rl_agent import PPOAgent
from data import DataHandler  # Import DataHandler for calculating metrics
from environment.reward import RiskAdjustedReward
from environment.trading_env import TradingEnvironment
from execution.binance_futures_orders import BinanceFuturesExecutor
from train import get_best_device, prepare_data

# Load environment variables
load_dotenv()


# Helper functions for Binance Futures trading
def get_price_precision(executor, symbol):
    """Get the price precision for a symbol from exchange info"""
    try:
        exchange_info = executor.client.exchange_info()
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                for filter_info in symbol_info['filters']:
                    if filter_info['filterType'] == 'PRICE_FILTER':
                        tick_size = float(filter_info['tickSize'])
                        return tick_size
        return 0.1  # Default fallback
    except Exception as e:
        print(f"Error getting price precision: {e}")
        return 0.1  # Default fallback

def get_quantity_precision(executor, symbol):
    """Get the quantity precision for a symbol from exchange info"""
    try:
        exchange_info = executor.client.exchange_info()
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                for filter_info in symbol_info['filters']:
                    if filter_info['filterType'] == 'LOT_SIZE':
                        step_size = float(filter_info['stepSize'])
                        return step_size
        return 0.001  # Default fallback
    except Exception as e:
        print(f"Error getting quantity precision: {e}")
        return 0.001  # Default fallback

def round_to_tick_size(price, tick_size):
    """Round a price to the nearest valid tick size"""
    inverse = 1.0 / tick_size
    return math.floor(price * inverse) / inverse

def round_to_step_size(quantity, step_size):
    """Round a quantity to the nearest valid step size"""
    inverse = 1.0 / step_size
    return math.floor(quantity * inverse) / inverse

# Initialize Binance Futures API
def init_binance_futures_api(use_testnet=True, leverage=5):
    """Initialize Binance Futures API connection."""
    try:
        executor = BinanceFuturesExecutor(
            use_testnet=use_testnet,
            trading_symbol="BTCUSDT",
            leverage=leverage
        )

        # Get account information
        account_info = executor.get_account_info()
        available_balance = account_info.get('availableBalance', 'N/A')
        print(f"Trading with Binance Futures {'testnet ' if use_testnet else ''}account: {available_balance} USDT available")

        return executor
    except Exception as e:
        raise ValueError(f"Failed to initialize Binance Futures API: {e}")


# Initialize Alpaca API
def init_alpaca_api():
    """Initialize Alpaca API connection."""
    api_key = os.getenv("ALPACA_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    base_url = "https://paper-api.alpaca.markets"  # Paper trading URL

    if not api_key or not api_secret:
        raise ValueError("Alpaca API key and secret must be set in .env file")

    return tradeapi.REST(api_key, api_secret, base_url, api_version="v2")


# Fetch latest data for a symbol from Binance Futures
def fetch_binance_futures_data(executor, timeframe="1h", limit=100):
    """Fetch market data from Binance Futures API.

    Args:
        executor: BinanceFuturesExecutor instance
        timeframe: Timeframe for klines ('1m', '3m', '5m', '15m', '30m', '1h', etc.)
        limit: Number of klines to fetch

    Returns:
        DataFrame with OHLCV data and calculated metrics
    """
    try:
        # Fetch klines from Binance Futures
        klines = executor.get_btc_klines(interval=timeframe, limit=limit)

        if not klines:
            print("Error: No klines data returned from Binance Futures")
            return None

        # Convert klines to DataFrame
        # Binance kline format: [Open time, Open, High, Low, Close, Volume, Close time, ...]
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Convert string values to float for price and volume data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['open_time'], unit='ms')

        # Keep only necessary columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

        # Initialize DataHandler and calculate all metrics
        dh = DataHandler()

        # Convert to the format expected by DataHandler
        df_for_metrics = df.copy()
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
        df_final = df_final.reset_index()

        print(f"Data processed successfully with all required metrics added")
        return df_final

    except Exception as e:
        print(f"Error fetching market data from Binance Futures: {e}")
        return None


# Fetch latest data for a symbol from Alpaca
def fetch_market_data(api, symbol, timeframe="1H", limit=100):
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
    if "/" not in symbol:
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
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    try:
        # Fetch bars from Alpaca
        bars = api.get_crypto_bars(
            alpaca_symbol, timeframe, start=start_str, end=end_str
        ).df

        # Format the data to match our expected format
        bars = bars.reset_index()

        # Ensure column names match what our model expects
        bars.rename(
            columns={
                "timestamp": "time",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
            inplace=True,
        )

        # Ensure the dataframe has all the required columns
        required_columns = ["time", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in bars.columns:
                raise ValueError(f"Missing required column: {col}")

        # Initialize DataHandler and calculate all metrics
        dh = DataHandler()

        # Convert to the format expected by DataHandler
        df_for_metrics = bars.copy()
        if "time" in df_for_metrics.columns:
            df_for_metrics = df_for_metrics.set_index("time")

        # Calculate technical indicators
        df_with_indicators = dh.calculate_technical_indicators(df_for_metrics)

        # Calculate risk metrics
        df_with_risk = dh.calculate_risk_metrics(df_with_indicators)

        # Identify trade setups
        df_with_setups = dh.identify_trade_setups(df_with_risk)

        # Fill NaN values
        df_final = df_with_setups.fillna(method="ffill")
        df_final = df_final.fillna(method="bfill")

        # Any remaining NaNs should be filled with zeros
        df_final = df_final.fillna(0)

        # Reset index to have time as a column
        if df_final.index.name == "timestamp":
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
        device=device,
    )

    # Load trained models
    agent.load_models(model_path)

    return agent


# Execute trade on Binance Futures
def execute_binance_futures_trade(executor, action_value, quantity=None, take_profit=None, stop_loss=None):
    """Execute a trade using Binance Futures API.

    Args:
        executor: BinanceFuturesExecutor instance
        action_value: Action value from agent (normalized between -1 and 1)
        quantity: Quantity to trade in BTC (if None, will calculate based on account balance)
        take_profit: Take profit percentage multiplier (relative to entry price)
        stop_loss: Stop loss percentage multiplier (relative to entry price)

    Returns:
        Dictionary containing order information
    """
    try:
        # Determine if we should buy or sell
        side = "BUY" if action_value > 0 else "SELL"

        # Get current BTC price
        btc_price = executor.get_btc_price()
        if btc_price == 0:
            print("Error: Could not get current BTC price")
            return None

        # Get price and quantity precision
        tick_size = get_price_precision(executor, executor.trading_symbol)
        step_size = get_quantity_precision(executor, executor.trading_symbol)

        # If quantity not provided, calculate based on account balance
        if quantity is None:
            # Get account information
            account_info = executor.get_account_info()
            available_balance = float(account_info.get('availableBalance', 0))

            # Use 2% of available balance for each trade
            trade_value = available_balance * 0.02

            # Calculate quantity in BTC
            raw_quantity = trade_value / btc_price

            # Round to valid step size
            quantity = round_to_step_size(raw_quantity, step_size)

            # Ensure minimum quantity (0.001 BTC is about $50-60)
            min_quantity = 0.001
            if quantity < min_quantity:
                quantity = min_quantity

        print(f"Executing {side} order for {quantity} BTC at ~${btc_price:.2f}")

        # If we have stop loss or take profit values, use the combined method
        if take_profit is not None or stop_loss is not None:
            # Execute market order first
            main_order = executor.execute_market_order(side=side, quantity=quantity)
            
            if "orderId" in main_order:
                # Wait a brief moment for the order to process
                time.sleep(1)
                
                # Get current position
                position = executor.get_btc_position()
                position_amt = float(position.get("positionAmt", 0))
                entry_price = float(position.get("entryPrice", btc_price))
                
                # Only proceed if we have a position
                if abs(position_amt) > 0:
                    # Calculate exit side (opposite of entry)
                    exit_side = "SELL" if side == "BUY" else "BUY"
                    
                    # Calculate stop loss and take profit prices
                    stop_loss_price = None
                    take_profit_price = None
                    
                    if stop_loss is not None:
                        if side == "BUY":
                            # For long positions, stop loss is below entry price
                            stop_loss_price = entry_price * (1 - stop_loss)
                        else:
                            # For short positions, stop loss is above entry price
                            stop_loss_price = entry_price * (1 + stop_loss)
                        # Round to valid tick size
                        stop_loss_price = round_to_tick_size(stop_loss_price, tick_size)
                        print(f"Setting stop loss at ${stop_loss_price:.2f}")
                    
                    if take_profit is not None:
                        if side == "BUY":
                            # For long positions, take profit is above entry price
                            take_profit_price = entry_price * (1 + take_profit)
                        else:
                            # For short positions, take profit is below entry price
                            take_profit_price = entry_price * (1 - take_profit)
                        # Round to valid tick size
                        take_profit_price = round_to_tick_size(take_profit_price, tick_size)
                        print(f"Setting take profit at ${take_profit_price:.2f}")
                    
                    # Execute combined stop loss and take profit order
                    sl_tp_orders = executor.execute_stop_loss_take_profit_order(
                        side=exit_side,
                        quantity=abs(position_amt),
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                        close_position=True  # Close the entire position
                    )
                    
                    if "orders" in sl_tp_orders:
                        print(f"Stop loss and take profit orders placed successfully")
                        # Add the SL/TP orders to the main order response
                        main_order["sl_tp_orders"] = sl_tp_orders["orders"]
                    else:
                        print(f"Failed to place stop loss and take profit orders: {sl_tp_orders}")
            
            return main_order
        else:
            # Just execute a market order without SL/TP
            return executor.execute_market_order(side=side, quantity=quantity)

    except Exception as e:
        print(f"Error executing Binance Futures trade: {e}")
        return None


# Execute trade on Alpaca
def execute_trade(api, symbol, action, quantity, take_profit=None, stop_loss=None):
    """Execute a trade using Alpaca API.

    Args:
        api: Alpaca API instance
        symbol: Trading symbol
        action: Action value from agent (normalized between -1 and 1)
        quantity: Quantity to trade (not used currently)
        take_profit: Take profit percentage multiplier (relative to entry price)
        stop_loss: Stop loss percentage multiplier (relative to entry price)

    Returns:
        Order information
    """
    # Convert crypto symbol format if needed (Alpaca now requires symbol in 'BTC/USD' format)
    if "/" not in symbol:
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
        side = "buy" if action > 0 else "sell"

        try:
            # Get current price to calculate stop loss and take profit prices
            latest_quote = api.get_latest_crypto_quotes(alpaca_symbol)
            if alpaca_symbol in latest_quote:
                bid_price = latest_quote[alpaca_symbol].bp
                ask_price = latest_quote[alpaca_symbol].ap
                current_price = (bid_price + ask_price) / 2

                print(
                    f"Executing {side} order for {safe_quantity} {alpaca_symbol} at ~${current_price:.2f}"
                )

                # Log the intended take profit and stop loss values
                if take_profit is not None and stop_loss is not None:
                    if side == "buy":
                        take_profit_price = current_price * (1 + take_profit)
                        stop_loss_price = current_price * (1 - stop_loss)
                    else:
                        take_profit_price = current_price * (1 - take_profit)
                        stop_loss_price = current_price * (1 + stop_loss)

                    print(f"Target take profit price: ${take_profit_price:.2f}")
                    print(f"Target stop loss price: ${stop_loss_price:.2f}")
                    print(
                        f"NOTE: For crypto, complex orders not supported. Monitor manually."
                    )
            else:
                print(
                    f"Could not get quotes for {alpaca_symbol}, submitting market order"
                )

            # For crypto, only implement the stop loss using a stop limit order
            if stop_loss is not None:
                # Use the stop_loss parameter for risk management
                if side == "buy":
                    # For buy orders, use stop_loss to exit at a loss if price falls
                    stop_price = current_price * (1 - stop_loss)
                    limit_price = stop_price * 0.99  # Set limit slightly below stop
                    exit_side = "sell"
                else:
                    # For sell orders, use stop_loss to exit at a loss if price rises
                    stop_price = current_price * (1 + stop_loss)
                    limit_price = stop_price * 1.01  # Set limit slightly above stop
                    exit_side = "buy"

                # Round prices to 2 decimal places and convert to strings
                stop_price_str = f"{stop_price:.2f}"
                limit_price_str = f"{limit_price:.2f}"

                print(
                    f"Setting stop limit order at stop=${stop_price_str}, limit=${limit_price_str}"
                )

            # Submit the main market order
            main_order = api.submit_order(
                symbol=alpaca_symbol,
                qty=safe_quantity,
                side=side,
                type="market",
                time_in_force="gtc",
            )

            # Wait a brief moment for the main order to process
            import time

            time.sleep(1)

            # Submit the stop limit order if we have stop_loss defined
            if stop_loss is not None:
                try:
                    stop_order = api.submit_order(
                        symbol=alpaca_symbol,
                        qty=safe_quantity,
                        side=exit_side,
                        type="stop_limit",
                        time_in_force="gtc",
                        stop_price=stop_price_str,
                        limit_price=limit_price_str,
                    )
                    print(f"Stop limit order submitted: {stop_order.id}")
                except Exception as e:
                    print(f"Error submitting stop limit order: {e}")

            return main_order

        except Exception as e:
            print(f"Error submitting order: {e}")
            return None

    except Exception as e:
        print(f"Error executing trade: {e}")
        return None


# Main inference function
def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run trading bot inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/parallel_experiment/best_checkpoint.pt",
        help="Path to trained model"
    )
    parser.add_argument("--symbol", type=str, default="BTC/USD", help="Trading symbol")
    parser.add_argument(
        "--initial_balance",
        type=float,
        default=10000,
        help="Initial balance for simulation",
    )
    parser.add_argument(
        "--commission", type=float, default=0.001, help="Commission rate"
    )
    parser.add_argument(
        "--max_leverage", type=float, default=3.0, help="Maximum leverage"
    )
    parser.add_argument(
        "--trade_interval", type=int, default=3600, help="Seconds between trades"
    )
    parser.add_argument(
        "--paper_trading", action="store_true", help="Enable paper trading with Alpaca"
    )
    parser.add_argument(
        "--binance_futures", action="store_true", help="Enable practice trading with Binance Futures"
    )
    parser.add_argument(
        "--binance_testnet", action="store_true", default=True, help="Use Binance Futures testnet (default: True)"
    )
    parser.add_argument(
        "--binance_leverage", type=int, default=5, help="Leverage for Binance Futures (default: 5)"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test data (if not using live data)",
    )

    args = parser.parse_args()

    # Initialize API clients
    api = None
    binance_executor = None

    # Initialize Alpaca API for paper trading
    if args.paper_trading:
        api = init_alpaca_api()
        account = api.get_account()
        print(f"Trading with Alpaca paper account: ${float(account.cash)} available")

    # Initialize Binance Futures API
    if args.binance_futures:
        binance_executor = init_binance_futures_api(
            use_testnet=args.binance_testnet,
            leverage=args.binance_leverage
        )

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
    elif args.binance_futures:
        # Fetch live data from Binance Futures
        data = fetch_binance_futures_data(binance_executor)
        if data is None or len(data) == 0:
            print("Error: No data available for inference")
            return
        print(f"Using live data from Binance Futures, {len(data)} rows")
    else:
        print("Error: Either test_data, paper_trading, or binance_futures must be specified")
        return

    # Create environment
    env = TradingEnvironment(
        data=data,
        reward_function=RiskAdjustedReward(),
        initial_balance=args.initial_balance,
        commission=args.commission,
        max_leverage=args.max_leverage,
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

            # Print stop loss and take profit if available in the action
            take_profit = None
            stop_loss = None
            if len(action) >= 4:
                # Scale down the raw model outputs to reasonable percentages
                # Convert from [0.5, 5.0] range to [0.01, 0.05] range (1% to 5%)
                raw_take_profit = action[2]
                raw_stop_loss = action[3]

                # Scale to more reasonable percentages (1-5%)
                take_profit = 0.01 + (raw_take_profit - 0.5) * 0.04 / 4.5
                stop_loss = 0.01 + (raw_stop_loss - 0.5) * 0.04 / 4.5

                print(
                    f"Raw model values - Take Profit: {raw_take_profit:.4f}, Stop Loss: {raw_stop_loss:.4f}"
                )
                print(
                    f"Scaled values - Take Profit: {take_profit:.4f} ({take_profit*100:.2f}%), Stop Loss: {stop_loss:.4f} ({stop_loss*100:.2f}%)"
                )

            # Execute trade based on the selected mode
            if args.paper_trading or args.binance_futures:
                # Only execute trade if action is significant
                if abs(action_value) > 0.1:
                    if args.paper_trading:
                        execute_trade(
                            api, args.symbol, action_value, None, take_profit, stop_loss
                        )
                    elif args.binance_futures:
                        execute_binance_futures_trade(
                            binance_executor, action_value, None, take_profit, stop_loss
                        )
                else:
                    print(f"Action value {action_value} too small, skipping trade")

            # Update environment
            next_observation, reward, done, truncated, info = env.step(action)

            # Update tracking variables
            observation = next_observation
            total_reward += reward
            step += 1

            # Print current status
            status_msg = f"Step: {step}, Action: {action_value:.4f}, "
            status_msg += f"Reward: {reward:.2f}, Total: {total_reward:.2f}, "
            status_msg += f"Balance: ${env.balance:.2f}"

            # Add position info
            status_msg += (
                f", Position: {env.position_size:.4f} @ ${env.position_price:.2f}"
            )

            # Add unrealized PnL if in a position
            if env.position_size != 0:
                unrealized_pnl = env._calculate_unrealized_pnl()
                status_msg += f", Unrealized PnL: ${unrealized_pnl:.2f}"

            print(status_msg)

            # Wait for next interval in live trading modes
            if (args.paper_trading or args.binance_futures) and not done:
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

    # Show final account status for live trading modes
    if args.paper_trading:
        try:
            account = api.get_account()
            positions = api.list_positions()

            print("\nFinal Alpaca Account Status:")
            print(f"Cash: ${float(account.cash):.2f}")
            print(f"Portfolio Value: ${float(account.portfolio_value):.2f}")

            print("\nOpen Positions:")
            for position in positions:
                print(
                    f"{position.symbol}: {position.qty} @ ${float(position.avg_entry_price):.2f} "
                    + f"(Current: ${float(position.current_price):.2f}, "
                    + f"P&L: ${float(position.unrealized_pl):.2f})"
                )
        except Exception as e:
            print(f"Error retrieving Alpaca account status: {e}")

    elif args.binance_futures:
        try:
            # Get account information
            account_info = binance_executor.get_account_info()

            print("\nFinal Binance Futures Account Status:")
            print(f"Available Balance: {account_info.get('availableBalance', 'N/A')} USDT")
            print(f"Total Wallet Balance: {account_info.get('totalWalletBalance', 'N/A')} USDT")

            # Get current BTC position
            position = binance_executor.get_btc_position()

            print("\nCurrent BTC Position:")
            position_amt = float(position.get('positionAmt', 0))
            if position_amt != 0:
                entry_price = float(position.get('entryPrice', 0))
                unrealized_profit = float(position.get('unRealizedProfit', 0))
                leverage = position.get('leverage', '1')

                print(f"Position Size: {position_amt} BTC")
                print(f"Entry Price: ${entry_price:.2f}")
                print(f"Unrealized P&L: ${unrealized_profit:.2f}")
                print(f"Leverage: {leverage}x")

                # Get current price to calculate current value
                current_price = binance_executor.get_btc_price()
                if current_price > 0:
                    print(f"Current Price: ${current_price:.2f}")
                    print(f"Current Position Value: ${abs(position_amt) * current_price:.2f}")
            else:
                print("No open BTC position")

            # Get open orders
            open_orders = binance_executor.get_open_orders()

            print("\nOpen Orders:")
            if open_orders:
                for order in open_orders:
                    print(f"Order ID: {order.get('orderId')}")
                    print(f"Type: {order.get('type')}")
                    print(f"Side: {order.get('side')}")
                    print(f"Price: ${float(order.get('price', 0)):.2f}")
                    print(f"Quantity: {order.get('origQty')}")
                    print("---")
            else:
                print("No open orders")

        except Exception as e:
            print(f"Error retrieving Binance Futures account status: {e}")


if __name__ == "__main__":
    main()
