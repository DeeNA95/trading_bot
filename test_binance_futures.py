"""
Test script for the Binance Futures executor.
"""

import logging
import math
import os
import time

from dotenv import load_dotenv

from execution.binance_futures_orders import BinanceFuturesExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env file
load_dotenv()


def get_price_precision(executor, symbol):
    """Get the price precision for a symbol from exchange info"""
    try:
        exchange_info = executor.client.exchange_info()
        for symbol_info in exchange_info["symbols"]:
            if symbol_info["symbol"] == symbol:
                for filter_info in symbol_info["filters"]:
                    if filter_info["filterType"] == "PRICE_FILTER":
                        tick_size = float(filter_info["tickSize"])
                        return tick_size
        return 0.1  # Default fallback
    except Exception as e:
        logging.error(f"Error getting price precision: {e}")
        return 0.1  # Default fallback


def get_quantity_precision(executor, symbol):
    """Get the quantity precision for a symbol from exchange info"""
    try:
        exchange_info = executor.client.exchange_info()
        for symbol_info in exchange_info["symbols"]:
            if symbol_info["symbol"] == symbol:
                for filter_info in symbol_info["filters"]:
                    if filter_info["filterType"] == "LOT_SIZE":
                        step_size = float(filter_info["stepSize"])
                        return step_size
        return 0.001  # Default fallback
    except Exception as e:
        logging.error(f"Error getting quantity precision: {e}")
        return 0.001  # Default fallback


def round_to_tick_size(price, tick_size):
    """Round a price to the nearest valid tick size"""
    inverse = 1.0 / tick_size
    return math.floor(price * inverse) / inverse


def round_to_step_size(quantity, step_size):
    """Round a quantity to the nearest valid step size"""
    inverse = 1.0 / step_size
    return math.floor(quantity * inverse) / inverse


def main():
    # Create a BinanceFuturesExecutor instance (using testnet)
    executor = BinanceFuturesExecutor(use_testnet=True)

    # Get account information
    account_info = executor.get_account_info()
    print("\nAccount Information:")
    print(f"Available Balance: {account_info.get('availableBalance', 'N/A')} USDT")
    print(f"Total Wallet Balance: {account_info.get('totalWalletBalance', 'N/A')} USDT")

    # Get current BTC price
    btc_price = executor.get_btc_price()
    print(f"\nCurrent BTC Price: {btc_price} USDT")

    # Get BTC position
    position = executor.get_btc_position()
    print("\nCurrent BTC Position:")
    print(f"Position Amount: {position.get('positionAmt', '0')} BTC")
    print(f"Entry Price: {position.get('entryPrice', '0')} USDT")
    print(f"Unrealized Profit: {position.get('unRealizedProfit', '0')} USDT")
    print(f"Leverage: {position.get('leverage', '0')}x")

    # Get price precision for BTCUSDT
    tick_size = get_price_precision(executor, "BTCUSDT")
    step_size = get_quantity_precision(executor, "BTCUSDT")
    print(f"\nBTCUSDT tick size: {tick_size}")
    print(f"BTCUSDT step size: {step_size}")

    # Place a test limit order
    print("\nPlacing a test limit order...")
    target_price = btc_price * 0.9  # 10% below current price
    limit_price = round_to_tick_size(target_price, tick_size)

    # Calculate quantity to meet minimum notional value of 100 USDT
    min_notional = 100.0
    min_quantity = min_notional / limit_price
    # Add a small buffer to ensure we're above the minimum
    quantity = round_to_step_size(min_quantity * 1.05, step_size)

    print(f"Target price: {target_price}, Rounded to tick size: {limit_price}")
    print(f"Minimum quantity needed: {min_quantity}, Using quantity: {quantity}")
    print(f"Notional value: {limit_price * quantity} USDT")

    limit_order = executor.execute_limit_order(
        side="BUY", quantity=quantity, price=limit_price
    )

    if "orderId" in limit_order:
        print(f"Limit order placed successfully: Order ID {limit_order['orderId']}")
        order_id = limit_order["orderId"]

        # Wait a moment for the order to be processed
        print("Waiting for order to be processed...")
        time.sleep(2)

        # Get open orders
        try:
            print("\nGetting open orders...")
            open_orders = executor.get_open_orders()
            print(f"Number of Open Orders: {len(open_orders)}")
            for order in open_orders:
                print(
                    f"Order ID: {order.get('orderId')}, Type: {order.get('type')}, Side: {order.get('side')}, Price: {order.get('price')}"
                )

            # Cancel the test order
            print(f"\nCancelling test order {order_id}...")
            cancel_result = executor.cancel_order(order_id)
            print(f"Cancel result: {cancel_result}")
        except Exception as e:
            print(f"\nError: {e}")
    else:
        print(f"Failed to place limit order: {limit_order}")


if __name__ == "__main__":
    main()
