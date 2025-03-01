#! /usr/bin/env python3

import os
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

# Load your API credentials from environment variables for security
load_dotenv()
ALPACA_API_KEY_ID = os.getenv("ALPACA_KEY")
ALPACA_API_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Debug information
print(f"API Key ID exists: {ALPACA_API_KEY_ID is not None}")
print(f"API Secret Key exists: {ALPACA_API_SECRET_KEY is not None}")

# Check if credentials are available
if not ALPACA_API_KEY_ID or not ALPACA_API_SECRET_KEY:
    print("Error: Alpaca API credentials are missing.")
    print("Please ensure you have a .env file with APCA_API_KEY_ID and APCA_API_SECRET_KEY defined.")
    print("Or set these environment variables before running the script.")
    exit(1)

try:
    # Initialize the trading client with your credentials
    trading_client = TradingClient(
        ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, paper=True
    )  # Set paper=False for live trading

    # Check if the trading client is connected
    account = trading_client.get_account()
    print(f"Trading client connected successfully. Account ID: {account.id}")
    print(f"Account status: {account.status}")
    print(f"Cash balance: ${account.cash}")
    print(f"Portfolio Value: {account.portfolio_value}")

except ValueError as e:
    print(f"Authentication error: {e}")
    print("Please check your API credentials.")
except Exception as e:
    print(f"An error occurred: {e}")
