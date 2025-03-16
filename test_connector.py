#!/usr/bin/env python3
from binance.um_futures import UMFutures
from dotenv import load_dotenv
import os

load_dotenv()
BINANCE_API_KEY = os.getenv("binance_api2")
BINANCE_SECRET = os.getenv("binance_secret2")
BINANCE_MAINNET_URL = "https://fapi.binance.com"

client = UMFutures(
    key=BINANCE_API_KEY, secret=BINANCE_SECRET, base_url=BINANCE_MAINNET_URL
)
print(client.balance())
