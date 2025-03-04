#! /usr/bin/env python3
from binance.um_futures import UMFutures
from dotenv import load_dotenv
import os

load_dotenv()
BINANCE_API_KEY = os.getenv("binance_future_testnet_api")
BINANCE_SECRET = os.getenv("binance_future_testnet_secret")
BINANCE_TESTNET_URL = os.getenv("testnest_url")

client = UMFutures(
    key=BINANCE_API_KEY, secret=BINANCE_SECRET, base_url=BINANCE_TESTNET_URL
)
print(client.balance())
