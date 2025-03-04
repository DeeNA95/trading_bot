from time import sleep
import os
from logging import getLogger

import polars as pl
from binance.error import ClientError
from binance.um_futures import UMFutures
from dotenv import load_dotenv


load_dotenv()

logger = getLogger(__name__)

BINANCE_API_KEY = os.getenv("binance_future_testnet_api")
BINANCE_API_SECRET = os.getenv("binance_future_testnet_secret")

client = UMFutures(key=BINANCE_API_KEY, secret=BINANCE_API_SECRET)

tp = 1e-2
sl = 1e-2
volume = 10
leverage = 2
trade_type = 'ISOLATED' #CROSS

def get_balance_usdt():
    try:
        response = client.balance(recvWindow=6000)
        for elem in response:
            if elem['asset'] == 'USDT':
                logger.info(f"USDT balance: {elem['balance']}")
    except:
        logger.exception("Error fetching balance")


print(get_balance_usdt())
