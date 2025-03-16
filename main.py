#!/usr/bin/env python3
from email.policy import strict
from binance.um_futures import UMFutures
from dotenv import load_dotenv
import os
import logging
import pandas as pd
import ta
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()
BINANCE_API_KEY = os.getenv("binance_future_testnet_api")
BINANCE_SECRET = os.getenv("binance_future_testnet_secret")
BINANCE_TESTNET_URL = os.getenv("testnest_url")

client = UMFutures(
    key=BINANCE_API_KEY, secret=BINANCE_SECRET, base_url=BINANCE_TESTNET_URL
)

tp = float(os.getenv("take_profit", 0.01))
sl = float(os.getenv("stop_loss", 0.01))
volume = float(os.getenv("volume", 50))
leverage = int(os.getenv("leverage", 2))
margin_type = os.getenv("margin_type", 'ISOLATED')

def get_balance_usdt():
    try:
        response = client.balance()
        for elem in response:
            if elem["asset"] == "USDT":
                return float(elem["balance"])
    except Exception as e:
        logger.error(f"Error getting balance: {e}")


def get_ticker_usdt():
    tickers = ["BTCUSDT","ETHUSDT"]
    response = client.ticker_price(tickers)
    # Extract just the symbol names
    return [item['symbol'] for item in response] if isinstance(response, list) else [response['symbol']]
print(get_ticker_usdt())

def klines(symbol):
    try:
        response = pd.DataFrame(client.klines(symbol=symbol, interval="15m"))
        response = response.iloc[:, 0:6]
        response.columns = ["time", "open", "high", "low", "close", "volume"]

        # Convert types
        response = response.set_index("time")
        response.index = pd.to_datetime(response.index, unit="ms")
        response = response.astype(float)

        # Sort by time
        response = response.sort_values("time")

        return response
    except Exception as e:
        logger.error(f"Error getting klines for {symbol}: {e}")


def set_leverage(symbol, level):
    try:
        response = client.change_leverage(
            symbol=symbol, leverage=level, recvWindow=6000
        )
        print(response)
    except Exception as e:
        logger.error(f"Error setting leverage for {symbol}: {e}")


def set_mode(symbol, margin_type):
    try:
        response = client.change_margin_type(
            symbol=symbol, marginType=margin_type, recvWindow=6000
        )
        print(response)
    except Exception as e:
        logger.error(f"Error setting margin type for {symbol}: {e}")


def get_price_precision(symbol):
    try:
        resp = client.exchange_info()
        for elem in resp["symbols"]:
            if elem["symbol"] == symbol:
                return elem["pricePrecision"]
    except Exception as e:
        logger.error(f"Error getting price precision for {symbol}: {e}")


def get_qty_precision(symbol):
    try:
        resp = client.exchange_info()
        for elem in resp["symbols"]:
            if elem["symbol"] == symbol:
                return elem["quantityPrecision"]
    except Exception as e:
        logger.error(f"Error getting quantity precision for {symbol}: {e}")


def open_order(symbol, side):

    price = float(client.ticker_price(symbol)["price"])
    qty_precision = get_qty_precision(symbol)
    price_precision = get_price_precision(symbol)
    qty = round(volume / price, qty_precision)

    if side == "buy":
        try:
            resp1 = client.new_order(
                symbol=symbol,
                side="BUY",
                type="LIMIT",
                timeInForce="GTC",
                quantity=qty,
                price=price,
            )
            print(symbol, side, "placing order")
            print(resp1)
            time.sleep(2)
            sl_price = round(price - price * sl, price_precision)
            resp2 = client.new_order(
                symbol=symbol,
                side="SELL",
                type="STOP_MARKET",
                timeInForce="GTC",
                quantity=qty,
                stopPrice=sl_price,
            )
            print(resp2)
            time.sleep(2)
            tp_price = round(price + price * tp, price_precision)
            resp3 = client.new_order(
                symbol=symbol,
                side="SELL",
                type="TAKE_PROFIT_MARKET",
                timeInForce="GTC",
                quantity=qty,
                stopPrice=tp_price,
            )
            print(resp3)
        except Exception as e:
            logger.error(f"Error opening buy order for {symbol}: {e}")

    if side == "sell":
        try:
            resp1 = client.new_order(
                symbol=symbol,
                side="SELL",
                type="LIMIT",
                timeInForce="GTC",
                quantity=qty,
                price=price,
            )
            print(symbol, side, "placing order")
            print(resp1)
            time.sleep(2)
            sl_price = round(price + price * sl, price_precision)
            resp2 = client.new_order(
                symbol=symbol,
                side="BUY",
                type="STOP_MARKET",
                timeInForce="GTC",
                quantity=qty,
                stopPrice=sl_price,
            )
            print(resp2)
            time.sleep(2)
            tp_price = round(price - price * tp, price_precision)
            resp3 = client.new_order(
                symbol=symbol,
                side="BUY",
                type="TAKE_PROFIT_MARKET",
                timeInForce="GTC",
                quantity=qty,
                stopPrice=tp_price,
            )
            print(resp3)
        except Exception as e:
            logger.error(f"Error opening sell order for {symbol}: {e}")


def check_positions():
    try:
        response = client.get_position_risk()
        positions = 0
        for elem in response:
            if float(elem["positionAmt"]) != 0:
                positions += 1
        return positions
    except Exception as e:
        logger.error(f"Error checking positions: {e}")


def close_open_positions(symbol):
    try:
        response = client.cancel_open_orders(symbol=symbol, recvWindow=6000)
        print(response)
    except Exception as e:
        logger.error(f"Error closing open positions for {symbol}: {e}")


def check_macd_ema(symbol):
    kl = klines(symbol)
    if (
        ta.trend.macd_diff(kl['close']).iloc[-1] > 0
        and ta.trend.macd_diff(kl['close']).iloc[-2] < 0
        and ta.trend.ema_indicator(kl['close'], window=200).iloc[-1] < kl['close'].iloc[-1]
    ):
        return "up"
    elif (
        ta.trend.macd_diff(kl['close']).iloc[-1] < 0
        and ta.trend.macd_diff(kl['close']).iloc[-2] > 0
        and ta.trend.ema_indicator(kl['close'], window=200).iloc[-1] > kl['close'].iloc[-1]
    ):
        return "down"
    else:
        return "none"


order = False
symbol = ""
symbols = get_ticker_usdt()

while True:
    positions = check_positions()
    print(f"positions open: {positions}")
    if positions == 0:
        order = False
        if symbol != "":
            close_open_positions(symbol)

    if order == False:
        for elem in symbols:
            signal = check_macd_ema(elem)
            print(signal, elem)
            if signal == "up":
                print(f"buy signal on {elem}")
                set_mode(elem, margin_type="ISOLATED")
                time.sleep(1)
                set_leverage(elem, leverage)
                time.sleep(1)
                print(f"buying {elem}")
                open_order(elem, side="buy")
                symbol = elem
                order = True
                break
            if signal == "down":
                print(f"sell signal on {elem}")
                set_mode(elem, margin_type="ISOLATED")
                time.sleep(1)
                set_leverage(elem, leverage)
                time.sleep(1)
                print(f"Selling {elem}")
                open_order(elem, side="sell")
                symbol = elem
                order = True
                break
    print('Sleeping')
    time.sleep(60)
