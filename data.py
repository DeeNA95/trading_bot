#!/usr/bin/env python3
import argparse
import logging
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv


class DataHandler:
    """
    Class for retrieving, processing, and preprocessing Binance Futures market data.
    This includes fetching raw klines data, adding technical and risk metrics,
    incorporating futures-specific metrics (funding rates, open interest, liquidations),
    and splitting the data into training and testing sets for machine learning.
    """

    def __init__(self):
        load_dotenv()

        binance_key = os.getenv("binance_future_testnet_api")
        binance_secret = os.getenv("binance_future_testnet_secret")

        if not binance_key or not binance_secret:
            raise ValueError(
                "Binance API credentials not found in environment variables. "
                "Ensure that binance_future_testnet_api and binance_future_testnet_secret are set in your .env file."
            )

        self.client = Client(
            api_key=binance_key,
            api_secret=binance_secret,
            requests_params={"timeout": 100},
        )
        logging.info("Binance Futures client initialized successfully.")

    def get_futures_data(self, symbol, interval="1m", start_time=None, end_time=None):
        """
        Fetch historical futures data using Binance klines endpoint.
        Data is retrieved in chunks to bypass the API row limit.
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(
                days=7
            )  # Default to 7 days for 1m data
        if end_time is None:
            end_time = datetime.now()

        all_data = []
        # Determine chunk size based on the interval
        if interval == "1m":
            chunk_size = timedelta(hours=12)
        elif interval in ["3m", "5m", "15m"]:
            chunk_size = timedelta(days=1)
        else:
            chunk_size = timedelta(days=7)

        current_start = start_time
        while current_start < end_time:
            current_end = min(current_start + chunk_size, end_time)
            try:
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=int(current_start.timestamp() * 1000),
                    endTime=int(current_end.timestamp() * 1000),
                    limit=1500,  # Maximum allowed by Binance
                )
            except BinanceAPIException as e:
                logging.error(f"Error fetching klines for {symbol}: {e}")
                break

            if not klines:
                logging.warning(
                    f"No data returned for {symbol} from {current_start} to {current_end}"
                )
                current_start = current_end
                continue

            df_chunk = pd.DataFrame(
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
            # Convert timestamps and cast data types
            df_chunk["open_time"] = pd.to_datetime(df_chunk["open_time"], unit="ms")
            df_chunk.set_index("open_time", inplace=True)
            df_chunk = df_chunk.astype(
                {
                    "open": "float",
                    "high": "float",
                    "low": "float",
                    "close": "float",
                    "volume": "float",
                    "quote_asset_volume": "float",
                    "number_of_trades": "int",
                    "taker_buy_base_asset_volume": "float",
                    "taker_buy_quote_asset_volume": "float",
                }
            )
            df_chunk["symbol"] = symbol
            all_data.append(df_chunk)
            # Advance current_start using the last close_time plus 1ms to avoid duplicates
            last_close = int(klines[-1][6]) + 1
            current_start = datetime.fromtimestamp(last_close / 1000)

        if not all_data:
            return pd.DataFrame()

        combined_data = pd.concat(all_data)
        combined_data.sort_index(inplace=True)
        logging.info(f"Retrieved {len(combined_data)} data points for {symbol}.")
        return combined_data

    def calculate_technical_indicators(self, df):
        """
        Calculate a set of technical indicators: SMA, RSI, Bollinger Bands, MACD,
        Stochastic Oscillator, ATR, and ADX.
        """
        result = df.copy()
        close = result["close"]

        # Simple Moving Averages
        result["sma_20"] = close.rolling(window=20).mean()
        result["sma_50"] = close.rolling(window=50).mean()

        # RSI calculation
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        result["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        result["bb_middle"] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        result["bb_upper"] = result["bb_middle"] + 2 * bb_std
        result["bb_lower"] = result["bb_middle"] - 2 * bb_std

        # MACD and Signal
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        result["macd"] = ema_12 - ema_26
        result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()

        # Stochastic Oscillator
        n = 14
        result["lowest_low"] = result["low"].rolling(window=n).min()
        result["highest_high"] = result["high"].rolling(window=n).max()
        result["stoch_k"] = 100 * (
            (close - result["lowest_low"])
            / (result["highest_high"] - result["lowest_low"])
        )
        result["stoch_d"] = result["stoch_k"].rolling(window=3).mean()

        # ATR
        high_low = result["high"] - result["low"]
        high_close = np.abs(result["high"] - close.shift())
        low_close = np.abs(result["low"] - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result["atr"] = true_range.rolling(14).mean()

        # ADX and Directional Indicators
        plus_dm = result["high"].diff()
        minus_dm = -result["low"].diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        plus_di = (
            100
            * pd.Series(plus_dm).ewm(alpha=1 / 14, adjust=False).mean()
            / result["atr"]
        )
        minus_di = (
            100
            * pd.Series(minus_dm).ewm(alpha=1 / 14, adjust=False).mean()
            / result["atr"]
        )
        result["plus_di"] = plus_di
        result["minus_di"] = minus_di
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        result["adx"] = pd.Series(dx).ewm(alpha=1 / 14, adjust=False).mean()

        logging.info("Calculating technical indicators...")
        return result

    def calculate_risk_metrics(self, df, window=20):
        """
        Calculate risk-related metrics: volatility, VaR, CVaR, max drawdown,
        Sharpe, Sortino, and Calmar ratios.
        """
        result = df.copy()
        result["daily_return"] = result["close"].pct_change()
        result["volatility"] = result["daily_return"].rolling(window=window).std()
        result["var_95"] = result["daily_return"].rolling(window=window).quantile(0.05)

        def calculate_cvar(returns):
            var = returns.quantile(0.05)
            return returns[returns <= var].mean()

        result["cvar_95"] = (
            result["daily_return"]
            .rolling(window=window)
            .apply(calculate_cvar, raw=False)
        )

        rolling_max = result["close"].rolling(window=window, min_periods=1).max()
        drawdown = result["close"] / rolling_max - 1.0
        result["max_drawdown"] = drawdown.rolling(window=window).min()

        result["sharpe_ratio"] = (
            result["daily_return"].rolling(window=window).mean()
            / result["daily_return"].rolling(window=window).std()
        ) * np.sqrt(252)

        downside = result["daily_return"].copy()
        downside[downside > 0] = 0
        result["sortino_ratio"] = (
            result["daily_return"].rolling(window=window).mean()
            / downside.rolling(window=window).std()
        ) * np.sqrt(252)

        result["calmar_ratio"] = (
            result["daily_return"].rolling(window=window).mean() * 252
        ) / result["max_drawdown"].abs()

        logging.info("Calculating risk metrics...")
        return result

    def identify_trade_setups(self, df):
        """
        Identify potential trade setups using trend, momentum, volatility, and strength signals.
        """
        result = df.copy()
        # Ensure required indicators are available
        for col in [
            "sma_20",
            "sma_50",
            "rsi",
            "macd",
            "macd_signal",
            "stoch_k",
            "stoch_d",
            "adx",
            "plus_di",
            "minus_di",
        ]:
            if col not in result.columns:
                result = self.calculate_technical_indicators(result)
                break

        # Initialize signals
        result["signal_trend"] = 0
        result["signal_momentum"] = 0
        result["signal_volatility"] = 0
        result["signal_strength"] = 0
        result["trade_setup"] = "none"

        # Trend signal: SMA crossover and directional movement
        result.loc[result["sma_20"] > result["sma_50"], "signal_trend"] = 1
        result.loc[result["sma_20"] < result["sma_50"], "signal_trend"] = -1
        result.loc[result["plus_di"] > result["minus_di"], "signal_trend"] = 1
        result.loc[result["plus_di"] < result["minus_di"], "signal_trend"] = -1

        # Momentum signal: RSI and MACD
        result.loc[result["rsi"] < 30, "signal_momentum"] = 1
        result.loc[result["rsi"] > 70, "signal_momentum"] = -1
        result.loc[result["macd"] > result["macd_signal"], "signal_momentum"] = 1
        result.loc[result["macd"] < result["macd_signal"], "signal_momentum"] = -1

        # Volatility signal based on ATR percentage
        result["atr_pct"] = result["atr"] / result["close"] * 100
        atr_mean = result["atr_pct"].rolling(window=20).mean()
        result.loc[result["atr_pct"] > atr_mean * 1.5, "signal_volatility"] = 1
        result.loc[result["atr_pct"] < atr_mean * 0.5, "signal_volatility"] = -1

        # Trend strength signal via ADX
        result.loc[result["adx"] > 25, "signal_strength"] = 1
        result.loc[result["adx"] < 20, "signal_strength"] = -1

        # Example trade setup rules:
        result.loc[
            (result["signal_trend"] == 1)
            & (result["signal_momentum"] == 1)
            & (result["signal_strength"] == 1),
            "trade_setup",
        ] = "strong_bullish"
        result.loc[
            (result["signal_trend"] == -1)
            & (result["signal_momentum"] == -1)
            & (result["signal_strength"] == 1),
            "trade_setup",
        ] = "strong_bearish"
        result.loc[
            (result["signal_trend"] == -1)
            & (result["signal_momentum"] == 1)
            & (result["rsi"] < 30),
            "trade_setup",
        ] = "bullish_reversal"
        result.loc[
            (result["signal_trend"] == 1)
            & (result["signal_momentum"] == -1)
            & (result["rsi"] > 70),
            "trade_setup",
        ] = "bearish_reversal"

        logging.info("Identifying trade setups...")
        return result

    def _fetch_futures_metric(
        self, api_method, params, time_col, resample_rule, cols_to_keep, value_cast=None
    ):
        """
        Generic method to fetch futures-specific metrics (funding, open interest, liquidations)
        and resample to match the main dataframe.
        """
        try:
            data = api_method(**params)
            if not data:
                return pd.DataFrame(columns=cols_to_keep)
            df = pd.DataFrame(data)
            df[time_col] = pd.to_datetime(df[time_col], unit="ms")
            df.set_index(time_col, inplace=True)
            if value_cast:
                for col, dtype in value_cast.items():
                    df[col] = df[col].astype(dtype)
            df = df.resample(resample_rule, closed="right", label="right").ffill()
            return df[cols_to_keep]
        except Exception as e:
            logging.error(f"Error fetching metric with {api_method.__name__}: {e}")
            return pd.DataFrame(columns=cols_to_keep)

    def add_futures_metrics(self, df, symbol, interval, start_time, end_time):
        """
        Add futures-specific metrics: funding rates, open interest, and liquidations.
        """
        result = df.copy()

        # Funding rates: Binance provides funding rates (typically every 8h)
        funding_params = {
            "symbol": symbol,
            "limit": 1000,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
        }
        funding_df = self._fetch_futures_metric(
            self.client.futures_funding_rate,
            funding_params,
            time_col="fundingTime",
            resample_rule=interval,
            cols_to_keep=["fundingRate"],
            value_cast={"fundingRate": float},
        )
        if not funding_df.empty:
            result = result.join(funding_df, how="left")
            result["fundingRate"] = result["fundingRate"].ffill()
            result["cumulative_funding"] = result["fundingRate"].cumsum()
        else:
            result["fundingRate"] = 0.0
            result["cumulative_funding"] = 0.0

        # Open interest: using futures_open_interest_hist endpoint
        oi_params = {
            "symbol": symbol,
            "period": interval,
            "limit": 1000,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
        }
        oi_df = self._fetch_futures_metric(
            self.client.futures_open_interest_hist,
            oi_params,
            time_col="timestamp",
            resample_rule=interval,
            cols_to_keep=["sumOpenInterest", "sumOpenInterestValue"],
            value_cast={"sumOpenInterest": float, "sumOpenInterestValue": float},
        )
        if not oi_df.empty:
            result = result.join(oi_df, how="left")
            result["sumOpenInterest"] = result["sumOpenInterest"].ffill()
            result["sumOpenInterestValue"] = result["sumOpenInterestValue"].ffill()
            result["oi_change"] = result["sumOpenInterest"].pct_change()
            result["oi_volume_ratio"] = result["sumOpenInterest"] / result[
                "volume"
            ].replace(0, np.nan)
        else:
            result["sumOpenInterest"] = 0.0
            result["sumOpenInterestValue"] = 0.0
            result["oi_change"] = 0.0
            result["oi_volume_ratio"] = 0.0

        # Liquidations: using futures_liquidation_orders endpoint
        liq_params = {
            "symbol": symbol,
            "limit": 1000,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
        }
        liq_df = self._fetch_futures_metric(
            self.client.futures_liquidation_orders,
            liq_params,
            time_col="time",
            resample_rule=interval,
            cols_to_keep=["price", "origQty"],
            value_cast={"price": float, "origQty": float},
        )
        if not liq_df.empty:
            liq_df["liquidation_value"] = liq_df["price"] * liq_df["origQty"]
            liq_resampled = (
                liq_df.resample(interval, closed="right", label="right")
                .agg({"liquidation_value": "sum", "origQty": "sum"})
                .fillna(0)
            )
            result = result.join(liq_resampled, how="left")
            result["liquidation_value"] = result["liquidation_value"].fillna(0)
            result["origQty"] = result["origQty"].fillna(0)
            result["liquidation_intensity"] = result["liquidation_value"] / result[
                "volume"
            ].replace(0, np.nan)
        else:
            result["liquidation_value"] = 0.0
            result["origQty"] = 0.0
            result["liquidation_intensity"] = 0.0

        logging.info("Adding futures-specific metrics...")
        return result

    def process_market_data(
        self, symbol, interval="1m", start_time=None, end_time=None, save_path=None
    ):
        """
        Comprehensive function to retrieve futures data, add all metrics (technical,
        risk, and futures-specific), and fill missing values.
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = (
                end_time - timedelta(days=7)
                if interval == "1m"
                else end_time - timedelta(days=60)
            )

        logging.info(
            f"Processing {symbol} data from {start_time} to {end_time} with interval {interval}."
        )

        df = self.get_futures_data(symbol, interval, start_time, end_time)
        if df.empty:
            logging.warning(f"No data retrieved for {symbol}.")
            raise ValueError(f"No data retrieved for {symbol}.")

        df = self.calculate_technical_indicators(df)
        df = self.add_futures_metrics(df, symbol, interval, start_time, end_time)
        df = self.calculate_risk_metrics(df)
        df = self.identify_trade_setups(df)
        df = df.ffill().bfill()

        if save_path:
            self.save_data_to_csv(df, save_path)
            logging.info(f"Data with metrics saved to {save_path}")

        return df

    def preprocess_data_for_ml(self, df):
        """
        Preprocess the data for machine learning:
          - Calculate technical, risk, and trade setup indicators.
          - Split the data: the last 10% is used as test data.
        Returns:
          train_data, test_data (DataFrames)
        """
        df = self.calculate_technical_indicators(df)
        df = self.calculate_risk_metrics(df)
        df = self.identify_trade_setups(df)
        df = df.ffill().bfill()

        split_idx = int(len(df) * 0.9)
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]
        return train_data, test_data

    def save_data_to_csv(self, df, file_path):
        """Save DataFrame to CSV."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.reset_index(inplace=True)
        df.to_csv(file_path, index=False)
        logging.info(f"Data saved to {file_path}")

    def save_train_test_data(self, train_data, test_data, path):
        """Save training and testing data as parquet files."""
        os.makedirs(path, exist_ok=True)
        train_file = os.path.join(path, "train_data.parquet")
        test_file = os.path.join(path, "test_data.parquet")
        train_data.to_parquet(train_file)
        test_data.to_parquet(test_file)
        logging.info(f"Training data saved to {train_file}")
        logging.info(f"Testing data saved to {test_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Binance Futures data processing utility"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Futures symbol (e.g., BTCUSDT)"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1m",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        help="Data interval",
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days of historical data to fetch"
    )
    parser.add_argument(
        "--start_date", type=str, default=None, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (defaults to now)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save processed data",
    )
    args = parser.parse_args()

    end_date = (
        datetime.now()
        if args.end_date is None
        else datetime.strptime(args.end_date, "%Y-%m-%d")
    )
    start_date = (
        end_date - timedelta(days=args.days)
        if args.start_date is None
        else datetime.strptime(args.start_date, "%Y-%m-%d")
    )

    data_handler = DataHandler()

    # Always process data: fetch, add metrics, and fill missing values
    processed_data = data_handler.process_market_data(
        args.symbol, args.interval, start_date, end_date
    )

    # Preprocess for machine learning (split into training and testing data: last 10% for testing)
    train_data, test_data = data_handler.preprocess_data_for_ml(processed_data)

    # Save the processed data as CSV and the train/test data as parquet files
    processed_csv = os.path.join(
        args.output_dir, f"{args.symbol}_{args.interval}_with_metrics.csv"
    )
    data_handler.save_data_to_csv(processed_data, processed_csv)
    data_handler.save_train_test_data(train_data, test_data, args.output_dir)
