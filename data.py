#!/usr/bin/env python3
import argparse
import logging
import os
import random
from datetime import datetime, timedelta
from time import sleep

import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from google.cloud import secretmanager, storage

logger = logging.getLogger(__name__)


class DataHandler:
    """
    Class for retrieving, processing, and preprocessing Binance Futures market data.
    This includes fetching raw klines data, adding technical and risk metrics,
    incorporating futures-specific metrics (funding rates, open interest, liquidations),
    """

    def __init__(self):

        try:
            self.gcloud_client = secretmanager.SecretManagerServiceClient()
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "seraphic-bliss-451413-c8")

            binance_key_response = self.gcloud_client.access_secret_version(
                name=f"projects/{project_id}/secrets/BINANCE_API_KEY/versions/latest"
            )
            binance_key = binance_key_response.payload.data.decode("UTF-8").strip()

            binance_secret_response = self.gcloud_client.access_secret_version(
                name=f"projects/{project_id}/secrets/BINANCE_SECRET_KEY/versions/latest"
            )
            binance_secret = binance_secret_response.payload.data.decode(
                "UTF-8"
            ).strip()

            logger.info(
                "Successfully retrieved Binance credentials from Google Secret Manager"
            )

        except Exception as e:
            logger.error(f"Could not retrieve from Google Secret Manager: {e}")
            load_dotenv()
            binance_key = os.getenv("binance_api")
            binance_secret = os.getenv("binance_secret")
            logger.info("Falling back to .env file for Binance credentials")

        if not binance_key or not binance_secret:
            raise ValueError(
                "Binance API credentials not found in environment variables. "
                "Ensure that binance_api and binance_secret are set in your .env file."
            )

        # init binance client
        self.client = Client(
            api_key=binance_key,
            api_secret=binance_secret,
            requests_params={"timeout": 100},
        )
        logger.info("Binance Futures client initialized successfully.")

    def get_futures_data(self, symbol, interval="1m", start_time=None, end_time=None):
        """
        Fetch historical futures data using Binance klines endpoint.
        Data is retrieved in chunks to bypass the API row limit.
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(
                days=30
            )  # default to 30 days for 1h data

        if end_time is None:
            end_time = datetime.now()

        all_data = []
        # Determine chunk size based on the interval
        if interval == "1m":
            chunk_size = timedelta(hours=12)  # 60x12 chunks
        elif interval in ["3m", "5m", "15m"]:
            chunk_size = timedelta(days=1)
        else:
            chunk_size = timedelta(days=7)
        ctr = 1
        current_start = start_time
        while current_start < end_time:

            print(f"Fetching chunk {ctr}")
            ctr += 1
            current_end = min(current_start + chunk_size, end_time)
            try:
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=int(current_start.timestamp() * 1000),  # since in ms
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
            logger.error("Returned Empty DataFrame")
            return pd.DataFrame()

        combined_data = pd.concat(all_data)
        combined_data.sort_index(inplace=True)
        logger.info(f"Retrieved {len(combined_data)} data points for {symbol}.")
        return combined_data

    @staticmethod
    def calculate_technical_indicators(df):
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
        ema_12 = close.ewm(span=12, adjust=False).mean()  # expo MA accross 12 timespans
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
        result["atr"] = result["atr"].bfill()
        # print('how many nas in atr',result["atr"].isna().sum())

        # ADX and Directional Indicators
        # Calculate the Positive Directional Movement (+DM)
        plus_dm = result["high"].diff()
        # Calculate the Negative Directional Movement (-DM)
        minus_dm = -result["low"].diff()

        # Apply conditions to determine +DM and -DM
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        # Convert to pandas Series to handle NaN values
        plus_dm = pd.Series(plus_dm, index=result.index)
        minus_dm = pd.Series(minus_dm, index=result.index)

        # Fill NaN values in +DM and -DM
        plus_dm.fillna(0, inplace=True)
        minus_dm.fillna(0, inplace=True)
        # Calculate the Positive Directional Indicator (+DI)
        plus_di = (
            100
            * pd.Series(plus_dm).ewm(alpha=1 / 14, adjust=False).mean()
            / result["atr"]
        )
        # print(plus_di)
        # Calculate the Negative Directional Indicator (-DI)
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

    @staticmethod
    def calculate_risk_metrics(df, interval="1h", window_size=60):
        """
        Calculate risk-related metrics: volatility, VaR, CVaR, max drawdown,
        Sharpe, Sortino, and Calmar ratios.
        """
        # Determine window size based on interval
        periods_per_day = DataHandler.get_periods_per_day(interval) # Use static call

        window = min(
            window_size, int(24 * periods_per_day)
        )  # 24 hours worth of data points

        result = df.copy()
        result["timely_return"] = result["close"].pct_change()
        # Use periods instead of time-based window
        result["volatility"] = result["timely_return"].rolling(window=window).std()
        result["var_95"] = result["timely_return"].rolling(window=window).quantile(0.05)

        def calculate_cvar(returns):  # continuous variance
            var = returns.quantile(0.05)
            return returns[returns <= var].mean()

        result["cvar_95"] = (
            result["timely_return"]
            .rolling(window=window)
            .apply(calculate_cvar, raw=False)
        )

        rolling_max = result["close"].rolling(window=window, min_periods=1).max()
        drawdown = result["close"] / rolling_max - 1.0
        result["max_drawdown"] = drawdown.rolling(
            window=window
        ).min()  # min over the window is max drawdown usually cause negative

        # Determine annualization factor based on interval
        if interval.endswith("m"):
            # For minute-based intervals
            minutes = int(interval[:-1])
            annualization_factor = np.sqrt(
                525600 / minutes
            )  # 525600 = minutes in a year (365*24*60)
        elif interval.endswith("h"):
            # For hour-based intervals
            hours = int(interval[:-1])
            annualization_factor = np.sqrt(
                8760 / hours
            )  # 8760 = hours in a year (365*24)
        elif interval.endswith("d"):
            # For day-based intervals
            days = int(interval[:-1])
            annualization_factor = np.sqrt(365 / days)  # 365 = days in a year
        else:
            # Default to daily for unknown intervals
            annualization_factor = np.sqrt(365)

        # Calculate ratios with appropriate annualization
        result["sharpe_ratio"] = (
            result["timely_return"].rolling(window=window).mean()
            / result["timely_return"].rolling(window=window).std()
        ) * annualization_factor

        downside = result["timely_return"].copy()
        downside[downside > 0] = 0
        result["sortino_ratio"] = (
            result["timely_return"].rolling(window=window).mean()
            / downside.rolling(window=window).std()
        ) * annualization_factor

        result["calmar_ratio"] = (
            result["timely_return"].rolling(window=window).mean()
            * (365 * 24 / DataHandler.get_periods_per_day(interval)) # Use static call
        ) / result["max_drawdown"].abs()

        # Calculate efficiency ratio
        risk_free_rate = 0.01  # Example risk-free rate, can be adjusted
        result["efficiency_ratio"] = (
            result["timely_return"].rolling(window=window).mean() / risk_free_rate
        ) * annualization_factor

        logger.info("Calculating risk metrics...")
        return result

    # Note: This helper is used by calculate_risk_metrics, making it static too.
    @staticmethod
    def get_periods_per_day(interval):
        """Helper function to calculate periods per day for a given interval"""
        if interval.endswith("m"):
            minutes = int(interval[:-1])
            return 24 * 60 / minutes
        elif interval.endswith("h"):
            hours = int(interval[:-1])
            return 24 / hours
        elif interval.endswith("d"):
            days = int(interval[:-1])
            return 1 / days
        else:
            return 1

    @staticmethod
    def identify_trade_setups(df):
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
                # Call the static method directly using the class name
                result = DataHandler.calculate_technical_indicators(result)
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

    @staticmethod
    def calculate_price_density(df, window=20, bins=10):
        """
        Calculate price density using a rolling window

        Args:
            df: DataFrame containing OHLC data
            window: Size of the rolling window (default: 20)
            bins: Number of price bins for density calculation (default: 10)

        Returns:
            DataFrame with 'price_density' column added.
        """
        # Define the helper function to apply to each rolling window
        def _calculate_density_for_window(window_data: np.ndarray, min_periods: int, num_bins: int) -> float:
            # Check if enough non-NaN values exist in the window
            # Note: rolling().apply() handles min_periods, but we double-check
            # or handle cases where the window might contain NaNs passed by `apply`
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) < min_periods:
                return np.nan

            try:
                # Ensure there's variation for histogram bins
                if np.nanmax(valid_data) == np.nanmin(valid_data):
                    # If all values are the same, density is concentrated in one bin (or 1/bins if we consider range)
                    # Let's return 1/bins as it occupies one bin.
                    return 1.0 / num_bins

                hist, _ = np.histogram(valid_data, bins=num_bins)
                populated_bins = np.sum(hist > 0)
                density = populated_bins / num_bins
                return density
            except Exception as e:
                # Handle potential numerical issues during histogram calculation
                # logger.warning(f"Could not calculate density for window: {e}") # Requires logger setup
                return np.nan

        # Apply the function over a rolling window on the 'close' prices
        # Ensure window size is at least 2 for histogram
        eff_window = max(2, window)
        min_periods_check = max(1, eff_window // 2) # Minimum valid data points needed in window

        # Use functools.partial to pass extra arguments to the apply function
        from functools import partial
        density_calculator = partial(_calculate_density_for_window, min_periods=min_periods_check, num_bins=bins)

        # Apply using raw=True for performance (passes numpy arrays)
        df['price_density'] = df['close'].rolling(window=eff_window, min_periods=min_periods_check).apply(density_calculator, raw=True)

        # Optional: Log number of NaNs created if logger is available
        # nan_count = df['price_density'].isnull().sum()
        # if nan_count > 0:
        #     logger.info(f"Price density calculation resulted in {nan_count} NaN values (likely due to insufficient window data).")

        return df

    # Removed calculate_fractal_dimension method as requested.

    @staticmethod
    def normalise_ohlc(df, window=20):
        """
        Normalise price-related columns of a dataframe using a rolling window and
        calculate additional complexity metrics

        Args:
            df: DataFrame containing OHLC and other price-related data
            window: Size of the rolling window for normalization (default: 20)

        Returns:
            DataFrame with normalized price-related values and complexity metrics
        """

        # Removed call to calculate_price_density (line 508). It's called separately in train.py now.
        # df["fractal_dimension"] = self.calculate_fractal_dimension(df, window) # Already removed
        # Calculate rolling mean and standard deviation
        rolling_mean = df["close"].rolling(window=window).mean()
        rolling_std = df["close"].rolling(window=window).std()

        result = df.copy()

        # Normalize OHLC values
        result["open"] = (df["open"] - rolling_mean) / rolling_std
        result["high"] = (df["high"] - rolling_mean) / rolling_std
        result["low"] = (df["low"] - rolling_mean) / rolling_std
        result["close"] = (df["close"] - rolling_mean) / rolling_std

        # Normalize moving averages and Bollinger Bands if they exist
        price_related_cols = [
            "sma_20",
            "sma_50",
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "lowest_low",
            "highest_high",
        ]

        for col in price_related_cols:
            if col in df.columns:
                result[col] = (df[col] - rolling_mean) / rolling_std
        return result

    def _fetch_futures_metric(
        self, api_method, params, time_col, resample_rule, cols_to_keep, value_cast=None
    ):
        """
        Generic method to fetch futures-specific metrics (funding, open interest, liquidations)
        and resample to match the main dataframe.

        Args:
            api_method: The Binance API method to call
            params: Parameters for the API method
            time_col: Name of the timestamp column
            resample_rule: String representing the resampling interval
            cols_to_keep: List of columns to keep in the result
            value_cast: Dictionary mapping column names to data types

        Returns:
            DataFrame containing the requested metrics
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

            # Use the provided resample_rule directly instead of trying to derive it
            df = df.resample(resample_rule, closed="right", label="right").ffill()
            return df[cols_to_keep]
        except Exception as e:
            logging.warning(f"Could not fetch metric with {api_method.__name__}: {e}")
            return pd.DataFrame(columns=cols_to_keep)

    def add_futures_metrics(self, df, symbol, interval, start_time, end_time):
        """
        Add futures-specific metrics: funding rates, open interest, and liquidations.

        Args:
            df: DataFrame containing OHLC data
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Data interval (e.g., '15m', '1h')
            start_time: Start time for data fetch
            end_time: End time for data fetch

        Returns:
            DataFrame with added futures-specific metrics
        """
        result = df.copy()

        # Format resample rule for pandas
        interval_map = {"m": "min", "h": "H", "d": "D"}
        freq_char = interval[-1]
        freq_unit = interval_map.get(freq_char, freq_char)
        formatted_resample_rule = interval[:-1] + freq_unit

        # Funding rates: Binance provides funding rates (typically every 8h)
        funding_params = {
            "symbol": symbol,
            "limit": 1000,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
            "interval": interval,
        }
        funding_df = self._fetch_futures_metric(
            self.client.futures_funding_rate,
            funding_params,
            time_col="fundingTime",
            resample_rule=formatted_resample_rule,
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
            "period": interval,  # Changed from 'period' to 'interval'
            "limit": 500,  # Changed from 1000 to 500
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
        }
        oi_df = self._fetch_futures_metric(
            self.client.futures_open_interest_hist,
            oi_params,
            time_col="timestamp",
            resample_rule=formatted_resample_rule,
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
            "limit": 100,  # Changed from 1000 to 100
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
        }
        liq_df = self._fetch_futures_metric(
            self.client.futures_liquidation_orders,
            liq_params,
            time_col="time",
            resample_rule=formatted_resample_rule,
            cols_to_keep=["price", "origQty"],
            value_cast={"price": float, "origQty": float},
        )
        if not liq_df.empty:
            liq_df["liquidation_value"] = liq_df["price"] * liq_df["origQty"]
            liq_resampled = (
                liq_df.resample(formatted_resample_rule, closed="right", label="right")
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

    def split_data_train_test(self, df, test_ratio=0.2, validation_ratio=0.0):
        """
        Split data into training, test, and optional validation sets.

        Args:
            df: DataFrame containing processed data
            test_ratio: Ratio of data to use for testing (default: 0.2)
            validation_ratio: Ratio of data to use for validation (default: 0.0)

        Returns:
            dict: Dictionary containing train, test, and optionally validation DataFrames
        """
        logger.info(f"Splitting data: test={test_ratio}, validation={validation_ratio}")

        # Sort by index to ensure temporal order is preserved
        df = df.sort_index()

        # Calculate split points
        total_rows = len(df)
        test_size = int(total_rows * test_ratio)
        validation_size = int(total_rows * validation_ratio)
        train_size = total_rows - test_size - validation_size

        # Perform splits
        train_df = df.iloc[:train_size]

        result = {"train": train_df}

        if test_size > 0:
            if validation_size > 0:
                # Three-way split: train, validation, test
                validation_df = df.iloc[train_size : train_size + validation_size]
                test_df = df.iloc[train_size + validation_size :]
                result["validation"] = validation_df
                result["test"] = test_df
            else:
                # Two-way split: train, test
                test_df = df.iloc[train_size:]
                result["test"] = test_df

        # Log split sizes
        for key, data in result.items():
            logger.info(
                f"{key.title()} set size: {len(data)} samples ({len(data)/total_rows:.1%})"
            )

        return result

    def process_market_data(
        self,
        symbol,
        interval="1h",
        start_time=None,
        end_time=None,
        save_path=None,
        split_data=False,
        test_ratio=0.2,
        validation_ratio=0.0,
    ):
        """
        Fetch, process, and optionally split market data for a given symbol and interval.
        This method now includes the full feature calculation pipeline needed for inference.
        """
        logger.info(f"Processing market data for {symbol} from {start_time} to {end_time}")

        # 1. Fetch raw klines data
        df = self.get_futures_data(symbol, interval, start_time, end_time)
        if df.empty:
            logger.error("Failed to fetch raw klines data.")
            return pd.DataFrame() # Return empty df if fetch fails

        # 2. Add futures-specific metrics (Funding, OI, Liquidations)
        # Note: Ensure start_time and end_time are appropriate for these metrics
        df = self.add_futures_metrics(df, symbol, interval, start_time, end_time)

        # 3. Calculate Technical Indicators
        df = DataHandler.calculate_technical_indicators(df) # Use static call

        # 4. Calculate Risk Metrics
        # Assuming window_size is needed - how to get it here? Use a default or pass it?
        # Let's use a default window size consistent with typical usage, e.g., 60 for 1m
        # TODO: Consider passing window_size as an argument if needed for consistency
        default_window_size = 60 # Example default
        df = DataHandler.calculate_risk_metrics(df, interval=interval, window_size=default_window_size) # Use static call

        # 5. Identify Trade Setups
        df = DataHandler.identify_trade_setups(df) # Use static call

        # 6. Calculate Price Density
        df = DataHandler.calculate_price_density(df, window=default_window_size) # Use static call

        # 7. Normalize OHLC and other price-related features
        df = self.normalise_ohlc(df, window=default_window_size)

        # 8. Final clean-up: Forward fill and drop NaNs introduced by calculations
        logger.info(f'Successfully updated market data with {(df.columns.tolist())} features')

        original_len = len(df)
        df = df.ffill().dropna()
        new_len = len(df)
        if original_len > new_len:
            logger.info(f"Dropped {original_len - new_len} rows containing NaNs after feature calculation.")

        if df.empty:
            logger.error("DataFrame became empty after feature calculation and dropna.")
            return pd.DataFrame()

        logger.info(f"Finished processing data. Shape: {df.shape}")

        # 9. Save data if path provided
        if save_path:
            self.save_data_to_csv(df, save_path)

        # 10. Split data if requested (less common for pure inference, but kept for consistency)
        if split_data:
            return self.split_data_train_test(df, test_ratio, validation_ratio)
        else:
            return df

    # Removed process_market_data method (lines ~744-838). # <-- This comment is now outdated
    # Feature calculation orchestration is now handled in train.py within the walk-forward loop. # <-- This comment is now outdated

    def save_data_to_csv(self, df, file_path): # Keep save method for now if needed elsewhere
        """Save DataFrame to CSV."""

        if file_path.startswith("gs://"):
            # parse bucket and blob path
            path_parts = file_path[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1] if len(path_parts) > 1 else ""

            csv_string = df.to_csv(index=False)

            # upload to GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.upload_from_string(csv_string, content_type="text/csv")
            logger.info(f"Data saved to GCS: {file_path}")

        else:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.reset_index(inplace=True)
            df.to_csv(file_path, index=False)
            logging.info(f"Data saved to {file_path}")


def parse_args():
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
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split data into training and testing sets",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Ratio of data to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.0,
        help="Ratio of data to use for validation (default: 0.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

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

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Create output filename
    base_filename = f"{args.symbol}_{args.interval}_with_metrics_3m"
    output_path = os.path.join(args.output_dir, f"{base_filename}.csv")

    # Process data with or without splitting
    if args.split:
        logging.info(
            f"Processing data with train/test split (test ratio: {args.test_ratio}, validation ratio: {args.validation_ratio})"
        )
        split_data = data_handler.process_market_data(
            symbol=args.symbol,
            interval=args.interval,
            start_time=start_date,
            end_time=end_date,
            save_path=output_path,
            split_data=True,
            test_ratio=args.test_ratio,
            validation_ratio=args.validation_ratio,
        )

        # Log split info
        for split_name, df in split_data.items():
            logging.info(f"{split_name.title()} set: {len(df)} rows")

        logging.info(f"Data processing and splitting complete.")
    else:
        # Process without splitting
        processed_data = data_handler.process_market_data(
            symbol=args.symbol,
            interval=args.interval,
            start_time=start_date,
            end_time=end_date,
            save_path=output_path,
        )

        logging.info(f"Processed data: {len(processed_data)} rows")
        logging.info(f"Data saved to {output_path}")
