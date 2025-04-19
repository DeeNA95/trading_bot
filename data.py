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
            PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "future-linker-456622-f8")

            BINANCE_KEY_response = self.gcloud_client.access_secret_version(
                name=f"projects/{PROJECT_ID}/secrets/BINANCE_API/versions/latest"
            )
            BINANCE_KEY = BINANCE_KEY_response.payload.data.decode("UTF-8").strip()

            BINANCE_SECRET_response = self.gcloud_client.access_secret_version(
                name=f"projects/{PROJECT_ID}/secrets/BINANCE_SECRET/versions/latest"
            )
            BINANCE_SECRET = BINANCE_SECRET_response.payload.data.decode(
                "UTF-8"
            ).strip()

            logger.info(
                "Successfully retrieved Binance credentials from Google Secret Manager"
            )

        except Exception as e:
            logger.error(f"Could not retrieve from Google Secret Manager: {e}")
            load_dotenv()
            BINANCE_KEY = os.getenv("BINANCE_KEY")
            BINANCE_SECRET = os.getenv("BINANCE_SECRET")
            logger.info("Falling back to .env file for Binance credentials")

        if not BINANCE_KEY or not BINANCE_SECRET:
            raise ValueError(
                "Binance API credentials not found in environment variables. "
                "Ensure that BINANCE_KEY and BINANCE_SECRET are set in your .env file."
            )

        # init binance client
        self.client = Client(
            api_key=BINANCE_KEY,
            api_secret=BINANCE_SECRET,
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
            sleep(0.1)

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

    def calculate_risk_metrics(self, df, interval="1h", window_size=60):
        """
        Calculate risk-related metrics: volatility, VaR, CVaR, max drawdown,
        Sharpe, Sortino, and Calmar ratios.
        """
        # Determine window size based on interval
        periods_per_day = self.get_periods_per_day(interval)

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
            * (365 * 24 / self.get_periods_per_day(interval))
        ) / result["max_drawdown"].abs()

        # Calculate efficiency ratio
        risk_free_rate = 0.01  # Example risk-free rate, can be adjusted
        result["efficiency_ratio"] = (
            result["timely_return"].rolling(window=window).mean() / risk_free_rate
        ) * annualization_factor

        logger.info("Calculating risk metrics...")
        return result

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

    def calculate_price_density(self, df, window=20, bins=10):
        """
        Calculate price density using a rolling window

        Args:
            df: DataFrame containing OHLC data
            window: Size of the rolling window (default: 20)
            bins: Number of price bins for density calculation (default: 10)

        Returns:
            Series with price density values
        """
        # Calculate price density for the entire DataFrame
        result = pd.Series(index=df.index, dtype=float)
        close_values = df["close"].values

        # For each position in the DataFrame
        for i in range(len(df)):
            # Skip positions that don't have enough data for the window
            if i < window:
                result.iloc[i] = np.nan
                continue

            # Extract the window of data
            window_data = close_values[i - window : i]

            # Skip if there's not enough data
            if len(window_data) < window / 2:
                result.iloc[i] = np.nan
                continue

            # Create histogram of prices in the window
            try:
                hist, _ = np.histogram(window_data, bins=bins)
                # Calculate density as the proportion of populated bins
                populated_bins = np.sum(hist > 0)
                result.iloc[i] = populated_bins / bins
            except Exception:
                # Handle any numerical issues
                result.iloc[i] = np.nan

        return result

    def calculate_fractal_dimension(self, df, window=20):
        """
        Calculate fractal dimension of price movements using box-counting method

        Args:
            df: DataFrame containing OHLC data
            window: Size of the rolling window (default: 20)

        Returns:
            Series with fractal dimension values
        """
        # Calculate fractal dimension for the entire DataFrame
        result = pd.Series(index=df.index, dtype=float)
        close_values = df["close"].values

        # For each position in the DataFrame
        for i in range(len(df)):
            # Skip positions that don't have enough data for the window
            if i < window:
                result.iloc[i] = np.nan
                continue

            # Extract the window of data
            window_data = close_values[i - window : i]

            # Skip if there's not enough data
            if len(window_data) < window / 2:
                result.iloc[i] = np.nan
                continue

            # Normalize the window data to [0,1] range
            min_val = np.min(window_data)
            max_val = np.max(window_data)
            if max_val == min_val:
                result.iloc[i] = 1.0  # Flat line has dimension 1
                continue

            norm_data = (window_data - min_val) / (max_val - min_val)

            # Use box-counting with multiple scales
            scales = [2, 4, 8, 16]
            log_counts = []
            log_scales = []

            for scale in scales:
                if scale >= len(norm_data):
                    continue

                # Count boxes at this scale
                box_count = 0
                for j in range(0, scale):
                    box_min = j / scale
                    box_max = (j + 1) / scale
                    # Check if any price falls within this box
                    if np.any((norm_data >= box_min) & (norm_data < box_max)):
                        box_count += 1

                if box_count > 0:
                    log_counts.append(np.log(box_count))
                    log_scales.append(np.log(scale))

            # Need at least 2 valid scales to calculate slope
            if len(log_counts) < 2:
                result.iloc[i] = np.nan
                continue

            # Linear regression to find the fractal dimension
            try:
                slope, _, _, _, _ = np.polyfit(log_scales, log_counts, 1, full=True)
                result.iloc[i] = slope
            except Exception:
                # Handle any numerical issues
                result.iloc[i] = np.nan

        return result

    def normalise_ohlc(self, df, window=20):
        """
        Normalise price-related columns of a dataframe using a rolling window and
        calculate additional complexity metrics

        Args:
            df: DataFrame containing OHLC and other price-related data
            window: Size of the rolling window for normalization (default: 20)

        Returns:
            DataFrame with normalized price-related values and complexity metrics
        """

        df["price_density"] = self.calculate_price_density(df, window)
        # df["fractal_dimension"] = self.calculate_fractal_dimension(df, window)
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
        Comprehensive function to retrieve futures data, add all metrics (technical,
        risk, and futures-specific), and fill missing values.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Data interval (e.g., '15m', '1h')
            start_time: Start time for data collection (default: depends on interval)
            end_time: End time for data collection (default: current time)
            save_path: Path to save processed data (optional)
            split_data: Whether to split data into training and test sets (default: False)
            test_ratio: Ratio of data to use for testing (default: 0.2)
            validation_ratio: Ratio of data to use for validation (default: 0.0)

        Returns:
            If split_data is False: DataFrame containing processed data
            If split_data is True: Dictionary containing train, test, and optionally validation DataFrames
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = (
                end_time - timedelta(days=7)
                if interval == "1m"
                else end_time - timedelta(days=60)
            )
        # Fetch raw data
        df = self.get_futures_data(
            symbol, interval, start_time, end_time
        )  # Pass start_time and end_time correctly
        if df.empty:
            logging.warning(f"No data retrieved for {symbol}.")
            raise ValueError(f"No data retrieved for {symbol}.")


        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)

        # Add futures-specific metrics
        df = self.add_futures_metrics(
            df, symbol, interval, start_time, end_time
        )  # Pass end_time correctly

        # Calculate risk metrics
        df = self.calculate_risk_metrics(df, interval)
        logging.debug(f"Data after calculating risk metrics:\n{df.head()}")

        # Identify trade setups
        df = self.identify_trade_setups(df)
        logging.debug(f"Data after identifying trade setups:\n{df.head()}")

        # Normalize OHLC data
        df = self.normalise_ohlc(df)
        logging.debug(f"Data after normalizing OHLC:\n{df.head()}")

        # Fill missing values
        df = df.ffill().bfill()
        logging.debug(f"Data after filling missing values:\n{df.head()}")

        # Check for NaN values
        if df.isnull().values.any():
            logging.warning(f"NaN values found in data:\n{df.isnull().sum()}")

        if save_path:
            # Base filename without extension
            base_path = save_path.rsplit(".", 1)[0] if "." in save_path else save_path

            if split_data:
                # Split the data
                split_sets = self.split_data_train_test(
                    df, test_ratio, validation_ratio
                )

                # Save each split
                for split_name, split_df in split_sets.items():
                    split_path = f"{base_path}_{split_name}.csv"
                    self.save_data_to_csv(split_df, split_path)
                    logging.info(f"{split_name.title()} data saved to {split_path}")

                # Return the split datasets
                return split_sets
            else:
                # Save the whole dataset
                self.save_data_to_csv(df, save_path)
                logging.info(f"Data with metrics saved to {save_path}")

                # Return the full dataset
                return df

        # If no save path is provided, just return the model data
        if split_data:
            return self.split_data_train_test(df, test_ratio, validation_ratio)
        else:
            return df

    def save_data_to_csv(self, df, file_path):
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
