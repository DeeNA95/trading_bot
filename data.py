#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import yfinance as yf

class DataHandler:
    """
    Class responsible for retrieving, processing, and normalizing market data
    from various sources including Alpaca (stocks and crypto) and Yahoo Finance.
    """

    def __init__(self, data_source="alpaca"):
        """
        Initialize the DataHandler with the specified data source.

        Args:
            data_source (str): The data source to use ('alpaca', 'yahoo')
        """
        self.data_source = data_source
        load_dotenv()

        # Initialize connections based on data source
        if data_source == "alpaca":
            self._init_alpaca()
        elif data_source == "yahoo":
            # No specific initialization needed for Yahoo Finance
            pass
        else:
            raise ValueError(f"Unsupported data source: {data_source}")

    def _init_alpaca(self):
        """Initialize the Alpaca data clients for stocks and crypto."""
        # Get API credentials from environment
        alpaca_key = os.getenv("ALPACA_KEY")
        alpaca_secret = os.getenv("ALPACA_SECRET_KEY")

        if not alpaca_key or not alpaca_secret:
            raise ValueError("Alpaca API credentials not found in environment variables")

        # Initialize stock data client (requires API keys)
        self.stock_client = StockHistoricalDataClient(alpaca_key, alpaca_secret)

        # Initialize crypto data client (no API keys required for crypto data)
        self.crypto_client = CryptoHistoricalDataClient()

    def get_stock_data(self, symbols, timeframe=TimeFrame.Day, start_date=None, end_date=None, limit=100):
        """
        Get historical stock price data for the specified symbols using Alpaca.

        Args:
            symbols (list): List of stock ticker symbols
            timeframe (TimeFrame): The timeframe for the data (default: Daily)
            start_date (datetime, optional): The start date for the data
            end_date (datetime, optional): The end date for the data
            limit (int): Maximum number of data points to retrieve

        Returns:
            DataFrame: Historical stock price data
        """
        if self.data_source != "alpaca":
            return self.get_yahoo_data(symbols, start_date, end_date)

        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            limit=limit
        )

        try:
            bars = self.stock_client.get_stock_bars(request_params)
            df = bars.df

            if len(symbols) == 1:
                # If only one symbol, remove the multi-index
                df = df.reset_index(level=0, drop=True)

            print(f"Successfully retrieved stock data for {symbols}")
            return df
        except Exception as e:
            print(f"Error retrieving stock data: {e}")
            # Fallback to Yahoo Finance if there's an error with Alpaca
            print("Falling back to Yahoo Finance...")
            return self.get_yahoo_data(symbols, start_date, end_date)

    def get_crypto_data(self, symbols, timeframe=TimeFrame.Day, start_date=None, end_date=None, limit=100):
        """
        Get historical cryptocurrency price data for the specified symbols using Alpaca.

        Args:
            symbols (list): List of crypto pairs (e.g., 'BTC/USD')
            timeframe (TimeFrame): The timeframe for the data (default: Daily)
            start_date (datetime, optional): The start date for the data
            end_date (datetime, optional): The end date for the data
            limit (int): Maximum number of data points to retrieve

        Returns:
            DataFrame: Historical crypto price data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        # Format crypto symbols correctly if needed (e.g., 'BTC' -> 'BTC/USD')
        formatted_symbols = []
        for symbol in symbols:
            if '/' not in symbol:
                formatted_symbols.append(f"{symbol}/USD")
            else:
                formatted_symbols.append(symbol)

        request_params = CryptoBarsRequest(
            symbol_or_symbols=formatted_symbols,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            limit=limit
        )

        try:
            bars = self.crypto_client.get_crypto_bars(request_params)
            df = bars.df

            if len(formatted_symbols) == 1:
                # If only one symbol, remove the multi-index
                df = df.reset_index(level=0, drop=True)

            print(f"Successfully retrieved crypto data for {formatted_symbols}")
            return df
        except Exception as e:
            print(f"Error retrieving crypto data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def get_yahoo_data(self, symbols, start_date=None, end_date=None):
        """
        Get historical data from Yahoo Finance.

        Args:
            symbols (list): List of ticker symbols
            start_date (datetime, optional): The start date for the data
            end_date (datetime, optional): The end date for the data

        Returns:
            DataFrame: Historical price data from Yahoo Finance
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        try:
            if len(symbols) == 1:
                # Single symbol
                ticker = yf.Ticker(symbols[0])
                df = ticker.history(start=start_date, end=end_date)
                # Rename columns to match Alpaca's format
                df.columns = [col.lower() for col in df.columns]
                # Ensure index is named 'timestamp' to match Alpaca's format
                df.index.name = 'timestamp'
                print(f"Successfully retrieved Yahoo Finance data for {symbols[0]}")
                return df
            else:
                # Multiple symbols
                data = yf.download(symbols, start=start_date, end=end_date)
                # Rename columns to match Alpaca's format
                data.columns = [col.lower() if isinstance(col, str) else (col[0].lower(), col[1].lower())
                               for col in data.columns]
                # Ensure index is named 'timestamp' to match Alpaca's format
                data.index.name = 'timestamp'
                print(f"Successfully retrieved Yahoo Finance data for {symbols}")
                return data
        except Exception as e:
            print(f"Error retrieving Yahoo Finance data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def save_data_to_csv(self, df, filename):
        """
        Save the DataFrame to a CSV file.

        Args:
            df (DataFrame): Data to save
            filename (str): Filename to save to
        """
        # Ensure the index is named 'timestamp'
        df_to_save = df.copy()
        if df_to_save.index.name != 'timestamp':
            df_to_save.index.name = 'timestamp'

        df_to_save.to_csv(filename)
        print(f"Data saved to {filename}")

    def load_data_from_csv(self, filename):
        """
        Load data from a CSV file.

        Args:
            filename (str): Filename to load from

        Returns:
            DataFrame: Loaded data
        """
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        # Ensure the index is named 'timestamp'
        df.index.name = 'timestamp'
        print(f"Data loaded from {filename}")
        return df

    def merge_data_sources(self, symbol, sources=None, start_date=None, end_date=None, timeframe=TimeFrame.Day):
        """
        Merge data for the same symbol from multiple sources.

        Args:
            symbol (str): The symbol to retrieve data for
            sources (list, optional): List of data sources to use ['alpaca', 'yahoo']. If None, uses all available.
            start_date (datetime, optional): The start date for the data
            end_date (datetime, optional): The end date for the data
            timeframe (TimeFrame): The timeframe for the data (default: Daily)

        Returns:
            DataFrame: Merged data from all sources with source indicators
        """
        if sources is None:
            sources = ['alpaca', 'yahoo']

        all_data = {}

        for source in sources:
            # Store original data source
            original_source = self.data_source

            try:
                # Temporarily switch data source
                self.data_source = source

                # Get data based on source and asset type
                if '/' in symbol:  # Crypto
                    if source == 'alpaca':
                        df = self.get_crypto_data([symbol], timeframe=timeframe,
                                                start_date=start_date, end_date=end_date)
                    else:
                        # Yahoo Finance uses different symbols for crypto
                        yahoo_symbol = symbol.split('/')[0] + '-' + symbol.split('/')[1]
                        df = self.get_yahoo_data([yahoo_symbol], start_date=start_date, end_date=end_date)
                else:  # Stock
                    df = self.get_stock_data([symbol], timeframe=timeframe,
                                           start_date=start_date, end_date=end_date)

                if not df.empty:
                    # Ensure index is named 'timestamp'
                    if df.index.name != 'timestamp':
                        df.index.name = 'timestamp'

                    # Add source column
                    df['data_source'] = source
                    all_data[source] = df
                    print(f"Retrieved {symbol} data from {source}")

            except Exception as e:
                print(f"Error retrieving {symbol} data from {source}: {e}")

            finally:
                # Restore original data source
                self.data_source = original_source

        if not all_data:
            print(f"No data retrieved for {symbol} from any source")
            return pd.DataFrame()

        # Merge data from all sources
        # Start with data from the first source
        merged_data = None
        source_priority = [s for s in sources if s in all_data]

        if not source_priority:
            return pd.DataFrame()

        # Use the first available source as base
        merged_data = all_data[source_priority[0]].copy()
        merged_data['source_priority'] = 1  # Priority 1 (highest)

        # Add data from other sources with lower priority
        for i, source in enumerate(source_priority[1:], 2):
            if source in all_data:
                source_data = all_data[source].copy()
                source_data['source_priority'] = i

                # Concatenate and sort by date
                merged_data = pd.concat([merged_data, source_data])

        # Sort by date and handle duplicates by keeping the highest priority source
        merged_data = merged_data.sort_index()

        # If there are duplicate indices (dates), keep the one with highest priority (lowest number)
        merged_data = merged_data.reset_index()
        # Ensure the column is named 'timestamp'
        if 'timestamp' not in merged_data.columns and 'index' in merged_data.columns:
            merged_data = merged_data.rename(columns={'index': 'timestamp'})

        merged_data = merged_data.sort_values(['timestamp', 'source_priority'])
        merged_data = merged_data.drop_duplicates('timestamp', keep='first')
        merged_data = merged_data.set_index('timestamp')

        # Drop the priority column
        merged_data = merged_data.drop('source_priority', axis=1)

        print(f"Successfully merged {symbol} data from {', '.join(source_priority)}")
        return merged_data

    def merge_multi_asset_data(self, symbols, columns=None, start_date=None, end_date=None,
                              timeframe=TimeFrame.Day, fill_method='ffill'):
        """
        Merge data for multiple assets into a single DataFrame.

        Args:
            symbols (list): List of symbols to retrieve data for
            columns (list, optional): List of columns to include. If None, includes all common columns.
            start_date (datetime, optional): The start date for the data
            end_date (datetime, optional): The end date for the data
            timeframe (TimeFrame): The timeframe for the data (default: Daily)
            fill_method (str): Method to fill missing values ('ffill', 'bfill', None)

        Returns:
            DataFrame: Merged data for all symbols
        """
        all_data = {}

        for symbol in symbols:
            try:
                # Determine if it's a crypto or stock symbol
                if '/' in symbol:  # Crypto
                    df = self.get_crypto_data([symbol], timeframe=timeframe,
                                            start_date=start_date, end_date=end_date)
                else:  # Stock
                    df = self.get_stock_data([symbol], timeframe=timeframe,
                                           start_date=start_date, end_date=end_date)

                if not df.empty:
                    # If specific columns are requested, filter the DataFrame
                    if columns:
                        available_cols = [col for col in columns if col in df.columns]
                        df = df[available_cols]

                    # Store the data
                    all_data[symbol] = df
                    print(f"Retrieved data for {symbol}")

            except Exception as e:
                print(f"Error retrieving data for {symbol}: {e}")

        if not all_data:
            print("No data retrieved for any symbol")
            return pd.DataFrame()

        # Create a multi-level column DataFrame
        merged_data = None

        for symbol, df in all_data.items():
            # Create a copy with multi-level columns
            symbol_df = df.copy()

            # Create MultiIndex columns with symbol as the first level
            symbol_df.columns = pd.MultiIndex.from_product([[symbol], symbol_df.columns])

            if merged_data is None:
                merged_data = symbol_df
            else:
                # Join with existing data
                merged_data = merged_data.join(symbol_df, how='outer')

        # Fill missing values if requested
        if fill_method:
            merged_data = merged_data.fillna(method=fill_method)

        print(f"Successfully merged data for {len(all_data)} symbols")
        return merged_data

    def align_timeframes(self, df, target_timeframe, method='ohlc'):
        """
        Resample data to a different timeframe.

        Args:
            df (DataFrame): Price data
            target_timeframe (str): Target timeframe as pandas offset string
                                   ('D' for daily, 'W' for weekly, 'M' for monthly, etc.)
            method (str): Resampling method ('ohlc' for OHLC bars, 'last' for last value)

        Returns:
            DataFrame: Resampled data
        """
        if df.empty:
            return df

        # Make sure the index is a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            print("DataFrame index is not a DatetimeIndex. Cannot resample.")
            return df

        # Resample based on the specified method
        if method == 'ohlc':
            # For OHLC data
            resampled = df.resample(target_timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum' if 'volume' in df.columns else None
            })

            # Remove None values from the aggregation dictionary
            resampled = resampled.dropna(axis=1, how='all')

            # For other columns, use the last value
            other_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            if other_cols:
                other_resampled = df[other_cols].resample(target_timeframe).last()
                resampled = pd.concat([resampled, other_resampled], axis=1)

        elif method == 'last':
            # Use the last value for all columns
            resampled = df.resample(target_timeframe).last()

            # Sum volume if it exists
            if 'volume' in df.columns:
                resampled['volume'] = df['volume'].resample(target_timeframe).sum()

        else:
            print(f"Unsupported resampling method: {method}")
            return df

        print(f"Data resampled to {target_timeframe} timeframe")
        return resampled

    def combine_with_external_data(self, df, external_data, join_column=None, how='left'):
        """
        Combine market data with external data sources (e.g., economic indicators, sentiment data).

        Args:
            df (DataFrame): Market price data
            external_data (DataFrame): External data to combine
            join_column (str, optional): Column to join on. If None, joins on index.
            how (str): Join method ('left', 'right', 'inner', 'outer')

        Returns:
            DataFrame: Combined data
        """
        if df.empty or external_data.empty:
            print("One of the DataFrames is empty. Cannot combine.")
            return df

        try:
            # If join_column is specified, join on that column
            if join_column:
                if join_column in df.columns and join_column in external_data.columns:
                    result = pd.merge(df, external_data, on=join_column, how=how)
                else:
                    print(f"Join column '{join_column}' not found in both DataFrames")
                    return df
            else:
                # Join on index
                result = df.join(external_data, how=how)

            print("Successfully combined with external data")
            return result

        except Exception as e:
            print(f"Error combining with external data: {e}")
            return df


    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for the given data.

        Args:
            df (DataFrame): Price data

        Returns:
            DataFrame: Price data with technical indicators
        """
        # Make a copy to avoid modifying the original
        result = df.copy()

        # Calculate Simple Moving Averages
        result['sma_20'] = result['close'].rolling(window=20).mean()
        result['sma_50'] = result['close'].rolling(window=50).mean()

        # Calculate Relative Strength Index (RSI)
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        result['rsi'] = 100 - (100 / (1 + rs))

        # Calculate Bollinger Bands
        result['bb_middle'] = result['close'].rolling(window=20).mean()
        result['bb_std'] = result['close'].rolling(window=20).std()
        result['bb_upper'] = result['bb_middle'] + 2 * result['bb_std']
        result['bb_lower'] = result['bb_middle'] - 2 * result['bb_std']

        # Calculate MACD
        result['ema_12'] = result['close'].ewm(span=12, adjust=False).mean()
        result['ema_26'] = result['close'].ewm(span=26, adjust=False).mean()
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()

        # Calculate Stochastic Oscillator
        n = 14  # Standard lookback period
        result['lowest_low'] = result['low'].rolling(window=n).min()
        result['highest_high'] = result['high'].rolling(window=n).max()
        result['stoch_k'] = 100 * ((result['close'] - result['lowest_low']) /
                                  (result['highest_high'] - result['lowest_low']))
        result['stoch_d'] = result['stoch_k'].rolling(window=3).mean()

        # Calculate Average True Range (ATR)
        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift())
        low_close = np.abs(result['low'] - result['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        result['atr'] = true_range.rolling(14).mean()

        # Calculate Average Directional Index (ADX)
        # True Range already calculated above
        # Plus Directional Movement (+DM)
        plus_dm = result['high'].diff()
        minus_dm = result['low'].diff().multiply(-1)
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        # Minus Directional Movement (-DM)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

        # Smooth the +DM, -DM and TR with Wilder's smoothing
        n = 14
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/n, adjust=False).mean() / result['atr']
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/n, adjust=False).mean() / result['atr']

        # Directional Index (DX)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

        # Average Directional Index (ADX)
        result['adx'] = pd.Series(dx).ewm(alpha=1/n, adjust=False).mean()
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di

        return result

    def preprocess_data_for_ml(self, df):
        """
        Preprocess data for machine learning.

        Args:
            df (DataFrame): Price data with indicators

        Returns:
            DataFrame: Processed data ready for ML models
        """
        # Calculate returns
        df['returns'] = df['close'].pct_change()

        # Normalize price-based features
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[f'{col}_norm'] = df[col] / df['close'].shift(1) - 1

        # Log transform volume
        if 'volume' in df.columns:
            df['volume_norm'] = df['volume'] / df['volume'].rolling(window=20).mean()

        # Handle NaN values
        df = df.dropna()

        return df

    def calculate_risk_metrics(self, df, window=20):
        """
        Calculate risk metrics for the given data.

        Args:
            df (DataFrame): Price data with indicators
            window (int): Window size for calculations

        Returns:
            DataFrame: Data with risk metrics
        """
        result = df.copy()

        # Calculate daily returns
        result['daily_return'] = result['close'].pct_change()

        # Volatility (standard deviation of returns)
        result['volatility'] = result['daily_return'].rolling(window=window).std()

        # Value at Risk (VaR) - 95% confidence level
        # Represents the maximum expected loss over a given time period
        result['var_95'] = result['daily_return'].rolling(window=window).quantile(0.05)

        # Conditional VaR (CVaR) / Expected Shortfall
        # Average loss when the loss exceeds VaR
        def calculate_cvar(returns):
            var = returns.quantile(0.05)
            return returns[returns <= var].mean()

        result['cvar_95'] = result['daily_return'].rolling(window=window).apply(
            calculate_cvar, raw=False)

        # Maximum Drawdown
        # Calculate rolling maximum
        rolling_max = result['close'].rolling(window=window, min_periods=1).max()
        # Calculate drawdown
        drawdown = (result['close'] / rolling_max - 1.0)
        # Calculate maximum drawdown
        result['max_drawdown'] = drawdown.rolling(window=window).min()

        # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        result['sharpe_ratio'] = (result['daily_return'].rolling(window=window).mean() /
                                 result['daily_return'].rolling(window=window).std()) * np.sqrt(252)

        # Sortino Ratio (only considers downside risk)
        downside_returns = result['daily_return'].copy()
        downside_returns[downside_returns > 0] = 0
        result['sortino_ratio'] = (result['daily_return'].rolling(window=window).mean() /
                                  downside_returns.rolling(window=window).std()) * np.sqrt(252)

        # Calmar Ratio (return / maximum drawdown)
        result['calmar_ratio'] = (result['daily_return'].rolling(window=window).mean() * 252 /
                                 result['max_drawdown'].abs())

        return result

    def identify_trade_setups(self, df):
        """
        Identify potential trade setups based on technical indicators.

        Args:
            df (DataFrame): Price data with technical indicators

        Returns:
            DataFrame: Data with trade setup signals
        """
        result = df.copy()

        # Ensure we have the necessary indicators
        if not all(indicator in result.columns for indicator in
                  ['sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
                   'stoch_k', 'stoch_d', 'adx', 'plus_di', 'minus_di']):
            result = self.calculate_technical_indicators(result)

        # Initialize signal columns
        result['signal_trend'] = 0  # 1 for bullish, -1 for bearish, 0 for neutral
        result['signal_momentum'] = 0
        result['signal_volatility'] = 0
        result['signal_strength'] = 0
        result['trade_setup'] = 'none'

        # Trend signals
        # Moving Average Crossover
        result.loc[result['sma_20'] > result['sma_50'], 'signal_trend'] = 1
        result.loc[result['sma_20'] < result['sma_50'], 'signal_trend'] = -1

        # Momentum signals
        # RSI overbought/oversold
        result.loc[result['rsi'] < 30, 'signal_momentum'] = 1  # Oversold, potential buy
        result.loc[result['rsi'] > 70, 'signal_momentum'] = -1  # Overbought, potential sell

        # MACD crossover
        result.loc[result['macd'] > result['macd_signal'], 'signal_momentum'] = 1
        result.loc[result['macd'] < result['macd_signal'], 'signal_momentum'] = -1

        # Stochastic crossover and overbought/oversold
        result.loc[(result['stoch_k'] < 20) & (result['stoch_k'] > result['stoch_d']), 'signal_momentum'] = 1
        result.loc[(result['stoch_k'] > 80) & (result['stoch_k'] < result['stoch_d']), 'signal_momentum'] = -1

        # Trend strength signals
        # ADX indicates trend strength (not direction)
        result.loc[result['adx'] > 25, 'signal_strength'] = 1  # Strong trend
        result.loc[result['adx'] < 20, 'signal_strength'] = -1  # Weak trend

        # Directional movement
        result.loc[result['plus_di'] > result['minus_di'], 'signal_trend'] = 1
        result.loc[result['plus_di'] < result['minus_di'], 'signal_trend'] = -1

        # Volatility signals based on ATR
        if 'atr' in result.columns:
            # Calculate ATR as percentage of price
            result['atr_pct'] = result['atr'] / result['close'] * 100
            # High volatility
            result.loc[result['atr_pct'] > result['atr_pct'].rolling(window=20).mean() * 1.5, 'signal_volatility'] = 1
            # Low volatility
            result.loc[result['atr_pct'] < result['atr_pct'].rolling(window=20).mean() * 0.5, 'signal_volatility'] = -1

        # Identify specific trade setups

        # Bullish setups
        # Strong uptrend with momentum confirmation
        result.loc[(result['signal_trend'] == 1) &
                  (result['signal_momentum'] == 1) &
                   (result['signal_strength'] == 1), 'trade_setup'] = 'strong_bullish'

        # Bullish reversal
        result.loc[(result['signal_trend'] == -1) &
                  (result['signal_momentum'] == 1) &
                   (result['rsi'] < 30), 'trade_setup'] = 'bullish_reversal'

        # Breakout with increased volatility
        result.loc[(result['close'] > result['highest_high'].shift(1)) &
                  (result['signal_volatility'] == 1) &
                  (result['volume'] > result['volume'].rolling(window=20).mean() * 1.5
                   if 'volume' in result.columns else True), 'trade_setup'] = 'bullish_breakout'

        # Bearish setups
        # Strong downtrend with momentum confirmation
        result.loc[(result['signal_trend'] == -1) &
                  (result['signal_momentum'] == -1) &
                   (result['signal_strength'] == 1), 'trade_setup'] = 'strong_bearish'

        # Bearish reversal
        result.loc[(result['signal_trend'] == 1) &
                  (result['signal_momentum'] == -1) &
                   (result['rsi'] > 70), 'trade_setup'] = 'bearish_reversal'

        # Breakdown with increased volatility
        result.loc[(result['close'] < result['lowest_low'].shift(1)) &
                  (result['signal_volatility'] == 1) &
                  (result['volume'] > result['volume'].rolling(window=20).mean() * 1.5
                   if 'volume' in result.columns else True), 'trade_setup'] = 'bearish_breakdown'

        # Range-bound setups
        # Low volatility, weak trend - potential range trading
        result.loc[(result['signal_strength'] == -1) &
                   (result['signal_volatility'] == -1), 'trade_setup'] = 'range_bound'

        # Calculate potential risk/reward for each setup
        result['potential_reward'] = np.nan
        result['potential_risk'] = np.nan
        result['risk_reward_ratio'] = np.nan

        # For bullish setups
        bullish_setups = result['trade_setup'].isin(['strong_bullish', 'bullish_reversal', 'bullish_breakout'])
        if 'atr' in result.columns:
            # Target is typically 2-3 ATRs for reward
            result.loc[bullish_setups, 'potential_reward'] = result['close'] + 2.5 * result['atr']
            # Stop loss is typically 1 ATR for risk
            result.loc[bullish_setups, 'potential_risk'] = result['close'] - 1 * result['atr']

        # For bearish setups
        bearish_setups = result['trade_setup'].isin(['strong_bearish', 'bearish_reversal', 'bearish_breakdown'])
        if 'atr' in result.columns:
            # Target is typically 2-3 ATRs for reward
            result.loc[bearish_setups, 'potential_reward'] = result['close'] - 2.5 * result['atr']
            # Stop loss is typically 1 ATR for risk
            result.loc[bearish_setups, 'potential_risk'] = result['close'] + 1 * result['atr']

        # Calculate risk/reward ratio where applicable
        valid_setups = result['trade_setup'] != 'none'
        result.loc[valid_setups, 'risk_reward_ratio'] = (
            (result.loc[valid_setups, 'potential_reward'] - result.loc[valid_setups, 'close']).abs() /
            (result.loc[valid_setups, 'potential_risk'] - result.loc[valid_setups, 'close']).abs()
        )

        return result

    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for the given data.

        Args:
            df (DataFrame): Price data

        Returns:
            DataFrame: Price data with technical indicators
        """
        # Make a copy to avoid modifying the original
        result = df.copy()

        # Calculate Simple Moving Averages
        result['sma_20'] = result['close'].rolling(window=20).mean()
        result['sma_50'] = result['close'].rolling(window=50).mean()

        # Calculate Relative Strength Index (RSI)
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        result['rsi'] = 100 - (100 / (1 + rs))

        # Calculate Bollinger Bands
        result['bb_middle'] = result['close'].rolling(window=20).mean()
        result['bb_std'] = result['close'].rolling(window=20).std()
        result['bb_upper'] = result['bb_middle'] + 2 * result['bb_std']
        result['bb_lower'] = result['bb_middle'] - 2 * result['bb_std']

        # Calculate MACD
        result['ema_12'] = result['close'].ewm(span=12, adjust=False).mean()
        result['ema_26'] = result['close'].ewm(span=26, adjust=False).mean()
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()

        # Calculate Stochastic Oscillator
        n = 14  # Standard lookback period
        result['lowest_low'] = result['low'].rolling(window=n).min()
        result['highest_high'] = result['high'].rolling(window=n).max()
        result['stoch_k'] = 100 * ((result['close'] - result['lowest_low']) /
                                    (result['highest_high'] - result['lowest_low']))
        result['stoch_d'] = result['stoch_k'].rolling(window=3).mean()

        # Calculate Average True Range (ATR)
        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift())
        low_close = np.abs(result['low'] - result['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        result['atr'] = true_range.rolling(14).mean()

        # Calculate Average Directional Index (ADX)
        # True Range already calculated above
        # Plus Directional Movement (+DM)
        plus_dm = result['high'].diff()
        minus_dm = result['low'].diff().multiply(-1)
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        # Minus Directional Movement (-DM)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

        # Smooth the +DM, -DM and TR with Wilder's smoothing
        n = 14
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/n, adjust=False).mean() / result['atr']
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/n, adjust=False).mean() / result['atr']

        # Directional Index (DX)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

        # Average Directional Index (ADX)
        result['adx'] = pd.Series(dx).ewm(alpha=1/n, adjust=False).mean()
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di

        return result

    def get_all_time_btc_data(self):
        """
        Get all-time BTC/USD data from Yahoo Finance and save it to a CSV file.

        Returns:
            DataFrame: All-time BTC/USD data
        """
        # Fetch all-time BTC/USD data from Yahoo Finance
        btc_data = self.get_yahoo_data(['BTC-USD'], start_date='2010-07-17', end_date=datetime.now())

        # Save the data to a CSV file
        self.save_data_to_csv(btc_data, 'data/BTC_USD_all_time.csv')

        return btc_data

    def preprocess_data_for_ml(self, df):
        """
        Preprocess data for machine learning.

        Args:
            df (DataFrame): Price data with indicators

        Returns:
            DataFrame: Processed data ready for ML models
        """
        # Calculate returns
        df['returns'] = df['close'].pct_change()

        # Normalize price-based features
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[f'{col}_norm'] = df[col] / df['close'].shift(1) - 1

        # Log transform volume
        if 'volume' in df.columns:
            df['volume_norm'] = df['volume'] / df['volume'].rolling(window=20).mean()

        # Handle NaN values
        df = df.dropna()

        return df

    def save_data_to_csv(self, df, filename):
        """
        Save the DataFrame to a CSV file.

        Args:
            df (DataFrame): Data to save
            filename (str): Filename to save to
        """
        # Ensure the index is named 'timestamp'
        df_to_save = df.copy()
        if df_to_save.index.name != 'timestamp':
            df_to_save.index.name = 'timestamp'

        df_to_save.to_csv(filename)
        print(f"Data saved to {filename}")

    def load_data_from_csv(self, filename):
        """
        Load data from a CSV file.

        Args:
            filename (str): Filename to load from

        Returns:
            DataFrame: Loaded data
        """
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        # Ensure the index is named 'timestamp'
        df.index.name = 'timestamp'
        print(f"Data loaded from {filename}")
        return df

    def merge_data_sources(self, symbol, sources=None, start_date=None, end_date=None, timeframe=TimeFrame.Day):
        """
        Merge data for the same symbol from multiple sources.

        Args:
            symbol (str): The symbol to retrieve data for
            sources (list, optional): List of data sources to use ['alpaca', 'yahoo']. If None, uses all available.
            start_date (datetime, optional): The start date for the data
            end_date (datetime, optional): The end date for the data
            timeframe (TimeFrame): The timeframe for the data (default: Daily)

        Returns:
            DataFrame: Merged data from all sources with source indicators
        """
        if sources is None:
            sources = ['alpaca', 'yahoo']

        all_data = {}

        for source in sources:
            # Store original data source
            original_source = self.data_source

            try:
                # Temporarily switch data source
                self.data_source = source

                # Get data based on source and asset type
                if '/' in symbol:  # Crypto
                    if source == 'alpaca':
                        df = self.get_crypto_data([symbol], timeframe=timeframe,
                                                start_date=start_date, end_date=end_date)
                    else:
                        # Yahoo Finance uses different symbols for crypto
                        yahoo_symbol = symbol.split('/')[0] + '-' + symbol.split('/')[1]
                        df = self.get_yahoo_data([yahoo_symbol], start_date=start_date, end_date=end_date)
                else:  # Stock
                    df = self.get_stock_data([symbol], timeframe=timeframe,
                                           start_date=start_date, end_date=end_date)

                if not df.empty:
                    # Ensure index is named 'timestamp'
                    if df.index.name != 'timestamp':
                        df.index.name = 'timestamp'

                    # Add source column
                    df['data_source'] = source
                    all_data[source] = df
                    print(f"Retrieved {symbol} data from {source}")

            except Exception as e:
                print(f"Error retrieving {symbol} data from {source}: {e}")

            finally:
                # Restore original data source
                self.data_source = original_source

        if not all_data:
            print(f"No data retrieved for {symbol} from any source")
            return pd.DataFrame()

        # Merge data from all sources
        # Start with data from the first source
        merged_data = None
        source_priority = [s for s in sources if s in all_data]

        if not source_priority:
            return pd.DataFrame()

        # Use the first available source as base
        merged_data = all_data[source_priority[0]].copy()
        merged_data['source_priority'] = 1  # Priority 1 (highest)

        # Add data from other sources with lower priority
        for i, source in enumerate(source_priority[1:], 2):
            if source in all_data:
                source_data = all_data[source].copy()
                source_data['source_priority'] = i

                # Concatenate and sort by date
                merged_data = pd.concat([merged_data, source_data])

        # Sort by date and handle duplicates by keeping the highest priority source
        merged_data = merged_data.sort_index()

        # If there are duplicate indices (dates), keep the one with highest priority (lowest number)
        merged_data = merged_data.reset_index()
        # Ensure the column is named 'timestamp'
        if 'timestamp' not in merged_data.columns and 'index' in merged_data.columns:
            merged_data = merged_data.rename(columns={'index': 'timestamp'})

        merged_data = merged_data.sort_values(['timestamp', 'source_priority'])
        merged_data = merged_data.drop_duplicates('timestamp', keep='first')
        merged_data = merged_data.set_index('timestamp')

        # Drop the priority column
        merged_data = merged_data.drop('source_priority', axis=1)

        print(f"Successfully merged {symbol} data from {', '.join(source_priority)}")
        return merged_data

    def merge_multi_asset_data(self, symbols, columns=None, start_date=None, end_date=None,
                              timeframe=TimeFrame.Day, fill_method='ffill'):
        """
        Merge data for multiple assets into a single DataFrame.

        Args:
            symbols (list): List of symbols to retrieve data for
            columns (list, optional): List of columns to include. If None, includes all common columns.
            start_date (datetime, optional): The start date for the data
            end_date (datetime, optional): The end date for the data
            timeframe (TimeFrame): The timeframe for the data (default: Daily)
            fill_method (str): Method to fill missing values ('ffill', 'bfill', None)

        Returns:
            DataFrame: Merged data for all symbols
        """
        all_data = {}

        for symbol in symbols:
            try:
                # Determine if it's a crypto or stock symbol
                if '/' in symbol:  # Crypto
                    df = self.get_crypto_data([symbol], timeframe=timeframe,
                                            start_date=start_date, end_date=end_date)
                else:  # Stock
                    df = self.get_stock_data([symbol], timeframe=timeframe,
                                           start_date=start_date, end_date=end_date)

                if not df.empty:
                    # If specific columns are requested, filter the DataFrame
                    if columns:
                        available_cols = [col for col in columns if col in df.columns]
                        df = df[available_cols]

                    # Store the data
                    all_data[symbol] = df
                    print(f"Retrieved data for {symbol}")

            except Exception as e:
                print(f"Error retrieving data for {symbol}: {e}")

        if not all_data:
            print("No data retrieved for any symbol")
            return pd.DataFrame()

        # Create a multi-level column DataFrame
        merged_data = None

        for symbol, df in all_data.items():
            # Create a copy with multi-level columns
            symbol_df = df.copy()

            # Create MultiIndex columns with symbol as the first level
            symbol_df.columns = pd.MultiIndex.from_product([[symbol], symbol_df.columns])

            if merged_data is None:
                merged_data = symbol_df
            else:
                # Join with existing data
                merged_data = merged_data.join(symbol_df, how='outer')

        # Fill missing values if requested
        if fill_method:
            merged_data = merged_data.fillna(method=fill_method)

        print(f"Successfully merged data for {len(all_data)} symbols")
        return merged_data

    def align_timeframes(self, df, target_timeframe, method='ohlc'):
        """
        Resample data to a different timeframe.

        Args:
            df (DataFrame): Price data
            target_timeframe (str): Target timeframe as pandas offset string
                                   ('D' for daily, 'W' for weekly, 'M' for monthly, etc.)
            method (str): Resampling method ('ohlc' for OHLC bars, 'last' for last value)

        Returns:
            DataFrame: Resampled data
        """
        if df.empty:
            return df

        # Make sure the index is a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            print("DataFrame index is not a DatetimeIndex. Cannot resample.")
            return df

        # Resample based on the specified method
        if method == 'ohlc':
            # For OHLC data
            resampled = df.resample(target_timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum' if 'volume' in df.columns else None
            })

            # Remove None values from the aggregation dictionary
            resampled = resampled.dropna(axis=1, how='all')

            # For other columns, use the last value
            other_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            if other_cols:
                other_resampled = df[other_cols].resample(target_timeframe).last()
                resampled = pd.concat([resampled, other_resampled], axis=1)

        elif method == 'last':
            # Use the last value for all columns
            resampled = df.resample(target_timeframe).last()

            # Sum volume if it exists
            if 'volume' in df.columns:
                resampled['volume'] = df['volume'].resample(target_timeframe).sum()

        else:
            print(f"Unsupported resampling method: {method}")
            return df

        print(f"Data resampled to {target_timeframe} timeframe")
        return resampled

    def combine_with_external_data(self, df, external_data, join_column=None, how='left'):
        """
        Combine market data with external data sources (e.g., economic indicators, sentiment data).

        Args:
            df (DataFrame): Market price data
            external_data (DataFrame): External data to combine
            join_column (str, optional): Column to join on. If None, joins on index.
            how (str): Join method ('left', 'right', 'inner', 'outer')

        Returns:
            DataFrame: Combined data
        """
        if df.empty or external_data.empty:
            print("One of the DataFrames is empty. Cannot combine.")
            return df

        try:
            # If join_column is specified, join on that column
            if join_column:
                if join_column in df.columns and join_column in external_data.columns:
                    result = pd.merge(df, external_data, on=join_column, how=how)
                else:
                    print(f"Join column '{join_column}' not found in both DataFrames")
                    return df
            else:
                # Join on index
                result = df.join(external_data, how=how)

            print("Successfully combined with external data")
            return result

        except Exception as e:
            print(f"Error combining with external data: {e}")
            return df


if __name__ == "__main__":
    # Example usage
    data_handler = DataHandler(data_source="alpaca")

    try:
        # Get historical stock data (last 200 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=200)

        # List of assets to retrieve
        stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        cryptos = ["BTC/USD", "ETH/USD", "SOL/USD"]

        # Retrieve stock data
        print("\n===== Retrieving Stock Data =====")
        for stock in stocks:
            stock_data = data_handler.get_stock_data([stock],
                                                   timeframe=TimeFrame.Day,
                                                   start_date=start_date,
                                                   end_date=end_date)

            if not stock_data.empty:
                # Save raw data
                data_handler.save_data_to_csv(stock_data, f"data/{stock}_raw.csv")

                # Calculate indicators and save
                stock_data_with_indicators = data_handler.calculate_technical_indicators(stock_data)
                data_handler.save_data_to_csv(stock_data_with_indicators, f"data/{stock}_indicators.csv")

                # Print sample
                print(f"\nSample data for {stock}:")
                print(stock_data.head(3))

        # Retrieve crypto data
        print("\n===== Retrieving Crypto Data =====")
        for crypto in cryptos:
            crypto_data = data_handler.get_crypto_data([crypto],
                                                     timeframe=TimeFrame.Day,
                                                     start_date=start_date,
                                                     end_date=end_date)

            if not crypto_data.empty:
                # Save raw data
                data_handler.save_data_to_csv(crypto_data, f"data/{crypto.replace('/', '_')}_raw.csv")

                # Calculate indicators and save
                crypto_data_with_indicators = data_handler.calculate_technical_indicators(crypto_data)
                data_handler.save_data_to_csv(crypto_data_with_indicators,
                                             f"data/{crypto.replace('/', '_')}_indicators.csv")

                # Print sample
                print(f"\nSample data for {crypto}:")
                print(crypto_data.head(3))

    except Exception as e:
        print(f"Error in main: {e}")

    # Example of using the new data merging methods
    try:
        print("\n===== Testing Data Merging Methods =====")

        # Example 1: Merge data from multiple sources for a single stock
        print("\nMerging data from multiple sources for AAPL:")
        merged_aapl = data_handler.merge_data_sources("AAPL",
                                                     sources=["alpaca", "yahoo"],
                                                     start_date=start_date,
                                                     end_date=end_date)
        if not merged_aapl.empty:
            print("Sample of merged AAPL data:")
            print(merged_aapl.head(3))
            print(f"Data sources in merged data: {merged_aapl['data_source'].unique()}")

        # Example 2: Merge data for multiple stocks
        print("\nMerging data for multiple stocks:")
        multi_stock = data_handler.merge_multi_asset_data(
            symbols=["AAPL", "MSFT", "GOOGL"],
            columns=["close", "volume"],
            start_date=start_date,
            end_date=end_date
        )
        if not multi_stock.empty:
            print("Sample of multi-stock data:")
            print(multi_stock.head(3))

        # Example 3: Align timeframes (convert daily to weekly data)
        print("\nResampling AAPL data from daily to weekly:")
        if not merged_aapl.empty:
            weekly_data = data_handler.align_timeframes(merged_aapl, 'W', method='ohlc')
            print("Sample of weekly AAPL data:")
            print(weekly_data.head(3))

    except Exception as e:
        print(f"Error testing data merging methods: {e}")
