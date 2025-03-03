#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import argparse
from binance.client import Client
import time
import logging
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

class DataHandler:
    """
    Class responsible for retrieving, processing, and normalizing market data
    from Binance futures.
    """

    def __init__(self):
        """
        Initialize the DataHandler.
        """
        load_dotenv()

        # Initialize Binance futures client
        self._init_binance_futures()

    def _init_binance_futures(self):
        """Initialize the Binance futures client."""
        # Get API credentials from environment
        binance_key = os.getenv("binance_future_testnet_api")
        binance_secret = os.getenv("binance_future_testnet_secret")

        if not binance_key or not binance_secret:
            raise ValueError(
                "Binance API credentials not found in environment variables.\n"
                "Please ensure you have BINANCE_API_KEY and BINANCE_SECRET_KEY "
                "set in your .env file in the project root directory."
            )

        # Initialize Binance futures client with increased timeout
        self.client = Client(api_key=binance_key, api_secret=binance_secret, requests_params={'timeout': 100})
        print("Binance Futures client initialized successfully with 100s timeout")

    def get_futures_data(self, symbols, interval="1m", start_time=None, end_time=None):
        """
        Get historical futures price data for the specified symbols using Binance.
        Implements pagination to retrieve more than the default 500 rows limit.

        Args:
            symbols (list): List of futures pairs (e.g., 'BTCUSDT')
            interval (str): The timeframe for the data (default: '1m')
            start_time (datetime, optional): The start time for the data
            end_time (datetime, optional): The end time for the data

        Returns:
            DataFrame: Historical futures price data
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=7)  # Default to 7 days for 1m data to avoid too much data
        if end_time is None:
            end_time = datetime.now()

        all_data = []

        for symbol in symbols:
            try:
                # Calculate time chunk size based on interval
                if interval == "1m":
                    chunk_size = timedelta(hours=12)  # 12-hour chunks for 1m data
                elif interval in ["3m", "5m", "15m"]:
                    chunk_size = timedelta(days=1)  # 1-day chunks for smaller timeframes
                else:
                    chunk_size = timedelta(days=7)  # 7-day chunks for larger timeframes
                
                print(f"Fetching {symbol} {interval} data from {start_time} to {end_time} in chunks...")
                
                # Initialize variables for pagination
                current_start = start_time
                symbol_data = []
                
                # Fetch data in chunks
                while current_start < end_time:
                    # Calculate end of current chunk
                    current_end = min(current_start + chunk_size, end_time)
                    
                    # Fetch data for current chunk
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval=interval,
                        startTime=int(current_start.timestamp() * 1000),
                        endTime=int(current_end.timestamp() * 1000),
                        limit=1500  # Use maximum limit allowed by Binance
                    )
                    
                    if not klines:
                        print(f"No data returned for {symbol} from {current_start} to {current_end}")
                        # Move to next chunk
                        current_start = current_end
                        continue
                    
                    print(f"Retrieved {len(klines)} klines for {symbol} from {current_start} to {current_end}")
                    
                    # Convert to DataFrame
                    df_chunk = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert timestamp to datetime
                    df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
                    
                    # Add to list of chunks
                    symbol_data.append(df_chunk)
                    
                    # Update start time for next chunk - use the last timestamp + 1ms to avoid duplicates
                    if len(klines) > 0:
                        last_timestamp = int(klines[-1][6]) + 1  # Use close_time (index 6) + 1ms
                        current_start = datetime.fromtimestamp(last_timestamp / 1000)
                    else:
                        current_start = current_end
                
                # Combine all chunks for this symbol
                if symbol_data:
                    df_symbol = pd.concat(symbol_data)
                    
                    # Remove duplicates based on timestamp
                    df_symbol = df_symbol.drop_duplicates(subset=['timestamp'])
                    
                    # Set timestamp as index
                    df_symbol.set_index('timestamp', inplace=True)
                    
                    # Convert data types
                    df_symbol = df_symbol.astype({
                        'open': 'float',
                        'high': 'float',
                        'low': 'float',
                        'close': 'float',
                        'volume': 'float',
                        'quote_asset_volume': 'float',
                        'number_of_trades': 'int',
                        'taker_buy_base_asset_volume': 'float',
                        'taker_buy_quote_asset_volume': 'float'
                    })
                    
                    # Add symbol column
                    df_symbol['symbol'] = symbol
                    
                    all_data.append(df_symbol)
                    print(f"Successfully retrieved {len(df_symbol)} total data points for {symbol}")
                else:
                    print(f"No data retrieved for {symbol}")

            except Exception as e:
                print(f"Error retrieving futures data for {symbol}: {e}")

        if not all_data:
            return pd.DataFrame()

        # Concatenate all data
        combined_data = pd.concat(all_data)
        
        # Sort by timestamp
        combined_data.sort_index(inplace=True)

        print(f"Successfully retrieved futures data for {symbols} with {len(combined_data)} total data points")
        return combined_data

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

        # Breakdown with decreased volatility
        result.loc[(result['close'] < result['lowest_low'].shift(1)) &
                  (result['signal_volatility'] == -1) &
                  (result['volume'] < result['volume'].rolling(window=20).mean() * 0.5
                   if 'volume' in result.columns else True), 'trade_setup'] = 'bearish_breakdown'

        return result

    def add_metrics(self, df):
        """
        Add various metrics to the DataFrame.

        Args:
            df (DataFrame): Price data

        Returns:
            DataFrame: Data with added metrics
        """
        result = df.copy()

        # Calculate daily returns
        result['daily_return'] = result['close'].pct_change()

        # Calculate volatility
        result['volatility'] = result['daily_return'].rolling(window=20).std()

        # Calculate Value at Risk (VaR) - 95% confidence level
        result['var_95'] = result['daily_return'].rolling(window=20).quantile(0.05)

        # Calculate Conditional VaR (CVaR) / Expected Shortfall
        def calculate_cvar(returns):
            var = returns.quantile(0.05)
            return returns[returns <= var].mean()

        result['cvar_95'] = result['daily_return'].rolling(window=20).apply(
            calculate_cvar, raw=False)

        # Calculate Maximum Drawdown
        rolling_max = result['close'].rolling(window=20, min_periods=1).max()
        drawdown = (result['close'] / rolling_max - 1.0)
        result['max_drawdown'] = drawdown.rolling(window=20).min()

        # Calculate Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        result['sharpe_ratio'] = (result['daily_return'].rolling(window=20).mean() /
                                   result['daily_return'].rolling(window=20).std()) * np.sqrt(252)

        # Calculate Sortino Ratio (only considers downside risk)
        downside_returns = result['daily_return'].copy()
        downside_returns[downside_returns > 0] = 0
        result['sortino_ratio'] = (result['daily_return'].rolling(window=20).mean() /
                                    downside_returns.rolling(window=20).std()) * np.sqrt(252)

        # Calculate Calmar Ratio (return / maximum drawdown)
        result['calmar_ratio'] = (result['daily_return'].rolling(window=20).mean() * 252 /
                                   result['max_drawdown'].abs())

        return result

    def process_market_data(self, symbol, interval="1m", start_time=None, end_time=None, save_path=None):
        """
        Comprehensive function to retrieve, process and add metrics to market data.
        Works with Binance futures data and automatically applies all metrics.

        Args:
            symbol (str): Trading symbol (e.g., "BTCUSDT")
            interval (str): Data interval ("1m" for 1 minute, "1h" for hourly, "1d" for daily)
            start_time (datetime): Start time for data, default is 7 days ago for 1m data
            end_time (datetime): End time for data, default is current time
            save_path (str): Optional path to save the processed data

        Returns:
            DataFrame: Processed data with all metrics
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            # For 1m data, limit to 7 days to avoid excessive data
            if interval == "1m":
                start_time = end_time - timedelta(days=7)
            else:
                start_time = end_time - timedelta(days=60)

        print(f"Retrieving {symbol} data from {start_time} to {end_time} with interval {interval}")

        # Get data
        df = self.get_futures_data([symbol], interval=interval, start_time=start_time, end_time=end_time)

        # Check if we have data
        if df.empty:
            raise ValueError(f"Could not retrieve data for {symbol}")

        print(f"Data retrieved, shape: {df.shape}")

        # Add all metrics
        # 1. Calculate technical indicators
        print("Calculating technical indicators...")
        df_with_indicators = self.calculate_technical_indicators(df)

        # 2. Add futures-specific metrics (funding rates, open interest, liquidations)
        print("Adding futures-specific metrics...")
        df_with_futures_metrics = self.add_futures_metrics(df_with_indicators, symbol, interval, start_time, end_time)

        # 3. Calculate risk metrics
        print("Calculating risk metrics...")
        df_with_risk = self.calculate_risk_metrics(df_with_futures_metrics)

        # 4. Identify trade setups
        print("Identifying trade setups...")
        df_with_setups = self.identify_trade_setups(df_with_risk)

        # 5. Handle NaN values
        print("Handling missing values...")
        # Forward-fill NaNs from rolling windows
        df_filled = df_with_setups.ffill()
        # Backward-fill any remaining NaNs
        df_filled = df_filled.bfill()

        # Save if path provided
        if save_path:
            self.save_data_to_csv(df_filled, save_path)

        return df_filled

    def save_data_to_csv(self, df, file_path):
        """
        Save DataFrame to CSV file.

        Args:
            df (DataFrame): DataFrame to save
            file_path (str): Path to save the file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Reset index if it's not already a column
        if df.index.name is not None:
            df = df.reset_index()

        # Save to CSV
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

    def load_data_from_csv(self, file_path):
        """
        Load data from CSV file.

        Args:
            file_path (str): Path to the CSV file

        Returns:
            DataFrame: Loaded data
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load data
        df = pd.read_csv(file_path)

        # If 'timestamp' column exists, set it as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        return df

    def get_funding_rates(self, symbol, start_time=None, end_time=None, limit=500):
        """
        Get historical funding rates for a futures symbol.
        
        Args:
            symbol (str): Futures symbol (e.g., 'BTCUSDT')
            start_time (datetime, optional): Start time for data retrieval
            end_time (datetime, optional): End time for data retrieval
            limit (int, optional): Maximum number of records to retrieve (default: 500, max: 1000)
            
        Returns:
            DataFrame: Historical funding rates data
        """
        try:
            params = {'symbol': symbol, 'limit': limit}
            
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
                
            funding_data = self.client.futures_funding_rate(**params)
            
            if not funding_data:
                print(f"No funding rate data retrieved for {symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(funding_data)
            
            # Convert timestamp to datetime
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            
            # Convert string columns to numeric
            df['fundingRate'] = df['fundingRate'].astype(float)
            if 'markPrice' in df.columns:
                df['markPrice'] = df['markPrice'].astype(float)
                
            # Set fundingTime as index
            df.set_index('fundingTime', inplace=True)
            
            print(f"Successfully retrieved {len(df)} funding rate records for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error retrieving funding rate data for {symbol}: {e}")
            return pd.DataFrame()
            
    def get_open_interest(self, symbol, interval="1h", start_time=None, end_time=None, limit=500):
        """
        Get historical open interest data for a futures symbol.
        
        Args:
            symbol (str): Futures symbol (e.g., 'BTCUSDT')
            interval (str): Data interval (e.g., '5m', '1h', '1d')
            start_time (datetime, optional): Start time for data retrieval
            end_time (datetime, optional): End time for data retrieval
            limit (int, optional): Maximum number of records to retrieve (default: 500, max: 1000)
            
        Returns:
            DataFrame: Historical open interest data
        """
        try:
            params = {
                'symbol': symbol, 
                'period': interval,
                'limit': limit
            }
            
            # Only add startTime if both start_time and end_time are provided
            # This avoids the "parameter 'startTime' is invalid" error
            if start_time and end_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
                params['endTime'] = int(end_time.timestamp() * 1000)
            elif end_time:  # If only end_time is provided
                params['endTime'] = int(end_time.timestamp() * 1000)
                # Calculate a default start_time (30 days before end_time)
                default_start = end_time - timedelta(days=30)
                params['startTime'] = int(default_start.timestamp() * 1000)
                
            open_interest_data = self.client.futures_open_interest_hist(**params)
            
            if not open_interest_data:
                print(f"No open interest data retrieved for {symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(open_interest_data)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert string columns to numeric
            df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
            df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            print(f"Successfully retrieved {len(df)} open interest records for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error retrieving open interest data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_liquidations(self, symbol=None, start_time=None, end_time=None, limit=500):
        """
        Get recent liquidation orders from Binance Futures.
        Note: This uses the futures_liquidation_orders endpoint which provides recent liquidations.
        For real-time liquidations, a WebSocket connection would be required.
        
        Args:
            symbol (str, optional): Futures symbol (e.g., 'BTCUSDT')
            start_time (datetime, optional): Start time for data retrieval
            end_time (datetime, optional): End time for data retrieval
            limit (int, optional): Maximum number of records to retrieve (default: 500, max: 1000)
            
        Returns:
            DataFrame: Recent liquidation orders
        """
        try:
            # Create an empty DataFrame as fallback for API permission issues
            empty_df = pd.DataFrame(columns=['time', 'symbol', 'side', 'price', 'origQty', 'liquidation_value'])
            empty_df.set_index('time', inplace=True)
            
            # Check if we have API permissions for this endpoint
            # Many Binance API keys don't have permissions for liquidation data
            try:
                # Test call with minimal parameters
                test_params = {'limit': 1}
                if symbol:
                    test_params['symbol'] = symbol
                self.client.futures_liquidation_orders(**test_params)
            except BinanceAPIException as e:
                if "API-key" in str(e) or "permissions" in str(e):
                    print(f"Your API key doesn't have permissions for liquidation data. Using empty DataFrame.")
                    return empty_df
                raise e
                
            # If we passed the permission check, proceed with the actual request
            params = {'limit': limit}
            
            if symbol:
                params['symbol'] = symbol
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
                
            liquidation_data = self.client.futures_liquidation_orders(**params)
            
            if not liquidation_data:
                print(f"No liquidation data retrieved{' for ' + symbol if symbol else ''}")
                return empty_df
                
            # Convert to DataFrame
            df = pd.DataFrame(liquidation_data)
            
            if df.empty:
                return empty_df
                
            # Convert timestamp to datetime
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            
            # Convert string columns to numeric
            df['price'] = df['price'].astype(float)
            df['origQty'] = df['origQty'].astype(float)
            df['avgPrice'] = df['avgPrice'].astype(float)
            
            # Calculate liquidation value in USD
            df['liquidation_value'] = df['price'] * df['origQty']
            
            # Set time as index
            df.set_index('time', inplace=True)
            
            print(f"Successfully retrieved {len(df)} liquidation records{' for ' + symbol if symbol else ''}")
            return df
            
        except Exception as e:
            print(f"Error retrieving liquidation data{' for ' + symbol if symbol else ''}: {e}")
            # Return empty DataFrame with correct columns
            empty_df = pd.DataFrame(columns=['time', 'symbol', 'side', 'price', 'origQty', 'liquidation_value'])
            empty_df.set_index('time', inplace=True)
            return empty_df
            
    def add_futures_metrics(self, df, symbol, interval="1m", start_time=None, end_time=None):
        """
        Add futures-specific metrics to the dataframe.
        
        Args:
            df (DataFrame): Price dataframe
            symbol (str): Futures symbol (e.g., 'BTCUSDT')
            interval (str): Data interval (e.g., '5m', '1h', '1d')
            start_time (datetime, optional): Start time for data retrieval
            end_time (datetime, optional): End time for data retrieval
            
        Returns:
            DataFrame: Price dataframe with added futures metrics
        """
        result = df.copy()
        
        # Get funding rates
        try:
            funding_df = self.get_funding_rates(symbol, start_time, end_time)
            
            # Add funding rates to the main dataframe
            if not funding_df.empty:
                # Resample funding rates to match the main dataframe's interval
                # Funding rates occur every 8 hours, we'll forward fill to have a value for each candle
                funding_df = funding_df.resample(interval, closed='right', label='right').ffill()
                
                # Join with main dataframe
                result = result.join(funding_df[['fundingRate']], how='left')
                
                # Forward fill funding rates (they remain constant until the next funding event)
                result['fundingRate'] = result['fundingRate'].ffill()
                
                # Calculate cumulative funding rate
                result['cumulative_funding'] = result['fundingRate'].cumsum()
            else:
                # If no funding rate data is available, add a default column with zeros
                print("No funding rate data available, adding default column with zeros")
                result['fundingRate'] = 0.0
                result['cumulative_funding'] = 0.0
        except Exception as e:
            # If there's an error getting funding rates, add default columns
            print(f"Error retrieving funding rate data: {e}. Adding default columns with zeros.")
            result['fundingRate'] = 0.0
            result['cumulative_funding'] = 0.0
        
        # Map price intervals to open interest intervals
        # Binance only provides open interest at specific intervals
        oi_interval_map = {
            '1m': '5m', '3m': '5m', '5m': '5m', '15m': '15m', '30m': '1h',
            '1h': '1h', '2h': '4h', '4h': '4h', '6h': '8h', '8h': '8h',
            '12h': '1d', '1d': '1d'
        }
        
        # Get open interest data
        try:
            oi_interval = oi_interval_map.get(interval, '1h')
            open_interest_df = self.get_open_interest(symbol, interval=oi_interval, start_time=start_time, end_time=end_time)
            
            # Add open interest to the main dataframe
            if not open_interest_df.empty:
                # Resample open interest to match the main dataframe's interval
                open_interest_df = open_interest_df.resample(interval, closed='right', label='right').ffill()
                
                # Join with main dataframe
                result = result.join(open_interest_df[['sumOpenInterest', 'sumOpenInterestValue']], how='left')
                
                # Forward fill open interest values
                result['sumOpenInterest'] = result['sumOpenInterest'].ffill()
                result['sumOpenInterestValue'] = result['sumOpenInterestValue'].ffill()
                
                # Calculate open interest change
                result['oi_change'] = result['sumOpenInterest'].pct_change()
                
                # Calculate open interest ratio (OI / Volume)
                result['oi_volume_ratio'] = result['sumOpenInterest'] / result['volume'].replace(0, np.nan)
            else:
                # If no open interest data is available, add default columns with zeros
                print("No open interest data available, adding default columns with zeros")
                result['sumOpenInterest'] = 0.0
                result['sumOpenInterestValue'] = 0.0
                result['oi_change'] = 0.0
                result['oi_volume_ratio'] = 0.0
        except Exception as e:
            # If there's an error getting open interest, add default columns
            print(f"Error retrieving open interest data: {e}. Adding default columns with zeros.")
            result['sumOpenInterest'] = 0.0
            result['sumOpenInterestValue'] = 0.0
            result['oi_change'] = 0.0
            result['oi_volume_ratio'] = 0.0
        
        # Get liquidation data
        try:
            liquidation_df = self.get_liquidations(symbol, start_time, end_time)
            
            # Add liquidation data to the main dataframe
            if not liquidation_df.empty:
                # Resample liquidation data to match the main dataframe's interval
                liquidation_resampled = liquidation_df.resample(interval, closed='right', label='right').agg({
                    'liquidation_value': 'sum',
                    'origQty': 'sum'
                }).fillna(0)
                
                # Join with main dataframe
                result = result.join(liquidation_resampled, how='left')
                
                # Fill NaN values with 0
                result['liquidation_value'] = result['liquidation_value'].fillna(0)
                result['origQty'] = result['origQty'].fillna(0)
                
                # Calculate liquidation intensity (liquidation value / volume)
                result['liquidation_intensity'] = result['liquidation_value'] / result['volume'].replace(0, np.nan)
            else:
                # If no liquidation data is available, add default columns with zeros
                print("No liquidation data available, adding default columns with zeros")
                result['liquidation_value'] = 0.0
                result['origQty'] = 0.0
                result['liquidation_intensity'] = 0.0
        except Exception as e:
            # If there's an error getting liquidation data, add default columns
            print(f"Error retrieving liquidation data: {e}. Adding default columns with zeros.")
            result['liquidation_value'] = 0.0
            result['origQty'] = 0.0
            result['liquidation_intensity'] = 0.0
        
        return result

if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Data processing utility for trading bot')
    parser.add_argument('--action', type=str, required=True,
                       choices=['fetch', 'metrics', 'all'],
                       help='Action to perform: fetch data only, add metrics only, or all')

    # Data source and symbol parameters
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Symbol to fetch data for (e.g., BTCUSDT, ETHUSDT)')
    parser.add_argument('--interval', type=str, default='1m',
                       choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                       help='Data interval (1m for 1 minute, 1h for hourly, 1d for daily)')

    # Date range parameters
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days of historical data to fetch')
    parser.add_argument('--start_date', type=str, default=None,
                       help='Start date in YYYY-MM-DD format (overrides days parameter if provided)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date in YYYY-MM-DD format (defaults to today)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Directory to save the processed data')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Input file for adding metrics to existing data')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine date range
    end_date = datetime.now()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    start_date = end_date - timedelta(days=args.days)
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')

    # Generate output filename
    interval_str = args.interval
    symbol_str = args.symbol

    # Initialize DataHandler
    data_handler = DataHandler()

    if args.action == 'fetch' or args.action == 'all':
        # Fetch and optionally process data
        output_file = f"{args.output_dir}/{symbol_str}_{interval_str}"
        if args.action == 'all':
            output_file += "_with_metrics.csv"
        else:
            output_file += "_raw.csv"

        print(f"Fetching futures data for {args.symbol} from {start_date.date()} to {end_date.date()} with interval {args.interval}")

        try:
            if args.action == 'all':
                # Process data with all metrics
                processed_data = data_handler.process_market_data(
                    symbol=args.symbol,
                    interval=args.interval,
                    start_time=start_date,
                    end_time=end_date,
                    save_path=output_file
                )
                print(f"\nSuccessfully processed {len(processed_data)} rows of {args.symbol} data")
                print(f"Data with metrics saved to {output_file}")
            else:
                # Just fetch raw data
                df = data_handler.get_futures_data(
                    symbols=[args.symbol],
                    interval=args.interval,
                    start_time=start_date,
                    end_time=end_date
                )

                if not df.empty:
                    data_handler.save_data_to_csv(df, output_file)
                    print(f"\nSuccessfully fetched {len(df)} rows of {args.symbol} data")
                    print(f"Raw data saved to {output_file}")
                else:
                    print(f"No data retrieved for {args.symbol}")

        except Exception as e:
            print(f"Error processing data: {e}")

    elif args.action == 'metrics':
        # Add metrics to existing data
        if args.input_file is None:
            print("Error: --input_file is required when action is 'metrics'")
            exit(1)

        if not os.path.exists(args.input_file):
            print(f"Error: Input file {args.input_file} does not exist")
            exit(1)

        output_file = f"{args.output_dir}/{symbol_str}_{interval_str}_with_metrics.csv"

        print(f"Adding metrics to data from {args.input_file}")

        try:
            # Load data
            df = data_handler.load_data_from_csv(args.input_file)

            # Add metrics
            print("Calculating technical indicators...")
            df_with_indicators = data_handler.calculate_technical_indicators(df)

            print("Calculating risk metrics...")
            df_with_risk = data_handler.calculate_risk_metrics(df_with_indicators)

            print("Identifying trade setups...")
            df_with_setups = data_handler.identify_trade_setups(df_with_risk)

            # Handle NaN values
            print("Handling missing values...")
            # Forward-fill NaNs from rolling windows
            df_filled = df_with_setups.ffill()
            # Backward-fill any remaining NaNs
            df_filled = df_filled.bfill()

            # Save processed data
            data_handler.save_data_to_csv(df_filled, output_file)
            print(f"\nSuccessfully added metrics to {len(df_filled)} rows of data")
            print(f"Processed data saved to {output_file}")

        except Exception as e:
            print(f"Error adding metrics to data: {e}")
