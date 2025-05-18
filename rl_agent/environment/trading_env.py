"""
Reinforcement learning trading environment for Binance Futures.

This module provides a gym-compatible environment for training reinforcement learning
agents to trade on Binance Futures. It supports both backtesting on historical data
and live trading with real-time market data.
"""

import json
import logging
import math
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import requests
import ta
from binance.um_futures import UMFutures
from dotenv import load_dotenv
from gymnasium import spaces

from .position_sizer import (BinanceFuturesPositionSizer,
                             VolatilityAdjustedPositionSizer)
# Import local modules
from .reward import FuturesRiskAdjustedReward
from .utils import calculate_adaptive_leverage # Added import

logger = logging.getLogger(__name__)
class BinanceFuturesCryptoEnv(gym.Env):
    """
    Enhanced cryptocurrency trading environment for Binance Futures BTCUSDT.

    Provides a gym environment that allows RL agents to interact with cryptocurrency
    data for training and live trading on Binance Futures.

    Attributes:
        df: DataFrame containing OHLCV data and indicators
        max_position: Maximum allowed position size
        leverage: Trading leverage
        trade_fee_percent: Trading fee percentage (taker fee)
        mode: Running mode (train or trade)
        dynamic_leverage: Whether to adjust leverage based on volatility
        use_risk_adjusted_rewards: Whether to use the FuturesRiskAdjustedReward system
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame = None,
        window_size: int = 30,
        max_position: int = 1,
        leverage: int = 2,
        max_leverage: int = 20,
        trade_fee_percent: float = 0.0004,  # 0.04% Binance taker fee
        initial_balance: float = 10000,
        indicators: List[str] = None,
        symbol: str = "BTCUSDT",
        mode: str = "train",
        base_url: str = "https://fapi.binance.com",
        margin_type: str = "ISOLATED",
        risk_reward_ratio: float = 1.5,
        stop_loss_percent: float = 0.01,
        dynamic_leverage: bool = True,
        use_risk_adjusted_rewards: bool = True,
        funding_rate_weight: float = 0.05,
        liquidation_penalty_weight: float = 1.75,
        open_interest_weight: float = 0.1,
        volatility_lookback: int = 24,  # Hours for volatility calculation
        data_fetch_interval: str = "15m",  # Interval for fetching market data
        include_funding_rate: bool = True,
        include_open_interest: bool = True,
        include_liquidation_data: bool = True,
        dry_run: bool = False,  # When True, don't execute actual trades
        slippage_fraction: float = 0.1, # Fraction of bar range (H-L) to use as slippage
        render_mode: str = None,
    ):
        """
                Initialize the Binance Futures crypto trading environment.

        > #file:`rl_agent/agent/ppo_agent.py`
                Args:
                    df: DataFrame containing OHLCV data (required for train mode)
                    window_size: The number of steps to include in the observation
                    max_position: Maximum allowed position size (1 = 100% of balance)
                    leverage: Trading leverage (1-125)
                    max_leverage: Maximum allowed leverage
                    trade_fee_percent: Trading fee percentage
                    initial_balance: Initial account balance in USDT
                    indicators: Technical indicators to use
                    symbol: Trading pair symbol (default: BTCUSDT)
                    mode: Running mode ('train' for backtesting, 'trade' for live trading)
                    base_url: Binance API URL (None for mainnet, or testnet URL)
                    margin_type: Margin type ('ISOLATED' or 'CROSSED')
                    risk_reward_ratio: Ratio of take profit to stop loss
                    stop_loss_percent: Stop loss percentage from entry
                    dynamic_leverage: Whether to adjust leverage based on volatility
                    use_risk_adjusted_rewards: Whether to use the FuturesRiskAdjustedReward system
                    funding_rate_weight: Weight for funding rate in reward calculation
                    liquidation_penalty_weight: Weight for liquidation risk in reward calculation
                    open_interest_weight: Weight for open interest changes in reward calculation
                    volatility_lookback: Hours to look back for volatility calculation
                    data_fetch_interval: Interval for fetching market data
                    include_funding_rate: Whether to include funding rate data
                    include_open_interest: Whether to include open interest data
                    include_liquidation_data: Whether to include liquidation data
                    dry_run: When True, don't execute actual trades
                    render_mode: Rendering mode for gym env
        """
        super().__init__()

        # Store environment parameters
        self.df = df
        self.window_size = window_size
        self.commission = trade_fee_percent
        self.initial_balance = initial_balance
        self.render_mode = render_mode

        # Initialize state with balance
        self.balance = self.initial_balance
        self.current_step = 0
        self.date = None

        # Set Binance Futures specific attributes
        self.symbol = symbol
        self.leverage = leverage
        self.max_leverage = max_leverage
        self.max_position = max_position
        self.mode = mode
        self.margin_type = margin_type
        self.risk_reward_ratio = risk_reward_ratio
        self.stop_loss_percent = stop_loss_percent

        # Enhanced features
        self.dynamic_leverage = dynamic_leverage
        self.use_risk_adjusted_rewards = use_risk_adjusted_rewards
        self.funding_rate_weight = funding_rate_weight
        self.liquidation_penalty_weight = liquidation_penalty_weight
        self.open_interest_weight = open_interest_weight
        self.volatility_lookback = volatility_lookback
        self.data_fetch_interval = data_fetch_interval
        self.include_funding_rate = include_funding_rate
        self.include_open_interest = include_open_interest
        self.include_liquidation_data = include_liquidation_data
        self.slippage_fraction = slippage_fraction # Store slippage fraction

        # Initialize position sizer
        self.position_sizer = BinanceFuturesPositionSizer(
            max_position_pct=self.max_position,
            position_sizer=VolatilityAdjustedPositionSizer(base_risk_pct=0.1),
            default_leverage=self.leverage,
            max_leverage=self.max_leverage,
            dynamic_leverage=self.dynamic_leverage,
        )

        # Advanced data storage
        self.funding_rates = []
        self.open_interest_history = []
        self.liquidation_data = []
        self.volatility_history = []
        self.current_funding_rate = 0.0
        self.current_open_interest = 0.0
        self.open_interest_change = 0.0
        self.current_volatility = 0.0

        # Initialize reward calculator
        if self.use_risk_adjusted_rewards:
            self.reward_calculator = FuturesRiskAdjustedReward(
                leverage_penalty=0.02,
                drawdown_penalty=0.2,
                liquidation_penalty=self.liquidation_penalty_weight,
                funding_rate_penalty=self.funding_rate_weight,
                max_leverage=self.max_leverage,
            )

        # Set the trading mode
        self.live_trading = mode == "trade"
        self.dry_run = dry_run

        # Setup Binance connection for live trading
        if self.live_trading:
            load_dotenv()
            use_testnet = base_url is None
            self.api_key = os.getenv(
                "binance_future_testnet_api" if use_testnet else "binance_api2"
            )
            self.api_secret = os.getenv(
                "binance_future_testnet_secret" if use_testnet else "binance_secret2"
            )

            self.client = UMFutures(
                key=self.api_key, secret=self.api_secret, base_url=base_url
            )

            # Initialize execution module
            from .execution import BinanceFuturesExecutor

            self.executor = BinanceFuturesExecutor(
                client=self.client,
                symbol=self.symbol,
                leverage=self.leverage,
                margin_type=self.margin_type,
                risk_reward_ratio=self.risk_reward_ratio,
                stop_loss_percent=self.stop_loss_percent,
                dry_run=self.dry_run,
            )

            # Position status tracking
            self.position_status = {
                "has_open_position": False,
                "sl_triggered": False,
                "tp_triggered": False,
            }

            # Add trade history tracking
            self.trade_history = []

            # Initialize Binance account
            self._init_binance_account()

        # Action and observation spaces
        # 0: Hold, 1: Buy/Long, 2: Sell/Short
        self.action_space = spaces.Discrete(3)

        # State space: OHLCV + indicators + account features (position, equity, etc.)
        if df is not None:
            state_dim = df.shape[1]  # Features from df only
        else:
            # If no df provided (like in live mode), use default dimension
            state_dim = 25  # OHLCV + common indicators + account features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, state_dim),
            dtype=np.float32,
        )

        # Track trading metrics
        self.trades = []
        self.current_position = 0
        self.entry_price = 0
        self.unrealized_pnl = 0
        self.max_account_value = self.initial_balance

        # Stop loss and take profit levels
        self.stop_loss = None
        self.take_profit = None

        # Create initial state
        self._create_initial_state()

    def _create_initial_state(self):
        """Create initial state for the environment."""
        if self.df is not None:
            # For backtesting, use the first window_size rows of the dataframe
            self.current_step = self.window_size - 1
            data = self.df.iloc[: self.window_size].copy()

            # Extract features
            price_features = data.values

            # Add account information
            balance = np.ones(self.window_size) * self.balance
            position = np.zeros(self.window_size)
            unrealized_pnl = np.zeros(self.window_size)

            # Combine into state and ensure it's float32
            # combined_data = np.column_stack(
            #     (
            #         price_features,
            #         # balance.reshape(-1, 1),
            #         # position.reshape(-1, 1),
            #         # unrealized_pnl.reshape(-1, 1),
            #     )
            # )
            combined_data = price_features
            # Attempt conversion, drop problematic columns if it fails
            try:
                self.state = combined_data.astype(np.float32)
            except ValueError as e:
                logger.warning(f"Initial state conversion failed: {e}. Attempting to drop problematic columns...")
                # TODO: Fix root cause of non-numeric data (e.g., 'none' string in 'trade_setup') upstream.
                # This block is a temporary workaround.

                import pandas as pd # Import locally for this block
                df_check = pd.DataFrame(combined_data)
                is_numeric = df_check.apply(lambda s: pd.to_numeric(s, errors='coerce').notna().all())
                problematic_cols_indices = [i for i, numeric in enumerate(is_numeric) if not numeric]

                if problematic_cols_indices:
                    logger.warning(f"Dropping non-numeric columns at indices: {problematic_cols_indices}")
                    combined_data_cleaned = np.delete(combined_data, problematic_cols_indices, axis=1)
                    try:
                        self.state = combined_data_cleaned.astype(np.float32)
                        logger.info(f"State created successfully after dropping columns. New state shape: {self.state.shape}")

                        # --- IMPORTANT: Redefine observation space based on the new shape ---
                        # Assuming the original definition was something like:
                        # num_features = df.shape[1] # Original number of features from input df
                        # state_shape = (window_size, num_features + 3) # + balance, position, pnl
                        # We need to adjust the number of features based on dropped columns
                        new_num_features = self.state.shape[1] # Get the actual number of columns in the final state
                        original_space_low = self.observation_space.low
                        original_space_high = self.observation_space.high

                        # Create new bounds based on the new shape.
                        # Assuming original bounds were uniform, replicate for the new shape.
                        # If bounds were feature-specific, this needs more complex logic.
                        new_low = np.full((self.window_size, new_num_features), original_space_low.flat[0], dtype=np.float32)
                        new_high = np.full((self.window_size, new_num_features), original_space_high.flat[0], dtype=np.float32)

                        self.observation_space = gym.spaces.Box(
                            low=new_low,
                            high=new_high,
                            dtype=np.float32,
                        )
                        logger.warning(f"Observation space redefined to shape: {self.observation_space.shape}")
                        # --- End Observation Space Redefinition ---

                    except ValueError as e2:
                        logger.error(f"State conversion failed EVEN AFTER dropping columns: {e2}")
                        logger.error("Problematic data likely persists or is widespread. Cannot initialize state.")
                        # Raise the error or handle appropriately (e.g., set state to zeros and hope?)
                        raise e2 # Re-raise the error as we can't recover
                else:
                    logger.error("ValueError occurred but could not identify problematic columns. Raising original error.")
                    raise e # Re-raise original error if columns weren't identified
        else:
            # For live trading, create an empty state that will be filled later
            self.state = np.zeros(
                (self.window_size, self.observation_space.shape[1]), dtype=np.float32
            )

    def _init_binance_account(self):
        """Initialize Binance Futures account with leverage and margin type."""
        if self.live_trading:
            try:
                # Set initial leverage using the utility function
                # Ensure volatility is updated first if needed for dynamic leverage
                if self.dynamic_leverage and self.current_volatility == 0.0:
                    self._update_volatility() # Make sure volatility is current

                adaptive_leverage = (
                    calculate_adaptive_leverage(
                        volatility=self.current_volatility,
                        max_leverage=self.max_leverage,
                        default_leverage=self.leverage
                    ) if self.dynamic_leverage
                    else self.leverage
                )
                # Store the calculated leverage being used
                self.leverage = adaptive_leverage

                self.client.change_leverage(
                    symbol=self.symbol, leverage=adaptive_leverage, recvWindow=6000
                )

                # Set margin type
                try:
                    self.client.change_margin_type(
                        symbol=self.symbol, marginType=self.margin_type, recvWindow=6000
                    )
                except Exception as e:
                    # Check if it's the "No need to change margin type" error
                    if "No need to change margin type" in str(e):
                        logging.getLogger("BinanceFuturesExecutor").info(
                            f"Margin type already set to {self.margin_type}"
                        )
                    else:
                        # If it's a different error, raise it
                        raise e

                # Get account info to verify settings
                account_info = self.client.account()

                # Initialize market data
                self._update_market_data()

                print(
                    f"Account initialized - Balance: {self._get_balance_usdt()} USDT, Leverage: {adaptive_leverage}x"
                )

            except Exception as e:
                print(f"Error initializing Binance account: {e}")

    # Removed duplicated _calculate_adaptive_leverage method. Using utils.calculate_adaptive_leverage instead.

    def _update_volatility(self):
        """Update market volatility calculation."""
        try:
            # Get recent price data
            if self.live_trading:
                # Convert lookback hours to number of candles based on interval
                interval_minutes = int(self.data_fetch_interval.replace("m", ""))
                candles_needed = int((self.volatility_lookback * 60) / interval_minutes)

                # Get klines data for volatility calculation
                klines = pd.DataFrame(
                    self.client.klines(
                        symbol=self.symbol,
                        interval=self.data_fetch_interval,
                        limit=min(1000, candles_needed),  # API limit is 1000
                    )
                )

                if klines.empty:
                    self.current_volatility = 0.0
                    return

                # Process klines
                klines = klines.iloc[:, 0:6]
                klines.columns = ["time", "open", "high", "low", "close", "volume"]
                klines = klines.astype(float)

                # Calculate returns
                close_prices = klines["close"].values
                returns = np.diff(np.log(close_prices))

                # Calculate volatility (standard deviation of returns)
                volatility = np.std(returns)

                # Scale the volatility to a 0-1 range (approximately)
                # High volatility is typically above 0.03-0.05 daily
                scaled_volatility = min(1.0, volatility * 20)

                self.current_volatility = scaled_volatility
                self.volatility_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "volatility": scaled_volatility,
                    }
                )
            else:
                # For backtesting, calculate from dataframe if available
                if self.df is not None:
                    current_idx = self.current_step
                    start_idx = max(
                        0, current_idx - (self.volatility_lookback * 4)
                    )  # Approximate candles for lookback

                    if start_idx < current_idx:
                        prices = self.df.iloc[start_idx:current_idx]["close"].values
                        if len(prices) > 1:
                            returns = np.diff(np.log(prices))
                            volatility = np.std(returns)
                            scaled_volatility = min(1.0, volatility * 20)
                            self.current_volatility = scaled_volatility
                            return

                # Default if calculation fails
                self.current_volatility = 0.1  # Moderate volatility by default

        except Exception as e:
            print(f"Error updating volatility: {e}")
            self.current_volatility = 0.1  # Default to moderate volatility on error

    def _get_balance_usdt(self):
        """Get USDT balance from Binance."""
        if self.live_trading:
            try:
                response = self.client.balance()
                for elem in response:
                    if elem["asset"] == "USDT":
                        return float(elem["balance"])
                return 0.0
            except Exception as e:
                print(f"Error getting balance: {e}")
                return 0.0
        else:
            return self.balance

    def _get_trading_params(self):
        """Get price and quantity precision for the symbol."""
        try:
            resp = self.client.exchange_info()
            price_precision = 0
            qty_precision = 0

            for elem in resp["symbols"]:
                if elem["symbol"] == self.symbol:
                    price_precision = elem["pricePrecision"]
                    qty_precision = elem["quantityPrecision"]
                    break

            return price_precision, qty_precision
        except Exception as e:
            print(f"Error getting trading parameters: {e}")
            return 8, 8  # Default precision values

    def _update_market_data(self):
        """Update market data including funding rate, open interest, and liquidation data."""
        if not self.live_trading:
            return

        try:
            # 1. Update funding rate
            if self.include_funding_rate:
                funding_rate = self._get_funding_rate()
                self.current_funding_rate = funding_rate
                self.funding_rates.append(
                    {"timestamp": datetime.now().isoformat(), "rate": funding_rate}
                )

            # 2. Update open interest
            if self.include_open_interest:
                new_open_interest = self._get_open_interest()
                # Calculate open interest change
                if self.current_open_interest > 0:
                    self.open_interest_change = (
                        new_open_interest - self.current_open_interest
                    ) / self.current_open_interest
                else:
                    self.open_interest_change = 0

                self.current_open_interest = new_open_interest
                self.open_interest_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "value": new_open_interest,
                        "change": self.open_interest_change,
                    }
                )

            # 3. Update liquidation data
            if self.include_liquidation_data:
                liquidation_data = self._get_liquidations()
                if liquidation_data:
                    self.liquidation_data.append(liquidation_data)

            # 4. Update volatility
            self._update_volatility()

            # 5. Dynamic leverage adjustment if needed
            if (
                self.dynamic_leverage
                and hasattr(self, "current_position")
                and self.current_position == 0
            ):  # Only adjust when not in a position
                adaptive_leverage = self._calculate_adaptive_leverage()
                if adaptive_leverage != self.leverage:
                    self.leverage = adaptive_leverage
                    # Update leverage on exchange
                    self.client.change_leverage(
                        symbol=self.symbol, leverage=self.leverage, recvWindow=6000
                    )
                    print(
                        f"Adjusted leverage to {self.leverage}x based on volatility of {self.current_volatility:.2%}"
                    )

        except Exception as e:
            print(f"Error updating market data: {e}")

    def _get_funding_rate(self):
        """Get current funding rate for the symbol."""
        try:
            # Get premium index (includes funding rate)
            response = self.client.mark_price(symbol=self.symbol)

            if "lastFundingRate" in response:
                return float(response["lastFundingRate"])
            return 0.0

        except Exception as e:
            print(f"Error getting funding rate: {e}")
            return 0.0

    def _get_open_interest(self):
        """Get current open interest for the symbol."""
        try:
            response = self.client.open_interest(symbol=self.symbol)

            if "openInterest" in response:
                return float(response["openInterest"])
            return 0.0

        except Exception as e:
            print(f"Error getting open interest: {e}")
            return 0.0

    def _get_liquidations(self):
        """Get recent liquidation data (if available)."""
        try:
            # Note: This is using a public API as Binance doesn't provide liquidation data directly in the API
            # For production, you might want to use a more reliable data source or websocket
            url = f"https://fapi.binance.com/fapi/v1/allForceOrders?symbol={self.symbol}&limit=50"
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()

                if data:
                    # Process and summarize liquidation data
                    current_time = datetime.now()
                    one_hour_ago = current_time - timedelta(hours=1)

                    # Filter for recent liquidations
                    recent_liquidations = [
                        item
                        for item in data
                        if datetime.fromtimestamp(item["time"] / 1000) > one_hour_ago
                    ]

                    # Summarize
                    long_liquidations = sum(
                        float(item["qty"])
                        for item in recent_liquidations
                        if item["side"] == "BUY"  # Liquidated shorts are bought
                    )

                    short_liquidations = sum(
                        float(item["qty"])
                        for item in recent_liquidations
                        if item["side"] == "SELL"  # Liquidated longs are sold
                    )

                    return {
                        "timestamp": current_time.isoformat(),
                        "long_liquidations": long_liquidations,
                        "short_liquidations": short_liquidations,
                        "total_liquidations": long_liquidations + short_liquidations,
                        "ratio": long_liquidations
                        / (short_liquidations + 1e-10),  # Prevent division by zero
                    }

            return None

        except Exception as e:
            print(f"Error getting liquidation data: {e}")
            return None

    def _calculate_liquidation_price(self):
        """Calculate approximate liquidation price based on position."""
        if self.current_position == 0 or self.entry_price == 0:
            return None

        # For Isolated margin, approximate liquidation calculation
        # This is a simplified calculation and may not match Binance's exact calculation
        maintenance_margin_rate = 0.005  # 0.5% for BTC, adjust as needed

        if self.current_position > 0:  # Long position
            liquidation_price = self.entry_price * (
                1 - (1 / self.leverage) + maintenance_margin_rate
            )
        else:  # Short position
            liquidation_price = self.entry_price * (
                1 + (1 / self.leverage) - maintenance_margin_rate
            )

        return liquidation_price

    def _calculate_position_size(self, price):
        """Calculate position size based on account balance, leverage, and risk management."""
        # Get trading parameters
        _, qty_precision = self._get_trading_params() if self.live_trading else (8, 8)

        # Calculate signal strength (could be modified to use actual signal strength)
        signal_strength = 1.0

        # Use position sizer to calculate position size
        position_result = self.position_sizer.calculate_position_size(
            account_balance=self.balance,
            current_price=price,
            signal_strength=signal_strength,
            volatility=self.current_volatility,
            leverage=self.leverage,
            qty_precision=qty_precision,
            atr=self._get_atr(),  # Pass ATR value for volatility-based sizing
        )

        # Update leverage if it was changed by the position sizer
        if self.dynamic_leverage and position_result["leverage"] != self.leverage:
            self.leverage = position_result["leverage"]

        return position_result["size_in_units"]

    def _get_atr(self):
        """Get Average True Range for the current market."""
        if self.live_trading:
            # Use ATR from recent market data
            try:
                klines = pd.DataFrame(
                    self.client.klines(
                        symbol=self.symbol,
                        interval=self.data_fetch_interval,
                        limit=14 + 1,  # ATR needs at least 14 candles
                    )
                )

                if klines.empty:
                    self.current_volatility = 0.0
                    return

                # Process klines
                klines = klines.iloc[:, 0:6]
                klines.columns = ["time", "open", "high", "low", "close", "volume"]
                klines = klines.astype(float)

                # Calculate returns
                close_prices = klines["close"].values
                returns = np.diff(np.log(close_prices))

                # Calculate volatility (standard deviation of returns)
                volatility = np.std(returns)

                # Scale the volatility to a 0-1 range (approximately)
                # High volatility is typically above 0.03-0.05 daily
                scaled_volatility = min(1.0, volatility * 20)

                self.current_volatility = scaled_volatility
                self.volatility_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "volatility": scaled_volatility,
                    }
                )
            except Exception as e:
                print(f"Error calculating ATR: {e}")

            # Default if calculation fails
            return self._get_current_price() * 0.01  # 1% of price as default
        else:
            # For backtesting, use ATR from dataframe if available
            if "atr" in self.df.columns:
                return self.df.iloc[self.current_step]["atr"]
            else:
                return (
                    self.df.iloc[self.current_step]["close"] * 0.01
                )  # 1% of price as default

    def _get_current_price(self):
        """Get current price."""
        if self.live_trading:
            try:
                ticker = self.client.ticker_price(symbol=self.symbol)
                return float(ticker["price"])
            except Exception as e:
                print(f"Error getting current price: {e}")
                return 0.0
        else:
            return self.df.iloc[self.current_step]["close"]

    def _execute_trade(
        self, action, scale_in=False, scale_out=False, scale_percentage=0.5
    ):
        """
        Execute a trade on Binance Futures using the executor module.

        Args:
            action: The action to take (0=hold, 1=buy/long, 2=sell/short)
            scale_in: Whether to scale into an existing position
            scale_out: Whether to scale out of an existing position
            scale_percentage: Percentage of position to scale in/out (0.5 = 50%)
        """
        if not self.live_trading:
            # In backtest mode, trades are handled by _handle_backtesting_trade
            return

        # Check position status first
        position_status = self.executor.check_position_status()

        # Update internal position tracking
        self.position_status["has_open_position"] = position_status["position_open"]

        # If position was closed by SL/TP, update trigger flags and record the trade
        if not position_status["position_open"] and position_status.get("trigger_type"):
            trigger_type = position_status["trigger_type"]

            if trigger_type == "stop_loss":
                self.position_status["sl_triggered"] = True
                self.position_status["tp_triggered"] = False
            elif trigger_type == "take_profit":
                self.position_status["sl_triggered"] = False
                self.position_status["tp_triggered"] = True

            # Record the trade in history
            self.trade_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "close",
                    "trigger": trigger_type,
                    "position_direction": self.current_position > 0 and 1 or -1,
                    "position_size": abs(self.current_position),
                    "entry_price": self.entry_price,
                }
            )

            # Reset internal position tracking
            self.current_position = 0
            self.entry_price = 0
            self.unrealized_pnl = 0

        # Handle scaling logic
        if self.position_status["has_open_position"] and (scale_in or scale_out):
            # Calculate position size/amount for scaling
            account_balance = self._get_balance_usdt()
            usdt_amount = account_balance * self.max_position * scale_percentage

            # Execute scaling operation through executor
            result = self.executor.execute_trade(
                action=action,
                usdt_amount=usdt_amount,
                scale_in=scale_in,
                scale_out=scale_out,
                scale_percentage=scale_percentage,
            )

            # Update position tracking based on scaling result
            if result["success"]:
                if result["action"] == "scale_in":
                    # Update position size and entry price from executor
                    position_direction = 1 if self.current_position > 0 else -1
                    self.current_position = position_direction * result["position_size"]
                    self.entry_price = result["entry_price"]
                    self.stop_loss = result.get("stop_loss", self.stop_loss)
                    self.take_profit = result.get("take_profit", self.take_profit)

                    # Record the scaling action
                    self.trade_history.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "action": "scale_in",
                            "position_direction": position_direction,
                            "added_size": result["quantity"],
                            "total_size": result["position_size"],
                            "avg_entry_price": result["entry_price"],
                            "stop_loss": result.get("stop_loss", self.stop_loss),
                            "take_profit": result.get("take_profit", self.take_profit),
                        }
                    )

                elif result["action"] == "scale_out":
                    # Update position size
                    position_direction = 1 if self.current_position > 0 else -1
                    if result["position_size"] <= 0:
                        # Position fully closed
                        self.current_position = 0
                        self.entry_price = 0
                        self.position_status["has_open_position"] = False
                    else:
                        # Position partially closed
                        self.current_position = (
                            position_direction * result["position_size"]
                        )

                    # Record the scaling action
                    self.trade_history.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "action": "scale_out",
                            "position_direction": position_direction,
                            "removed_size": result["quantity"],
                            "remaining_size": result["position_size"],
                            "price": result["price"],
                        }
                    )

            return

        # Skip if we already have an open position (not scaling) or if action is 0
        if self.position_status["has_open_position"] or action == 0:
            return

        # Calculate USDT amount to allocate for new positions
        account_balance = self._get_balance_usdt()
        usdt_amount = account_balance * self.max_position

        # Execute trade through executor
        result = self.executor.execute_trade(action=action, usdt_amount=usdt_amount)

        # If trade was successful, update internal tracking
        if result["success"] and result["action"] not in ["hold", "error"]:
            # Update position tracking
            position_direction = 1 if result["action"] == "buy" else -1
            self.current_position = position_direction * result["quantity"]
            self.entry_price = result["price"]

            # Update stop loss and take profit levels
            self.stop_loss = result["stop_loss"]
            self.take_profit = result["take_profit"]

            # Update position status
            self.position_status["has_open_position"] = True
            self.position_status["sl_triggered"] = False
            self.position_status["tp_triggered"] = False

            # Record trade in history
            self.trade_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": result["action"],
                    "position_direction": position_direction,
                    "position_size": result["quantity"],
                    "entry_price": result["price"],
                    "stop_loss": result["stop_loss"],
                    "take_profit": result["take_profit"],
                }
            )

    def _handle_backtesting_trade(self, action, price, exit_price: Optional[float] = None):
        """
        Handle trade simulation in backtesting mode.
        Uses `exit_price` if provided (for SL/TP), otherwise uses `price`.
        `price` is used as the entry price if opening a new position.
        """
        # --- Slippage Calculation ---
        slippage = 0
        if self.slippage_fraction > 0 and self.current_step < len(self.df):
            current_bar = self.df.iloc[self.current_step]
            bar_range = current_bar["high"] - current_bar["low"]
            # Ensure bar_range is non-negative and finite
            if pd.notna(bar_range) and bar_range >= 0:
                slippage = bar_range * self.slippage_fraction
        # --- End Slippage Calculation ---

        # Determine the base price for execution (SL/TP price or current bar's close)
        base_exec_price = exit_price if exit_price is not None else price
        realized_pnl = 0
        transaction_cost = 0

        if action == 1:  # Buy/Long action
            if self.current_position < 0: # Closing existing short
                # Apply slippage to exit price (buying back short) - pay more
                exec_price = base_exec_price + slippage / 2
                realized_pnl = self.current_position * (exec_price - self.entry_price) * self.leverage
                self.balance += realized_pnl
                transaction_cost += abs(self.current_position * exec_price * self.commission) # Exit fee

                self.trades.append({
                    "entry_price": self.entry_price, "exit_price": exec_price,
                    "position": self.current_position, "pnl": realized_pnl, "timestamp": self.date,
                })
                self.current_position = 0
                self.entry_price = 0
                self.stop_loss = None
                self.take_profit = None

            if exit_price is None: # Only open new long if not forced exit by SL/TP
                # Apply slippage to entry price (buying long) - pay more
                entry_exec_price = price + slippage / 2
                position_size = self._calculate_position_size(entry_exec_price) # Size based on execution price
                fee = position_size * entry_exec_price * self.commission
                transaction_cost += fee
                self.balance -= fee
                self.current_position = position_size
                self.entry_price = entry_exec_price # Store actual execution price
                # SL/TP based on actual entry execution price
                self.stop_loss = entry_exec_price * (1 - self.stop_loss_percent)
                self.take_profit = entry_exec_price * (1 + self.stop_loss_percent * self.risk_reward_ratio)

        elif action == 2:  # Sell/Short action
            if self.current_position > 0: # Closing existing long
                # Apply slippage to exit price (selling long) - receive less
                exec_price = base_exec_price - slippage / 2
                realized_pnl = self.current_position * (exec_price - self.entry_price) * self.leverage
                self.balance += realized_pnl
                transaction_cost += abs(self.current_position * exec_price * self.commission) # Exit fee

                self.trades.append({
                    "entry_price": self.entry_price, "exit_price": exec_price,
                    "position": self.current_position, "pnl": realized_pnl, "timestamp": self.date,
                })
                self.current_position = 0
                self.entry_price = 0
                self.stop_loss = None
                self.take_profit = None

            if exit_price is None: # Only open new short if not forced exit by SL/TP
                # Apply slippage to entry price (selling short) - receive less
                entry_exec_price = price - slippage / 2
                position_size = self._calculate_position_size(entry_exec_price) # Size based on execution price
                fee = position_size * entry_exec_price * self.commission
                transaction_cost += fee
                self.balance -= fee
                self.current_position = -position_size
                self.entry_price = entry_exec_price # Store actual execution price
                # SL/TP based on actual entry execution price
                self.stop_loss = entry_exec_price * (1 + self.stop_loss_percent)
                self.take_profit = entry_exec_price * (1 - self.stop_loss_percent * self.risk_reward_ratio)

        elif action == 0 and exit_price is not None: # Hold action BUT SL/TP triggered closing position
             if self.current_position > 0: # Closing existing long
                # Apply slippage to exit price (selling long) - receive less
                exec_price = base_exec_price - slippage / 2
                realized_pnl = self.current_position * (exec_price - self.entry_price) * self.leverage
                self.balance += realized_pnl
                transaction_cost += abs(self.current_position * exec_price * self.commission) # Exit fee
                self.trades.append({
                    "entry_price": self.entry_price, "exit_price": exec_price,
                    "position": self.current_position, "pnl": realized_pnl, "timestamp": self.date,
                })
             elif self.current_position < 0: # Closing existing short
                # Apply slippage to exit price (buying back short) - pay more
                exec_price = base_exec_price + slippage / 2
                realized_pnl = self.current_position * (exec_price - self.entry_price) * self.leverage
                self.balance += realized_pnl
                transaction_cost += abs(self.current_position * exec_price * self.commission) # Exit fee
                self.trades.append({
                    "entry_price": self.entry_price, "exit_price": exec_price,
                    "position": self.current_position, "pnl": realized_pnl, "timestamp": self.date,
                })
             # Reset position state after SL/TP close during hold
             self.current_position = 0
             self.entry_price = 0
             self.stop_loss = None
             self.take_profit = None

        # Return PnL and cost for reward calculation if needed
        return realized_pnl, transaction_cost

    def step(self, action, scale_in=False, scale_out=False, scale_percentage=0.5):
        """
        Take a step in the environment.

        Args:
            action: 0 = Hold, 1 = Buy/Long, 2 = Sell/Short
            scale_in: Whether to scale into an existing position (add to it)
            scale_out: Whether to scale out of an existing position (reduce it)
            scale_percentage: Percentage to scale in/out (0.5 = 50%)

        Returns:
            tuple: (next_state, reward, done, truncated, info)
        """
        # Store previous state info for reward calculation
        prev_account_value = self.balance + self.unrealized_pnl

        # Check for existing positions and modify action if needed
        if (
            self.live_trading
            and self.position_status["has_open_position"]
            and action != 0
            and not scale_in
            and not scale_out
        ):
            # If we have an open position and not scaling, force hold action
            print(f"Position already open, ignoring action {action} and forcing HOLD")
            action = 0

        is_trade = action != 0 or self.current_position != 0

        # Get current data and price
        if self.live_trading:
            # Update market data (funding rates, open interest, etc.)
            self._update_market_data()
            current_price = float(self.client.ticker_price(self.symbol)["price"])
        else:
            # In backtesting mode, move to next step
            self.current_step += 1
            if self.current_step >= len(self.df.index) - 1:
                # End of data
                return self.state, 0, True, False, {"status": "Terminal state reached"}

            current_price = self.df.iloc[self.current_step]["close"]
            if isinstance(self.df.index, pd.DatetimeIndex):
                self.date = self.df.index[self.current_step]
            else:
                self.date = None

            # For backtesting, try to get funding rate from data if available
            if "funding_rate" in self.df.columns:
                self.current_funding_rate = self.df.iloc[self.current_step][
                    "funding_rate"
                ]

            # For backtesting, try to get open interest from data if available
            if "open_interest" in self.df.columns:
                new_oi = self.df.iloc[self.current_step]["open_interest"]
                self.open_interest_change = (
                    (new_oi - self.current_open_interest) / self.current_open_interest
                    if self.current_open_interest > 0
                    else 0
                )
                self.current_open_interest = new_oi

            # --- SL/TP Check using NEXT bar's data (Fix for Lookahead Bias) ---
            sl_tp_triggered = False
            sl_tp_exit_price = None
            # Start with the agent's intended action, may be overridden by SL/TP
            forced_action = action

            # Only check if we have a position and there's a next bar available
            if self.current_position != 0 and self.current_step + 1 < len(self.df):
                next_bar = self.df.iloc[self.current_step + 1]
                next_high = next_bar["high"]
                next_low = next_bar["low"]
                # next_open = next_bar["open"] # Keep for potential future use (e.g., tie-breaking)

                sl_hit = False
                tp_hit = False

                if self.current_position > 0: # Long position check
                    if self.stop_loss is not None and next_low <= self.stop_loss:
                        sl_hit = True
                    if self.take_profit is not None and next_high >= self.take_profit:
                        tp_hit = True
                elif self.current_position < 0: # Short position check
                    if self.stop_loss is not None and next_high >= self.stop_loss:
                        sl_hit = True
                    if self.take_profit is not None and next_low <= self.take_profit:
                        tp_hit = True

                # Determine exit price and forced action, prioritizing SL
                if sl_hit:
                    sl_tp_triggered = True
                    # Assume fill at SL price. More complex models could use next_open or SL price.
                    sl_tp_exit_price = self.stop_loss
                    forced_action = 2 if self.current_position > 0 else 1 # Force close action
                    # print(f"Debug: SL Hit at step {self.current_step+1}. Price: {sl_tp_exit_price}") # Optional debug
                elif tp_hit:
                    sl_tp_triggered = True
                    # Assume fill at TP price. More complex models could use next_open or TP price.
                    sl_tp_exit_price = self.take_profit
                    forced_action = 2 if self.current_position > 0 else 1 # Force close action
                    # print(f"Debug: TP Hit at step {self.current_step+1}. Price: {sl_tp_exit_price}") # Optional debug

            # ---------------------------------------------------------------------

        # Calculate unrealized PnL *before* the potential trade action
        # Use current_price from the *current* step for this calculation
        old_unrealized_pnl = 0
        if self.current_position != 0:
            old_unrealized_pnl = (
                self.current_position
                * (current_price - self.entry_price)
                * self.leverage
            )

        # Position direction before action (used for info dict)
        old_position_direction = (
            1 if self.current_position > 0 else (-1 if self.current_position < 0 else 0)
        )

        # Execute the action (either agent's original 'action' or 'forced_action' from SL/TP)
        realized_pnl = 0
        transaction_cost = 0
        if self.live_trading:
            # Live trading uses its own execution logic which should handle SL/TP via exchange orders
            # We pass the original agent action here, assuming SL/TP is managed by the executor/exchange.
            self._execute_trade(action, scale_in, scale_out, scale_percentage)
            # PnL/costs would need to be updated based on executor feedback or API calls
        else:
            # Backtesting: Use the potentially forced action and SL/TP exit price.
            # current_price is used for new entries if SL/TP didn't trigger an exit.
            realized_pnl, transaction_cost = self._handle_backtesting_trade(
                forced_action, current_price, exit_price=sl_tp_exit_price
            )

        # Calculate new unrealized PnL *after* the trade action
        # Use current_price from the *current* step
        if self.current_position != 0:
            self.unrealized_pnl = (
                self.current_position
                * (current_price - self.entry_price)
                * self.leverage
            )
        else:
            self.unrealized_pnl = 0

        # Note: realized_pnl and transaction_cost are now directly returned by _handle_backtesting_trade
        # for backtesting mode. Live mode would need separate handling.

        # Update state with new market data
        if not self.live_trading:
            # Create observation space with position, balance, and unrealized_pnl added
            self._update_state()
        else:
            # In live trading mode, fetch and update the state from Binance API
            self._update_live_state()

        # Calculate account value
        account_value = self.balance + self.unrealized_pnl

        # Calculate max account value for drawdown
        self.max_account_value = max(account_value, self.max_account_value)
        drawdown = (
            (self.max_account_value - account_value) / self.max_account_value
            if self.max_account_value > 0
            else 0
        )

        # Calculate liquidation price
        liquidation_price = self._calculate_liquidation_price()

        # Prepare info dictionary with enhanced data
        info = {
            "account_value": account_value,
            "balance": self.balance,
            "position": self.current_position,
            "position_direction": (
                1
                if self.current_position > 0
                else (-1 if self.current_position < 0 else 0)
            ),
            "entry_price": self.entry_price,
            "current_price": current_price,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "drawdown": drawdown,
            "trade_count": len(self.trades),
            "leverage": self.leverage,
            "liquidation_price": liquidation_price,
            "funding_rate": self.current_funding_rate,
            "open_interest": self.current_open_interest,
            "open_interest_change": self.open_interest_change,
            "volatility": self.current_volatility,
            "is_trade": is_trade,
            "transaction_cost": transaction_cost,
            "date": self.date if not self.live_trading else datetime.now(),
            "new_balance": account_value,
        }

        # Add position status for live trading
        if self.live_trading:
            info.update(
                {
                    "has_open_position": self.position_status["has_open_position"],
                    "sl_triggered": self.position_status["sl_triggered"],
                    "tp_triggered": self.position_status["tp_triggered"],
                    "scale_in": scale_in,
                    "scale_out": scale_out,
                    "scale_percentage": scale_percentage,
                }
            )

        # Calculate reward
        if self.use_risk_adjusted_rewards:
            # Use the specialized futures reward function
            reward = self.reward_calculator.calculate_reward(
                state=self.state,
                action=action,
                next_state=self.state,  # Not actually used in our implementation
                info=info,
            )
        else:
            # Simple reward based on PnL difference
            reward = (
                self.unrealized_pnl - old_unrealized_pnl
                if realized_pnl == 0
                else realized_pnl
            )

            # Apply risk-adjustment to reward: penalize for high drawdowns
            drawdown_penalty = 1.0 - drawdown
            reward *= drawdown_penalty

        # Check for done conditions
        done = False
        truncated = False

        # Bankruptcy check - if account value is too low
        if account_value <= self.initial_balance * 0.1:  # 90% loss
            done = True
            reward -= 100  # Extra penalty for bankruptcy

        # End of data check for backtesting
        if not self.live_trading and self.current_step >= len(self.df) - 1:
            done = True

        return self.state, reward, done, truncated, info

    def _update_state(self):
        """Update the state with current market data and account information."""
        if not self.live_trading:
            # For backtesting, get data from the dataframe
            data = self.df.iloc[
                self.current_step - self.window_size + 1 : self.current_step + 1
            ].copy()

            # Make sure we have window_size rows
            if len(data) < self.window_size:
                # If we don't have enough rows, prepend the first row multiple times
                missing_rows = self.window_size - len(data)
                padding = pd.concat([data.iloc[[0]]] * missing_rows)
                data = pd.concat([padding, data])

            # Extract features
            price_features = data.values

            # Account information (balance, position, unrealized_pnl) is no longer part of the model's state.
            # It can still be tracked by the environment for reward calculation or logging if needed.

            # The state now consists only of price_features.
            try:
                self.state = price_features.astype(np.float32)
            except ValueError as e:
                logger.warning(f"Initial state conversion failed: {e}. Attempting to drop problematic columns...")
                # TODO: Fix root cause of non-numeric data (e.g., 'none' string in 'trade_setup') upstream.
                # This block is a temporary workaround.

                import pandas as pd # Import locally for this block
                df_check = pd.DataFrame(combined_data)
                is_numeric = df_check.apply(lambda s: pd.to_numeric(s, errors='coerce').notna().all())
                problematic_cols_indices = [i for i, numeric in enumerate(is_numeric) if not numeric]

                if problematic_cols_indices:
                    logger.warning(f"Dropping non-numeric columns at indices: {problematic_cols_indices}")
                    combined_data_cleaned = np.delete(combined_data, problematic_cols_indices, axis=1)
                    try:
                        self.state = combined_data_cleaned.astype(np.float32)
                        logger.info(f"State created successfully after dropping columns. New state shape: {self.state.shape}")

                        # --- IMPORTANT: Redefine observation space based on the new shape ---
                        # Assuming the original definition was something like:
                        # num_features = df.shape[1] # Original number of features from input df
                        # state_shape = (window_size, num_features + 3) # + balance, position, pnl
                        # We need to adjust the number of features based on dropped columns
                        new_num_features = self.state.shape[1] # Get the actual number of columns in the final state
                        original_space_low = self.observation_space.low
                        original_space_high = self.observation_space.high

                        # Create new bounds based on the new shape.
                        # Assuming original bounds were uniform, replicate for the new shape.
                        # If bounds were feature-specific, this needs more complex logic.
                        new_low = np.full((self.window_size, new_num_features), original_space_low.flat[0], dtype=np.float32)
                        new_high = np.full((self.window_size, new_num_features), original_space_high.flat[0], dtype=np.float32)

                        self.observation_space = gym.spaces.Box(
                            low=new_low,
                            high=new_high,
                            dtype=np.float32,
                        )
                        logger.warning(f"Observation space redefined to shape: {self.observation_space.shape}")
                        # --- End Observation Space Redefinition ---

                    except ValueError as e2:
                        logger.error(f"State conversion failed EVEN AFTER dropping columns: {e2}")
                        logger.error("Problematic data likely persists or is widespread. Cannot initialize state.")
                        # Raise the error or handle appropriately (e.g., set state to zeros and hope?)
                        raise e2 # Re-raise the error as we can't recover

        else:
            # For live trading, fetch data from Binance API
            self._update_live_state()

    def _update_live_state(self):
        """Update state with live market data from Binance."""
        try:
            # 1. Get recent OHLCV data
            klines = pd.DataFrame(
                self.client.klines(
                    symbol=self.symbol,
                    interval=self.data_fetch_interval,
                    limit=self.window_size,
                )
            )

            if klines.empty:
                print("Warning: Empty klines data received")
                return

            # Process klines
            klines = klines.iloc[:, 0:6]
            klines.columns = ["time", "open", "high", "low", "close", "volume"]
            klines = klines.astype(float)

            # 2. Calculate technical indicators
            indicators_df = self._calculate_indicators(klines)

            # 3. Combine OHLCV and indicators
            feature_data = indicators_df.values

            # 4. Add account information
            balance = np.ones(self.window_size) * self.balance
            position = np.ones(self.window_size) * self.current_position
            unrealized_pnl = np.ones(self.window_size) * self.unrealized_pnl

            # 5. Combine into state and ensure it's float32
            combined_data = np.column_stack(
                (
                    feature_data,
                    balance.reshape(-1, 1),
                    position.reshape(-1, 1),
                    unrealized_pnl.reshape(-1, 1),
                )
            )
            try:
                self.state = combined_data.astype(np.float32)
            except ValueError as e:
                logger.warning(f"Initial state conversion failed: {e}. Attempting to drop problematic columns...")
                # TODO: Fix root cause of non-numeric data (e.g., 'none' string in 'trade_setup') upstream.
                # This block is a temporary workaround.

                import pandas as pd # Import locally for this block
                df_check = pd.DataFrame(combined_data)
                is_numeric = df_check.apply(lambda s: pd.to_numeric(s, errors='coerce').notna().all())
                problematic_cols_indices = [i for i, numeric in enumerate(is_numeric) if not numeric]

                if problematic_cols_indices:
                    logger.warning(f"Dropping non-numeric columns at indices: {problematic_cols_indices}")
                    combined_data_cleaned = np.delete(combined_data, problematic_cols_indices, axis=1)
                    try:
                        self.state = combined_data_cleaned.astype(np.float32)
                        logger.info(f"State created successfully after dropping columns. New state shape: {self.state.shape}")

                        # --- IMPORTANT: Redefine observation space based on the new shape ---
                        # Assuming the original definition was something like:
                        # num_features = df.shape[1] # Original number of features from input df
                        # state_shape = (window_size, num_features + 3) # + balance, position, pnl
                        # We need to adjust the number of features based on dropped columns
                        new_num_features = self.state.shape[1] # Get the actual number of columns in the final state
                        original_space_low = self.observation_space.low
                        original_space_high = self.observation_space.high

                        # Create new bounds based on the new shape.
                        # Assuming original bounds were uniform, replicate for the new shape.
                        # If bounds were feature-specific, this needs more complex logic.
                        new_low = np.full((self.window_size, new_num_features), original_space_low.flat[0], dtype=np.float32)
                        new_high = np.full((self.window_size, new_num_features), original_space_high.flat[0], dtype=np.float32)

                        self.observation_space = gym.spaces.Box(
                            low=new_low,
                            high=new_high,
                            dtype=np.float32,
                        )
                        logger.warning(f"Observation space redefined to shape: {self.observation_space.shape}")
                        # --- End Observation Space Redefinition ---

                    except ValueError as e2:
                        logger.error(f"State conversion failed EVEN AFTER dropping columns: {e2}")
                        logger.error("Problematic data likely persists or is widespread. Cannot initialize state.")
                        # Raise the error or handle appropriately (e.g., set state to zeros and hope?)
                        raise e2 # Re-raise the error as we can't recover
            # Update date for reference
            self.date = datetime.now()

        except Exception as e:
            print(f"Error updating live state: {e}")

    def _calculate_indicators(self, data):
        """Calculate technical indicators for the given OHLCV data."""
        df = data.copy()

        # Add basic indicators
        try:
            # MACD
            macd = ta.trend.MACD(df["close"])
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_diff"] = macd.macd_diff()

            # RSI
            df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()

            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df["close"])
            df["bb_high"] = bollinger.bollinger_hband()
            df["bb_mid"] = bollinger.bollinger_mavg()
            df["bb_low"] = bollinger.bollinger_lband()

            # EMA
            df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
            df["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)

            # ATR (Average True Range) - volatility indicator
            df["atr"] = ta.volatility.AverageTrueRange(
                df["high"], df["low"], df["close"], window=14
            ).average_true_range()

            # Add market data if available
            if self.include_funding_rate:
                df["funding_rate"] = self.current_funding_rate

            if self.include_open_interest:
                df["open_interest"] = self.current_open_interest
                df["open_interest_change"] = self.open_interest_change

            # Fill NaN values that may result from indicators that need longer lookbacks
            df = df.ffill()
            df.fillna(0, inplace=True)

            return df

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            # If indicators fail, just return original data
            return data

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment.

        Returns:
            tuple: (state, info)
        """
        # Reset trading metrics
        self.trades = []
        self.current_position = 0
        self.entry_price = 0
        self.unrealized_pnl = 0
        self.max_account_value = self.initial_balance
        self.balance = self.initial_balance

        # Reset position status for live trading
        if self.live_trading:
            # Check for any open positions on reset
            if hasattr(self, "executor"):
                position_status = self.executor.check_position_status()
                self.position_status["has_open_position"] = position_status[
                    "position_open"
                ]

                # If there's an open position, we need to sync our state with it
                if self.position_status["has_open_position"]:
                    self.current_position = position_status.get(
                        "position_size", 0
                    ) * position_status.get("position_direction", 0)
                    self.entry_price = position_status.get("entry_price", 0)
                    print(
                        f"Reset with existing position: {self.current_position} @ {self.entry_price}"
                    )
                else:
                    self.position_status["sl_triggered"] = False
                    self.position_status["tp_triggered"] = False

        if not self.live_trading:
            # For backtesting, reset to initial step
            self.current_step = self.window_size - 1

            # Random start point for training
            if self.mode == "train" and len(self.df) > self.window_size + 100:
                self.current_step = np.random.randint(
                    self.window_size, len(self.df) - 100
                )

            # Set initial date if index is datetime
            if isinstance(self.df.index, pd.DatetimeIndex):
                self.date = self.df.index[self.current_step]
            else:
                self.date = None

            # Reset state
            self._update_state()
            info = {"status": "Environment reset"}
            return self.state, info
        else:
            # For live trading, initialize with current market data
            # Fetch initial data and set state
            self._update_live_state()
            info = {"status": "Live environment reset"}
            return self.state, info

    def render(self):
        """Render the environment state."""
        if self.render_mode != "human":
            return

        if self.current_position == 0:
            position_type = "FLAT"
        elif self.current_position > 0:
            position_type = "LONG"
        else:
            position_type = "SHORT"

        print(
            f"Date: {self.date if hasattr(self, 'date') and self.date is not None else datetime.now()}"
        )
        print(f"Balance: {self.balance:.2f} USDT")
        print(
            f"Position: {abs(self.current_position):.8f} {self.symbol} {position_type}"
        )
        print(f"Entry Price: {self.entry_price:.2f} USDT")
        print(f"Unrealized PnL: {self.unrealized_pnl:.2f} USDT")
        print(f"Total Trades: {len(self.trades)}")

        # Calculate win rate
        if len(self.trades) > 0:
            winning_trades = sum(1 for trade in self.trades if trade["pnl"] > 0)
            win_rate = winning_trades / len(self.trades)
            print(f"Win Rate: {win_rate:.2%}")

        if self.stop_loss is not None and self.take_profit is not None:
            print(f"Stop Loss: {self.stop_loss:.2f} USDT")
            print(f"Take Profit: {self.take_profit:.2f} USDT")

    def close(self):
        """Close the environment, releasing resources."""
        pass
