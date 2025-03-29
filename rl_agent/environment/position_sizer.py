"""
Position sizing strategies for risk management in trading environments.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .utils import calculate_adaptive_leverage # Added import


class PositionSizer(ABC):
    """Abstract base class for position sizing strategies."""

    @abstractmethod
    def calculate_position_size(
        self,
        account_balance: float,
        signal_strength: float,
        volatility: float,
        **kwargs: Any,
    ) -> float:
        """
        Calculate the position size based on account balance and market conditions.

        Args:
            account_balance: Current account balance.
            signal_strength: Strength of the trading signal (between -1 and 1).
            volatility: Market volatility measure (e.g., ATR).
            **kwargs: Additional parameters.

        Returns:
            Recommended position size as a fraction of account balance.
        """
        pass


class FixedFractionPositionSizer(PositionSizer):
    """Position sizer that allocates a fixed fraction of the account balance."""

    def __init__(self, max_risk_pct: float = 0.02):
        """
        Initialize the fixed fraction position sizer.

        Args:
            max_risk_pct: Maximum percentage of account balance to risk per trade.
        """
        self.max_risk_pct = max_risk_pct

    def calculate_position_size(
        self,
        account_balance: float,
        signal_strength: float,
        volatility: float,
        risk_per_trade: Optional[float] = None,
        **kwargs: Any,
    ) -> float:
        """
        Calculate position size based on fixed fraction risk model.

        Args:
            account_balance: Current account balance.
            signal_strength: Strength of the trading signal (between -1 and 1).
            volatility: Market volatility measure (e.g., ATR).
            risk_per_trade: Optional override for risk percentage per trade.

        Returns:
            Recommended position size as a fraction of account balance.
        """
        risk_pct = risk_per_trade if risk_per_trade is not None else self.max_risk_pct
        signal_scale = abs(signal_strength)
        # Avoid division by zero in volatility
        vol_scale = 1.0 / max(volatility, 0.001)
        position_size = account_balance * risk_pct * signal_scale * vol_scale
        return min(position_size, account_balance)


class KellyPositionSizer(PositionSizer):
    """Position sizer based on the Kelly Criterion."""

    def __init__(self, max_kelly_pct: float = 0.2, win_rate_window: int = 50):
        """
        Initialize the Kelly position sizer.

        Args:
            max_kelly_pct: Maximum fraction of the Kelly Criterion to use.
            win_rate_window: Number of past trades to consider for win rate calculation.
        """
        self.max_kelly_pct = max_kelly_pct
        self.win_rate_window = win_rate_window
        self.past_trades: List[float] = []

    def update_trade_history(self, trade_result: float) -> None:
        """
        Update the trade history with the result of a trade.

        Args:
            trade_result: Profit/loss from the trade as a percentage.
        """
        self.past_trades.append(trade_result)
        if len(self.past_trades) > self.win_rate_window:
            self.past_trades.pop(0)

    def calculate_position_size(
        self,
        account_balance: float,
        signal_strength: float,
        volatility: float,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> float:
        """
        Calculate position size based on the Kelly Criterion.

        Args:
            account_balance: Current account balance.
            signal_strength: Strength of the trading signal (between -1 and 1).
            volatility: Market volatility measure (e.g., ATR).
            win_rate: Optional probability of winning; if None, it is computed from past trades.
            avg_win_loss_ratio: Optional average win/loss ratio; if None, computed from past trades.

        Returns:
            Recommended position size as a fraction of account balance.
        """
        if win_rate is None or avg_win_loss_ratio is None:
            if not self.past_trades:
                win_rate = 0.5
                avg_win_loss_ratio = 1.0
            else:
                wins = [t for t in self.past_trades if t > 0]
                losses = [t for t in self.past_trades if t <= 0]
                win_rate = (
                    len(wins) / len(self.past_trades) if self.past_trades else 0.5
                )
                avg_win = np.mean(wins) if wins else 0
                avg_loss = abs(np.mean(losses)) if losses else 1.0
                avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        # Kelly formula: f* = (bp - q) / b, where b = avg_win_loss_ratio, q = 1 - p
        kelly_pct = (
            (win_rate - (1 - win_rate) / avg_win_loss_ratio)
            if avg_win_loss_ratio > 0
            else 0
        )
        kelly_pct = max(0, min(kelly_pct, self.max_kelly_pct))
        # Scale by absolute signal strength
        kelly_pct *= abs(signal_strength)
        # Adjust for volatility: higher volatility reduces position size
        volatility_factor = 1.0 / (1.0 + volatility)
        kelly_pct *= volatility_factor
        position_size = account_balance * kelly_pct
        return position_size


class VolatilityAdjustedPositionSizer(PositionSizer):
    """Position sizer that adjusts position size based on ATR volatility."""

    def __init__(self, base_risk_pct: float = 0.2, volatility_scale: float = 2.0):
        """
        Initialize the volatility adjusted position sizer.

        Args:
            base_risk_pct: Base percentage of account to risk per trade.
            volatility_scale: Scaling factor for adjusting risk based on volatility.
        """
        self.base_risk_pct = base_risk_pct
        self.volatility_scale = volatility_scale

    def calculate_position_size(
        self,
        account_balance: float,
        signal_strength: float,
        volatility: float,
        price: Optional[float] = None,
        atr: Optional[float] = None,
        **kwargs: Any,
    ) -> float:
        """
        Calculate position size based on volatility adjustments.

        Args:
            account_balance: Current account balance.
            signal_strength: Strength of the trading signal (between -1 and 1).
            volatility: Market volatility measure (normalized).
            price: Current asset price.
            atr: Average True Range value.

        Returns:
            Recommended position size as a fraction of account balance.
        """
        # Adjust risk percentage inversely with volatility
        adj_risk_pct = self.base_risk_pct / (1 + self.volatility_scale * volatility)
        risk_amount = account_balance * adj_risk_pct

        if price is not None and atr is not None:
            # Determine stop loss distance using ATR scaled by volatility_scale
            stop_distance = self.volatility_scale * atr
            position_size_units = (
                risk_amount / stop_distance if stop_distance > 0 else 0
            )
            # Convert to fraction of account balance
            position_fraction = (position_size_units * price) / account_balance
            return min(position_fraction, 1.0)
        return adj_risk_pct * abs(signal_strength)


class BinanceFuturesPositionSizer:
    """
    Position sizing strategy for Binance Futures trading.

    Handles leverage, account balance, and precision rules for the Binance Futures exchange.
    """

    def __init__(
        self,
        max_position_pct: float = 1.0,
        position_sizer: PositionSizer = None,
        default_leverage: int = 2,
        max_leverage: int = 20,
        dynamic_leverage: bool = True,
    ):
        """
        Initialize the Binance Futures position sizer.

        Args:
            max_position_pct: Maximum percentage of account balance to use (1.0 = 100%)
            position_sizer: Strategy implementation for position sizing (FixedFraction by default)
            default_leverage: Default leverage to use
            max_leverage: Maximum allowed leverage
            dynamic_leverage: Whether to adjust leverage based on volatility
        """
        self.max_position_pct = max_position_pct
        self.position_sizer = position_sizer or VolatilityAdjustedPositionSizer()
        self.default_leverage = default_leverage
        self.max_leverage = max_leverage
        self.dynamic_leverage = dynamic_leverage

    def calculate_position_size(
        self,
        account_balance: float,
        current_price: float,
        signal_strength: float = 1.0,
        volatility: float = 0.1,
        leverage: Optional[int] = None,
        qty_precision: int = 8,
        **kwargs: Any,
    ) -> Dict[str, Union[float, int]]:
        """
        Calculate position size for Binance Futures trading.

        Args:
            account_balance: Current account balance in USDT
            current_price: Current price of the asset
            signal_strength: Strength of the trading signal (between -1 and 1)
            volatility: Market volatility measure
            leverage: Trading leverage to use (falls back to default if None)
            qty_precision: Quantity precision for the symbol
            **kwargs: Additional parameters passed to the underlying position sizer

        Returns:
            Dictionary containing position size in units, USD value, and leverage
        """
        # Get adaptive leverage if dynamic leverage is enabled
        # Determine leverage to use
        if leverage is not None:
            # Use explicitly provided leverage
            used_leverage = leverage
        elif self.dynamic_leverage:
            # Calculate dynamically using the utility function
            used_leverage = calculate_adaptive_leverage(
                volatility=volatility,
                max_leverage=self.max_leverage,
                default_leverage=self.default_leverage
                # volatility_factor can use default from utility function
            )
        else:
            # Use the default static leverage
            used_leverage = self.default_leverage


        # Calculate position size as percentage of account
        position_pct = self.position_sizer.calculate_position_size(
            account_balance=account_balance,
            signal_strength=signal_strength,
            volatility=volatility,
            **kwargs,
        )

        # Apply maximum position percentage constraint
        position_pct = min(position_pct, self.max_position_pct)

        # Calculate dollar value of position
        position_value = account_balance * position_pct

        # Calculate position size in units
        position_size_units = (position_value * used_leverage) / current_price

        # Round to appropriate precision
        position_size_units = round(position_size_units, qty_precision)

        return {
            "size_in_usd": position_value * used_leverage,
            "size_in_units": position_size_units,
            "leverage": used_leverage,
            "position_pct": position_pct,
        }

    # Removed duplicated _calculate_adaptive_leverage method. Using utils.calculate_adaptive_leverage instead.
