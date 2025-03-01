"""
Position sizing strategies for risk management.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any


class PositionSizer(ABC):
    """Abstract base class for position sizing strategies."""
    
    @abstractmethod
    def calculate_position_size(self, 
                               account_balance: float, 
                               signal_strength: float, 
                               volatility: float, 
                               **kwargs) -> float:
        """Calculate the position size based on account balance and market conditions.
        
        Args:
            account_balance: Current account balance
            signal_strength: Strength of the trading signal (between -1 and 1)
            volatility: Market volatility measure (e.g., ATR)
            
        Returns:
            Recommended position size as a fraction of account balance
        """
        pass


class FixedFractionPositionSizer(PositionSizer):
    """Position sizer that allocates a fixed fraction of the account balance."""
    
    def __init__(self, max_risk_pct: float = 0.02):
        """Initialize the fixed fraction position sizer.
        
        Args:
            max_risk_pct: Maximum percentage of account balance to risk per trade
        """
        self.max_risk_pct = max_risk_pct
        
    def calculate_position_size(self, 
                               account_balance: float, 
                               signal_strength: float, 
                               volatility: float, 
                               risk_per_trade: Optional[float] = None, 
                               **kwargs) -> float:
        """Calculate position size based on fixed fraction risk model.
        
        Args:
            account_balance: Current account balance
            signal_strength: Strength of the trading signal (between -1 and 1)
            volatility: Market volatility measure (e.g., ATR)
            risk_per_trade: Override for max_risk_pct if provided
            
        Returns:
            Recommended position size as a fraction of account balance
        """
        # Use provided risk_per_trade or fall back to max_risk_pct
        risk_pct = risk_per_trade if risk_per_trade is not None else self.max_risk_pct
        
        # Scale position size by signal strength
        signal_scale = abs(signal_strength)
        
        # Scale position size inversely with volatility
        # Higher volatility = smaller position
        vol_scale = 1.0 / max(volatility, 0.001)
        
        # Calculate position size
        position_size = account_balance * risk_pct * signal_scale * vol_scale
        
        # Ensure position size is not more than account balance
        return min(position_size, account_balance)


class KellyPositionSizer(PositionSizer):
    """Position sizer based on the Kelly Criterion."""
    
    def __init__(self, 
                max_kelly_pct: float = 0.2, 
                win_rate_window: int = 50):
        """Initialize the Kelly position sizer.
        
        Args:
            max_kelly_pct: Maximum percentage of Kelly to use (0.5 = "half Kelly")
            win_rate_window: Number of past trades to consider for win rate calculation
        """
        self.max_kelly_pct = max_kelly_pct
        self.win_rate_window = win_rate_window
        self.past_trades = []
        
    def update_trade_history(self, trade_result: float):
        """Update the trade history.
        
        Args:
            trade_result: Profit/loss from the trade as a percentage
        """
        self.past_trades.append(trade_result)
        if len(self.past_trades) > self.win_rate_window:
            self.past_trades.pop(0)
    
    def calculate_position_size(self, 
                               account_balance: float, 
                               signal_strength: float, 
                               volatility: float, 
                               win_rate: Optional[float] = None, 
                               avg_win_loss_ratio: Optional[float] = None, 
                               **kwargs) -> float:
        """Calculate position size based on Kelly Criterion.
        
        Args:
            account_balance: Current account balance
            signal_strength: Strength of the trading signal (between -1 and 1)
            volatility: Market volatility measure (e.g., ATR)
            win_rate: Probability of winning (if None, calculated from past_trades)
            avg_win_loss_ratio: Average win/loss ratio (if None, calculated from past_trades)
            
        Returns:
            Recommended position size as a fraction of account balance
        """
        # Calculate win rate and avg_win_loss_ratio from past trades if not provided
        if win_rate is None or avg_win_loss_ratio is None:
            if not self.past_trades:
                # Not enough history, use conservative defaults
                win_rate = 0.5
                avg_win_loss_ratio = 1.0
            else:
                # Calculate from past trades
                wins = [t for t in self.past_trades if t > 0]
                losses = [t for t in self.past_trades if t <= 0]
                
                win_rate = len(wins) / len(self.past_trades) if len(self.past_trades) > 0 else 0.5
                
                avg_win = np.mean(wins) if wins else 0
                avg_loss = abs(np.mean(losses)) if losses else 1
                avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # Kelly formula: f* = (bp - q) / b = p - q/b
        # where p = win rate, q = 1-p, b = avg_win_loss_ratio
        kelly_pct = win_rate - (1 - win_rate) / avg_win_loss_ratio if avg_win_loss_ratio > 0 else 0
        
        # Apply max Kelly fraction and ensure it's positive
        kelly_pct = max(0, min(kelly_pct, self.max_kelly_pct))
        
        # Scale by signal strength
        kelly_pct *= abs(signal_strength)
        
        # Adjust for volatility - reduce position size in high volatility
        volatility_factor = 1.0 / (1.0 + volatility)
        kelly_pct *= volatility_factor
        
        # Calculate position size
        position_size = account_balance * kelly_pct
        
        return position_size


class VolatilityAdjustedPositionSizer(PositionSizer):
    """Position sizer that adjusts based on ATR volatility."""
    
    def __init__(self, base_risk_pct: float = 0.01, volatility_scale: float = 2.0):
        """Initialize the volatility adjusted position sizer.
        
        Args:
            base_risk_pct: Base percentage of account to risk
            volatility_scale: Scaling factor for volatility adjustments
        """
        self.base_risk_pct = base_risk_pct
        self.volatility_scale = volatility_scale
        
    def calculate_position_size(self, 
                               account_balance: float, 
                               signal_strength: float, 
                               volatility: float, 
                               price: float = None, 
                               atr: float = None, 
                               **kwargs) -> float:
        """Calculate position size based on ATR and account balance.
        
        Args:
            account_balance: Current account balance
            signal_strength: Strength of the trading signal (between -1 and 1)
            volatility: Market volatility measure (normalized)
            price: Current price of the asset
            atr: Average True Range value
            
        Returns:
            Recommended position size as a fraction of account balance
        """
        # Scale the risk percentage inversely with volatility
        adj_risk_pct = self.base_risk_pct / (1 + self.volatility_scale * volatility)
        
        # Calculate risk amount
        risk_amount = account_balance * adj_risk_pct
        
        # If we have price and ATR, we can calculate position size in units
        if price is not None and atr is not None:
            # Use ATR to determine stop loss distance
            stop_distance = self.volatility_scale * atr
            
            # Calculate position size in units
            position_size = risk_amount / stop_distance if stop_distance > 0 else 0
            
            # Convert to account fraction
            position_fraction = (position_size * price) / account_balance
            return min(position_fraction, 1.0)
        
        # Otherwise return the risk percentage directly
        return adj_risk_pct * abs(signal_strength)
