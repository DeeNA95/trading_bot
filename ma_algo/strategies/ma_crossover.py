import numpy as np
import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class MACrossoverStrategy:
    """
    Moving Average Crossover Strategy with integrated risk management
    """
    
    def __init__(self, 
                 short_window: int = 50, 
                 long_window: int = 200,
                 risk_per_trade: float = 0.01,
                 atr_multiplier: float = 2.0):
        """
        Initialize MA crossover strategy parameters
        
        Args:
            short_window: Short-term MA window (default: 50)
            long_window: Long-term MA window (default: 200)
            risk_per_trade: % of account to risk per trade (default: 1%)
            atr_multiplier: Multiplier for ATR-based stop loss (default: 2.0)
        """
        self.short_window = short_window
        self.long_window = long_window
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.position = 0  # -1, 0, 1
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy
        
        Args:
            df: DataFrame with OHLCV data
        Returns:
            DataFrame with added technical features
        """
        df = df.copy()
        
        # Calculate MAs
        df['ma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['ma_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # Calculate ATR for volatility measurement
        df['tr'] = self._true_range(df)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Calculate crossover signals
        df['crossover'] = np.where(df['ma_short'] > df['ma_long'], 1, -1)
        df['signal'] = df['crossover'].diff().fillna(0)
        
        return df

    def generate_signal(self, 
                       row: pd.Series, 
                       account_balance: float) -> Tuple[int, float]:
        """
        Generate trading signal with position sizing
        
        Args:
            row: Latest market data row
            account_balance: Current account balance
        Returns:
            tuple: (direction, quantity)
        """
        if pd.isna(row['ma_short']) or pd.isna(row['ma_long']):
            return 0, 0.0
        
        # Crossover logic
        if row['ma_short'] > row['ma_long'] and self.position <= 0:
            direction = 1
        elif row['ma_short'] < row['ma_long'] and self.position >= 0:
            direction = -1
        else:
            return 0, 0.0
        
        # Position sizing with ATR-based stop loss
        atr = row['atr']
        if np.isnan(atr) or atr == 0:
            return 0, 0.0
            
        risk_amount = account_balance * self.risk_per_trade
        position_size = risk_amount / (self.atr_multiplier * atr)
        
        # Convert to appropriate lot size
        position_size = self._adjust_to_lot_size(position_size, row['step_size'])
        
        self.position = direction
        return direction, position_size

    def _true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1)

    def _adjust_to_lot_size(self, qty: float, step_size: float) -> float:
        """Round quantity to exchange's step size"""
        if step_size <= 0:
            return qty
        return round(qty / step_size) * step_size
