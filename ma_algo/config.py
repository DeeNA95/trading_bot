from dataclasses import dataclass

@dataclass
class MAConfig:
    short_window: int = 50
    long_window: int = 200
    risk_per_trade: float = 0.01  # 1% of account per trade
    atr_period: int = 14
    atr_multiplier: float = 2.0
    max_leverage: int = 5
    commission: float = 0.0004  # 0.04% per trade
    slippage: float = 0.0001  # 0.01%
