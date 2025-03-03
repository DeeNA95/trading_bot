import pandas as pd
from .strategies.ma_crossover import MACrossoverStrategy
from .config import MAConfig

def backtest_ma_strategy(data: pd.DataFrame, config: MAConfig = None) -> dict:
    """
    Backtest MA crossover strategy on historical data
    """
    config = config or MAConfig()
    strategy = MACrossoverStrategy(
        short_window=config.short_window,
        long_window=config.long_window,
        risk_per_trade=config.risk_per_trade,
        atr_multiplier=config.atr_multiplier
    )
    
    df = strategy.calculate_features(data)
    
    # Initialize backtest variables
    balance = 10000.0  # Starting balance
    position = 0
    trades = []
    
    for i, row in df.iterrows():
        if pd.isna(row['ma_short']) or pd.isna(row['ma_long']):
            continue
            
        direction, size = strategy.generate_signal(row, balance)
        
        if direction != 0 and size > 0:
            # Simulate trade execution
            price = row['close'] * (1 + config.slippage)
            cost = size * price
            fee = cost * config.commission
            
            if direction == 1:
                balance -= (cost + fee)
                position += size
            else:
                balance += (cost - fee)
                position -= size
            
            trades.append({
                'timestamp': row.name,
                'direction': direction,
                'size': size,
                'price': price,
                'fee': fee
            })
    
    # Calculate performance metrics
    return {
        'final_balance': balance,
        'return_pct': (balance - 10000) / 100,
        'num_trades': len(trades),
        'max_drawdown': 0.0,  # Implement actual calculation
        'sharpe_ratio': 0.0   # Implement actual calculation
    }
