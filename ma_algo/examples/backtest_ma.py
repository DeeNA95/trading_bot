import pandas as pd
from ma_algo.backtest import backtest_ma_strategy
from ma_algo.config import MAConfig

if __name__ == "__main__":
    # Load historical data
    data = pd.read_parquet('path/to/historical_data.parquet')
    
    # Configure strategy
    config = MAConfig(
        short_window=50,
        long_window=200,
        risk_per_trade=0.02,
        atr_multiplier=1.5
    )
    
    # Run backtest
    results = backtest_ma_strategy(data, config)
    
    print(f"Final Balance: {results['final_balance']:.2f}")
    print(f"Total Return: {results['return_pct']:.2f}%")
    print(f"Trades Executed: {results['num_trades']}")
