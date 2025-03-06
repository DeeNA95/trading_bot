# Reinforcement Learning Trading Bot for Binance Futures

An advanced reinforcement learning agent for algorithmic trading on Binance Futures, designed for crypto market trading with a focus on BTC/USDT.

## Features

- Deep reinforcement learning agent with Proximal Policy Optimization (PPO)
- Support for various neural network architectures (CNN, LSTM, Transformer)
- Live trading on Binance Futures with proper risk management
- Position sizing based on volatility and risk metrics
- Integration of market data signals (funding rates, volatility)
- Stop loss and take profit management

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/trading_bot.git
cd trading_bot
```

2. Create a virtual environment and install dependencies:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. Set up your API keys:
   - Create a `.env` file in the project root
   - Add your Binance API keys (for testnet and/or mainnet):

```
# For testnet
binance_future_testnet_api=your_testnet_api_key
binance_future_testnet_secret=your_testnet_secret_key
testnest_url=https://testnet.binancefuture.com

# For mainnet (when ready for live trading)
binance_future_api=your_mainnet_api_key
binance_future_secret=your_mainnet_secret_key
```

## Usage

### 1. Generate training data

```bash
# Generate data for BTCUSDT with 15-minute candlesticks for the past 30 days
python data_generator.py --symbol BTCUSDT --interval 15m --days 30 --output data/btcusdt_15m.csv

# Split data into training and testing sets
python data_generator.py --symbol BTCUSDT --interval 15m --days 60 --split
```

### 2. Train the agent

```bash
# Train with default parameters
python train.py --data data/btcusdt_15m.csv --symbol BTCUSDT --model lstm --episodes 100

# Train with custom parameters
python train.py --data data/btcusdt_15m.csv --symbol BTCUSDT --model transformer --episodes 200 --leverage 3 --window 50
```

### 3. Run backtests

```bash
# Backtest a trained model
python main.py backtest --data data/test_BTCUSDT_15m.csv --model models/BTCUSDT_lstm_20250101_120000.pt --model_type lstm
```

### 4. Live trading

```bash
# Live trading on testnet
python main.py live --model models/BTCUSDT_lstm_20250101_120000.pt --model_type lstm --interval 60 --testnet

# Live trading on mainnet (use with caution!)
python main.py live --model models/BTCUSDT_lstm_20250101_120000.pt --model_type lstm --interval 60
```

## Project Structure

- `rl_agent/` - Core RL agent implementation
  - `agent/` - PPO agent and neural network models
  - `environment/` - Trading environment, position sizing, and execution
- `data_generator.py` - Script for fetching and preprocessing data
- `train.py` - Script for training the RL agent
- `main.py` - Script for backtesting and live trading

## Model Architectures

### CNN
Best for detecting visual patterns in the price charts.

### LSTM
Effective for sequence modeling and capturing temporal dependencies in the time series data.

### Transformer
Advanced architecture that captures long-range dependencies and complex relationships in market data.

## Risk Management

The bot implements several risk management strategies:

1. **Position Sizing**: Adjusts position size based on account balance and market volatility
2. **Stop Loss/Take Profit**: Automatic TP/SL orders with configurable risk-reward ratio
3. **Dynamic Leverage**: Adjusts leverage based on market volatility
4. **Risk-Adjusted Rewards**: Penalizes excess risk-taking during training

## Performance Monitoring

During training and backtesting, the bot tracks various performance metrics:

- Total return and drawdown
- Win rate and average trade profitability
- Sharpe ratio and other risk-adjusted metrics

## Disclaimer

This software is for educational and research purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred by using this software. Always start with paper trading before committing real capital.

## License

[MIT License](LICENSE)