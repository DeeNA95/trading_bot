# Trading Bot Inference

This document explains how to use the inference script to run the trained trading bot model.

## Prerequisites

- Trained model files in the `models/` directory
- Binance API credentials in `.env` file (for live trading mode)

## Setup

1. Ensure your `.env` file contains Binance API credentials:

```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

2. Install required dependencies:

```
pipenv install binance-futures-connector python-binance
```

## Running Inference

### Using Test Data

To test the model on historical data without making any trades:

```
make inference
```

or

```
pipenv run python inference.py --test_data data/BTC_USD_complete.csv
```

### Live Trading with Binance Futures

To run the model with live trading on Binance Futures:

```
make inference-binance
```

or

```
pipenv run python inference.py --binance_futures --trade_interval 3600
```

This will execute trades on your Binance Futures account, with a 1-hour interval between trades.

## Command Line Arguments

- `--model_path`: Path to the trained model (default: 'models/best')
- `--symbol`: Trading symbol (default: 'BTCUSDT')
- `--initial_balance`: Initial balance for simulation (default: 10000)
- `--commission`: Commission rate (default: 0.001)
- `--max_leverage`: Maximum leverage (default: 3.0)
- `--trade_interval`: Seconds between trades in live trading mode (default: 3600)
- `--binance_futures`: Enable live trading with Binance Futures
- `--test_data`: Path to test data (if not using live data)

## Example Usage

Test with different model:
```
pipenv run python inference.py --test_data data/BTC_USD_complete.csv --model_path models/checkpoint_1000
```

Live trade with different symbol and 30-minute intervals:
```
pipenv run python inference.py --binance_futures --symbol ETHUSDT --trade_interval 1800
```

## Stopping the Inference

To stop the inference script while it's running, press `Ctrl+C`.
