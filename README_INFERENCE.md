# Trading Bot Inference

This document explains how to use the inference script to run the trained trading bot model.

## Prerequisites

- Trained model files in the `models/` directory
- Alpaca API credentials in `.env` file (for paper trading mode)

## Setup

1. Ensure your `.env` file contains Alpaca API credentials:

```
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here
```

2. Install required dependencies:

```
pipenv install alpaca-trade-api
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

### Paper Trading with Alpaca

To run the model with paper trading on Alpaca:

```
make inference-paper
```

or

```
pipenv run python inference.py --paper_trading --trade_interval 3600
```

This will execute trades on your Alpaca paper trading account, with a 1-hour interval between trades.

## Command Line Arguments

- `--model_path`: Path to the trained model (default: 'models/best')
- `--symbol`: Trading symbol (default: 'BTC/USD')
- `--initial_balance`: Initial balance for simulation (default: 10000)
- `--commission`: Commission rate (default: 0.001)
- `--max_leverage`: Maximum leverage (default: 3.0)
- `--trade_interval`: Seconds between trades in paper trading mode (default: 3600)
- `--paper_trading`: Enable paper trading with Alpaca
- `--test_data`: Path to test data (if not using live data)

## Example Usage

Test with different model:
```
pipenv run python inference.py --test_data data/BTC_USD_complete.csv --model_path models/checkpoint_1000
```

Paper trade with different symbol and 30-minute intervals:
```
pipenv run python inference.py --paper_trading --symbol ETH/USD --trade_interval 1800
```

## Stopping the Inference

To stop the inference script while it's running, press `Ctrl+C`.
