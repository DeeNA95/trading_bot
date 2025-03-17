# Binance Futures Trading Bot - Inference

This document explains how to use the inference script to run your trained RL agent on Binance Futures paper trading.

## Prerequisites

1. Make sure you have a trained model (`.pt` file) in the `models` directory
2. Ensure your `.env` file contains the following variables:
   ```
   # Binance Futures API Keys for Testnet (Paper Trading)
   binance_future_testnet_api=YOUR_TESTNET_API_KEY_HERE
   binance_future_testnet_secret=YOUR_TESTNET_SECRET_KEY_HERE
   ```

## Setting Up API Keys

1. Create a Binance Futures Testnet account at https://testnet.binancefuture.com
2. Generate API keys from your testnet account dashboard
3. Copy the `.env.template` file to `.env`:
   ```bash
   cp .env.template .env
   ```
4. Edit the `.env` file and replace the placeholders with your actual API keys:
   ```
   binance_future_testnet_api=YOUR_TESTNET_API_KEY_HERE
   binance_future_testnet_secret=YOUR_TESTNET_SECRET_KEY_HERE
   ```

## Running in Dry-Run Mode

To test the inference without executing actual trades:

```bash
./run_inference.sh
```

This will:
- Load the best model
- Connect to Binance Futures testnet
- Run in dry-run mode (no actual trades)
- Print actions and account information to the console and log file

## Running Live Trading

When you're ready to execute actual trades on the testnet:

```bash
./run_live_trading.sh
```

This will:
- Load the best model
- Connect to Binance Futures testnet
- Execute real trades on the testnet
- Print actions and account information to the console and log file

## Customizing Parameters

You can customize the trading parameters by editing the scripts or running the inference script directly:

```bash
pipenv run python inference.py \
  --model_path models/your_model.pt \
  --model_type lstm \
  --symbol BTCUSDT \
  --window_size 24 \
  --leverage 2 \
  --interval 15m \
  --initial_balance 10000 \
  --stop_loss 0.01 \
  --risk_reward 1.5 \
  --sleep_time 60
```

## Parameters Explained

- `model_path`: Path to the trained model file
- `model_type`: Model architecture (cnn, lstm, transformer)
- `symbol`: Trading pair symbol
- `window_size`: Observation window size
- `leverage`: Trading leverage
- `interval`: Data fetch interval
- `initial_balance`: Initial balance for the trading account
- `stop_loss`: Stop loss percentage (0.01 = 1%)
- `risk_reward`: Risk-reward ratio (take profit to stop loss)
- `dry_run`: Run in dry-run mode (no actual trades)
- `sleep_time`: Sleep time between iterations in seconds

## Logs

The inference script logs all actions and results to:
- Console output
- `inference.log` file

## Stopping the Bot

To stop the bot, press `Ctrl+C` in the terminal where it's running.
