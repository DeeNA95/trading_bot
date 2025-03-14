#! /bin/sh

nohup pipenv run python inference.py \
  --model_path gs://crypto_trading_models/LSTM/BACKTESTED/ETH/ETHUSDT_lstm_20250314_094627.pt \
  --model_type lstm \
  --symbol ETHUSDT \
  --window_size 24 \
  --leverage 20 \
  --interval 15m \
  --initial_balance 1.54 \
  --stop_loss 0.01 \
  --risk_reward 1.5 \
  --sleep_time 600 2>&1 &
