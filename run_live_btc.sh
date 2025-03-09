#! /bin/sh

nohup pipenv run python inference.py \
  --model_path gs://crypto_trading_models/LSTM/POSITION_GAINERS/BTC/best_model.pt \
  --model_type lstm \
  --symbol BTCUSDT \
  --window_size 24 \
  --leverage 20 \
  --interval 15m \
  --initial_balance 12 \
  --stop_loss 0.01 \
  --risk_reward 1.5 \
  --sleep_time 60 2>&1 &
