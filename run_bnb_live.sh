#! /bin/sh

nohup pipenv run python inference.py \
  --model_path gs://crypto_trading_models/LSTM/BNB/best_model.pt \
  --model_type lstm \
  --symbol BNBUSDT \
  --window_size 24 \
  --leverage 20 \
  --interval 15m \
  --initial_balance 12 \
  --stop_loss 0.05 \
  --risk_reward 1.5 \
  --sleep_time 60
