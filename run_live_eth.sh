#! /bin/sh

nohup pipenv run python inference.py \
  --model_path gs://crypto_trading_models/TRANSFORMERS/ETH/UNCLIPPED/best_model.pt \
  --model_type transformer \
  --symbol ETHUSDT \
  --window_size 60 \
  --leverage 20 \
  --interval 1m \
  --initial_balance 3 \
  --stop_loss_percent 0.005 \
  --risk_reward_ratio 1.5 \
  --sleep_time 600 2>&1 &
