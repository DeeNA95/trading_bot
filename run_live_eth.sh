#! /bin/sh

# nohup
 pipenv run python inference.py \
  --model_path gs://crypto_trading_models/TRANSFORMERS/ETH/IUIN/best_model.pt \
  --model_type transformer \
  --symbol ETHUSDT \
  --window_size 24 \
  --leverage 20 \
  --interval 15m \
  --initial_balance 3 \
  --stop_loss_percent 0.01 \
  --risk_reward_ratio 1.5 \
  --sleep_time 60 2>&1 &
