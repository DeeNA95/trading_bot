#! /bin/zsh

pipenv run python inference.py \
  --model_path models/XRP/lstm/best_model.pt \
  --model_type lstm \
  --symbol XRPUSDT \
  --window_size 24 \
  --leverage 20 \
  --interval 15m \
  --initial_balance 12 \
  --stop_loss 0.05 \
  --risk_reward 1.1 \
  --sleep_time 60
