#!/bin/bash
# Run inference with the best model in dry-run mode

nohup pipenv run python inference.py \
  --model_path models/best_model.pt \
  --model_type lstm \
  --symbol BTCUSDT \
  --window_size 24 \
  --leverage 2 \
  --interval 15m \
  --initial_balance 10000 \
  --stop_loss 0.01 \
  --risk_reward 1.5 \
  --dry_run \
  --sleep_time 60
