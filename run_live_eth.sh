#! /bin/sh

# nohup
python inference.py \
  --model_path gs://ctrading/models_lstm_eth/fold_5/best_model.pt \
  --model_type lstm \
  --symbol ETHUSDT \
  --window_size 60 \
  --leverage 20 \
  --interval 1m \
  --initial_balance 3 \
  --stop_loss_percent 0.005 \
  --risk_reward_ratio 1.5 \
  --dry_run \
  --sleep_time 60 2>&1 &
