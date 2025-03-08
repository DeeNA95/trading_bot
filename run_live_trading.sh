#!/bin/bash
# Run live trading with the best model on Binance Futures testnet

nohup pipenv run python inference.py \
  --model_path lstm_btc/best_model.pt \
  --model_type lstm \
  --symbol BTCUSDT \
  --window_size 24 \
  --leverage 20 \
  --interval 15m \
  --initial_balance 12 \
  --stop_loss 0.05 \
  --risk_reward 1.5 \
  --sleep_time 60
