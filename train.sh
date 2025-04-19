#!/bin/bash

# Define save path with descriptive name
SAVE_DIR="gs://ctrading/models/enc_mha_run_$(date +%Y%m%d_%H%M%S)"
echo "Saving models to: $SAVE_DIR"

python train.py \
    --train_data gs://ctrading/data/eth/ETHUSDT_1m_with_metrics.csv \
    --symbol ETHUSDT \
    --interval 1m \
    --window 60 \
    --episodes 500 \
    --batch_size 1024 \
    --update_freq 2048 \
    --lr 5e-5 \
    --save_path "$SAVE_DIR" \
    --device auto \
    --n_splits 5 \
    --val_ratio 0.15 \
    --eval_freq 20 \
    --leverage 20 \
    --max_position 1.0 \
    --balance 100 \
    --risk_reward 1.5 \
    --stop_loss 0.01 \
    --trade_fee 0.0004 \
