#!/bin/bash

# Define save path with descriptive name
SAVE_DIR="gs://ctrading/models/enc_mha_run_$(date +%Y%m%d_%H%M%S)"
echo "Saving models to: $SAVE_DIR"


python train.py \
  --train_data gs://ctrading/data/eth/ETHUSDT_1m_with_metrics_6y.parquet \
  --symbol ETHUSDT \
  --interval 1m \
  --window 60 \
  --leverage 20 \
  --balance 10 \
  --architecture encoder_only \
  --n_encoder_layers 8 \
  --dropout 0.2 \
  --n_heads 8 \
  --ffn_type moe \
  --n_experts 2 \
  --ffn_dim 128 \
  --lr 3e-5 \
  --batch_size 2048 \
  --episodes 2000 \
  --save_path gs://models/large_encoder
