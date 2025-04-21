#!/bin/bash



python train.py \
  --train_data gs://ctrading/data/eth/ETHUSDT_1m_with_metrics_6y.parquet \
  --symbol ETHUSDT \
  --interval 1m \
  --window 64 \
  --leverage 20 \
  --balance 10 \
  --architecture encoder_only \
  --n_encoder_layers 16 \
  --dropout 0.2 \
  --n_heads 16 \
  --ffn_type moe \
  --n_experts 8 \
  --ffn_dim 128 \
  --lr 3e-5 \
  --episodes 500 \
  --save_path gs://ctrading/models/large_encoder/$(date +%Y%m%d_%H%M%S) \
  --embedding_dim 256 \
