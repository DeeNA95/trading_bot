#!/bin/bash



nohup python train.py \
  --train_data gs://btrading/data/eth/ETHUSDT_1m_with_metrics_6y.parquet \
  --symbol ETHUSDT \
  --interval 1m \
  --window 80 \
  --leverage 20 \
  --balance 10 \
  --architecture encoder_only \
  --n_encoder_layers 16 \
  --dropout 0.15 \
  --n_heads 16 \
  --ffn_type moe \
  --n_experts 16 \
  --ffn_dim 256 \
  --lr 1e-5 \
  --episodes 200 \
  --save_path gs://btrading/models/large_encoder/mha_16l_16h_moe16$(date +%Y%m%d_%H%M%S) \
  --embedding_dim 320 \
  --attention_type mha
