#!/bin/bash

# Complex Pyramidal Attention Decoder-Only Model with ResNet Feature Extraction
# and Enhanced Actor-Critic Heads

# Run with nohup to handle disconnections and redirect output
# The '&' at the end allows the shell to continue being used
nohup python train.py \
  --train_data gs://btrading/data/eth/ETHUSDT_1m_with_metrics_6y.parquet \
  --symbol ETHUSDT \
  --interval 1m \
  --window 256 \
  --leverage 20 \
  --balance 10 \
  --architecture decoder_only \
  --n_decoder_layers 16 \
  --dropout 0.2 \
  --n_heads 16 \
  --attention_type pyr \
  --ffn_type moe \
  --n_experts 16 \
  --top_k 5 \
  --ffn_dim 512 \
  --lr 3e-4 \
  --episodes 200 \
  --batch_size 256 \
  --embedding_dim 320 \
  --feature_extractor_type resnet \
  --feature_extractor_dim 128 \
  --feature_extractor_layers 3 \
  --use_skip_connections \
  --head_hidden_dim 256 \
  --head_n_layers 3 \
  --head_use_layer_norm \
  --head_use_residual \
  --residual_scale 1.2 \
  --use_gated_residual \
  --use_final_norm \
  --gradient_accumulation_steps 8 \
  --save_path gs://btrading/models/pyramidal_decoder/pyr_16l_16h_moe16_resnet$(date +%Y%m%d_%H%M%S) > nohup.out 2>&1 &

# Print a message with the process ID
echo "Training started in background with PID $!"
echo "You can monitor the progress with: tail -f nohup.out"
