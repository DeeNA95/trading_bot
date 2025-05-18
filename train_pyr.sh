#!/bin/bash

nohup python train.py \
  --train_data gs://btrading/data/eth/ETHUSDT_1m_with_metrics_6y.parquet \
  --symbol ETHUSDT \
  --interval 1m \
  --window 256 \
  --leverage 20 \
  --balance 10000 \
  --core_model_type transformer \
  --architecture decoder_only \
  --attention_type pyr \
  --n_heads 16 \
  --ffn_type moe \
  --n_experts 8 \
  --top_k 2 \
  --dropout 0.2 \
  --lr 3e-4 \
  --use_lr_scheduler \
  --policy_clip 0.3 \
  --episodes 200 \
  --batch_size 1024 \
  --embedding_dim 128 \
  --feature_extractor_type resnet \
  --feature_extractor_dim 128 \
  --feature_extractor_layers 3 \
  --use_skip_connections \
  --head_hidden_dim 256 \
  --head_n_layers 3 \
  --head_use_layer_norm \
  --head_use_residual \
  --residual_scale 0.9 \
  --use_gated_residual \
  --use_final_norm \
  --gradient_accumulation_steps 8 \
  --entropy_coef 0.1 \
  --n_epochs 4 \
  --weight_decay 0.001 \
  --save_path gs://btrading/models/pyraformer_decoder/pyr_d16h_moe8k2_$(date +%Y%m%d_%H%M%S) > nohup_pyr.out 2>&1 &

# Print a message with the process ID
echo "Pyraformer Decoder training started in background with PID $!"
echo "You can monitor the progress with: tail -f nohup_pyr.out"
