#!/bin/bash

nohup python train.py \
  --train_data gs://btrading/data/eth/ETHUSDT_1m_with_metrics_6y.parquet \
  --symbol ETHUSDT \
  --interval 1m \
  --window 256 \
  --leverage 20 \
  --balance 10000 \
  --core_model_type lstm \
  --lstm_hidden_dim 128 \
  --lstm_num_layers 2 \
  --lstm_dropout 0.1 \
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
  --save_path gs://btrading/models/lstm_core/lstm_h128_l2_$(date +%Y%m%d_%H%M%S) > nohup.out 2>&1 &

# Print a message with the process ID
echo "Training started in background with PID $!"
echo "You can monitor the progress with: tail -f nohup.out"
