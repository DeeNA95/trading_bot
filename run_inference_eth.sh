#!/bin/zsh

# Inference script for ETHUSDT using the model from fold 1


echo "Starting ETHUSDT inference..."

python3.12 inference.py \
    --model_path gs://btrading/models/large_encoder/20250421_225336/fold_2/best_model.pt   \
    --symbol ETHUSDT \
    --interval 1m \
    --window_size 60 \
    --leverage 20 \
    --stop_loss_percent 0.005 \
    --risk_reward_ratio 2.0 \
    --sleep_time 600
echo "Inference script finished."
