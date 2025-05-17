#!/bin/zsh

# Inference script for ETHUSDT using the model from fold 1


echo "Starting ETHUSDT inference..."

python3.12 inference.py \
    --model_path gs://btrading/models/pyramidal_decoder/pyr_16l_16h_moe16_resnet20250517_215003/fold_5/best_model.pt \
    --scaler_path gs://btrading/models/pyramidal_decoder/pyr_16l_16h_moe16_resnet20250517_215003/fold_5/scaler_fold_5.joblib \
    --symbol ETHUSDT \
    --interval 1m \
    --window_size 128 \
    --leverage 125 \
    --stop_loss_percent 0.005 \
    --risk_reward_ratio 2.0 \
    --sleep_time 60 \
    --initial_balance 42 \
    --dry_run
echo "Inference script finished."
