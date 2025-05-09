#!/bin/zsh

# Inference script for ETHUSDT using the model from fold 1


echo "Starting ETHUSDT inference..."

python3.12 inference.py \
    --model_path gs://btrading/models/pyramidal_decoder/pyr_16l_16h_moe16_resnet20250504_204019/fold_3/best_model.pt \
    --scaler_path gs://btrading/models/pyramidal_decoder/pyr_16l_16h_moe16_resnet20250504_204019/fold_3/scaler_fold_3.joblib \
    --symbol ETHUSDT \
    --interval 1m \
    --window_size 512 \
    --leverage 125 \
    --stop_loss_percent 0.005 \
    --risk_reward_ratio 2.0 \
    --sleep_time 600 \
    --initial_balance 42 
echo "Inference script finished."
