#!/bin/zsh

# Inference script for ETHUSDT using the model from fold 1


echo "Starting ETHUSDT inference..."

python3.12 inference.py \
    --model_path best_model.pt \
    --scaler_path scaler_fold_3.joblib \
    --symbol ETHUSDT \
    --interval 1m \
    --window_size 512 \
    --leverage 125 \
    --stop_loss_percent 0.005 \
    --risk_reward_ratio 2.0 \
    --sleep_time 600 \
    --initial_balance 42 
echo "Inference script finished."
