#!/bin/zsh

# Inference script for ETHUSDT using the model from fold 1


echo "Starting ETHUSDT inference..."

python3.12 inference.py \
    --model_path gs://btrading/models/large_encoder/mha_16l_16h_moe1620250423_212253/fold_5/best_model.pt \
    --scaler_path gs://btrading/models/large_encoder/mha_16l_16h_moe1620250423_212253/fold_5/scaler_fold_5.joblib \
    --symbol ETHUSDT \
    --interval 1m \
    --window_size 80 \
    --leverage 20 \
    --stop_loss_percent 0.005 \
    --risk_reward_ratio 2.0 \
    --sleep_time 600
echo "Inference script finished."
