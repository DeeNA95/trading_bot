#!/bin/bash

# Inference script for ETHUSDT using the model from fold 1

# Ensure pipenv environment is active
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || exit 1 # Change to script directory to ensure pipenv works

echo "Starting ETHUSDT inference..."

pipenv run python inference.py \
    --model_path models/test_run_walk_forward/fold_4/best_model.pt  \
    --symbol ETHUSDT \
    --interval 1m \
    --window_size 60 \
    --model_type transformer \
    --dry_run \
    --leverage 10 \
    --stop_loss_percent 0.005 \
    --risk_reward_ratio 2.0 \
    --sleep_time 30

echo "Inference script finished."
