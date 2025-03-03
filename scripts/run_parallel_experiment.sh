#!/bin/bash
# Script to run multiple training jobs with different configurations

# Create experiment directory
EXPERIMENT_DIR="experiments/$(date +%Y%m%d_%H%M%S)"
mkdir -p $EXPERIMENT_DIR

# Copy configuration file for reference
cp configs/default_futures_configs.json $EXPERIMENT_DIR/

# Extract model names from the config file
MODEL_NAMES=($(jq -r '.[].model_name' configs/default_futures_configs.json))

# Run each model configuration separately
for model_name in "${MODEL_NAMES[@]}"; do
  echo "Starting training for model: $model_name"
  
  # Run training for this model
  pipenv run python train.py \
    --model_name "$model_name" \
    --reward_type futures \
    --n_episodes 500 \
    --data_path data/BTCUSDT_1m_with_metrics.csv
    
  echo "Completed training for model: $model_name"
done

# Create a summary file
echo "Experiment completed at $(date)" > $EXPERIMENT_DIR/summary.txt
echo "Models trained:" >> $EXPERIMENT_DIR/summary.txt
for model_name in "${MODEL_NAMES[@]}"; do
  echo "- $model_name" >> $EXPERIMENT_DIR/summary.txt
done

echo "Experiment results saved to $EXPERIMENT_DIR"
