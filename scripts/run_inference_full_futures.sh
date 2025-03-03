#!/bin/bash

# Set the model name
MODEL_NAME="futures_model"

# Set the path to the best model
MODEL_PATH="models/${MODEL_NAME}/best"

# Run inference
pipenv run python inference.py \
  --model_path "${MODEL_PATH}" \
  --binance_futures \
  --symbol BTCUSDT
