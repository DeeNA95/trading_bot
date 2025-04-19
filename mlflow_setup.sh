#! /bin/bash
# This script sets up the MLflow environment for the project.

mlflow server \
  --backend-store-uri postgresql+psycopg2://mlflow_user:avrutd123@127.0.0.1:5432/mlflow_db \
  --default-artifact-root gs://crypto_trading_models \
  --host 127.0.0.1 \
  --port 5001

#run this first
#cloud-sql-proxy seraphic-bliss-451413-c8:asia-southeast1:mlflow-pg-instance
