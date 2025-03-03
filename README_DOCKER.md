# Dockerized Trading Bot

This document explains how to use Docker to run the trading bot.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system

## Getting Started

1. Build the Docker image:

```bash
make docker-build
```

## Training the Model

### Standard Training

To run the standard training process:

```bash
make docker-train
```

This will:
- Train the model with 2000 episodes
- Use batch size of 128
- Save checkpoints every 200 episodes
- Mount volumes for models, logs, and data for persistence

### Parallel Training

To run training with parallelization:

```bash
make docker-train-parallel
```

This allows Docker to automatically use all available CPU cores.

## Running Inference

### With Test Data

To test the model using historical data:

```bash
make docker-inference-test
```

### Live Trading

To run the model with live trading through Binance Futures:

```bash
make docker-inference-binance
```

Make sure your `.env` file contains valid Binance API credentials:

```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

## Customizing Training Parameters

You can customize parameters by editing the docker-compose.yml file or by running Docker commands directly:

```bash
docker run -v ./data:/app/data -v ./models:/app/models -v ./logs:/app/logs trading-bot:latest train.py --n_episodes 5000 --batch_size 256
```

## Viewing Logs

Logs will be available in the mounted logs directory and also through Docker's logging:

```bash
docker logs trading-bot-training
```

## Using GPU Acceleration (If Available)

The Docker setup will automatically detect and use MPS (Apple Silicon) or CUDA if available in the host system.
