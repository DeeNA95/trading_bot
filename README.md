# Reinforcement Learning Trading Bot

A Bitcoin trading bot that uses reinforcement learning to make trading decisions.

## Installation

This project uses `pipenv` for dependency management. To set up the environment, follow these steps:

```bash
# Install pipenv if not already installed
pip install pipenv

# Install dependencies
pipenv install

# Activate the environment
pipenv shell
```

## Running Tests

The project includes both unit tests and integration tests. To run all tests:

```bash
python run_tests.py
```

To run specific test types:

```bash
# Run only unit tests
python run_tests.py --type unit

# Run only integration tests
python run_tests.py --type integration

# Run with verbose output
python run_tests.py --verbose
```

You can also run individual test files using unittest directly:

```bash
python -m unittest tests/risk/test_position_sizing.py
python -m unittest tests/integration/test_backtest_risk.py
```

## Training the Agent

To train the reinforcement learning agent, use the `train.py` script. The script requires a model name and supports various parameters for customizing the training process.

### Basic Training

```bash
python train.py --model_name my_bitcoin_model
```

### Training with Custom Parameters

```bash
python train.py --model_name my_bitcoin_model \
                --data_path data/BTC_USD_hourly_with_metrics.csv \
                --reward_type risk_adjusted \
                --initial_balance 100000 \
                --commission 0.01 \
                --n_episodes 1000 \
                --batch_size 64
```

### Resuming Training

To resume training from the latest checkpoint:

```bash
python train.py --model_name my_bitcoin_model --resume
```

To resume from a specific checkpoint:

```bash
python train.py --model_name my_bitcoin_model --resume_checkpoint checkpoint_100
```

### Key Parameters

- `--model_name`: (Required) Name of the model to train
- `--data_path`: Path to the data file (default: 'data/BTC_USD_hourly_with_metrics.csv')
- `--reward_type`: Type of reward function to use (choices: 'simple', 'sharpe', 'risk_adjusted', 'tunable', default: 'tunable')
- `--initial_balance`: Initial balance for the trading environment
- `--commission`: Commission rate for trading
- `--max_leverage`: Maximum leverage allowed
- `--n_episodes`: Number of episodes to train for
- `--batch_size`: Batch size for PPO training
- `--checkpoint_interval`: Number of episodes between checkpoints
- `--resume`: Flag to resume training from the latest checkpoint
- `--resume_checkpoint`: Specific checkpoint to resume from

### Reward Function Tuning

The bot uses a tunable reward function by default, which allows for customizing various reward components. You can adjust these parameters to optimize the agent's learning:

```bash
python train.py --model_name tuned_model \
                --realized_pnl_weight 1.0 \
                --unrealized_pnl_weight 0.1 \
                --trade_penalty 0.0002 \
                --leverage_penalty 0.001 \
                --drawdown_penalty 0.1 \
                --track_reward_components
```

For systematic reward function tuning, use the provided utility script:

```bash
./scripts/tune_rewards.py --base_model_name tuning_experiment \
                          --n_episodes 100 \
                          --trade_penalties 0.0001 0.0005 0.001
```

You can also use a predefined set of experiments from a JSON configuration file:

```bash
./scripts/tune_rewards.py --base_model_name tuning_experiment \
                          --n_episodes 100 \
                          --experiments_file configs/reward_experiments.json
```

To visualize reward components from a trained model:

```bash
./scripts/visualize_reward_components.py --model_name my_model
```

### Parallel Training

To train multiple models in parallel with different configurations:

```bash
python train.py --parallel --config_file configs/parallel_config.json --max_parallel_jobs 4
```

This will train multiple models simultaneously using the configurations specified in the JSON file. Results and visualizations will be saved to the `parallel_training_results` directory.

You can also use the Makefile targets:

```bash
# Train multiple models in parallel
make train-multi-parallel

# Using Docker
make docker-train-multi-parallel
```

Example configuration file:
```json
[
  {
    "model_name": "baseline",
    "n_episodes": 500,
    "reward_type": "tunable",
    "realized_pnl_weight": 1.0,
    "unrealized_pnl_weight": 0.1,
    "trade_penalty": 0.0002
  },
  {
    "model_name": "high_unrealized",
    "n_episodes": 500,
    "reward_type": "tunable",
    "realized_pnl_weight": 1.0,
    "unrealized_pnl_weight": 0.3,
    "trade_penalty": 0.0002
  }
]
```

## Project Structure

- `agent/`: Reinforcement learning agent implementation
- `environment/`: Trading environment and reward functions
- `risk/`: Risk management strategies
- `backtest/`: Backtesting framework
- `tests/`: Unit and integration tests
