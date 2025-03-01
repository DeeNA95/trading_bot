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

## Project Structure

- `agent/`: Reinforcement learning agent implementation
- `environment/`: Trading environment and reward functions
- `risk/`: Risk management strategies
- `backtest/`: Backtesting framework
- `tests/`: Unit and integration tests
