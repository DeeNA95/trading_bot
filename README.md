# Reinforcement Learning Trading Bot for Binance Futures

An advanced reinforcement learning agent for algorithmic trading on Binance Futures, designed for crypto market trading with a focus on BTC/USDT.

## Features

- Deep reinforcement learning agent with Proximal Policy Optimization (PPO)
- Support for various neural network architectures (CNN, LSTM, Transformer)
- Live trading on Binance Futures with proper risk management
- Position sizing based on volatility and risk metrics (fixed fraction, Kelly Criterion, volatility-adjusted)
- Integration of market data signals (funding rates, open interest, volatility)
- Stop loss and take profit management (handled by the execution logic)
- Dynamic leverage adjustment based on market conditions

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/DeeNA95/trading_bot.git
   cd trading_bot
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   - Create a `.env` file in the project root
   - Add your Binance API keys (for testnet and/or mainnet):

   ```
   # For testnet
   BINANCE_KEY=your_testnet_api_key
   BINANCE_SECRET=your_testnet_secret_key
   testnet_url=https://testnet.binancefuture.com

   # For mainnet (when ready for live trading)
   BINANCE_KEY=your_mainnet_api_key
   BINANCE_SECRET=your_mainnet_secret_key
   ```

## Usage

### 1. Generate training data

```bash
# Generate data for BTCUSDT with 15-minute candlesticks for the past 30 days
python data.py --symbol BTCUSDT --interval 15m --days 30 --output data/btcusdt_15m.csv

# Split data into training and testing sets
python data.py --symbol BTCUSDT --interval 15m --days 60 --split --output data/btcusdt_15m_split.csv
```

### 2. Train the agent

```bash
# Train with default parameters (often specified in a config file)
# Example using a simplified command structure, actual training might involve a config file:
python train.py --data_path data/btcusdt_15m_train.csv --symbol BTCUSDT --model_type lstm --train_episodes 100 --output_dir models/

# Train an encoder-only transformer (example)
# This typically involves setting num_decoder_layers to 0 in the model configuration.
# If using a JSON config: "num_decoder_layers": 0
python train.py --data_path data/btcusdt_15m_train.csv --symbol BTCUSDT --model_type transformer --train_episodes 200 --leverage 3 --window_size 50 --output_dir models/ --model_config '{"num_encoder_layers": 6, "num_decoder_layers": 0, "embedding_dim": 128}'
```
*Note: `train.py` accepts many arguments. For complex model configurations like specifying layer numbers, using a JSON config file passed via the `--config` argument or providing a JSON string to `--model_config` (as shown above) is recommended. The exact parameters for `--model_config` should match the `ModelConfig` dataclass fields.*

### 3. Run backtests

```bash
# Backtest a trained model
python backtest.py --symbol BTCUSDT --data_path data/btcusdt_15m_test.csv --model_path models/BTCUSDT_lstm_YYYYMMDD_HHMMSS/model.pt --model_type lstm --output_dir results/
```

### 4. Live trading

```bash
# Live trading on testnet
python inference.py --symbol BTCUSDT --model_path models/BTCUSDT_lstm_YYYYMMDD_HHMMSS/model.pt --model_type lstm --interval 1m --testnet --output_dir logs/

# Live trading on mainnet (use with caution!)
python inference.py --symbol BTCUSDT --model_path models/BTCUSDT_lstm_YYYYMMDD_HHMMSS/model.pt --model_type lstm --interval 1m --output_dir logs/
```

## Project Structure

- `rl_agent/` - Core RL agent implementation
  - `agent/` - PPO agent ([`ppo_agent.py`](rl_agent/agent/ppo_agent.py:0)), neural network models ([`models.py`](rl_agent/agent/models.py:0)), LSTM core ([`lstm_core.py`](rl_agent/agent/lstm_core.py:0)), feature extractors ([`feature_extractors.py`](rl_agent/agent/feature_extractors.py:0)), various attention mechanisms ([`attention/`](rl_agent/agent/attention/)), and transformer blocks ([`blocks/`](rl_agent/agent/blocks/))
  - `environment/` - Trading environment ([`trading_env.py`](rl_agent/environment/trading_env.py:0)), position sizing strategies ([`position_sizer.py`](rl_agent/environment/position_sizer.py:0)), trade execution logic ([`execution.py`](rl_agent/environment/execution.py:0)), and reward functions ([`reward.py`](rl_agent/environment/reward.py:0))
- `training/` - Scripts and utilities for training models
  - [`trainer.py`](training/trainer.py:0) - Main training loop and evaluation logic
  - [`model_factory.py`](training/model_factory.py:0) - Creates model instances (including `DynamicTransformerCore`)
  - [`load_preprocess_data.py`](training/load_preprocess_data.py:0) - Data loading for training
- [`data.py`](data.py:0) - Script for fetching, preprocessing, and saving market data
- [`train.py`](train.py:0) - Script for initiating the training process for the RL agent
- [`backtest.py`](backtest.py:0) - Script for backtesting trained models on historical data
- [`inference.py`](inference.py:0) - Script for running live trading with trained models
- [`requirements.txt`](requirements.txt:0) - Python package dependencies
- [`.env_example`](.env_example:0) - Example environment file for API keys (user should create `.env`)

## Model Architectures

### CNN
Convolutional Neural Networks are used for feature extraction from market data, capable of detecting spatial hierarchies of patterns. Implemented via various extractor classes like [`BasicConvExtractor`](rl_agent/agent/feature_extractors.py:14), [`ResNetExtractor`](rl_agent/agent/feature_extractors.py:246), and [`InceptionExtractor`](rl_agent/agent/feature_extractors.py:401).

### LSTM
Long Short-Term Memory networks are effective for sequence modeling and capturing temporal dependencies in time series data. The [`LSTMCore`](rl_agent/agent/lstm_core.py:5) provides the LSTM implementation.

### Transformer
Advanced architecture that captures long-range dependencies and complex relationships in market data using self-attention mechanisms. The implementation includes [`EncoderBlock`](rl_agent/agent/blocks/encoder_block.py:10) and [`DecoderBlock`](rl_agent/agent/blocks/decoder_block.py:10), various attention types (e.g., [`MultiHeadAttention`](rl_agent/agent/attention/multi_head_attention.py:6), [`PyramidalAttention`](rl_agent/agent/attention/pyramidal_attention.py:7)), and is dynamically constructed via [`DynamicTransformerCore`](training/model_factory.py:80) in the [`model_factory.py`](training/model_factory.py:0).

## Risk Management

The bot implements several risk management strategies:

1.  **Position Sizing**: Adjusts position size based on account balance, market volatility, and chosen strategy (e.g., Fixed Fraction, Kelly Criterion, Volatility-Adjusted). See [`position_sizer.py`](rl_agent/environment/position_sizer.py:0).
2.  **Stop Loss/Take Profit**: While not explicitly managed by discrete TP/SL orders placed on the exchange by default, the agent learns to exit positions. Custom logic for TP/SL can be integrated into the [`BinanceFuturesExecutor`](rl_agent/environment/execution.py:20) or trading environment.
3.  **Dynamic Leverage**: Adjusts leverage based on market volatility or other factors, configured in the environment. See [`calculate_adaptive_leverage`](rl_agent/environment/utils.py:9) and [`BinanceFuturesCryptoEnv`](rl_agent/environment/trading_env.py:33).
4.  **Risk-Adjusted Rewards**: Penalizes excess risk-taking during training through custom reward functions. See [`reward.py`](rl_agent/environment/reward.py:0).

## Performance Monitoring

During training and backtesting, the bot tracks various performance metrics:

- Total return and drawdown
- Win rate and average trade profitability
- Sharpe ratio and other risk-adjusted metrics
- These are typically logged and can be visualized using tools like TensorBoard or W&B (if integrated).

## Disclaimer

This software is for educational and research purposes only. Use at your own risk. The author(s) is(are) not responsible for any financial losses incurred by using this software. Always start with paper trading (testnet) before committing real capital.

## License

[MIT License](LICENSE)
