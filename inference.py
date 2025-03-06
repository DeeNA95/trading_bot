#!/usr/bin/env python
"""
Inference script for running a trained RL agent on Binance Futures paper trading.

This script loads a trained model and uses it to make trading decisions on
Binance Futures paper trading environment using the testnet API.
"""

import os
import time
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from dotenv import load_dotenv

from rl_agent.environment.trading_env import BinanceFuturesCryptoEnv
from rl_agent.agent.ppo_agent import PPOAgent
from rl_agent.agent.models import ActorCriticCNN, ActorCriticLSTM, ActorCriticTransformer
from data import DataHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("inference")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a trained RL agent on Binance Futures")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model file")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["cnn", "lstm", "transformer"],
                        help="Type of model architecture")
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Trading symbol")
    parser.add_argument("--window_size", type=int, default=24,
                        help="Observation window size")
    parser.add_argument("--leverage", type=int, default=2,
                        help="Trading leverage")
    parser.add_argument("--interval", type=str, default="15m",
                        help="Data fetch interval")
    parser.add_argument("--initial_balance", type=float, default=10000,
                        help="Initial balance for the trading account")
    parser.add_argument("--stop_loss", type=float, default=0.01,
                        help="Stop loss percentage")
    parser.add_argument("--risk_reward", type=float, default=1.5,
                        help="Risk-reward ratio (take profit to stop loss)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run in dry-run mode (no actual trades)")
    parser.add_argument("--sleep_time", type=int, default=60,
                        help="Sleep time between iterations in seconds")

    return parser.parse_args()

def check_api_keys():
    """
    Check if the required API keys are set in the .env file.

    Returns:
        bool: True if keys are set, False otherwise
    """
    # api_key = os.getenv("binance_future_testnet_api")
    # api_secret = os.getenv("binance_future_testnet_secret")
    api_key = os.getenv("binance_api2")
    api_secret = os.getenv("binance_secret2")

    if not api_key or not api_secret:
        logger.error("Binance Futures testnet API keys not found in .env file")
        logger.error("Please set binance_future_testnet_api and binance_future_testnet_secret in your .env file")
        logger.error("You can use the .env.template file as a reference")
        return False

    logger.info("API keys found in .env file")
    return True

class CustomBinanceFuturesCryptoEnv(BinanceFuturesCryptoEnv):
    """
    Enhanced BinanceFuturesCryptoEnv using DataHandler for comprehensive data processing.
    This ensures all 57 metrics from the data.py module are available during inference.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the DataHandler for processing market data
        self.data_handler = DataHandler()
        # Track the last time we updated the data
        self.last_update_time = datetime.now() - timedelta(minutes=10)
        # Store processed data frames
        self.processed_df = None

    def _update_live_state(self):
        """
        Override the _update_live_state method to use DataHandler for processing.
        This ensures all the same metrics as in training data are calculated.
        """
        try:
            # Only fetch new data if enough time has passed (avoid API rate limits)
            current_time = datetime.now()
            time_diff = (current_time - self.last_update_time).total_seconds()

            # Determine fetch interval in seconds
            interval_map = {
                "1m": 60, "3m": 180, "5m": 300, "15m": 900,
                "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
                "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400
            }
            interval_seconds = interval_map.get(self.data_fetch_interval, 900)  # Default to 15m

            # Fetch new data if interval has passed or processed_df is None
            if time_diff >= interval_seconds * 0.8 or self.processed_df is None:
                logger.info(f"Fetching new market data for {self.symbol}...")

                # Set time range for data fetch (window_size + buffer candles)
                end_time = current_time
                start_time = end_time - timedelta(
                    minutes=int(interval_seconds/60) * (self.window_size + 10)
                )

                # Use DataHandler to get and process data with all metrics
                raw_data = self.data_handler.get_futures_data(
                    symbol=self.symbol,
                    interval=self.data_fetch_interval,
                    start_time=start_time,
                    end_time=end_time
                )

                # Apply all the same transformations as in data.py
                processed_data = self.data_handler.calculate_technical_indicators(raw_data)
                processed_data = self.data_handler.calculate_risk_metrics(processed_data)
                processed_data = self.data_handler.identify_trade_setups(processed_data)

                # Try to add futures-specific metrics if possible
                try:
                    processed_data = self.data_handler.add_futures_metrics(
                        processed_data, self.symbol, self.data_fetch_interval,
                        start_time, end_time
                    )
                except Exception as e:
                    logger.warning(f"Could not add futures metrics: {e}")

                # Fill missing values
                processed_data = processed_data.ffill().bfill()

                # Replace any remaining NaN values with zeros
                processed_data = processed_data.fillna(0)

                # Store the processed dataframe
                self.processed_df = processed_data
                self.last_update_time = current_time

                num_features = processed_data.shape[1]
                logger.info(f"Data updated successfully with {num_features} features")
                if num_features < 57:
                    logger.warning(f"Expected 57 features, but got only {num_features}. Some metrics may be missing.")
            # Get the last window_size rows from processed data
            if len(self.processed_df) >= self.window_size:
                data = self.processed_df.iloc[-self.window_size:].copy()
            else:
                # If we don't have enough data, pad with the first row
                missing_rows = self.window_size - len(self.processed_df)
                first_row = self.processed_df.iloc[[0]]
                padding = pd.concat([first_row] * missing_rows)
                data = pd.concat([padding, self.processed_df])

            # Drop non-numeric columns like 'symbol' that can't be converted to float
            if 'symbol' in data.columns:
                data = data.drop(columns=['symbol'])

            # Convert any remaining string columns to numeric if possible
            for col in data.select_dtypes(include=['object']).columns:
                if col == 'trade_setup':
                    # Convert trade_setup to numeric (0=none, 1=bullish, 2=bearish, etc)
                    setup_map = {
                        'none': 0,
                        'strong_bullish': 1,
                        'strong_bearish': 2,
                        'bullish_reversal': 3,
                        'bearish_reversal': 4
                    }
                    data[col] = data[col].map(setup_map).fillna(0)
                else:
                    # Try to convert other object columns to numeric
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

            # Extract features
            feature_data = data.values

            # Add account information
            balance = np.ones(self.window_size) * self.balance
            position = np.ones(self.window_size) * self.current_position
            unrealized_pnl = np.ones(self.window_size) * self.unrealized_pnl

            # Combine into state and ensure it's float32
            try:
                combined_data = np.column_stack(
                    (
                        feature_data,
                        balance.reshape(-1, 1),
                        position.reshape(-1, 1),
                        unrealized_pnl.reshape(-1, 1),
                    )
                )
                self.state = combined_data.astype(np.float32)
                logger.info(f"State successfully updated with shape {self.state.shape}")
            except ValueError as e:
                # Debug information
                logger.error(f"Error creating state: {e}")

                # Check for NaN or infinite values in each column
                nan_columns = []
                inf_columns = []
                for col in data.columns:
                    if data[col].isnull().any():
                        nan_columns.append(col)
                    if np.isinf(data[col].replace([np.inf, -np.inf], np.nan)).any():
                        inf_columns.append(col)

                if nan_columns:
                    logger.error(f"Columns with NaN values: {nan_columns}")
                if inf_columns:
                    logger.error(f"Columns with infinite values: {inf_columns}")

                # Check column data types
                logger.error(f"Data types in feature_data: {[(col, data[col].dtype) for col in data.columns]}")

                # Identify problem columns that can't be converted to float32
                problem_columns = []
                for col in data.columns:
                    try:
                        np.array(data[col].values, dtype=np.float32)
                    except:
                        problem_columns.append(col)

                if problem_columns:
                    logger.error(f"Problem columns: {problem_columns}")
                    # Try to fix by dropping problematic columns
                    data = data.drop(columns=problem_columns)
                    feature_data = data.values

                    # Try again with the fixed data
                    combined_data = np.column_stack(
                        (
                            feature_data,
                            balance.reshape(-1, 1),
                            position.reshape(-1, 1),
                            unrealized_pnl.reshape(-1, 1),
                        )
                    )
                    self.state = combined_data.astype(np.float32)
                    logger.info(f"Recovered by dropping problem columns. New state shape: {self.state.shape}")

            # Update date for reference
            self.date = current_time

            # Extract market data for environment
            if not data.empty:
                if 'volatility' in data.columns:
                    self.current_volatility = data.iloc[-1]['volatility']
                if 'fundingRate' in data.columns:
                    self.current_funding_rate = data.iloc[-1]['fundingRate']
                if 'sumOpenInterest' in data.columns:
                    self.current_open_interest = data.iloc[-1]['sumOpenInterest']
                if 'oi_change' in data.columns:
                    self.open_interest_change = data.iloc[-1]['oi_change']

        except Exception as e:
            logger.error(f"Error updating live state: {e}", exc_info=True)


def load_agent(args):
    """
    Load the trained agent from a model file.

    Args:
        args: Command line arguments

    Returns:
        Loaded PPO agent
    """
    logger.info(f"Loading model from {args.model_path}")

    # First load the model to get its input shape
    try:
        # Try with weights_only=False
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        logger.warning(f"Could not load with weights_only=False, trying with weights_only=True: {e}")
        # If that fails, try with weights_only=True
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=True)

    # Extract the input shape based on model type
    input_dim = None
    model_state = checkpoint['model_state_dict']

    if args.model_type.lower() == 'lstm':
        if 'lstm.weight_ih_l0' in model_state:
            lstm_weights = model_state['lstm.weight_ih_l0']
            input_dim = lstm_weights.shape[1]
        else:
            raise ValueError("LSTM weights not found in model state dict")
    elif args.model_type.lower() == 'cnn':
        # For CNN, find the first conv layer to determine input shape
        for key in model_state:
            if 'cnn_layers.0.weight' in key:
                conv_weights = model_state[key]
                input_dim = conv_weights.shape[1] * args.window_size
                break
        if input_dim is None:
            # Fallback method: try to infer from linear layer dimensions
            for key in model_state:
                if 'actor.0.weight' in key:
                    # Infer from the first linear layer in actor network
                    linear_weights = model_state[key]
                    input_dim = linear_weights.shape[1]
                    break
        if input_dim is None:
            raise ValueError("Could not detect input dimension from CNN model")
    elif args.model_type.lower() == 'transformer':
        # For transformer, look at input embedding layer
        if 'input_embedding.weight' in model_state:
            embedding_weights = model_state['input_embedding.weight']
            input_dim = embedding_weights.shape[1] * args.window_size
        else:
            raise ValueError("Transformer input embedding weights not found")

    if input_dim is None:
        raise ValueError(f"Could not determine input dimension for model type: {args.model_type}")

    logger.info(f"Detected input dimension from model: {input_dim}")

    # Create a custom environment that uses DataHandler for comprehensive data processing
    env = CustomBinanceFuturesCryptoEnv(
        window_size=args.window_size,
        symbol=args.symbol,
        leverage=args.leverage,
        mode="trade",  # Live trading mode
        # base_url="https://testnet.binancefuture.com",  # Use testnet
        base_url='https://fapi.binance.com',  # Use mainnet
        margin_type="ISOLATED",
        risk_reward_ratio=args.risk_reward,
        stop_loss_percent=args.stop_loss,
        data_fetch_interval=args.interval,
        initial_balance=args.initial_balance,
        dry_run=args.dry_run
    )

    # Create a custom agent with the correct input shape
    class CustomPPOAgent(PPOAgent):
        def __init__(self, env, model_type, hidden_dim, save_dir, input_dim, lr=3e-4):
            # Pass lr parameter explicitly to avoid TypeError
            super().__init__(env=env, model_type=model_type, hidden_dim=hidden_dim, lr=lr, save_dir=save_dir)
            # Store model type and input dimension
            self.model_type = model_type
            self.hidden_dim = hidden_dim

            # Handle input dimension based on model type
            if model_type.lower() == 'lstm':
                # For LSTM, input_dim is the feature dimension for each timestep
                self.expected_feature_dim = input_dim
                self.input_shape = (env.window_size, input_dim)
            elif model_type.lower() == 'cnn':
                # For CNN, typical shapes are (channels, timesteps) or (timesteps, channels)
                # We need to ensure the dimension matches what the model expects
                features_per_timestep = input_dim // env.window_size
                self.expected_feature_dim = input_dim
                self.input_shape = (features_per_timestep, env.window_size)
                logger.info(f"CNN input shape: {self.input_shape} (features_per_timestep, window_size)")
            else:
                # For transformer and other models
                self.expected_feature_dim = input_dim
                self.input_shape = (env.window_size, input_dim // env.window_size)

            # Recreate the model based on model_type
            self._initialize_model()

        def _initialize_model(self):
            """Initialize the model with the correct input shape."""
            # Based on the model type, initialize the appropriate model
            if self.model_type.lower() == 'cnn':
                self.model = ActorCriticCNN(
                    input_shape=self.input_shape,
                    action_dim=self.action_dim,
                    hidden_dim=self.hidden_dim,
                    device=self.device
                )
            elif self.model_type.lower() == 'lstm':
                self.model = ActorCriticLSTM(
                    input_shape=self.input_shape,  # Use the exact shape from the trained model
                    action_dim=self.action_dim,
                    hidden_dim=self.hidden_dim,
                    device=self.device
                )
            elif self.model_type.lower() == 'transformer':
                self.model = ActorCriticTransformer(
                    input_shape=self.input_shape,
                    action_dim=self.action_dim,
                    hidden_dim=self.hidden_dim,
                    device=self.device
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Recreate optimizer with the new model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        def choose_action(self, state):
            """Override to handle input dimension mismatch and check for NaN values."""
            # Check for NaN values in the state
            if np.isnan(state).any():
                logger.warning(f"NaN values detected in the state, replacing with zeros")
                state = np.nan_to_num(state, nan=0.0)

            # If state dimension doesn't match expected, pad or transform it
            if state.shape[1] != self.expected_feature_dim:
                # Log dimension mismatch
                logger.info(f"State dimension mismatch. Got {state.shape[1]}, expected {self.expected_feature_dim}")
                # Pad the state to match expected dimensions
                padded_state = self._pad_state(state)

                # Double-check for NaNs in padded state
                if np.isnan(padded_state).any():
                    logger.warning(f"NaN values detected in padded state, replacing with zeros")
                    padded_state = np.nan_to_num(padded_state, nan=0.0)

                try:
                    return super().choose_action(padded_state)
                except Exception as e:
                    logger.error(f"Error in choose_action with padded state: {e}")
                    # As a fallback, return a conservative action (HOLD)
                    return 0, np.array([1.0, 0.0, 0.0]), None
            else:
                try:
                    return super().choose_action(state)
                except Exception as e:
                    logger.error(f"Error in choose_action: {e}")
                    # As a fallback, return a conservative action (HOLD)
                    return 0, np.array([1.0, 0.0, 0.0]), None

        def _pad_state(self, state):
            """Pad the state to match the expected dimensions."""
            # Get current and expected dimensions
            window_size, current_features = state.shape
            padding_size = self.expected_feature_dim - current_features

            if padding_size <= 0:
                return state

            # Create padding of zeros
            padding = np.zeros((window_size, padding_size), dtype=np.float32)

            # Concatenate state with padding
            padded_state = np.concatenate([state, padding], axis=1)

            logger.info(f"Padded state from {current_features} to {self.expected_feature_dim} features")

            # For CNN models, need to reshape the state differently
            if self.model_type.lower() == 'cnn':
                # For CNN, need to reshape to [batch_size, channels, height, width]
                # Since the model was trained with a specific input shape, we need to match it
                logger.info(f"Reshaping for CNN model. Input shape: {self.input_shape}")

                # The CNN likely expects input as [batch, channels, timesteps]
                # where channels = features per timestep
                features_per_timestep = self.input_shape[1]

                # If we have more features than expected, truncate
                if padded_state.shape[1] > self.expected_feature_dim:
                    padded_state = padded_state[:, :self.expected_feature_dim]

                # Reshape to [1, channels, timesteps]
                try:
                    # Add batch dimension first
                    batched = np.expand_dims(padded_state, axis=0)

                    # CNN models typically expect [batch, channels, time] format
                    # Reshape to match the expected format
                    # For 1D CNN: [batch, channels, timesteps]
                    # where channels = features per timestep
                    reshaped = np.transpose(batched, (0, 2, 1))
                    logger.info(f"CNN input reshaped to {reshaped.shape}")
                    return reshaped
                except Exception as e:
                    logger.error(f"Error reshaping for CNN: {e}. Using original padded state.")

            return padded_state

    # Create agent with the appropriate model type and input dimension
    agent = CustomPPOAgent(
        env=env,
        model_type=args.model_type,
        hidden_dim=128,
        save_dir=os.path.dirname(args.model_path),
        input_dim=input_dim
    )

    # Load the trained model
    agent.load(args.model_path)
    logger.info(f"Model loaded successfully with input shape {agent.input_shape}")

    # For CNN models, let's add extra debugging to verify the expected shape
    if args.model_type.lower() == 'cnn':
        # Inspect the first convolutional layer to confirm input shape expectations
        if hasattr(agent.model, 'cnn_layers') and len(agent.model.cnn_layers) > 0:
            first_conv = agent.model.cnn_layers[0]
            logger.info(f"First CNN layer info: in_channels={first_conv.in_channels}, "
                        f"out_channels={first_conv.out_channels}, kernel_size={first_conv.kernel_size}")
            logger.info(f"This confirms the model expects input with {first_conv.in_channels} channels")

            # Update expected feature dim if needed based on actual model architecture
            agent.expected_feature_dim = first_conv.in_channels * agent.input_shape[1]
            logger.info(f"Updated expected_feature_dim to {agent.expected_feature_dim} based on CNN architecture")

    return agent, env

def run_inference(agent, env, args):
    """
    Run the inference loop.

    Args:
        agent: Trained PPO agent
        env: Trading environment
        args: Command line arguments
    """
    logger.info(f"Starting inference on {args.symbol} with {args.model_type} model")
    logger.info(f"Trading parameters: Leverage={args.leverage}, Stop Loss={args.stop_loss*100}%, "
                f"Risk-Reward={args.risk_reward}, Interval={args.interval}")

    if args.dry_run:
        logger.info("Running in DRY RUN mode - no actual trades will be executed")

    # Reset the environment
    state, _ = env.reset()

    # Log state dimensions to ensure we're getting the right data
    logger.info(f"State shape: {state.shape}")
    if isinstance(env, CustomBinanceFuturesCryptoEnv) and env.processed_df is not None:
        logger.info(f"Number of features in processed data: {env.processed_df.shape[1]}")
        # Show the feature names for debugging
        logger.info(f"Features: {list(env.processed_df.columns)}")

    # Main inference loop
    try:
        while True:
            # Get action from the agent
            action, action_prob, _ = agent.choose_action(state)

            # Log the action probabilities
            action_names = ["HOLD", "BUY/LONG", "SELL/SHORT"]

            # Check if action_prob is already a scalar (not an array)
            if isinstance(action_prob, (float, np.float32, np.float64)):
                logger.info(f"Action probability: {action_prob:.4f}")
            else:
                # If it's an array, log each probability
                logger.info(f"Action probabilities: {action_names[0]}: {action_prob[0]:.4f}, "
                            f"{action_names[1]}: {action_prob[1]:.4f}, {action_names[2]}: {action_prob[2]:.4f}")

            logger.info(f"Selected action: {action_names[action]}")

            # Take the action in the environment
            next_state, reward, done, _, info = env.step(action)

            # Log the result
            logger.info(f"Reward: {reward:.4f}")
            logger.info(f"Account value: {info['account_value']:.2f} USDT")
            logger.info(f"Position: {info['position']:.8f} {args.symbol} "
                        f"(Direction: {'+' if info['position_direction'] > 0 else '-' if info['position_direction'] < 0 else '0'})")

            if info['position_direction'] != 0:
                logger.info(f"Entry price: {info['entry_price']:.2f} USDT")
                logger.info(f"Unrealized PnL: {info['unrealized_pnl']:.2f} USDT")
                if 'liquidation_price' in info and info['liquidation_price'] > 0:
                    logger.info(f"Liquidation price: {info['liquidation_price']:.2f} USDT")

            # Update state
            state = next_state

            # Sleep to avoid excessive API calls
            logger.info(f"Sleeping for {args.sleep_time} seconds...")
            time.sleep(args.sleep_time)

    except KeyboardInterrupt:
        logger.info("Inference stopped by user")
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
    finally:
        # Clean up
        env.close()
        logger.info("Inference ended")

def main():
    """Main function."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    args = parse_args()

    # Check if API keys are set
    if not check_api_keys():
        return

    # Load agent and environment
    agent, env = load_agent(args)

    # Run inference
    run_inference(agent, env, args)

if __name__ == "__main__":
    main()
