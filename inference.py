#!/usr/bin/env python
"""
Inference script for running a trained RL agent on Binance Futures.

This script loads a trained model and uses it to make trading decisions on
Binance Futures paper trading environment using the testnet API.
"""

import argparse
import logging
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from google.cloud import secretmanager, storage

from data import DataHandler
from rl_agent.agent.models import (ActorCriticCNN, ActorCriticLSTM,
                                   ActorCriticTransformer)
from rl_agent.agent.ppo_agent import PPOAgent
from rl_agent.environment.trading_env import BinanceFuturesCryptoEnv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("inference.log"), logging.StreamHandler()],
)
logger = logging.getLogger("inference")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with a trained RL agent on Binance Futures"
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model file"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="lstm",
        choices=["cnn", "lstm", "transformer"],
        help="Type of model architecture",
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument(
        "--window_size", type=int, default=24, help="Observation window size"
    )
    parser.add_argument("--leverage", type=int, default=2, help="Trading leverage")
    parser.add_argument(
        "--interval", type=str, default="15m", help="Data fetch interval"
    )
    parser.add_argument(
        "--initial_balance",
        type=float,
        default=10000,
        help="Initial balance for the trading account",
    )
    parser.add_argument(
        "--stop_loss", type=float, default=0.01, help="Stop loss percentage"
    )
    parser.add_argument(
        "--risk_reward",
        type=float,
        default=1.5,
        help="Risk-reward ratio (take profit to stop loss)",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Run in dry-run mode (no actual trades)"
    )
    parser.add_argument(
        "--sleep_time",
        type=int,
        default=60,
        help="Sleep time between iterations in seconds",
    )
    parser.add_argument(
        "--allow_scaling",
        action="store_true",
        help="Allow position scaling (in/out) during inference",
    )

    return parser.parse_args()


def check_api_keys():
    """
    Check if the required API keys are set and return them.

    Returns:
        tuple: (api_key, api_secret) if keys are found, raises ValueError otherwise
    """
    binance_key = None
    binance_secret = None

    try:
        gcloud_client = secretmanager.SecretManagerServiceClient()
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "seraphic-bliss-451413-c8")

        binance_key_response = gcloud_client.access_secret_version(
            name=f"projects/{project_id}/secrets/BINANCE_API_KEY/versions/latest"
        )
        binance_key = binance_key_response.payload.data.decode("UTF-8").strip()

        binance_secret_response = gcloud_client.access_secret_version(
            name=f"projects/{project_id}/secrets/BINANCE_SECRET_KEY/versions/latest"
        )
        binance_secret = binance_secret_response.payload.data.decode("UTF-8").strip()

        logger.info(
            "Successfully retrieved Binance credentials from Google Secret Manager"
        )

    except Exception as e:
        logger.error(f"Could not retrieve from Google Secret Manager: {e}")
        load_dotenv()
        # Try multiple possible environment variable names
        binance_key = (
            os.getenv("binance_api")
            or os.getenv("binance_future_api")
            or os.getenv("binance_api2")
        )
        binance_secret = (
            os.getenv("binance_secret")
            or os.getenv("binance_future_secret")
            or os.getenv("binance_secret2")
        )
        logger.info("Falling back to .env file for Binance credentials")

    if not binance_key or not binance_secret:
        raise ValueError(
            "Binance API credentials not found in environment variables. "
            "Ensure that API credentials are set in your .env file or Google Secret Manager."
        )

    # Set these in environment variables for other components to use
    os.environ["binance_api"] = binance_key
    os.environ["binance_secret"] = binance_secret
    os.environ["binance_future_api"] = binance_key
    os.environ["binance_future_secret"] = binance_secret
    os.environ["binance_api2"] = binance_key
    os.environ["binance_secret2"] = binance_secret

    return binance_key, binance_secret


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

    def process_inference_data(self, symbol, interval, window_size, start_time=None, end_time=None):
        """
        Comprehensive function to process market data for inference.
        This ensures all the same metrics as in training data are calculated.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Data interval (e.g., '15m', '1h')
            window_size: Number of candles needed for observation window
            start_time: Start time for data fetch (default: calculated based on window_size)
            end_time: End time for data fetch (default: current time)
            
        Returns:
            DataFrame: Processed data with all metrics
        """
        # Set default times if not provided
        if end_time is None:
            end_time = datetime.now()
            
        if start_time is None:
            # Calculate start time based on interval and window size plus buffer
            interval_map = {
                "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
                "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
                "8h": 28800, "12h": 43200, "1d": 86400,
            }
            interval_seconds = interval_map.get(interval, 900)  # Default to 15m
            buffer_candles = 10  # Extra candles as buffer
            
            # Set start time to window_size + buffer candles ago
            start_time = end_time - timedelta(
                seconds=interval_seconds * (window_size + buffer_candles)
            )
        
        logger.info(f"Processing inference data for {symbol} from {start_time} to {end_time}")
        
        try:
            # Fetch raw data
            raw_data = self.data_handler.get_futures_data(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
            )
            
            if raw_data.empty:
                logger.error(f"No data retrieved for {symbol}")
                return pd.DataFrame()
                
            # Process data with all available metrics
            logger.info("Calculating technical indicators...")
            processed_data = self.data_handler.calculate_technical_indicators(raw_data)
            
            logger.info("Calculating risk metrics...")
            processed_data = self.data_handler.calculate_risk_metrics(processed_data, interval=interval)
            
            logger.info("Identifying trade setups...")
            processed_data = self.data_handler.identify_trade_setups(processed_data)
            
            # Try to add futures-specific metrics
            try:
                logger.info("Adding futures-specific metrics...")
                processed_data = self.data_handler.add_futures_metrics(
                    processed_data,
                    symbol,
                    interval,
                    start_time,
                    end_time,
                )
            except Exception as e:
                logger.warning(f"Could not add futures metrics: {e}")
            
            # Normalize data
            logger.info("Normalizing OHLC and price-related data...")
            processed_data = self.data_handler.normalise_ohlc(processed_data)
            
            # Fill missing values
            processed_data = processed_data.ffill().bfill()
            
            # Replace any remaining NaN values with zeros
            processed_data = processed_data.fillna(0)
            
            num_features = processed_data.shape[1]
            logger.info(f"Data processed successfully with {num_features} features")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing inference data: {e}", exc_info=True)
            return pd.DataFrame()

    def _update_live_state(self):
        """
        Override the _update_live_state method to use the process_inference_data function.
        """
        try:
            # Only fetch new data if enough time has passed (avoid API rate limits)
            current_time = datetime.now()
            time_diff = (current_time - self.last_update_time).total_seconds()

            # Determine fetch interval in seconds
            interval_map = {
                "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
                "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
                "8h": 28800, "12h": 43200, "1d": 86400,
            }
            interval_seconds = interval_map.get(
                self.data_fetch_interval, 900
            )  # Default to 15m

            # Fetch new data if interval has passed or processed_df is None
            if time_diff >= interval_seconds * 0.8 or self.processed_df is None:
                logger.info(f"Fetching new market data for {self.symbol}...")

                # Process market data using the consolidated function
                self.processed_df = self.process_inference_data(
                    symbol=self.symbol,
                    interval=self.data_fetch_interval,
                    window_size=self.window_size
                )
                
                self.last_update_time = current_time

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
            if "symbol" in data.columns:
                data = data.drop(columns=["symbol"])

            # Convert any remaining string columns to numeric if possible
            for col in data.select_dtypes(include=["object"]).columns:
                if col == "trade_setup":
                    # Convert trade_setup to numeric (0=none, 1=bullish, 2=bearish, etc)
                    setup_map = {
                        "none": 0,
                        "strong_bullish": 1,
                        "strong_bearish": 2,
                        "bullish_reversal": 3,
                        "bearish_reversal": 4,
                    }
                    data[col] = data[col].map(setup_map).fillna(0)
                else:
                    # Try to convert other object columns to numeric
                    data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

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
                logger.error(
                    f"Data types in feature_data: {[(col, data[col].dtype) for col in data.columns]}"
                )

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
                    logger.info(
                        f"Recovered by dropping problem columns. New state shape: {self.state.shape}"
                    )

            # Update date for reference
            self.date = current_time

            # Extract market data for environment
            if not data.empty:
                if "volatility" in data.columns:
                    self.current_volatility = data.iloc[-1]["volatility"]
                if "fundingRate" in data.columns:
                    self.current_funding_rate = data.iloc[-1]["fundingRate"]
                if "sumOpenInterest" in data.columns:
                    self.current_open_interest = data.iloc[-1]["sumOpenInterest"]
                if "oi_change" in data.columns:
                    self.open_interest_change = data.iloc[-1]["oi_change"]
                if "price_density" in data.columns:
                    self.price_density = data.iloc[-1]["price_density"]
                if "fractal_dimension" in data.columns:
                    self.fractal_dimension = data.iloc[-1]["fractal_dimension"]

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
    if args.model_path.startswith("gs://"):
        try:
            # Import Google Cloud Storage
            import io

            from google.cloud import storage

            # Parse bucket and blob path
            path_parts = args.model_path[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1] if len(path_parts) > 1 else ""

            # Download from GCS to a buffer
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            buffer = io.BytesIO()
            blob.download_to_file(buffer)
            buffer.seek(0)

            # Try to load with weights_only=False
            try:
                checkpoint = torch.load(buffer, map_location="cpu", weights_only=False)
            except Exception as e:
                logger.warning(
                    f"Could not load with weights_only=False, trying with weights_only=True: {e}"
                )
                buffer.seek(0)
                checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)

            logger.info(f"Model loaded from GCS: {args.model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model from GCS: {e}")
    else:
        # Load from local file
        try:
            # Try with weights_only=False
            checkpoint = torch.load(
                args.model_path, map_location="cpu", weights_only=False
            )
        except Exception as e:
            logger.warning(
                f"Could not load with weights_only=False, trying with weights_only=True: {e}"
            )
            # If that fails, try with weights_only=True
            checkpoint = torch.load(
                args.model_path, map_location="cpu", weights_only=True
            )

    # Extract the input shape based on model type
    input_dim = None
    model_state = checkpoint["model_state_dict"]

    if args.model_type.lower() == "lstm":
        if "lstm.weight_ih_l0" in model_state:
            lstm_weights = model_state["lstm.weight_ih_l0"]
            input_dim = lstm_weights.shape[1]
        else:
            raise ValueError("LSTM weights not found in model state dict")
    elif args.model_type.lower() == "cnn":
        # For CNN, find the first conv layer to determine input shape
        for key in model_state:
            if "cnn_layers.0.weight" in key:
                conv_weights = model_state[key]
                input_dim = conv_weights.shape[1] * args.window_size
                break
        if input_dim is None:
            # Fallback method: try to infer from linear layer dimensions
            for key in model_state:
                if "actor.0.weight" in key:
                    # Infer from the first linear layer in actor network
                    linear_weights = model_state[key]
                    input_dim = linear_weights.shape[1]
                    break
        if input_dim is None:
            raise ValueError("Could not detect input dimension from CNN model")
    elif args.model_type.lower() == "transformer":
        # For transformer, look at input embedding layer
        if "input_embedding.weight" in model_state:
            embedding_weights = model_state["input_embedding.weight"]
            input_dim = embedding_weights.shape[1] * args.window_size
        else:
            raise ValueError("Transformer input embedding weights not found")

    if input_dim is None:
        raise ValueError(
            f"Could not determine input dimension for model type: {args.model_type}"
        )

    logger.info(f"Detected input dimension from model: {input_dim}")

    # Create a custom environment that uses DataHandler for comprehensive data processing
    env = CustomBinanceFuturesCryptoEnv(
        window_size=args.window_size,
        symbol=args.symbol,
        leverage=args.leverage,
        mode="trade",  # Live trading mode
        # base_url="https://testnet.binancefuture.com",  # Use testnet
        base_url="https://fapi.binance.com",  # Use mainnet
        margin_type="ISOLATED",
        risk_reward_ratio=args.risk_reward,
        stop_loss_percent=args.stop_loss,
        data_fetch_interval=args.interval,
        initial_balance=args.initial_balance,
        dry_run=args.dry_run,
    )

    # Create a custom agent with the correct input shape
    class CustomPPOAgent(PPOAgent):
        def __init__(self, env, model_type, hidden_dim, save_dir, input_dim, lr=3e-4):
            # Pass lr parameter explicitly to avoid TypeError
            super().__init__(
                env=env,
                model_type=model_type,
                hidden_dim=hidden_dim,
                lr=lr,
                save_dir=save_dir,
            )
            # Store model type and input dimension
            self.model_type = model_type
            self.hidden_dim = hidden_dim

            # Handle input dimension based on model type
            if model_type.lower() == "lstm":
                # For LSTM, input_dim is the feature dimension for each timestep
                self.expected_feature_dim = input_dim
                self.input_shape = (env.window_size, input_dim)
            elif model_type.lower() == "cnn":
                # For CNN, typical shapes are (channels, timesteps) or (timesteps, channels)
                # We need to ensure the dimension matches what the model expects
                features_per_timestep = input_dim // env.window_size
                self.expected_feature_dim = input_dim
                self.input_shape = (features_per_timestep, env.window_size)
                logger.info(
                    f"CNN input shape: {self.input_shape} (features_per_timestep, window_size)"
                )
            else:
                # For transformer and other models
                self.expected_feature_dim = input_dim
                self.input_shape = (env.window_size, input_dim // env.window_size)

            # Recreate the model based on model_type
            self._initialize_model()

        def _initialize_model(self):
            """Initialize the model with the correct input shape."""
            # Based on the model type, initialize the appropriate model
            if self.model_type.lower() == "cnn":
                self.model = ActorCriticCNN(
                    input_shape=self.input_shape,
                    action_dim=self.action_dim,
                    hidden_dim=self.hidden_dim,
                    device=self.device,
                )
            elif self.model_type.lower() == "lstm":
                self.model = ActorCriticLSTM(
                    input_shape=self.input_shape,  # Use the exact shape from the trained model
                    action_dim=self.action_dim,
                    hidden_dim=self.hidden_dim,
                    device=self.device,
                )
            elif self.model_type.lower() == "transformer":
                self.model = ActorCriticTransformer(
                    input_shape=self.input_shape,
                    action_dim=self.action_dim,
                    hidden_dim=self.hidden_dim,
                    device=self.device,
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Recreate optimizer with the new model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        def choose_action(self, state):
            """Override to handle input dimension mismatch and check for NaN values."""
            # Check for NaN values in the state
            if np.isnan(state).any():
                logger.warning(
                    f"NaN values detected in the state, replacing with zeros"
                )
                state = np.nan_to_num(state, nan=0.0)

            # If state dimension doesn't match expected, pad or transform it
            if state.shape[1] != self.expected_feature_dim:
                # Log dimension mismatch
                logger.info(
                    f"State dimension mismatch. Got {state.shape[1]}, expected {self.expected_feature_dim}"
                )
                # Pad the state to match expected dimensions
                padded_state = self._pad_state(state)

                # Double-check for NaNs in padded state
                if np.isnan(padded_state).any():
                    logger.warning(
                        f"NaN values detected in padded state, replacing with zeros"
                    )
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

            logger.info(
                f"Padded state from {current_features} to {self.expected_feature_dim} features"
            )

            # For CNN models, need to reshape the state differently
            if self.model_type.lower() == "cnn":
                # For CNN, need to reshape to [batch_size, channels, height, width]
                # Since the model was trained with a specific input shape, we need to match it
                logger.info(f"Reshaping for CNN model. Input shape: {self.input_shape}")

                # The CNN likely expects input as [batch, channels, timesteps]
                # where channels = features per timestep
                features_per_timestep = self.input_shape[1]

                # If we have more features than expected, truncate
                if padded_state.shape[1] > self.expected_feature_dim:
                    padded_state = padded_state[:, : self.expected_feature_dim]

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
                    logger.error(
                        f"Error reshaping for CNN: {e}. Using original padded state."
                    )

            return padded_state

    # Create agent with the appropriate model type and input dimension
    agent = CustomPPOAgent(
        env=env,
        model_type=args.model_type,
        hidden_dim=128,
        save_dir=os.path.dirname(args.model_path),
        input_dim=input_dim,
    )

    # Load the trained model
    agent.load(args.model_path)
    logger.info(f"Model loaded successfully with input shape {agent.input_shape}")

    # For CNN models, let's add extra debugging to verify the expected shape
    if args.model_type.lower() == "cnn":
        # Inspect the first convolutional layer to confirm input shape expectations
        if hasattr(agent.model, "cnn_layers") and len(agent.model.cnn_layers) > 0:
            first_conv = agent.model.cnn_layers[0]
            logger.info(
                f"First CNN layer info: in_channels={first_conv.in_channels}, "
                f"out_channels={first_conv.out_channels}, kernel_size={first_conv.kernel_size}"
            )
            logger.info(
                f"This confirms the model expects input with {first_conv.in_channels} channels"
            )

            # Update expected feature dim if needed based on actual model architecture
            agent.expected_feature_dim = first_conv.in_channels * agent.input_shape[1]
            logger.info(
                f"Updated expected_feature_dim to {agent.expected_feature_dim} based on CNN architecture"
            )

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
    logger.info(
        f"Trading parameters: Leverage={args.leverage}, Stop Loss={args.stop_loss*100}%, "
        f"Risk-Reward={args.risk_reward}, Interval={args.interval}"
    )

    if args.dry_run:
        logger.info("Running in DRY RUN mode - no actual trades will be executed")

    if args.allow_scaling:
        logger.info(
            "Position scaling is ENABLED - will attempt to scale in/out of positions"
        )
    else:
        logger.info(
            "Position scaling is DISABLED - all trades will open/close full positions"
        )

    # Reset the environment
    state, _ = env.reset()

    # Log state dimensions to ensure we're getting the right data
    logger.info(f"State shape: {state.shape}")
    if isinstance(env, CustomBinanceFuturesCryptoEnv) and env.processed_df is not None:
        logger.info(
            f"Number of features in processed data: {env.processed_df.shape[1]}"
        )
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
                logger.info(
                    f"Action probabilities: {action_names[0]}: {action_prob[0]:.4f}, "
                    f"{action_names[1]}: {action_prob[1]:.4f}, {action_names[2]}: {action_prob[2]:.4f}"
                )

            logger.info(f"Selected action: {action_names[action]}")

            # Initialize info if it's the first loop iteration
            if "info" not in locals():
                info = {"position_direction": 0, "unrealized_pnl": 0}

            # Determine if we should scale in/out based on position and action
            scale_in = False
            scale_out = False
            scale_percentage = 0.5  # Default to 50%

            # Only consider scaling if the flag is enabled
            if args.allow_scaling:
                # Get current position info
                has_position = (
                    info["position_direction"] != 0
                    if "position_direction" in info
                    else False
                )
                position_direction = (
                    info["position_direction"] if "position_direction" in info else 0
                )
                unrealized_pnl = (
                    info["unrealized_pnl"] if "unrealized_pnl" in info else 0
                )

                # Determine scaling based on PnL and action consistency
                if has_position:
                    # For scaling into winning positions (matching direction and positive PnL)
                    if (
                        action == 1 and position_direction > 0 and unrealized_pnl > 0
                    ) or (
                        action == 2 and position_direction < 0 and unrealized_pnl > 0
                    ):
                        scale_in = True
                        logger.info(
                            f"Scaling INTO winning position by {scale_percentage*100:.0f}%"
                        )

                    # For scaling out of losing positions (PnL negative)
                    elif unrealized_pnl < 0:
                        scale_out = True
                        logger.info(
                            f"Scaling OUT OF losing position by {scale_percentage*100:.0f}%"
                        )
            elif (
                "info" in locals()
                and "position_direction" in info
                and info["position_direction"] != 0
            ):
                # If scaling is disabled but we have a position and matching action, log the decision
                position_direction = info["position_direction"]
                if (action == 1 and position_direction > 0) or (
                    action == 2 and position_direction < 0
                ):
                    logger.info(
                        "Matching position detected but scaling is disabled - not scaling in"
                    )

            # Take the action in the environment with scaling parameters
            next_state, reward, done, _, info = env.step(
                action,
                scale_in=scale_in,
                scale_out=scale_out,
                scale_percentage=scale_percentage,
            )

            # Log the result
            logger.info(f"Reward: {reward:.4f}")
            logger.info(f"Account value: {info['account_value']:.2f} USDT")
            logger.info(
                f"Position: {info['position']:.8f} {args.symbol} "
                f"(Direction: {'+' if info['position_direction'] > 0 else '-' if info['position_direction'] < 0 else '0'})"
            )

            # Log scaling actions if they occurred
            if "scale_in" in info and info["scale_in"]:
                logger.info(
                    f"Successfully scaled INTO position by {info['scale_percentage']*100:.0f}%"
                )
            if "scale_out" in info and info["scale_out"]:
                logger.info(
                    f"Successfully scaled OUT OF position by {info['scale_percentage']*100:.0f}%"
                )

            if info["position_direction"] != 0:
                logger.info(f"Entry price: {info['entry_price']:.2f} USDT")
                logger.info(f"Unrealized PnL: {info['unrealized_pnl']:.2f} USDT")
                if "liquidation_price" in info and info["liquidation_price"] > 0:
                    logger.info(
                        f"Liquidation price: {info['liquidation_price']:.2f} USDT"
                    )

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
    load_dotenv()  # TODO: take away when settled on cloud provider

    # Parse arguments
    args = parse_args()

    # Check if API keys are set and get them
    api_key, api_secret = check_api_keys()

    # Set these in environment variables for other components to use
    os.environ["binance_api"] = api_key
    os.environ["binance_secret"] = api_secret
    os.environ["binance_future_api"] = api_key
    os.environ["binance_future_secret"] = api_secret
    os.environ["binance_api2"] = api_key
    os.environ["binance_secret2"] = api_secret

    # Load agent and environment
    agent, env = load_agent(args)

    # Run inference
    run_inference(agent, env, args)


if __name__ == "__main__":
    main()
