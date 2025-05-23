#!/usr/bin/env python
"""
Simplified inference script for running a trained RL agent on Binance Futures.
Handles model loading, data fetching, and trade execution while focusing on inference.
"""

import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple, Union

from networkx import volume

import numpy as np
import pandas as pd
import torch
from binance.um_futures import UMFutures
from dotenv import load_dotenv
from google.cloud import secretmanager, storage

from data import DataHandler
from rl_agent.agent.models import (ActorCriticCNN, ActorCriticLSTM,
                                   ActorCriticTransformer)
from rl_agent.environment.execution import BinanceFuturesExecutor

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('inference.log'), logging.StreamHandler()]
)
logger = logging.getLogger('inference')

def check_api_keys() -> Tuple[str, str]:
    """
    Check if the required API keys are set and return them.

    Returns:
        tuple: (api_key, api_secret) if keys are found, raises ValueError otherwise
    """
    BINANCE_KEY = None
    BINANCE_secret = None

    try:
        gcloud_client = secretmanager.SecretManagerServiceClient()
        PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "future-linker-456622-f8")

        BINANCE_KEY_response = gcloud_client.access_secret_version(
            name=f"projects/{PROJECT_ID}/secrets/BINANCE_API/versions/latest"
        )
        BINANCE_KEY = BINANCE_KEY_response.payload.data.decode("UTF-8").strip()

        BINANCE_secret_response = gcloud_client.access_secret_version(
            name=f"projects/{PROJECT_ID}/secrets/BINANCE_SECRET/versions/latest"
        )
        BINANCE_secret = BINANCE_secret_response.payload.data.decode(
            "UTF-8"
        ).strip()

        logger.info(
            "Successfully retrieved Binance credentials from Google Secret Manager"
        )

    except Exception as e:
        logger.error(f"Could not retrieve from Google Secret Manager: {e}")
        load_dotenv()
        BINANCE_KEY = os.getenv("BINANCE_KEY")
        BINANCE_secret = os.getenv("BINANCE_secret")
        logger.info("Falling back to .env file for Binance credentials")

    if not BINANCE_KEY or not BINANCE_secret:
        raise ValueError(
            "Binance API credentials not found in environment variables. "
            "Ensure that BINANCE_KEY and BINANCE_secret are set in your .env file."
        )



    # Set these in environment variables for other components to use
    os.environ["binance_api"] = BINANCE_KEY
    os.environ["BINANCE_secret"] = BINANCE_secret
    os.environ["binance_future_api"] = BINANCE_KEY
    os.environ["binance_future_secret"] = BINANCE_secret
    os.environ["binance_api2"] = BINANCE_KEY
    os.environ["BINANCE_secret2"] = BINANCE_secret

    return BINANCE_KEY, BINANCE_secret

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Run inference with a trained RL agent on Binance Futures'
    )

    # Model parameters
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='transformer',
        choices=['cnn', 'lstm', 'transformer'],
        help='Type of model architecture'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run inference on'
    )

    # Trading parameters
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading symbol'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='15m',
        choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'],
        help='Data interval'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=24,
        help='Number of candles for observation window'
    )
    parser.add_argument(
        '--leverage',
        type=int,
        default=2,
        help='Trading leverage'
    )
    parser.add_argument(
        '--risk_reward_ratio',
        type=float,
        default=1.5,
        help='Ratio of take profit to stop loss'
    )
    parser.add_argument(
        '--stop_loss_percent',
        type=float,
        default=0.01,
        help='Stop loss percentage from entry'
    )
    parser.add_argument(
        '--initial_balance',
        type=float,
        default=10000,
        help='Initial balance for trading'
    )
    parser.add_argument(
        '--base_url',
        type=str,
        default='https://fapi.binance.com',
        choices=['https://fapi.binance.com', 'https://testnet.binancefuture.com'],
        help='Binance Futures base URL'
    )

    # Execution parameters
    parser.add_argument(
        '--sleep_time',
        type=int,
        default=60,
        help='Time to sleep between iterations in seconds'
    )
    parser.add_argument(
        '--allow_scaling',
        action='store_true',
        help='Allow position scaling'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Run in dry-run mode (no actual trades)'
    )

    return parser.parse_args()

class InferenceAgent:
    """
    Lightweight agent for model inference in production.
    Handles model loading, data processing, and trade execution.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize inference agent with components needed for prediction and execution.

        Args:
            args: Parsed command line arguments containing configuration parameters
        """
        # Check and set up API keys
        api_key, api_secret = check_api_keys()

        self.model_path = args.model_path
        self.model_type = args.model_type.lower()
        self.symbol = args.symbol
        self.interval = args.interval
        self.window_size = args.window_size
        self.leverage = args.leverage
        self.risk_reward_ratio = args.risk_reward_ratio
        self.stop_loss_percent = args.stop_loss_percent
        self.initial_balance = args.initial_balance
        self.dry_run = args.dry_run
        self.base_url = args.base_url

        # Set device automatically if not specified
        if args.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = args.device

        logger.info(f'Initializing InferenceAgent with {self.model_type} model on {self.device}')

        # Initialize components
        self.model = self._load_model()
        self.data_handler = DataHandler()

        # Initialize executor with API configuration
        self.executor = BinanceFuturesExecutor(
            client=UMFutures(
                base_url=self.base_url,
                key=api_key,
                secret=api_secret
            ),
            symbol=self.symbol,
            leverage=self.leverage,
            risk_reward_ratio=self.risk_reward_ratio,
            stop_loss_percent=self.stop_loss_percent,
            dry_run=self.dry_run
        )

        # Track last update time
        self.last_update_time = datetime.now() - timedelta(minutes=10)
        self.processed_df = pd.DataFrame()  # Initialize as empty DataFrame instead of None

    def _load_model(self) -> Union[ActorCriticCNN, ActorCriticLSTM, ActorCriticTransformer]:
        """
        Load the trained model weights and initialize architecture.

        Returns:
            Initialized model with loaded weights
        """
        try:
            # Handle Google Cloud Storage paths
            if self.model_path.startswith('gs://'):
                try:
                    from google.cloud import storage
                    import io
                except ImportError:
                    raise ImportError('google-cloud-storage is required to load from GCS. Install with pip install google-cloud-storage')

                # Parse bucket and blob path
                path_parts = self.model_path[5:].split('/', 1)
                bucket_name = path_parts[0]
                blob_path = path_parts[1] if len(path_parts) > 1 else ''

                # Download from GCS to a buffer
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)

                buffer = io.BytesIO()
                blob.download_to_file(buffer)
                buffer.seek(0)

                # Load state dict from buffer with weights_only=False
                state_dict = torch.load(buffer, map_location=self.device, weights_only=False)
                logger.info(f'Successfully loaded model from GCS: {self.model_path}')
            else:
                # Load from local file
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
                logger.info(f'Successfully loaded model from local file: {self.model_path}')

            # Extract model configuration
            model_config = state_dict.get('model_config', {})
            input_dim = model_config.get('input_dim', 60)  # Default to standard feature dim
            hidden_dim = model_config.get('hidden_dim', 128)

            # Initialize appropriate model architecture
            if self.model_type == 'cnn':
                model = ActorCriticCNN(
                    input_shape=(self.window_size, input_dim),
                    action_dim=3,  # HOLD, BUY/LONG, SELL/SHORT
                    hidden_dim=hidden_dim,
                    device=self.device
                )
            elif self.model_type == 'lstm':
                model = ActorCriticLSTM(
                    input_shape=(self.window_size, input_dim),
                    action_dim=3,
                    hidden_dim=hidden_dim,
                    device=self.device
                )
            else:  # default to transformer
                model = ActorCriticTransformer(
                    input_shape=(self.window_size, input_dim),
                    action_dim=3,
                    hidden_dim=hidden_dim,
                    device=self.device
                )

            # Load weights
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)  # For older models that saved state dict directly

            model = model.to(self.device)
            model.eval()  # Set to evaluation mode

            return model

        except Exception as e:
            logger.error(f'Failed to load model: {e}')
            raise

    def update_market_data(self) -> bool:
        """
        Fetch and process latest market data.

        Returns:
            bool: True if data was updated successfully
        """
        try:
            current_time = datetime.now()

            # Calculate the interval in seconds
            interval_map = {
                '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600,
                '8h': 28800, '12h': 43200, '1d': 86400
            }
            interval_seconds = interval_map.get(self.interval, 900)  # Default to 15m

            # Only update if enough time has passed
            time_diff = (current_time - self.last_update_time).total_seconds()
            if time_diff < interval_seconds * 0.8 and not self.processed_df.empty:
                return True

            # Calculate time range for data fetch
            end_time = current_time
            start_time = end_time - timedelta(
                seconds=interval_seconds * (self.window_size + 1)  # Add buffer
            )

            logger.info(f'Fetching market data for {self.symbol} from {start_time} to {end_time}')

            # Use process_market_data for comprehensive data processing
            self.processed_df = self.data_handler.process_market_data(
                symbol=self.symbol,
                interval=self.interval,
                start_time=start_time,
                end_time=end_time
            )

            if self.processed_df.empty:
                logger.error(f'No data retrieved for {self.symbol}')
                return False

            self.last_update_time = current_time
            logger.info(f'Successfully updated market data with {len(self.processed_df.columns)} features')
            return True

        except Exception as e:
            logger.error(f'Error updating market data: {e}')
            return False

    def get_current_position(self):
        """Get current position and PnL from Binance"""
        try:
            # Get current position using correct method name
            position = self.executor.client.get_position_information(symbol=self.symbol)
            if position and len(position) > 0:
                position = position[0]
                return {
                    'position': float(position['positionAmt']),
                    'unrealized_pnl': float(position['unRealizedProfit']),
                    'leverage': float(position['leverage'])
                }
            return {'position': 0, 'unrealized_pnl': 0, 'leverage': 1}
        except Exception as e:
            logger.error(f'Error getting position: {e}')
            return {'position': 0, 'unrealized_pnl': 0, 'leverage': 1}

    def prepare_state(self) -> Optional[np.ndarray]:
        """
        Prepare the current state for inference.

        Returns:
            numpy array of the current state or None if preparation fails
        """
        try:
            if self.processed_df.empty or len(self.processed_df) < self.window_size:
                return None

            # Get the last window_size rows
            data = self.processed_df.iloc[-self.window_size:].copy()
            logger.info(f'Original columns: {data.columns}')

            # Drop non-numeric columns and Unnamed: 0 if present
            columns_to_drop = ['symbol']
            if 'Unnamed: 0' in data.columns:
                columns_to_drop.append('Unnamed: 0')
            if columns_to_drop:
                data = data.drop(columns=columns_to_drop)

            # Get current position and PnL
            position_info = self.get_current_position()

            # Add environment features
            data['balance'] = self.initial_balance # Use initial balance for now
            data['position'] = position_info['position']
            data['unrealized_pnl'] = position_info['unrealized_pnl']

            # Convert any remaining string columns to numeric
            for col in data.select_dtypes(include=['object']).columns:
                if col == 'trade_setup':
                    setup_map = {
                        'none': 0.0,
                        'strong_bullish': 1.0,
                        'strong_bearish': -1.0,
                        'bullish_reversal': 0.5,
                        'bearish_reversal': -0.5,
                    }
                    data[col] = data[col].map(setup_map).fillna(0)
                else:
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)


            # Convert to numpy array and handle any remaining NaN values
            state = data.values.astype(np.float32)
            state = np.nan_to_num(state, nan=0.0)  # Replace NaN with 0

            # Log feature count for debugging
            current_features = state.shape[1]
            logger.info(f'State shape: {state.shape}')

            return state

        except Exception as e:
            logger.error(f'Error preparing state: {e}')
            return None

    def predict(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Make a prediction for the given state.

        Args:
            state: Environment state as numpy array

        Returns:
            Dictionary containing action probabilities and value estimate
        """
        with torch.no_grad():
            try:
                # Convert state to tensor and ensure correct shape
                state_tensor = torch.FloatTensor(state).to(self.device)
                if len(state_tensor.shape) == 2:  # If shape is [window_size, features]
                    state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension

                # Get model predictions
                logits, value = self.model(state_tensor)  # Model returns 2 values
                action_probs = torch.softmax(logits, dim=-1)

                # Select action with highest probability
                action = int(torch.argmax(action_probs, dim=-1).item())

                return {
                    'action': action,
                    'action_probs': action_probs.cpu().numpy(),
                    'value': value.item()
                }

            except Exception as e:
                logger.error(f'Prediction failed: {e}')
                raise

    def run_inference_loop(
        self,
        sleep_time: int = 60,
        allow_scaling: bool = False
    ) -> None:
        """
        Run the main inference loop.

        Args:
            sleep_time: Time to sleep between iterations in seconds
            allow_scaling: Whether to allow position scaling
        """
        logger.info(
            f'Starting inference on {self.symbol} with {self.model_type} model\n'
            f'Trading parameters: Leverage={self.leverage}, '
            f'Stop Loss={self.stop_loss_percent*100}%, '
            f'Risk-Reward={self.risk_reward_ratio}, Interval={self.interval}'
        )

        if self.dry_run:
            logger.info('Running in DRY RUN mode - no actual trades will be executed')

        try:
            while True:
                # Update market data
                if not self.update_market_data():
                    logger.error('Failed to update market data, sleeping...')
                    time.sleep(sleep_time)
                    continue

                # Prepare state
                state = self.prepare_state()
                if state is None:
                    logger.error('Failed to prepare state, sleeping...')
                    time.sleep(sleep_time)
                    continue

                # Get prediction
                prediction = self.predict(state)
                action = prediction['action']
                action_probs = prediction['action_probs']

                # Log action probabilities
                action_names = ['HOLD', 'BUY/LONG', 'SELL/SHORT']
                probs_str = ', '.join(
                    f'{name}: {prob:.4f}'
                    for name, prob in zip(action_names, action_probs[0])
                )
                logger.info(f'Action probabilities: {probs_str}')
                logger.info(f'Selected action: {action_names[action]}')

                # Check position status
                position_status = self.executor.check_position_status()

                # Execute trade based on prediction
                if position_status['trigger_type'] in ['stop_loss', 'take_profit']:
                    logger.info(
                        f'Position was closed by {position_status["trigger_type"]}, '
                        'waiting for next signal...'
                    )
                else:
                    # Calculate trade size based on current balance
                    account_value = self.initial_balance  # In real trading, get this from exchange
                    trade_result = self.executor.execute_trade(
                        action=action,
                        usdt_amount=account_value,
                        scale_in=allow_scaling,
                        scale_out=allow_scaling
                    )

                    if trade_result['success']:
                        if trade_result['action'] not in ['hold', 'error']:
                            logger.info(
                                f'Trade executed: {trade_result["action"]} '
                                f'{trade_result["quantity"]} {self.symbol} @ {trade_result["price"]}'
                            )
                    else:
                        logger.error(f'Trade failed: {trade_result.get("message", "Unknown error")}')

                # Sleep before next iteration
                logger.info(f'Sleeping for {sleep_time} seconds...')
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info('Inference stopped by user')
        except Exception as e:
            logger.error(f'Error in inference loop: {e}')
        finally:
            # Clean up
            if not self.dry_run:
                self.executor.close_position()
            logger.info('Inference ended')

def main():
    """Main function to run the inference agent."""
    # Parse command line arguments
    args = parse_args()

    try:
        # Initialize agent
        agent = InferenceAgent(args)

        # Run inference loop
        agent.run_inference_loop(
            sleep_time=args.sleep_time,
            allow_scaling=args.allow_scaling
        )
    except KeyboardInterrupt:
        logger.info('Inference stopped by user')
    except Exception as e:
        logger.error(f'Error running inference: {e}')
        raise

if __name__ == '__main__':
    main()
