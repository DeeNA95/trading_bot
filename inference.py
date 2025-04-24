#!/usr/bin/env python
"""
Simplified inference script for running a trained RL agent on Binance Futures.
Handles model loading, data fetching, and trade execution while focusing on inference.
"""
import torch
from rl_agent.agent.models import ActorCriticWrapper

import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple, Union


from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import pandas as pd
import torch
try:
    import torch_xla.core.xla_model as xm
    _HAS_XLA = True
except ImportError:
    _HAS_XLA = False
from binance.um_futures import UMFutures
from dotenv import load_dotenv
from google.cloud import secretmanager, storage

from data import DataHandler
from rl_agent.agent.models import (ActorCriticWrapper)  # Corrected import
from rl_agent.agent.attention.multi_head_attention import MultiHeadAttention
from rl_agent.agent.feedforward import FeedForward, MixtureOfExperts
from rl_agent.agent.embeddings import TimeEmbedding
from rl_agent.agent.blocks.encoder_block import EncoderBlock
from rl_agent.agent.blocks.decoder_block import DecoderBlock
from rl_agent.environment.execution import BinanceFuturesExecutor
import torch.nn as nn
from torch.nn import Sequential, Conv1d, GELU, ReLU, Linear, BatchNorm1d, LayerNorm, ModuleList, Dropout
from training.model_factory import DynamicTransformerCore

torch.serialization.add_safe_globals([
    ActorCriticWrapper, Sequential, Conv1d, GELU, ReLU, Linear, BatchNorm1d, LayerNorm, DynamicTransformerCore, TimeEmbedding, ModuleList,
    EncoderBlock, DecoderBlock, MultiHeadAttention, Dropout, MixtureOfExperts, FeedForward
])


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
        PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "zeta-turbine-457610-h4")

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
        BINANCE_secret = os.getenv("BINANCE_SECRET")
        logger.info("Falling back to .env file for Binance credentials")

    if not BINANCE_KEY or not BINANCE_secret:
        raise ValueError(
            "Binance API credentials not found in environment variables. "
            "Ensure that BINANCE_KEY and BINANCE_secret are set in your .env file."
        )




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
        '--scaler_path',
        type=str,
        required=True,
        help='Path to the saved scaler file (.joblib)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda','mps', 'xla'],
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
        self.scaler_path = args.scaler_path # Store scaler path
        self.symbol = args.symbol
        self.interval = args.interval
        self.window_size = args.window_size
        self.leverage = args.leverage
        self.risk_reward_ratio = args.risk_reward_ratio
        self.stop_loss_percent = args.stop_loss_percent
        self.initial_balance = args.initial_balance
        self.dry_run = args.dry_run
        self.base_url = args.base_url

        # Set device automatically if not specified, prioritizing TPU > GPU > MPS > CPU
        if args.device == 'auto':
            if _HAS_XLA:
                self.device = xm.xla_device()
                logger.info("Using TPU (XLA) device.")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA (GPU) device.")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                 self.device = torch.device("mps")
                 logger.info("Using MPS (Apple Silicon) device.")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU device.")
        else:
            # Use the device specified by the user argument
            self.device = torch.device(args.device)
            if args.device == 'xla' and not _HAS_XLA:
                 logger.warning("XLA device specified but torch_xla not found. Falling back to CPU.")
                 self.device = torch.device("cpu")

        logger.info(f'Initializing InferenceAgent on device: {self.device}')

        # Initialize components
        self.model = self._load_model()
        self.scaler = self._load_scaler() # Load the scaler
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

    def _load_model(self) -> Union[ActorCriticWrapper]:  # Corrected type hint
        """
        Load the trained model weights and initialize architecture based on args.
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

                # Load the checkpoint dictionary from buffer
                checkpoint = torch.load(buffer, map_location=self.device)
                logger.info(f'Successfully loaded checkpoint dictionary from GCS: {self.model_path}')
            else:
                # Load the checkpoint dictionary from local file
                checkpoint = torch.load(self.model_path, map_location=self.device)
                logger.info(f'Successfully loaded checkpoint dictionary from local file: {self.model_path}')

            # --- Reconstruct model from saved config ---
            if 'model_config' not in checkpoint or checkpoint['model_config'] is None:
                 raise ValueError("Model config not found in checkpoint. Cannot reconstruct model.")

            config_dict = checkpoint['model_config']
            # Convert ModelConfig dict back to ModelConfig object if needed by factory,
            # or just use the dict directly if factory accepts it.
            # Assuming DynamicTransformerCore can be built from config_dict

            # Determine input_dim dynamically or use a reasonable default/saved value if possible
            # For now, using the previous hardcoded value, but ideally, this should be saved/inferred
            # Check if n_features was saved in the config
            input_dim = config_dict.get('n_features', 59) # Use n_features from config if available
            embedding_dim = config_dict.get('embedding_dim', 128)
            dropout_rate = config_dict.get('dropout', 0.1)
            window_size = config_dict.get('window_size', self.window_size)
            architecture = config_dict.get('architecture', 'encoder_only')
            n_encoder_layers = config_dict.get('n_encoder_layers', 4)
            n_decoder_layers = config_dict.get('n_decoder_layers', 0) # Default 0 if not encoder_decoder

            logger.info(f"Reconstructing model: arch={architecture}, input_dim={input_dim}, embed_dim={embedding_dim}")

            # --- Replicate component creation from create_model ---

            # Attention
            attention_mapping = {"mha": MultiHeadAttention} # Add others if needed
            attention_class = attention_mapping.get(config_dict.get('attention_type', 'mha'))
            if not attention_class: raise ValueError(f"Unsupported attention type: {config_dict.get('attention_type')}")
            attention_args = {
                "embedding_dim": embedding_dim,
                "n_heads": config_dict.get('n_heads', 8),
                "dropout": dropout_rate
            }
            # Add specific args like n_latents, n_groups if needed based on attention_type

            # FFN
            ffn_mapping = {"standard": FeedForward, "moe": MixtureOfExperts} # Add others if needed
            ffn_class = ffn_mapping.get(config_dict.get('ffn_type', 'standard'))
            if not ffn_class: raise ValueError(f"Unsupported FFN type: {config_dict.get('ffn_type')}")
            ffn_args = {"dropout": dropout_rate}
            if config_dict.get('ffn_dim'): ffn_args["dim_feedforward"] = config_dict['ffn_dim']
            if config_dict.get('ffn_type') == "moe":
                 if config_dict.get('n_experts'): ffn_args["num_experts"] = config_dict['n_experts']
                 if config_dict.get('top_k'): ffn_args["top_k"] = config_dict['top_k']

            # Norm
            norm_mapping = {"layer_norm": nn.LayerNorm} # Add others if needed
            norm_class = norm_mapping.get(config_dict.get('norm_type', 'layer_norm'))
            if not norm_class: raise ValueError(f"Unsupported norm type: {config_dict.get('norm_type')}")
            norm_args = {} # Add specific args if needed

            # Shared components
            dropout_emb = nn.Dropout(dropout_rate)

            # Initialize component holders
            encoder_time_embedding = None
            encoder_blocks = None
            encoder_norm = None
            decoder_time_embedding = None
            decoder_blocks = None
            decoder_norm = None

            # Build Encoder components if needed
            if architecture in ["encoder_only", "encoder_decoder"]:
                if n_encoder_layers <= 0: raise ValueError("n_encoder_layers must be > 0")
                encoder_time_embedding = TimeEmbedding(embedding_dim, max_len=window_size)
                encoder_blocks = nn.ModuleList()
                for _ in range(n_encoder_layers):
                    block = EncoderBlock(
                        embedding_dim=embedding_dim,
                        attention_class=attention_class, attention_args=attention_args.copy(),
                        ffn_class=ffn_class, ffn_args=ffn_args.copy() if ffn_args else None,
                        norm_class=norm_class, norm_args=norm_args,
                        dropout=dropout_rate
                    )
                    encoder_blocks.append(block)
                encoder_norm = norm_class(embedding_dim, **norm_args)

            # Build Decoder components if needed
            if architecture in ["decoder_only", "encoder_decoder"]:
                if n_decoder_layers <= 0: raise ValueError("n_decoder_layers must be > 0")
                decoder_time_embedding = TimeEmbedding(embedding_dim, max_len=window_size)
                decoder_blocks = nn.ModuleList()
                for _ in range(n_decoder_layers):
                     block = DecoderBlock(
                         embedding_dim=embedding_dim,
                         self_attention_class=attention_class, self_attention_args=attention_args.copy(),
                         cross_attention_class=attention_class if architecture == "encoder_decoder" else None,
                         cross_attention_args=attention_args.copy() if architecture == "encoder_decoder" else None,
                         ffn_class=ffn_class, ffn_args=ffn_args.copy() if ffn_args else None,
                         norm_class=norm_class, norm_args=norm_args,
                         dropout=dropout_rate
                     )
                     decoder_blocks.append(block)
                decoder_norm = norm_class(embedding_dim, **norm_args)

            # Instantiate the dynamic core with built components
            core_transformer = DynamicTransformerCore(
                architecture=architecture,
                encoder_time_embedding=encoder_time_embedding,
                encoder_blocks=encoder_blocks,
                encoder_norm=encoder_norm,
                decoder_time_embedding=decoder_time_embedding,
                decoder_blocks=decoder_blocks,
                decoder_norm=decoder_norm,
                dropout_emb=dropout_emb,
                window_size=window_size,
                use_causal_mask_encoder=(architecture == "encoder_only"), # Match create_model logic
                use_causal_mask_decoder=True
            )

            # Create ActorCriticWrapper
            model = ActorCriticWrapper(
                input_shape=(window_size, input_dim),
                action_dim=3, # Assuming 3 actions (Hold, Buy, Sell) - Should ideally be saved/inferred
                transformer_core=core_transformer,
                feature_extractor_hidden_dim=config_dict.get('feature_extractor_dim', 64),
                embedding_dim=config_dict.get('embedding_dim', 128),
                device=self.device # Pass device here
            )
            # --- End Reconstruction ---

            # Load weights into the reconstructed model
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Successfully loaded model_state_dict into reconstructed model.")
            else:
                 raise ValueError("Checkpoint does not contain 'model_state_dict'.")


            # Ensure the model is on the correct device and in evaluation mode
            model = model.to(self.device) # Ensure model is explicitly moved to the target device
            model.eval() # MOVED HERE

            logger.info(f"Successfully loaded and reconstructed model from checkpoint on {self.device}.")
            return model

        except Exception as e: # Corrected indentation
            logger.error(f'Failed to load and reconstruct model: {e}')
            raise # Corrected indentation

    # --- _load_scaler method definition START ---
    def _load_scaler(self) -> StandardScaler:
        """Loads the pre-fitted scaler from the specified path."""
        logger.info(f"Loading scaler from {self.scaler_path}...")
        try:
            # Handle GCS paths for scaler
            if self.scaler_path.startswith('gs://'):
                try:
                    from google.cloud import storage
                    import io
                except ImportError:
                    raise ImportError('google-cloud-storage is required to load scaler from GCS.')

                path_parts = self.scaler_path[5:].split('/', 1)
                bucket_name = path_parts[0]
                blob_path = path_parts[1]

                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)

                buffer = io.BytesIO()
                blob.download_to_file(buffer)
                buffer.seek(0)
                scaler = joblib.load(buffer)
                logger.info("Scaler loaded successfully from GCS.")
            else:
                # Load from local path
                if not os.path.exists(self.scaler_path):
                    raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}")
                scaler = joblib.load(self.scaler_path)
                logger.info("Scaler loaded successfully from local file.")

            # Basic validation
            if not hasattr(scaler, 'transform'):
                 raise TypeError("Loaded object is not a valid scaler (missing transform method).")
            if not hasattr(scaler, 'mean_') or scaler.mean_ is None:
                 logger.warning("Loaded scaler does not appear to be fitted (missing mean_). Scaling might be incorrect.")

            return scaler
        except Exception as e:
            logger.error(f'Failed to load scaler: {e}. Inference will proceed without scaling!', exc_info=True)
            # Return an identity scaler or raise error depending on desired behavior
            class IdentityScaler:
                 def transform(self, X): return X
                 def fit(self, X): pass
                 def fit_transform(self, X): return X
            return IdentityScaler() # Return dummy scaler that does nothing
    # --- _load_scaler method definition END ---

    def update_market_data(self) -> bool: # Corrected indentation
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
                seconds=interval_seconds * (self.window_size + 200)  # Add buffer
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
            return True

        except Exception as e:
            logger.error(f'Error updating market data: {e}')
            return False

    def get_current_position(self):
        """Get current position and PnL from Binance"""
        try:
            # Get account information which includes positions
            account_info = self.executor.client.account()
            positions = account_info.get('positions', [])

            # Find the position for the specific symbol
            position_data = None
            for p in positions:
                if p.get('symbol') == self.symbol:
                    position_data = p
                    break # Found the symbol, exit loop

            if position_data:
                # Extract relevant data
                pos_amt = float(position_data.get('positionAmt', 0.0))
                unrealized_pnl = float(position_data.get('unrealizedProfit', 0.0))
                leverage = float(position_data.get('leverage', 1.0)) # Default leverage to 1 if not found

                # Filter out negligible positions often left by Binance
                # (Adjust threshold if necessary)
                if abs(pos_amt) < 1e-7: # Example threshold, adjust based on symbol's precision
                     pos_amt = 0.0
                     unrealized_pnl = 0.0 # Reset PnL if position is negligible

                return {
                    'position': pos_amt,
                    'unrealized_pnl': unrealized_pnl,
                    'leverage': leverage
                }
            else:
                # Symbol not found in positions, assume zero position
                logger.warning(f"Position data for symbol {self.symbol} not found in account info.")
                return {'position': 0.0, 'unrealized_pnl': 0.0, 'leverage': 1.0}

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
            # logger.info(f'Original columns: {data.columns}')
            data_cleaned = data.drop(columns=['close_time','symbol','trade_setup'], errors='ignore').copy()

            # Store index and columns before scaling
            original_index = data_cleaned.index
            original_columns = data_cleaned.columns

            # Scale the data (returns numpy array)
            scaled_np_data = self.scaler.transform(data_cleaned)

            # Convert scaled numpy array back to DataFrame
            scaled_data_df = pd.DataFrame(scaled_np_data, index=original_index, columns=original_columns)

            # Get current position and PnL
            position_info = self.get_current_position()

            # Add environment features to the DataFrame
            scaled_data_df['balance'] = self.initial_balance # Use initial balance for now
            scaled_data_df['position'] = position_info['position']
            scaled_data_df['unrealized_pnl'] = position_info['unrealized_pnl']

            # Convert any remaining string columns to numeric (should ideally not happen after scaling)
            for col in scaled_data_df.select_dtypes(include=['object']).columns:
                 logger.warning(f"Column '{col}' has object type after scaling, converting to numeric.")
                 scaled_data_df[col] = pd.to_numeric(scaled_data_df[col], errors='coerce').fillna(0)

            # --- Ensure final feature set matches expectation (no 'trade_setup' etc.) ---
            # This check is now less critical as we dropped before scaling, but good practice.
            final_columns = scaled_data_df.columns # Keep track of final columns
            # logger.info(f"Final columns before converting to numpy: {final_columns.tolist()}")

            # Convert final DataFrame to numpy array and handle any remaining NaN values
            state = scaled_data_df.values.astype(np.float32)
            state = np.nan_to_num(state, nan=0.0)  # Replace NaN with 0

            # Log feature count for debugging
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
            f'Starting inference on {self.symbol}\n'
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
        # finally:
        #     # Clean up
        #     if not self.dry_run:
        #         self.executor.close_position()
        #     logger.info('Inference ended')

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
