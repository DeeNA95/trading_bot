#!/usr/bin/env python
"""
Backtesting script for trained RL models.
Loads models from GCP and runs backtests on historical data.
"""

import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('backtest.log'), logging.StreamHandler()]
)
logger = logging.getLogger('backtest')

# Try to import Google Cloud Storage
try:
    from google.cloud import storage  # type: ignore
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning('Google Cloud Storage not available. Install with: pip install google-cloud-storage')

from data import DataHandler
from rl_agent.agent.models import (ActorCriticCNN, ActorCriticLSTM,
                                 ActorCriticTransformer)

# Load environment variables
load_dotenv()

class BacktestAgent:
    """
    Agent for running backtests with trained RL models.
    Handles model loading, data processing, and performance evaluation.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize backtest agent with components needed for testing.

        Args:
            args: Parsed command line arguments containing configuration parameters
        """
        self.model_path = args.model_path
        self.model_type = args.model_type.lower()
        self.symbol = args.symbol
        self.interval = args.interval
        self.window_size = args.window_size
        self.initial_balance = args.initial_balance
        self.leverage = args.leverage
        self.risk_reward_ratio = args.risk_reward_ratio
        self.stop_loss_percent = args.stop_loss_percent
        self.gcp_data_path = args.gcp_data_path

        # Set device automatically if not specified
        if args.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = args.device

        logger.info(f'Initializing BacktestAgent with {self.model_type} model on {self.device}')

        # Initialize components
        self.model = self._load_model()
        self.data_handler = DataHandler()

        # Initialize storage client for GCP
        self.storage_client = storage.Client()

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
                # Load from local file with weights_only=False
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
                logger.info(f'Successfully loaded model from local file: {self.model_path}')

            # Extract model configuration
            model_config = state_dict.get('model_config', {})
            input_dim = model_config.get('input_dim', 56)  # Default to 56 features to match training
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
                model.load_state_dict(state_dict)

            model = model.to(self.device)
            model.eval()  # Set to evaluation mode

            return model

        except Exception as e:
            logger.error(f'Failed to load model: {e}')
            raise

    def load_historical_data(self) -> pd.DataFrame:
        """
        Load historical data from GCP for backtesting.

        Returns:
            DataFrame containing historical data
        """
        try:
            # Parse bucket and blob path from GCP path
            path_parts = self.gcp_data_path[5:].split('/', 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1] if len(path_parts) > 1 else ''

            # Load data from GCP
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            # Download to temporary file
            local_path = f'temp_{self.symbol}_{self.interval}.csv'
            blob.download_to_filename(local_path)

            # Load data
            df = pd.read_csv(local_path)

            # Clean up temporary file
            os.remove(local_path)

            # Use DataHandler to process the data
            processed_df = self.data_handler.process_market_data(
                symbol=self.symbol,
                interval=self.interval,
                start_time=pd.to_datetime('2025-03-13'),
                end_time=pd.to_datetime('2025-03-21')
            )

            if not isinstance(processed_df, pd.DataFrame):
                raise ValueError("DataHandler returned invalid data type")

            logger.info(f'Loaded {len(processed_df)} rows of historical data')
            return processed_df

        except Exception as e:
            logger.error(f'Error loading historical data: {e}')
            raise

    def prepare_state(self, data: pd.DataFrame, index: int) -> Optional[np.ndarray]:
        """
        Prepare state for model input.

        Args:
            data: Historical data DataFrame
            index: Current index in the data

        Returns:
            numpy array of the current state or None if preparation fails
        """
        try:
            if index < self.window_size:
                return None

            # Get the window of data
            window_data = data.iloc[index-self.window_size:index].copy()

            # Drop non-numeric columns
            columns_to_drop = ['symbol']
            if 'Unnamed: 0' in window_data.columns:
                columns_to_drop.append('Unnamed: 0')
            window_data = window_data.drop(columns=columns_to_drop)

            # Add environment features
            window_data['balance'] = self.initial_balance
            window_data['position'] = 0  # Will be updated based on previous actions
            window_data['unrealized_pnl'] = 0  # Will be updated based on previous actions

            # Convert any remaining string columns to numeric
            for col in window_data.select_dtypes(include=['object']).columns:
                if col == 'trade_setup':
                    setup_map = {
                        'none': 0.0,
                        'strong_bullish': 1.0,
                        'strong_bearish': -1.0,
                        'bullish_reversal': 0.5,
                        'bearish_reversal': -0.5,
                    }
                    window_data[col] = window_data[col].map(setup_map)
                else:
                    window_data[col] = pd.to_numeric(window_data[col], errors='coerce')

                # Fill NaN values with 0
                window_data[col] = window_data[col].fillna(0)

            # Normalize data
            for col in window_data.columns:
                median = window_data[col].median()
                iqr = window_data[col].quantile(0.75) - window_data[col].quantile(0.25)
                if iqr > 0:
                    window_data[col] = (window_data[col] - median) / (iqr + 1e-8)
                else:
                    window_data[col] = window_data[col] - median

            # Convert to numpy array and handle any remaining NaN values
            state = window_data.values.astype(np.float32)
            state = np.nan_to_num(state, nan=0.0)

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
                logits, value = self.model(state_tensor)
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

    def run_backtest(self) -> Dict[str, Any]:
        """
        Run backtest on historical data.

        Returns:
            Dictionary containing backtest results and metrics
        """
        try:
            # Load historical data
            data = self.load_historical_data()
            logger.info(f'Starting backtest with {len(data)} data points')

            # Initialize tracking variables
            balance = self.initial_balance
            position = 0
            trades = []
            equity_curve = []
            total_pnl = 0
            trade_count = 0

            # Run through historical data
            for i in range(len(data)):
                # Prepare state
                state = self.prepare_state(data, i)
                if state is None:
                    continue

                # Get model prediction
                prediction = self.predict(state)
                action = prediction['action']
                action_names = ['HOLD', 'BUY/LONG', 'SELL/SHORT']
                logger.debug(f'Action probabilities: {prediction["action_probs"]}')
                logger.debug(f'Selected action: {action_names[action]}')

                # Execute trade based on prediction
                current_price = data.iloc[i]['close']

                if action == 1:  # BUY/LONG
                    if position <= 0:  # Close short if exists
                        if position < 0:
                            pnl = (current_price - entry_price) * abs(position)
                            balance += pnl
                            total_pnl += pnl
                            trade_count += 1
                            logger.info(f'[Trade {trade_count}] CLOSE SHORT: Price={current_price:.2f}, PnL={pnl:.2f}, Balance={balance:.2f}')
                            trades.append({
                                'action': 'close_short',
                                'price': current_price,
                                'pnl': pnl
                            })
                        # Open long
                        position = 1
                        entry_price = current_price
                        trade_count += 1
                        logger.info(f'[Trade {trade_count}] OPEN LONG: Price={current_price:.2f}, Balance={balance:.2f}')
                        trades.append({
                            'action': 'open_long',
                            'price': current_price,
                            'pnl': 0
                        })

                elif action == 2:  # SELL/SHORT
                    if position >= 0:  # Close long if exists
                        if position > 0:
                            pnl = (entry_price - current_price) * position
                            balance += pnl
                            total_pnl += pnl
                            trade_count += 1
                            logger.info(f'[Trade {trade_count}] CLOSE LONG: Price={current_price:.2f}, PnL={pnl:.2f}, Balance={balance:.2f}')
                            trades.append({
                                'action': 'close_long',
                                'price': current_price,
                                'pnl': pnl
                            })
                        # Open short
                        position = -1
                        entry_price = current_price
                        trade_count += 1
                        logger.info(f'[Trade {trade_count}] OPEN SHORT: Price={current_price:.2f}, Balance={balance:.2f}')
                        trades.append({
                            'action': 'open_short',
                            'price': current_price,
                            'pnl': 0
                        })

                # Record equity
                unrealized_pnl = 0
                if position != 0:
                    if position > 0:
                        unrealized_pnl = (current_price - entry_price) * position
                    else:
                        unrealized_pnl = (entry_price - current_price) * abs(position)

                equity_curve.append({
                    'equity': balance + unrealized_pnl
                })

                # Log progress every 1000 iterations
                if (i + 1) % 1000 == 0:
                    logger.info(f'Progress: {i+1}/{len(data)} iterations, Current Balance: {balance:.2f}, Total PnL: {total_pnl:.2f}')

            # Calculate performance metrics
            equity_df = pd.DataFrame(equity_curve)
            returns = equity_df['equity'].pct_change().dropna()

            metrics = {
                'initial_balance': self.initial_balance,
                'final_balance': balance,
                'total_return': (balance - self.initial_balance) / self.initial_balance,
                'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(equity_df['equity']),
                'total_trades': len(trades),
                'win_rate': self._calculate_win_rate(trades),
                'equity_curve': equity_curve,
                'trades': trades
            }

            logger.info(f'\nBacktest completed:')
            logger.info(f'Total trades: {trade_count}')
            logger.info(f'Total PnL: {total_pnl:.2f}')
            logger.info(f'Final balance: {balance:.2f}')
            logger.info(f'Return: {(balance - self.initial_balance) / self.initial_balance * 100:.2f}%')

            return metrics

        except Exception as e:
            logger.error(f'Error running backtest: {e}')
            raise

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve."""
        rolling_max = equity.expanding().max()
        drawdowns = equity / rolling_max - 1
        min_drawdown = drawdowns.min()
        return float(abs(min_drawdown)) if isinstance(min_drawdown, (int, float)) else 0.0

    def _calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trade history."""
        if not trades:
            return 0.0
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        return winning_trades / len(trades)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run backtest with trained RL model')

    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model file')
    parser.add_argument('--model_type', type=str, default='transformer',
                      choices=['cnn', 'lstm', 'transformer'],
                      help='Type of model architecture')
    parser.add_argument('--device', type=str, default='auto',
                      choices=['auto', 'cpu', 'cuda'],
                      help='Device to run inference on')

    # Data parameters
    parser.add_argument('--gcp_data_path', type=str, required=True,
                      help='Full GCP path to historical data file (e.g., gs://your-bucket/data/BTCUSDT_15m.csv)')

    # Trading parameters
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                      help='Trading symbol')
    parser.add_argument('--interval', type=str, default='15m',
                      choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'],
                      help='Data interval')
    parser.add_argument('--window_size', type=int, default=60,
                      help='Number of candles for observation window')
    parser.add_argument('--leverage', type=int, default=20,
                      help='Trading leverage')
    parser.add_argument('--risk_reward_ratio', type=float, default=1.5,
                      help='Ratio of take profit to stop loss')
    parser.add_argument('--stop_loss_percent', type=float, default=0.01,
                      help='Stop loss percentage from entry')
    parser.add_argument('--initial_balance', type=float, default=3,
                      help='Initial balance for trading')

    return parser.parse_args()

def main():
    """Main function to run the backtest."""
    # Parse command line arguments
    args = parse_args()

    try:
        # Initialize agent
        agent = BacktestAgent(args)

        # Run backtest
        results = agent.run_backtest()

        # Print results
        logger.info('\nBacktest Results:')
        logger.info(f'Initial Balance: ${results["initial_balance"]:,.2f}')
        logger.info(f'Final Balance: ${results["final_balance"]:,.2f}')
        logger.info(f'Total Return: {results["total_return"]*100:.2f}%')
        logger.info(f'Sharpe Ratio: {results["sharpe_ratio"]:.2f}')
        logger.info(f'Max Drawdown: {results["max_drawdown"]*100:.2f}%')
        logger.info(f'Total Trades: {results["total_trades"]}')
        logger.info(f'Win Rate: {results["win_rate"]*100:.2f}%')

        # Save results to file
        results_file = f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        pd.DataFrame(results['trades']).to_csv(f'trades_{results_file}.csv', index=False)
        pd.DataFrame(results['equity_curve']).to_csv(f'equity_{results_file}.csv', index=False)

        logger.info(f'\nDetailed results saved to {results_file}')

    except Exception as e:
        logger.error(f'Error running backtest: {e}')
        raise

if __name__ == '__main__':
    main()
