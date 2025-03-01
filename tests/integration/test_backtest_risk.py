"""
Integration tests for backtesting and risk management.
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests.mocks.backtest_mock import MockBacktestSimulator
from risk.position_sizing import FixedFractionPositionSizer, KellyPositionSizer
from unittest.mock import MagicMock, patch


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, position_sizer, action_sequence=None):
        """Initialize with a position sizer and optional action sequence.
        
        Args:
            position_sizer: Position sizing strategy
            action_sequence: Optional sequence of actions to return
        """
        self.position_sizer = position_sizer
        self.action_sequence = action_sequence or []
        self.current_idx = 0
        
    def choose_action(self, state):
        """Choose an action based on the current state.
        
        Args:
            state: Environment state
            
        Returns:
            Action, log probability, and value
        """
        if self.action_sequence and self.current_idx < len(self.action_sequence):
            action = self.action_sequence[self.current_idx]
            self.current_idx += 1
        else:
            # Default action: random
            action = np.random.uniform(-1, 1, 4)
            
        # Apply position sizing
        price = getattr(state, 'price', 100.0)
            
        balance = 10000.0  # Dummy balance
        
        # Adjust size component based on position sizer
        position_size = self.position_sizer.calculate_position_size(
            account_balance=balance,
            signal_strength=0.8,
            volatility=1.0,
            price=price
        )
        
        # Scale to [-1, 1]
        scaled_size = min(position_size / balance, 1.0) * np.sign(action[0])
        
        # Update action with calculated size
        if len(action) > 1:
            action[1] = scaled_size
        
        return action, np.array([0.1]), np.array([100.0])


class TestBacktestRiskIntegration(unittest.TestCase):
    """Test the integration of backtesting and risk management."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test dataset
        dates = pd.date_range(start='2022-01-01', periods=100)
        
        # Generate a simple trend with some volatility
        prices = np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)
        
        self.data = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.uniform(0, 5, 100),
            'low': prices - np.random.uniform(0, 5, 100),
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 100),
            'rsi': np.random.uniform(0, 100, 100),
            'macd': np.random.uniform(-5, 5, 100),
            'macd_signal': np.random.uniform(-5, 5, 100),
            'atr': np.random.uniform(1, 5, 100),
        }, index=dates)
        
        # Initialize backtest simulator
        self.simulator = MockBacktestSimulator(
            initial_balance=10000.0,
            commission=0.001
        )
        
    def test_fixed_fraction_sizing_integration(self):
        """Test integration of fixed fraction position sizing with backtesting."""
        # Create fixed fraction position sizer
        position_sizer = FixedFractionPositionSizer(max_risk_pct=0.05)
        
        # Create mock agent with the position sizer
        agent = MockAgent(position_sizer=position_sizer)
        
        # Run backtest
        metrics = self.simulator.run_backtest(
            agent=agent,
            start_idx=0,
            end_idx=50,
            window_size=10
        )
        
        # Check that backtest ran successfully
        self.assertIn('total_return', metrics)
        self.assertIn('total_trades', metrics)
        
    def test_kelly_sizing_integration(self):
        """Test integration of Kelly position sizing with backtesting."""
        # Create Kelly position sizer
        position_sizer = KellyPositionSizer(max_kelly_pct=0.5)
        
        # Create predetermined win rate and risk:reward
        position_sizer.calculate_position_size = MagicMock(return_value=1000.0)
        
        # Create mock agent with the position sizer
        agent = MockAgent(position_sizer=position_sizer)
        
        # Run backtest
        metrics = self.simulator.run_backtest(
            agent=agent,
            start_idx=0,
            end_idx=50,
            window_size=10
        )
        
        # Check that backtest ran successfully
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        
        # Check that position sizer was called
        self.assertGreater(position_sizer.calculate_position_size.call_count, 0)
        
    def test_different_sizing_strategies(self):
        """Test and compare different position sizing strategies."""
        results = {}
        
        # Test fixed fraction sizer with different fractions
        for risk_pct in [0.01, 0.05, 0.1]:
            position_sizer = FixedFractionPositionSizer(max_risk_pct=risk_pct)
            agent = MockAgent(position_sizer=position_sizer)
            
            # Reset simulator metrics
            self.simulator.reset_metrics()
            
            # Run backtest
            metrics = self.simulator.run_backtest(
                agent=agent,
                start_idx=0,
                end_idx=50,
                window_size=10
            )
            
            results[f'fixed_fraction_{risk_pct}'] = {
                'total_return': metrics['total_return'],
                'max_drawdown': metrics['max_drawdown'],
                'sharpe_ratio': metrics['sharpe_ratio']
            }
        
        # Test with Kelly sizer (if we had real win rates and risk:reward)
        position_sizer = KellyPositionSizer(max_kelly_pct=0.5)
        position_sizer.calculate_position_size = MagicMock(return_value=500.0)
        agent = MockAgent(position_sizer=position_sizer)
        
        # Reset simulator metrics
        self.simulator.reset_metrics()
        
        # Run backtest
        metrics = self.simulator.run_backtest(
            agent=agent,
            start_idx=0,
            end_idx=50,
            window_size=10
        )
        
        results['kelly'] = {
            'total_return': metrics['total_return'],
            'max_drawdown': metrics['max_drawdown'],
            'sharpe_ratio': metrics['sharpe_ratio']
        }
        
        # Check that we have results for all strategies
        self.assertEqual(len(results), 4)


if __name__ == '__main__':
    unittest.main()
