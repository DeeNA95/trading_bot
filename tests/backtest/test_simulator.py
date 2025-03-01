"""
Unit tests for the backtest simulator.
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests.mocks.backtest_mock import MockBacktestSimulator
from tests.mocks.agent_mock import MockAgent


class TestBacktestSimulator(unittest.TestCase):
    """Test cases for the backtest simulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = MockBacktestSimulator(
            initial_balance=10000.0,
            commission=0.001
        )
        self.agent = MockAgent(input_dim=10, output_dim=4)
        
    def test_initialization(self):
        """Test simulator initialization."""
        self.assertEqual(self.simulator.initial_balance, 10000.0)
        self.assertEqual(self.simulator.commission, 0.001)
        
    def test_run_backtest(self):
        """Test running a backtest."""
        # Run the backtest
        metrics = self.simulator.run_backtest(
            agent=self.agent,
            start_idx=0,
            end_idx=100,
            window_size=10
        )
        
        # Check that metrics include expected keys
        for key in ['total_return', 'sharpe_ratio', 'max_drawdown', 'total_trades']:
            self.assertIn(key, metrics)
            
        # Check position history
        self.assertTrue(len(self.simulator.position_history) > 0)
        
    def test_reset_metrics(self):
        """Test resetting simulator metrics."""
        # Run a backtest to populate history
        self.simulator.run_backtest(self.agent)
        
        # Verify that position history is populated
        self.assertTrue(len(self.simulator.position_history) > 0)
        
        # Reset metrics
        self.simulator.reset_metrics()
        
        # Verify that position history is empty after reset
        self.assertEqual(len(self.simulator.position_history), 0)
        
    def test_multiple_backtests(self):
        """Test running multiple backtests sequentially."""
        # Run first backtest
        metrics1 = self.simulator.run_backtest(self.agent)
        
        # Reset simulator
        self.simulator.reset_metrics()
        
        # Run second backtest
        metrics2 = self.simulator.run_backtest(self.agent)
        
        # Both should have metrics
        self.assertIn('total_return', metrics1)
        self.assertIn('total_return', metrics2)


if __name__ == '__main__':
    unittest.main()
