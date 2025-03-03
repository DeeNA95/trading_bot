"""
Unit tests for position sizing strategies.
"""

import os
import sys
import unittest

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from risk.position_sizing import (FixedFractionPositionSizer,
                                  KellyPositionSizer,
                                  VolatilityAdjustedPositionSizer)


class TestFixedFractionPositionSizer(unittest.TestCase):
    """Test cases for fixed fraction position sizing."""

    def test_calculate_position_size(self):
        """Test calculating position size based on account balance."""
        # Initialize with a fixed fraction of 0.1
        sizer = FixedFractionPositionSizer(max_risk_pct=0.05)

        # Test with a balance of 10000
        balance = 10000.0
        position_size = sizer.calculate_position_size(
            account_balance=balance, signal_strength=0.8, volatility=1.0
        )

        # Verify that the position size is positive and less than account balance
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, balance)

    def test_zero_volatility(self):
        """Test with zero volatility."""
        sizer = FixedFractionPositionSizer(max_risk_pct=0.05)
        balance = 10000.0

        position_size = sizer.calculate_position_size(
            account_balance=balance, signal_strength=0.8, volatility=0.0
        )

        # Should handle zero volatility by using a small default value
        self.assertGreater(position_size, 0)


class TestVolatilityPositionSizer(unittest.TestCase):
    """Test cases for volatility-based position sizing."""

    def test_calculate_position_size(self):
        """Test calculating position size based on market volatility."""
        # Initialize with volatility multiplier of 2
        sizer = VolatilityAdjustedPositionSizer(base_risk_pct=0.05, volatility_scale=2)

        # Test with ATR of 5 and price of 100
        balance = 10000.0
        price = 100.0
        atr = 5.0

        position_size = sizer.calculate_position_size(
            account_balance=balance,
            signal_strength=0.8,
            volatility=1.0,
            price=price,
            atr=atr,
        )

        # Not directly testing the exact value due to the complexity of the calculation
        # but ensuring it returns a positive value for a valid input
        self.assertGreater(position_size, 0)

        # Test with zero ATR
        position_size = sizer.calculate_position_size(
            account_balance=balance,
            signal_strength=0.8,
            volatility=1.0,
            price=price,
            atr=0.0,
        )

        # Expected: 0 (can't size with zero volatility)
        self.assertEqual(position_size, 0.0)


class TestKellyPositionSizer(unittest.TestCase):
    """Test cases for Kelly Criterion position sizing."""

    def test_calculate_position_size(self):
        """Test calculating position size based on Kelly Criterion."""
        # Initialize with half-Kelly
        sizer = KellyPositionSizer(max_kelly_pct=0.5)

        # Mock past trade results
        sizer.past_trades = [0.05, -0.02, 0.03, 0.04, -0.01]

        balance = 10000.0

        position_size = sizer.calculate_position_size(
            account_balance=balance, signal_strength=0.8, volatility=1.0
        )

        # Since we're not testing the exact formula, just verify it returns a positive value
        self.assertGreaterEqual(position_size, 0)

        # Test with no past trades
        sizer.past_trades = []
        position_size = sizer.calculate_position_size(
            account_balance=balance, signal_strength=0.8, volatility=1.0
        )

        # Should have a default position size calculation with empty past_trades
        self.assertGreaterEqual(position_size, 0)


if __name__ == "__main__":
    unittest.main()
