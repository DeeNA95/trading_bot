"""
Utility functions for the RL trading environment components.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_adaptive_leverage(
    volatility: float,
    max_leverage: int,
    default_leverage: int,
    volatility_factor: float = 5.0 # Default factor used previously
) -> int:
    """
    Calculate adaptive leverage based on market volatility.
    Lower leverage is used when volatility is high.

    Args:
        volatility: Market volatility measure (expected range approx 0-1).
        max_leverage: Maximum allowed leverage.
        default_leverage: Default leverage if volatility is zero or calculation fails.
        volatility_factor: Adjusts sensitivity to volatility.

    Returns:
        Integer leverage value between 1 and max_leverage.
    """
    adaptive_leverage = default_leverage # Default value
    try:
        if volatility > 0:
            # Ensure leverage doesn't go below 1
            leverage_raw = max_leverage * (1 / (1 + volatility * volatility_factor))
            adaptive_leverage = max(1, min(max_leverage, int(leverage_raw)))
        # else: use default_leverage already assigned
    except Exception as e:
        logger.warning(f"Error calculating adaptive leverage: {e}. Falling back to default {default_leverage}.")
        # Fallback to default leverage on any error

    return adaptive_leverage
