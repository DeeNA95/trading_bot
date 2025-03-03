import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MAExecutor:
    """Handles order execution for MA strategies"""
    
    def __init__(self, exchange_client, config):
        self.exchange = exchange_client
        self.config = config
        self.position = 0

    def execute_order(self, direction: int, quantity: float) -> Optional[Dict]:
        """
        Execute market order with risk checks
        """
        if quantity <= 0:
            logger.warning("Invalid order quantity")
            return None

        try:
            # Calculate position size with leverage
            max_position = self.exchange.get_balance() * self.config.max_leverage
            if quantity > max_position:
                logger.warning(f"Position size {quantity} exceeds max {max_position}")
                quantity = max_position

            # Execute order
            order = self.exchange.create_market_order(
                symbol="BTC/USDT",
                side="buy" if direction > 0 else "sell",
                amount=quantity,
                params={
                    "leverage": self.config.max_leverage,
                    "reduceOnly": False
                }
            )
            self.position = direction
            return order
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return None
