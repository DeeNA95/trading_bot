"""
Order execution using the Alpaca API.
"""

import os
import time
from typing import Dict, List, Optional, Union, Any
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus


class AlpacaExecutor:
    """Order execution via Alpaca API."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 paper: bool = True):
        """Initialize the Alpaca executor.
        
        Args:
            api_key: Alpaca API key (default: from environment)
            api_secret: Alpaca API secret (default: from environment)
            paper: Whether to use paper trading (default: True)
        """
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY')
        self.api_secret = api_secret or os.environ.get('ALPACA_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not provided")
            
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=paper)
        
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        account = self.trading_client.get_account()
        return {
            'cash': float(account.cash),
            'equity': float(account.equity),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
        }
        
    def execute_order(self, 
                     symbol: str,
                     side: str,
                     qty: float,
                     order_type: str = 'market',
                     take_profit_price: Optional[float] = None,
                     stop_loss_price: Optional[float] = None,
                     time_in_force: str = 'day') -> Dict[str, Any]:
        """Execute a trading order.
        
        Args:
            symbol: Asset symbol
            side: Order side ('buy' or 'sell')
            qty: Order quantity
            order_type: Order type ('market', 'limit', etc.)
            take_profit_price: Optional take profit price
            stop_loss_price: Optional stop loss price
            time_in_force: Time in force ('day', 'gtc', 'ioc', 'fok')
            
        Returns:
            Order details
        """
        # Map side to Alpaca enum
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        
        # Map time in force to Alpaca enum
        if time_in_force.lower() == 'day':
            tif = TimeInForce.DAY
        elif time_in_force.lower() == 'gtc':
            tif = TimeInForce.GTC
        else:
            tif = TimeInForce.DAY
            
        # Create market order
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif
        )
        
        # Add take profit and stop loss if specified
        if take_profit_price is not None:
            order_request.take_profit = TakeProfitRequest(
                limit_price=take_profit_price
            )
            
        if stop_loss_price is not None:
            order_request.stop_loss = StopLossRequest(
                stop_price=stop_loss_price
            )
            
        # Submit order
        try:
            order = self.trading_client.submit_order(order_request)
            
            # Format response
            order_details = {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'qty': float(order.qty) if order.qty else None,
                'status': order.status.value,
                'created_at': order.created_at,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'filled_qty': float(order.filled_qty) if order.filled_qty else None,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
            }
            
            return order_details
            
        except Exception as e:
            return {'error': str(e)}
            
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Status of the cancellation
        """
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return {'success': True, 'order_id': order_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get current position for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Position details or empty dict if no position
        """
        try:
            position = self.trading_client.get_position(symbol)
            
            return {
                'symbol': position.symbol,
                'qty': float(position.qty),
                'avg_entry_price': float(position.avg_entry_price),
                'market_value': float(position.market_value),
                'cost_basis': float(position.cost_basis),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'current_price': float(position.current_price),
            }
        except Exception:
            # No position for this symbol
            return {}
            
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close an open position.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Status of the position closure
        """
        try:
            response = self.trading_client.close_position(symbol)
            
            return {
                'symbol': response.symbol,
                'qty': float(response.qty) if response.qty else None,
                'status': 'closed',
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
