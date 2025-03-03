"""
Order execution using the Kraken API.
"""

import os
import time
from typing import Dict, List, Optional, Union, Any
import krakenex
from pykrakenapi import KrakenAPI


class KrakenExecutor:
    """Order execution via Kraken API."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None):
        """Initialize the Kraken executor.
        
        Args:
            api_key: Kraken API key (default: from environment)
            api_secret: Kraken API secret (default: from environment)
        """
        self.api_key = api_key or os.environ.get('KRAKEN_API_KEY')
        self.api_secret = api_secret or os.environ.get('KRAKEN_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Kraken API credentials not provided")
            
        # Initialize the Kraken API client
        api = krakenex.API(self.api_key, self.api_secret)
        self.kraken = KrakenAPI(api)
        
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information.
        
        Returns:
            Dict containing account balance information
        """
        try:
            balance = self.kraken.get_account_balance()
            return {currency: float(amount) for currency, amount in balance.items()}
        except Exception as e:
            return {'error': str(e)}
        
    def execute_order(self, 
                     symbol: str,
                     side: str,
                     qty: float,
                     order_type: str = 'market',
                     take_profit_price: Optional[float] = None,
                     stop_loss_price: Optional[float] = None,
                     time_in_force: str = 'GTC') -> Dict[str, Any]:
        """Execute a trading order with optional take profit and stop loss.
        
        Args:
            symbol: Asset pair (e.g., 'XBTUSD')
            side: Order side ('buy' or 'sell')
            qty: Order quantity
            order_type: Order type ('market', 'limit')
            take_profit_price: Optional take profit price
            stop_loss_price: Optional stop loss price
            time_in_force: Time in force ('GTC', 'IOC', 'GTD')
            
        Returns:
            Order details
        """
        try:
            # Standardize the symbol format for Kraken
            symbol = self._format_symbol(symbol)
            
            # Prepare order parameters
            params = {
                'pair': symbol,
                'type': side.lower(),
                'ordertype': self._map_order_type(order_type),
                'volume': str(qty),
                'timeinforce': time_in_force
            }
            
            # Add close order parameters for take profit and stop loss
            if take_profit_price is not None or stop_loss_price is not None:
                close_params = {}
                
                if take_profit_price is not None:
                    # Configure take profit
                    close_params['ordertype'] = 'limit'
                    close_params['price'] = str(take_profit_price)
                
                if stop_loss_price is not None:
                    # If both TP and SL are specified, use price2 for SL
                    if take_profit_price is not None:
                        close_params['price2'] = str(stop_loss_price)
                    else:
                        # Only SL specified
                        close_params['ordertype'] = 'stop-loss'
                        close_params['price'] = str(stop_loss_price)
                
                # Add close parameters to the order
                params['close'] = close_params
            
            # Submit the order
            response = self.kraken.query_private('AddOrder', params)
            
            # Format the response
            order_details = {
                'txid': response.get('txid', [None])[0],
                'description': response.get('descr', {}).get('order', ''),
                'status': 'submitted',
                'symbol': symbol,
                'side': side.lower(),
                'qty': qty,
                'take_profit': take_profit_price,
                'stop_loss': stop_loss_price
            }
            
            return order_details
            
        except Exception as e:
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order.
        
        Args:
            order_id: Transaction ID of the order to cancel
            
        Returns:
            Status of the cancellation
        """
        try:
            response = self.kraken.query_private('CancelOrder', {'txid': order_id})
            return {'success': response.get('count', 0) > 0, 'order_id': order_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_open_orders(self) -> Dict[str, Any]:
        """Get all open orders.
        
        Returns:
            Dictionary of open orders
        """
        try:
            response = self.kraken.query_private('OpenOrders')
            return response.get('open', {})
        except Exception as e:
            return {'error': str(e)}
    
    def get_closed_orders(self) -> Dict[str, Any]:
        """Get closed orders.
        
        Returns:
            Dictionary of closed orders
        """
        try:
            response = self.kraken.query_private('ClosedOrders')
            return response.get('closed', {})
        except Exception as e:
            return {'error': str(e)}
    
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol to Kraken's expected format.
        
        Args:
            symbol: Symbol to format (e.g., 'BTC/USD' or 'BTCUSD')
            
        Returns:
            Formatted symbol (e.g., 'XBTUSD')
        """
        # Remove slash if present
        symbol = symbol.replace('/', '')
        
        # Convert common symbols to Kraken format
        if symbol.upper().startswith('BTC'):
            symbol = 'XBT' + symbol[3:]
        
        return symbol
    
    def _map_order_type(self, order_type: str) -> str:
        """Map generic order type to Kraken-specific order type.
        
        Args:
            order_type: Generic order type ('market', 'limit', etc.)
            
        Returns:
            Kraken-specific order type
        """
        order_type_map = {
            'market': 'market',
            'limit': 'limit',
            'stop': 'stop-loss',
            'stop_limit': 'stop-loss-limit',
            'take_profit': 'take-profit',
            'take_profit_limit': 'take-profit-limit'
        }
        
        return order_type_map.get(order_type.lower(), 'market')
