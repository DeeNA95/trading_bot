"""
Order execution using the Binance API.
"""

import os
import time
import json
import requests
from typing import Dict, List, Optional, Union, Any
from binance.client import Client
from binance.exceptions import BinanceAPIException


class BinanceExecutor:
    """Order execution via Binance API."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 testnet: bool = False):
        """Initialize the Binance executor.
        
        Args:
            api_key: Binance API key (default: from environment)
            api_secret: Binance API secret (default: from environment)
            testnet: Whether to use the testnet (default: False)
        """
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY')
        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API credentials not provided")
            
        try:
            self.client = Client(self.api_key, self.api_secret, testnet=testnet)
            # Test connection with a simple API call
            self.client.get_system_status()
        except BinanceAPIException as e:
            if "API-key format invalid" in str(e):
                raise ValueError("Invalid Binance API key format")
            elif "Signature for this request is not valid" in str(e):
                raise ValueError("Invalid Binance API secret")
            elif "Invalid API-key, IP, or permissions for action" in str(e):
                # Get current IP and provide helpful error message
                current_ip = self._get_current_ip()
                raise ValueError(
                    f"IP restriction error: Your current IP ({current_ip}) is not allowed to access the Binance API. "
                    f"Please add this IP to your API key's IP whitelist in your Binance account settings."
                )
            else:
                raise
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information.
        
        Returns:
            Dict containing account information
        """
        try:
            account = self.client.get_account()
            
            # Format the response
            balances = {}
            for asset in account['balances']:
                if float(asset['free']) > 0 or float(asset['locked']) > 0:
                    balances[asset['asset']] = {
                        'free': float(asset['free']),
                        'locked': float(asset['locked']),
                        'total': float(asset['free']) + float(asset['locked'])
                    }
            
            return {
                'balances': balances,
                'can_trade': account['canTrade'],
                'can_withdraw': account['canWithdraw'],
                'can_deposit': account['canDeposit']
            }
        except BinanceAPIException as e:
            self._handle_api_exception(e)
            return {'error': str(e)}
        
    def execute_order(self, 
                     symbol: str,
                     side: str,
                     qty: float,
                     order_type: str = 'MARKET',
                     price: Optional[float] = None,
                     time_in_force: str = 'GTC') -> Dict[str, Any]:
        """Execute a trading order.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: Order side ('BUY' or 'SELL')
            qty: Order quantity
            order_type: Order type ('MARKET', 'LIMIT', etc.)
            price: Price for limit orders
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
            
        Returns:
            Order details
        """
        try:
            # Prepare order parameters
            params = {
                'symbol': symbol.upper(),
                'side': side.upper(),
                'type': order_type.upper(),
                'quantity': qty
            }
            
            # Add price for limit orders
            if order_type.upper() == 'LIMIT' and price is not None:
                params['price'] = price
                params['timeInForce'] = time_in_force
            
            # Submit the order
            response = self.client.create_order(**params)
            
            return response
        except BinanceAPIException as e:
            self._handle_api_exception(e)
            return {'error': str(e)}
    
    def create_oco_order(self,
                        symbol: str,
                        side: str,
                        qty: float,
                        price: float,
                        stop_price: float,
                        stop_limit_price: Optional[float] = None) -> Dict[str, Any]:
        """Create an OCO (One-Cancels-the-Other) order.
        
        This allows setting both take profit and stop loss at the same time.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: Order side ('BUY' or 'SELL')
            qty: Order quantity
            price: Limit price (take profit)
            stop_price: Stop price (stop loss trigger)
            stop_limit_price: Stop limit price (optional)
            
        Returns:
            Order details
        """
        try:
            params = {
                'symbol': symbol.upper(),
                'side': side.upper(),
                'quantity': qty,
                'price': price,
                'stopPrice': stop_price
            }
            
            if stop_limit_price:
                params['stopLimitPrice'] = stop_limit_price
                params['stopLimitTimeInForce'] = 'GTC'
            
            response = self.client.create_oco_order(**params)
            return response
        except BinanceAPIException as e:
            self._handle_api_exception(e)
            return {'error': str(e)}
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an open order.
        
        Args:
            symbol: Trading pair
            order_id: Order ID to cancel
            
        Returns:
            Status of the cancellation
        """
        try:
            response = self.client.cancel_order(symbol=symbol.upper(), orderId=order_id)
            return response
        except BinanceAPIException as e:
            self._handle_api_exception(e)
            return {'success': False, 'error': str(e)}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders.
        
        Args:
            symbol: Optional trading pair to filter by
            
        Returns:
            List of open orders
        """
        try:
            if symbol:
                return self.client.get_open_orders(symbol=symbol.upper())
            else:
                return self.client.get_open_orders()
        except BinanceAPIException as e:
            self._handle_api_exception(e)
            return [{'error': str(e)}]
    
    def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get information about a specific order.
        
        Args:
            symbol: Trading pair
            order_id: Order ID to retrieve
            
        Returns:
            Order details
        """
        try:
            return self.client.get_order(symbol=symbol.upper(), orderId=order_id)
        except BinanceAPIException as e:
            self._handle_api_exception(e)
            return {'error': str(e)}
    
    def _get_current_ip(self) -> str:
        """Get the current public IP address.
        
        Returns:
            Current public IP address
        """
        try:
            response = requests.get('https://api.ipify.org')
            return response.text
        except Exception:
            return "unknown"
    
    def _handle_api_exception(self, exception: BinanceAPIException) -> None:
        """Handle Binance API exceptions with helpful messages.
        
        Args:
            exception: The Binance API exception
        """
        if exception.code == -2015 or "Invalid API-key, IP, or permissions" in str(exception):
            current_ip = self._get_current_ip()
            print(f"IP RESTRICTION ERROR: Your current IP ({current_ip}) is not whitelisted in your Binance API settings.")
            print("Please add this IP to your API key's allowed IPs in your Binance account.")
            print("Alternatively, consider using a VPN with a static IP that is already whitelisted.")
