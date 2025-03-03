"""
Order execution using the Coinbase Advanced Trade API.
"""

import os
import time
import hmac
import hashlib
import base64
import json
import requests
from typing import Dict, List, Optional, Union, Any
from datetime import datetime


class CoinbaseExecutor:
    """Order execution via Coinbase Advanced Trade API."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None):
        """Initialize the Coinbase executor.
        
        Args:
            api_key: Coinbase API key (default: from environment)
            api_secret: Coinbase API secret (default: from environment)
        """
        self.api_key = api_key or os.environ.get('COINBASE_API_KEY')
        self.api_secret = api_secret or os.environ.get('COINBASE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Coinbase API credentials not provided")
            
        self.base_url = "https://api.exchange.coinbase.com"
        
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information.
        
        Returns:
            Dict containing account information
        """
        endpoint = "/accounts"
        response = self._send_request("GET", endpoint)
        
        if 'accounts' in response:
            # Format the response
            accounts = {}
            for account in response['accounts']:
                accounts[account['currency']] = {
                    'balance': float(account['balance']),
                    'available': float(account['available']),
                    'hold': float(account['hold'])
                }
            return accounts
        return response
        
    def execute_order(self, 
                     symbol: str,
                     side: str,
                     qty: float,
                     order_type: str = 'market',
                     price: Optional[float] = None,
                     time_in_force: str = 'GTC') -> Dict[str, Any]:
        """Execute a trading order.
        
        Note: Coinbase Advanced Trade doesn't directly support take profit and stop loss
        in a single order. These need to be implemented as separate orders after the
        main order is filled.
        
        Args:
            symbol: Product ID (e.g., 'BTC-USD')
            side: Order side ('buy' or 'sell')
            qty: Order quantity
            order_type: Order type ('market', 'limit')
            price: Price for limit orders
            time_in_force: Time in force ('GTC', 'GTT', 'IOC', 'FOK')
            
        Returns:
            Order details
        """
        endpoint = "/orders"
        
        # Prepare order data
        order_data = {
            "product_id": symbol,
            "side": side.lower(),
            "type": order_type.lower(),
            "size": str(qty)
        }
        
        # Add price for limit orders
        if order_type.lower() == 'limit' and price is not None:
            order_data["price"] = str(price)
            order_data["time_in_force"] = time_in_force
        
        response = self._send_request("POST", endpoint, data=order_data)
        return response
    
    def create_stop_order(self,
                         symbol: str,
                         side: str,
                         qty: float,
                         stop_price: float,
                         limit_price: Optional[float] = None) -> Dict[str, Any]:
        """Create a stop order (stop loss or stop entry).
        
        Args:
            symbol: Product ID (e.g., 'BTC-USD')
            side: Order side ('buy' or 'sell')
            qty: Order quantity
            stop_price: Stop trigger price
            limit_price: Optional limit price (for stop-limit orders)
            
        Returns:
            Order details
        """
        endpoint = "/orders"
        
        # Determine order type
        order_type = "stop_limit" if limit_price else "stop"
        
        # Prepare order data
        order_data = {
            "product_id": symbol,
            "side": side.lower(),
            "type": order_type,
            "size": str(qty),
            "stop": "loss" if side.lower() == "sell" else "entry",
            "stop_price": str(stop_price)
        }
        
        # Add limit price if provided
        if limit_price:
            order_data["price"] = str(limit_price)
        
        response = self._send_request("POST", endpoint, data=order_data)
        return response
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Status of the cancellation
        """
        endpoint = f"/orders/{order_id}"
        response = self._send_request("DELETE", endpoint)
        
        if response.get('success', False):
            return {'success': True, 'order_id': order_id}
        return {'success': False, 'error': response.get('message', 'Unknown error')}
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get information about a specific order.
        
        Args:
            order_id: Order ID to retrieve
            
        Returns:
            Order details
        """
        endpoint = f"/orders/{order_id}"
        return self._send_request("GET", endpoint)
    
    def get_orders(self, status: str = 'all') -> Dict[str, Any]:
        """Get orders with the specified status.
        
        Args:
            status: Order status ('open', 'pending', 'done', 'all')
            
        Returns:
            List of orders
        """
        endpoint = f"/orders?status={status}"
        return self._send_request("GET", endpoint)
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
        """Generate the signature for API authentication.
        
        Args:
            timestamp: Current timestamp
            method: HTTP method
            request_path: API endpoint path
            body: Request body (for POST requests)
            
        Returns:
            Base64-encoded signature
        """
        message = timestamp + method + request_path + body
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode('ascii'),
            hashlib.sha256
        )
        return base64.b64encode(signature.digest()).decode('utf-8')
    
    def _send_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Send a request to the Coinbase API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data for POST requests
            
        Returns:
            API response
        """
        url = self.base_url + endpoint
        timestamp = str(int(time.time()))
        
        # Prepare request body
        body = ''
        if data and method == 'POST':
            body = json.dumps(data)
        
        # Generate signature
        signature = self._generate_signature(timestamp, method, endpoint, body)
        
        # Prepare headers
        headers = {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'Content-Type': 'application/json'
        }
        
        # Send request
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, headers=headers, data=body)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)
            else:
                return {'error': f'Unsupported method: {method}'}
            
            # Parse response
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'error': f'API error: {response.status_code}',
                    'message': response.text
                }
                
        except Exception as e:
            return {'error': str(e)}
