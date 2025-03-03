"""
Binance Futures order execution module for the trading bot.
Handles futures trading operations using Binance Futures API.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple

from binance.um_futures import UMFutures
from binance.error import ClientError

logger = logging.getLogger(__name__)

class BinanceFuturesExecutor:
    """
    Executor class for Binance Futures trading operations.
    Specialized for BTC futures trading on USDT-Margined (UM) futures.
    """
    
    def __init__(self, use_testnet: bool = True, trading_symbol: str = "BTCUSDT", leverage: int = 5):
        """
        Initialize the Binance Futures executor.
        
        Args:
            use_testnet: Whether to use the testnet environment (default: True)
            trading_symbol: Trading pair symbol (default: BTCUSDT)
            leverage: Default leverage to use (default: 5)
        """
        self.use_testnet = use_testnet
        self.trading_symbol = trading_symbol
        self.default_leverage = leverage
        
        # Load API credentials from environment variables
        if use_testnet:
            self.api_key = os.environ.get('binance_future_testnet_api')
            self.api_secret = os.environ.get('binance_future_testnet_secret')
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.api_key = os.environ.get('binance_api')
            self.api_secret = os.environ.get('binance_secret')
            self.base_url = None  # Use default production URL
        
        # Initialize USDT-Margined futures client
        self.client = UMFutures(
            key=self.api_key,
            secret=self.api_secret,
            base_url=self.base_url
        )
        
        # Set leverage for BTC futures
        try:
            self.client.change_leverage(
                symbol=self.trading_symbol, 
                leverage=self.default_leverage
            )
            logger.info(f"Set leverage to {self.default_leverage}x for {self.trading_symbol}")
        except ClientError as e:
            logger.error(f"Failed to set leverage: {e}")
        
        logger.info(f"Initialized Binance {'Testnet ' if use_testnet else ''}Futures executor for {self.trading_symbol}")

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary containing account information
        """
        try:
            return self.client.account()
        except ClientError as e:
            logger.error(f"Failed to get account info: {e}")
            return {"error": str(e)}

    def get_btc_position(self) -> Dict[str, Any]:
        """
        Get BTC position risk information.
            
        Returns:
            Dictionary containing BTC position risk information
        """
        try:
            positions = self.client.get_position_risk(symbol=self.trading_symbol)
            return positions[0] if positions else {}
        except ClientError as e:
            logger.error(f"Failed to get BTC position: {e}")
            return {"error": str(e)}

    def get_btc_price(self) -> float:
        """
        Get current BTC price.
        
        Returns:
            Current BTC price as float
        """
        try:
            ticker = self.client.ticker_price(symbol=self.trading_symbol)
            return float(ticker['price'])
        except (ClientError, KeyError, ValueError) as e:
            logger.error(f"Failed to get BTC price: {e}")
            return 0.0

    def get_btc_klines(self, interval: str = "1h", limit: int = 100) -> List[List]:
        """
        Get BTC candlestick/kline data.
        
        Args:
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of klines to return (max 1500)
            
        Returns:
            List of klines
        """
        try:
            return self.client.klines(symbol=self.trading_symbol, interval=interval, limit=limit)
        except ClientError as e:
            logger.error(f"Failed to get BTC klines: {e}")
            return []

    def change_leverage(self, leverage: int) -> Dict[str, Any]:
        """
        Change leverage for BTC futures.
        
        Args:
            leverage: Leverage value (1-125)
            
        Returns:
            Response from the API
        """
        try:
            response = self.client.change_leverage(
                symbol=self.trading_symbol, 
                leverage=leverage
            )
            self.default_leverage = leverage
            logger.info(f"Changed leverage to {leverage}x for {self.trading_symbol}")
            return response
        except ClientError as e:
            logger.error(f"Failed to change leverage: {e}")
            return {"error": str(e)}

    def change_margin_type(self, margin_type: str) -> Dict[str, Any]:
        """
        Change margin type for BTC futures.
        
        Args:
            margin_type: Margin type, either "ISOLATED" or "CROSSED"
            
        Returns:
            Response from the API
        """
        try:
            return self.client.change_margin_type(
                symbol=self.trading_symbol, 
                marginType=margin_type
            )
        except ClientError as e:
            logger.error(f"Failed to change margin type: {e}")
            return {"error": str(e)}

    def execute_market_order(self, side: str, quantity: float) -> Dict[str, Any]:
        """
        Execute a market order for BTC futures.
        
        Args:
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity in BTC
            
        Returns:
            Dictionary containing order information
        """
        try:
            response = self.client.new_order(
                symbol=self.trading_symbol,
                side=side,
                type="MARKET",
                quantity=quantity
            )
            logger.info(f"Executed {side} MARKET order for {quantity} BTC futures")
            return response
        except ClientError as e:
            logger.error(f"Failed to execute market order: {e}")
            return {"error": str(e)}

    def execute_limit_order(self, side: str, quantity: float, price: float, 
                          time_in_force: str = "GTC") -> Dict[str, Any]:
        """
        Execute a limit order for BTC futures.
        
        Args:
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity in BTC
            price: Order price in USDT
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
            
        Returns:
            Dictionary containing order information
        """
        try:
            response = self.client.new_order(
                symbol=self.trading_symbol,
                side=side,
                type="LIMIT",
                quantity=quantity,
                price=price,
                timeInForce=time_in_force
            )
            logger.info(f"Executed {side} LIMIT order for {quantity} BTC futures at {price} USDT")
            return response
        except ClientError as e:
            logger.error(f"Failed to execute limit order: {e}")
            return {"error": str(e)}

    def execute_stop_market_order(self, side: str, quantity: float, 
                               stop_price: float) -> Dict[str, Any]:
        """
        Execute a stop market order for BTC futures.
        
        Args:
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity in BTC
            stop_price: Stop trigger price
            
        Returns:
            Dictionary containing order information
        """
        try:
            response = self.client.new_order(
                symbol=self.trading_symbol,
                side=side,
                type="STOP_MARKET",
                quantity=quantity,
                stopPrice=stop_price,
            )
            logger.info(f"Executed {side} STOP_MARKET order for {quantity} BTC futures at stop price {stop_price} USDT")
            return response
        except ClientError as e:
            logger.error(f"Failed to execute stop market order: {e}")
            return {"error": str(e)}

    def execute_take_profit_market_order(self, side: str, quantity: float, 
                                      stop_price: float) -> Dict[str, Any]:
        """
        Execute a take profit market order for BTC futures.
        
        Args:
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity in BTC
            stop_price: Stop trigger price
            
        Returns:
            Dictionary containing order information
        """
        try:
            response = self.client.new_order(
                symbol=self.trading_symbol,
                side=side,
                type="TAKE_PROFIT_MARKET",
                quantity=quantity,
                stopPrice=stop_price,
            )
            logger.info(f"Executed {side} TAKE_PROFIT_MARKET order for {quantity} BTC futures at stop price {stop_price} USDT")
            return response
        except ClientError as e:
            logger.error(f"Failed to execute take profit market order: {e}")
            return {"error": str(e)}

    def execute_stop_loss_take_profit_order(self, side: str, quantity: float, 
                                       position_side: str = "BOTH", stop_loss_price: float = None, 
                                       take_profit_price: float = None, close_position: bool = False) -> Dict[str, Any]:
        """
        Execute a combined stop loss and take profit order for BTC futures.
        This creates a strategy order with both stop loss and take profit in a single API call.
        
        Args:
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity in BTC
            position_side: Position side ('LONG', 'SHORT', or 'BOTH')
            stop_loss_price: Stop loss trigger price
            take_profit_price: Take profit trigger price
            close_position: Whether to close the entire position
            
        Returns:
            Dictionary containing order information
        """
        try:
            # Validate inputs
            if stop_loss_price is None and take_profit_price is None:
                logger.error("Both stop_loss_price and take_profit_price cannot be None")
                return {"error": "Both stop_loss_price and take_profit_price cannot be None"}
            
            # Prepare parameters
            params = {
                "symbol": self.trading_symbol,
                "side": side,
                "positionSide": position_side,
                "workingType": "MARK_PRICE",  # Use mark price for triggering
                "priceProtect": "TRUE",  # Enable price protection
                "reduceOnly": "TRUE" if close_position else "FALSE"
            }
            
            # Add quantity unless we're closing the position
            if not close_position:
                params["quantity"] = quantity
            
            # Create list to store order responses
            orders = []
            
            # Place stop loss order if price is provided
            if stop_loss_price is not None:
                stop_loss_params = params.copy()
                stop_loss_params["type"] = "STOP_MARKET"
                stop_loss_params["stopPrice"] = stop_loss_price
                stop_loss_params["closePosition"] = "TRUE" if close_position else "FALSE"
                
                try:
                    stop_loss_response = self.client.new_order(**stop_loss_params)
                    logger.info(f"Executed {side} STOP_MARKET order at stop price {stop_loss_price} USDT")
                    orders.append(stop_loss_response)
                except ClientError as e:
                    logger.error(f"Failed to execute stop loss order: {e}")
                    orders.append({"error": str(e)})
            
            # Place take profit order if price is provided
            if take_profit_price is not None:
                take_profit_params = params.copy()
                take_profit_params["type"] = "TAKE_PROFIT_MARKET"
                take_profit_params["stopPrice"] = take_profit_price
                take_profit_params["closePosition"] = "TRUE" if close_position else "FALSE"
                
                try:
                    take_profit_response = self.client.new_order(**take_profit_params)
                    logger.info(f"Executed {side} TAKE_PROFIT_MARKET order at stop price {take_profit_price} USDT")
                    orders.append(take_profit_response)
                except ClientError as e:
                    logger.error(f"Failed to execute take profit order: {e}")
                    orders.append({"error": str(e)})
            
            return {"orders": orders}
            
        except Exception as e:
            logger.error(f"Failed to execute stop loss/take profit orders: {e}")
            return {"error": str(e)}

    def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dictionary containing cancellation information
        """
        try:
            response = self.client.cancel_order(
                symbol=self.trading_symbol, 
                orderId=order_id
            )
            logger.info(f"Cancelled order {order_id}")
            return response
        except ClientError as e:
            logger.error(f"Failed to cancel order: {e}")
            return {"error": str(e)}

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get all open orders for BTC futures.
        
        Returns:
            List of dictionaries containing open orders information
        """
        try:
            # According to the API docs, we can get all open orders for a specific symbol
            # without providing an orderId
            params = {"symbol": self.trading_symbol}
            response = self.client.get_open_orders(**params)
            return response
        except ClientError as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    def get_order_info(self, order_id: int) -> Dict[str, Any]:
        """
        Get information about a specific order.
        
        Args:
            order_id: Order ID to query
            
        Returns:
            Dictionary containing order information
        """
        try:
            return self.client.get_order(
                symbol=self.trading_symbol, 
                orderId=order_id
            )
        except ClientError as e:
            logger.error(f"Failed to get order info: {e}")
            return {"error": str(e)}
    
    def close_all_positions(self) -> Dict[str, Any]:
        """
        Close all open BTC futures positions.
        
        Returns:
            Dictionary containing order information
        """
        try:
            position = self.get_btc_position()
            if "positionAmt" in position and float(position["positionAmt"]) != 0:
                position_amt = float(position["positionAmt"])
                side = "SELL" if position_amt > 0 else "BUY"
                quantity = abs(position_amt)
                
                response = self.client.new_order(
                    symbol=self.trading_symbol,
                    side=side,
                    type="MARKET",
                    quantity=quantity,
                    reduceOnly="true"
                )
                logger.info(f"Closed {quantity} BTC futures position with {side} order")
                return response
            else:
                logger.info("No open positions to close")
                return {"info": "No open positions to close"}
        except (ClientError, KeyError) as e:
            logger.error(f"Failed to close positions: {e}")
            return {"error": str(e)}
