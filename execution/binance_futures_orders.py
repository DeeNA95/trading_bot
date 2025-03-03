#!/usr/bin/env python3
"""
Binance Futures order execution module for the trading bot.
Handles futures trading operations using Binance Futures API.
Supports USDT-margined futures trading for BTCUSDT with logic to mimic OCO functionality via separate conditional orders.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from binance.error import ClientError
from binance.um_futures import UMFutures

logger = logging.getLogger(__name__)


class BinanceFuturesExecutor:
    """
    Executor class for Binance Futures trading operations.
    Specialized for BTC futures trading on USDT-Margined (UM) futures.
    """

    def __init__(
        self,
        use_testnet: bool = True,
        trading_symbol: str = "BTCUSDT",
        leverage: int = 2,
    ):
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
            self.api_key = os.environ.get("binance_future_testnet_api")
            self.api_secret = os.environ.get("binance_future_testnet_secret")
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.api_key = os.environ.get("binance_api")
            self.api_secret = os.environ.get("binance_secret")
            self.base_url = None  # Use default production URL

        # Initialize USDT-Margined futures client
        self.client = UMFutures(
            key=self.api_key, secret=self.api_secret, base_url=self.base_url
        )

        # Set leverage for the trading symbol
        try:
            self.client.change_leverage(
                symbol=self.trading_symbol, leverage=self.default_leverage
            )
            logger.info(
                f"Set leverage to {self.default_leverage}x for {self.trading_symbol}"
            )
        except ClientError as e:
            logger.error(f"Failed to set leverage: {e}")

        logger.info(
            f"Initialized Binance {'Testnet ' if use_testnet else ''}Futures executor for {self.trading_symbol}"
        )

    def get_account_info(self) -> Dict[str, Any]:
        """Retrieve account information."""
        try:
            return self.client.account()
        except ClientError as e:
            logger.error(f"Failed to get account info: {e}")
            return {"error": str(e)}

    def get_btc_position(self) -> Dict[str, Any]:
        """Retrieve BTC position risk information."""
        try:
            positions = self.client.get_position_risk(symbol=self.trading_symbol)
            return positions[0] if positions else {}
        except ClientError as e:
            logger.error(f"Failed to get BTC position: {e}")
            return {"error": str(e)}

    def get_btc_price(self) -> float:
        """Retrieve the current BTC price."""
        try:
            ticker = self.client.ticker_price(symbol=self.trading_symbol)
            return float(ticker["price"])
        except (ClientError, KeyError, ValueError) as e:
            logger.error(f"Failed to get BTC price: {e}")
            return 0.0

    def get_btc_klines(self, interval: str = "1h", limit: int = 100) -> List[List]:
        """
        Retrieve BTC candlestick/kline data.

        Args:
            interval: Kline interval (e.g., 1m, 1h, 1d)
            limit: Number of klines to return (max 1500)

        Returns:
            List of klines.
        """
        try:
            return self.client.klines(
                symbol=self.trading_symbol, interval=interval, limit=limit
            )
        except ClientError as e:
            logger.error(f"Failed to get BTC klines: {e}")
            return []

    def change_leverage(self, leverage: int) -> Dict[str, Any]:
        """
        Change leverage for BTC futures.

        Args:
            leverage: Leverage value (1-125)

        Returns:
            API response.
        """
        try:
            response = self.client.change_leverage(
                symbol=self.trading_symbol, leverage=leverage
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
            margin_type: Either "ISOLATED" or "CROSSED"

        Returns:
            API response.
        """
        try:
            return self.client.change_margin_type(
                symbol=self.trading_symbol, marginType=margin_type
            )
        except ClientError as e:
            logger.error(f"Failed to change margin type: {e}")
            return {"error": str(e)}

    def execute_market_order(
        self, side: str, quantity: float, close_position: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a market order.

        Args:
            side: 'BUY' or 'SELL'
            quantity: Order quantity in BTC
            close_position: If True, the order is intended to close an existing position

        Returns:
            API response.
        """
        try:
            params = {
                "symbol": self.trading_symbol,
                "side": side,
                "type": "MARKET",
                "quantity": quantity,
            }
            if close_position:
                params["reduceOnly"] = True
            response = self.client.new_order(**params)
            logger.info(
                f"Executed {side} MARKET order for {quantity} BTC (close_position={close_position})"
            )
            return response
        except ClientError as e:
            logger.error(f"Failed to execute market order: {e}")
            return {"error": str(e)}

    def execute_limit_order(
        self, side: str, quantity: float, price: float, time_in_force: str = "GTC"
    ) -> Dict[str, Any]:
        """
        Execute a limit order.

        Args:
            side: 'BUY' or 'SELL'
            quantity: Order quantity in BTC
            price: Order price in USDT
            time_in_force: 'GTC', 'IOC', or 'FOK'

        Returns:
            API response.
        """
        try:
            response = self.client.new_order(
                symbol=self.trading_symbol,
                side=side,
                type="LIMIT",
                quantity=quantity,
                price=price,
                timeInForce=time_in_force,
            )
            logger.info(
                f"Executed {side} LIMIT order for {quantity} BTC at {price} USDT"
            )
            return response
        except ClientError as e:
            logger.error(f"Failed to execute limit order: {e}")
            return {"error": str(e)}

    def execute_stop_market_order(
        self, side: str, quantity: float, stop_price: float
    ) -> Dict[str, Any]:
        """
        Execute a stop market order (for stop loss).

        Args:
            side: 'BUY' or 'SELL'
            quantity: Order quantity in BTC
            stop_price: Trigger price for the stop order

        Returns:
            API response.
        """
        try:
            response = self.client.new_order(
                symbol=self.trading_symbol,
                side=side,
                type="STOP_MARKET",
                quantity=quantity,
                stopPrice=stop_price,
            )
            logger.info(
                f"Executed {side} STOP_MARKET order for {quantity} BTC at stop price {stop_price} USDT"
            )
            return response
        except ClientError as e:
            logger.error(f"Failed to execute stop market order: {e}")
            return {"error": str(e)}

    def execute_take_profit_market_order(
        self, side: str, quantity: float, stop_price: float
    ) -> Dict[str, Any]:
        """
        Execute a take profit market order.

        Args:
            side: 'BUY' or 'SELL'
            quantity: Order quantity in BTC
            stop_price: Trigger price for the take profit order

        Returns:
            API response.
        """
        try:
            response = self.client.new_order(
                symbol=self.trading_symbol,
                side=side,
                type="TAKE_PROFIT_MARKET",
                quantity=quantity,
                stopPrice=stop_price,
            )
            logger.info(
                f"Executed {side} TAKE_PROFIT_MARKET order for {quantity} BTC at trigger price {stop_price} USDT"
            )
            return response
        except ClientError as e:
            logger.error(f"Failed to execute take profit market order: {e}")
            return {"error": str(e)}

    def execute_stop_loss_take_profit_order(
        self,
        side: str,
        quantity: float,
        position_side: str = "BOTH",
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        close_position: bool = False,
    ) -> Dict[str, Any]:
        """
        Mimic an OCO order by placing separate conditional orders for stop loss and take profit.

        Steps:
          1. Cancel any existing open orders for the trading symbol.
          2. Generate unique client order IDs for both orders.
          3. Place a STOP_MARKET order if stop_loss_price is provided.
          4. Place a TAKE_PROFIT_MARKET order if take_profit_price is provided.
          5. Return responses and client order IDs for further monitoring.

        Args:
            side: 'BUY' or 'SELL'
            quantity: Order quantity in BTC
            position_side: 'LONG', 'SHORT', or 'BOTH'
            stop_loss_price: Price level to trigger stop loss (optional)
            take_profit_price: Price level to trigger take profit (optional)
            close_position: If True, orders are intended to close an existing position.

        Returns:
            Dictionary with order responses, client order IDs, and an "is_oco" flag.
        """
        if stop_loss_price is None and take_profit_price is None:
            logger.error("Both stop_loss_price and take_profit_price cannot be None")
            return {
                "error": "Both stop_loss_price and take_profit_price cannot be None"
            }

        # Cancel existing open orders first.
        try:
            open_orders = self.get_open_orders()
            if open_orders:
                logger.info(f"Canceling {len(open_orders)} existing open orders")
                self.client.cancel_open_orders(symbol=self.trading_symbol)
                time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Error canceling open orders: {e}")

        # Generate unique client order IDs.
        sl_client_order_id = (
            f"sl_{int(time.time() * 1000)}" if stop_loss_price is not None else None
        )
        tp_client_order_id = (
            f"tp_{int(time.time() * 1000)}" if take_profit_price is not None else None
        )

        # Base parameters common to both orders.
        base_params = {
            "symbol": self.trading_symbol,
            "side": side,
            "positionSide": position_side,
            "workingType": "MARK_PRICE",  # Trigger orders by mark price.
            "priceProtect": "TRUE",
            "timeInForce": "GTC",
        }
        if close_position:
            base_params["closePosition"] = "TRUE"
        else:
            base_params["quantity"] = quantity
            base_params["reduceOnly"] = "TRUE"

        order_responses = []

        # Place Stop Loss order if provided.
        if stop_loss_price is not None:
            sl_params = base_params.copy()
            sl_params["type"] = "STOP_MARKET"
            sl_params["stopPrice"] = stop_loss_price
            sl_params["newClientOrderId"] = sl_client_order_id
            try:
                sl_response = self.client.new_order(**sl_params)
                logger.info(f"Placed STOP_MARKET order at {stop_loss_price} USDT")
                order_responses.append(
                    {"order_type": "STOP_MARKET", "response": sl_response}
                )
            except ClientError as e:
                logger.error(f"Failed to place stop loss order: {e}")
                order_responses.append({"order_type": "STOP_MARKET", "error": str(e)})

        # Place Take Profit order if provided.
        if take_profit_price is not None:
            tp_params = base_params.copy()
            tp_params["type"] = "TAKE_PROFIT_MARKET"
            tp_params["stopPrice"] = take_profit_price
            tp_params["newClientOrderId"] = tp_client_order_id
            try:
                tp_response = self.client.new_order(**tp_params)
                logger.info(
                    f"Placed TAKE_PROFIT_MARKET order at {take_profit_price} USDT"
                )
                order_responses.append(
                    {"order_type": "TAKE_PROFIT_MARKET", "response": tp_response}
                )
            except ClientError as e:
                logger.error(f"Failed to place take profit order: {e}")
                order_responses.append(
                    {"order_type": "TAKE_PROFIT_MARKET", "error": str(e)}
                )

        return {
            "orders": order_responses,
            "sl_client_order_id": sl_client_order_id,
            "tp_client_order_id": tp_client_order_id,
            "is_oco": True,
        }

    def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """
        Cancel an order by order ID.

        Args:
            order_id: The ID of the order to cancel.

        Returns:
            API response.
        """
        try:
            response = self.client.cancel_order(
                symbol=self.trading_symbol, orderId=order_id
            )
            logger.info(f"Cancelled order {order_id}")
            return response
        except ClientError as e:
            logger.error(f"Failed to cancel order: {e}")
            return {"error": str(e)}

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Retrieve all open orders for the trading symbol.

        Returns:
            List of open orders.
        """
        try:
            return self.client.get_open_orders(symbol=self.trading_symbol)
        except ClientError as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_order_info(self, order_id: int) -> Dict[str, Any]:
        """
        Get details for a specific order by order ID.

        Args:
            order_id: The order ID.

        Returns:
            Order details as a dictionary.
        """
        try:
            return self.client.get_order(symbol=self.trading_symbol, orderId=order_id)
        except ClientError as e:
            logger.error(f"Failed to get order info: {e}")
            return {"error": str(e)}

    def get_position_info(self) -> Dict[str, Any]:
        """Alias for get_btc_position()."""
        return self.get_btc_position()

    def close_all_positions(self) -> Dict[str, Any]:
        """
        Close all open positions for the trading symbol.

        Returns:
            API response for closing the position.
        """
        try:
            position = self.get_btc_position()
            if "positionAmt" in position and float(position["positionAmt"]) != 0:
                position_amt = abs(float(position["positionAmt"]))
                side = "SELL" if float(position["positionAmt"]) > 0 else "BUY"
                response = self.client.new_order(
                    symbol=self.trading_symbol,
                    side=side,
                    type="MARKET",
                    quantity=position_amt,
                    reduceOnly="true",
                )
                logger.info(f"Closed position of {position_amt} BTC with {side} order")
                return response
            else:
                logger.info("No open positions to close")
                return {"info": "No open positions to close"}
        except (ClientError, KeyError) as e:
            logger.error(f"Failed to close positions: {e}")
            return {"error": str(e)}

    def cancel_order_by_client_id(self, client_order_id: str) -> Dict[str, Any]:
        """
        Cancel an order by its client order ID.

        Args:
            client_order_id: The client order ID.

        Returns:
            API response.
        """
        try:
            response = self.client.cancel_order(
                symbol=self.trading_symbol, origClientOrderId=client_order_id
            )
            logger.info(f"Canceled order with client order ID {client_order_id}")
            return response
        except ClientError as e:
            logger.error(
                f"Failed to cancel order with client ID {client_order_id}: {e}"
            )
            return {"error": str(e)}

    def check_order_status(self, client_order_id: str) -> Dict[str, Any]:
        """
        Check the status of an order by its client order ID.

        Args:
            client_order_id: The client order ID.

        Returns:
            Order status information.
        """
        try:
            return self.client.get_order(
                symbol=self.trading_symbol, origClientOrderId=client_order_id
            )
        except ClientError as e:
            logger.error(
                f"Failed to check order status for client ID {client_order_id}: {e}"
            )
            return {"error": str(e)}

    def manage_oco_orders(
        self, sl_client_order_id: str, tp_client_order_id: str
    ) -> Dict[str, Any]:
        """
        Monitor and manage the conditional orders (mimicking OCO).
        If one order is filled, cancel the other.

        Args:
            sl_client_order_id: Client order ID of the stop loss order.
            tp_client_order_id: Client order ID of the take profit order.

        Returns:
            Dictionary with the result and details of the filled order.
        """
        try:
            # Check stop loss order status
            if sl_client_order_id:
                sl_status = self.check_order_status(sl_client_order_id)
                if sl_status.get("status") == "FILLED":
                    logger.info("Stop loss order filled; canceling take profit order.")
                    if tp_client_order_id:
                        self.cancel_order_by_client_id(tp_client_order_id)
                    return {"result": "stop_loss_filled", "order": sl_status}

            # Check take profit order status
            if tp_client_order_id:
                tp_status = self.check_order_status(tp_client_order_id)
                if tp_status.get("status") == "FILLED":
                    logger.info("Take profit order filled; canceling stop loss order.")
                    if sl_client_order_id:
                        self.cancel_order_by_client_id(sl_client_order_id)
                    return {"result": "take_profit_filled", "order": tp_status}

            return {"result": "orders_active"}
        except Exception as e:
            logger.error(f"Failed to manage OCO orders: {e}")
            return {"error": str(e)}

    def get_ticker_info(self) -> Dict[str, Any]:
        """
        Retrieve current ticker information.

        Returns:
            Dictionary with trading symbol and current price.
        """
        return {"symbol": self.trading_symbol, "price": self.get_btc_price()}

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get details for a specific order by order ID.

        Args:
            order_id: The order ID.

        Returns:
            Order details.
        """
        try:
            return self.client.query_order(symbol=self.trading_symbol, orderId=order_id)
        except ClientError as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return {"error": str(e)}


# # If this module is run directly, you can add test calls or debugging output here.
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     executor = BinanceFuturesExecutor(use_testnet=True)

#     # Example usage:
#     account_info = executor.get_account_info()
#     logger.info(f"Account Info: {account_info}")

#     btc_price = executor.get_btc_price()
#     logger.info(f"Current BTC Price: {btc_price} USDT")

#     # Mimic placing an OCO-like order
#     # (For example: SELL order to close a long position with a stop loss at 30,000 and take profit at 40,000)
#     oco_result = executor.execute_stop_loss_take_profit_order(
#         side="SELL", quantity=0.001, stop_loss_price=30000.0, take_profit_price=40000.0
#     )
#     logger.info(f"OCO Order Result: {oco_result}")

#     # You can then call manage_oco_orders periodically to check order statuses
#     # For instance:
#     if oco_result.get("is_oco"):
#         management_result = executor.manage_oco_orders(
#             sl_client_order_id=oco_result.get("sl_client_order_id"),
#             tp_client_order_id=oco_result.get("tp_client_order_id"),
#         )
#         logger.info(f"OCO Management Result: {management_result}")
