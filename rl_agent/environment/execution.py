"""
Trade execution logic for Binance Futures.

This module handles the execution of trades on Binance Futures, including
position entry, stop-loss, and take-profit orders.
"""
import os
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
from binance.um_futures import UMFutures
from dotenv import load_dotenv
load_dotenv()
API = os.getenv("binance_api2")
SECRET = os.getenv('binance_secret2')
class BinanceFuturesExecutor:
    """
    Executes and tracks trades on Binance Futures with proper risk management.

    This class encapsulates the logic for:
    - Opening positions with appropriate leverage and margin
    - Setting stop-loss and take-profit orders
    - Tracking open positions and orders
    - Handling position exits and cleanup
    """

    def __init__(
        self,
        client: UMFutures = UMFutures(
            base_url="https://fapi.binance.com",
            key = API,
            secret = SECRET

            ),
        symbol: str = "BTCUSDT",
        leverage: int = 2,
        margin_type: str = "ISOLATED",
        risk_reward_ratio: float = 1.5,
        stop_loss_percent: float = 0.1,
        recv_window: int = 6000,
        dry_run: bool = False,
        max_trade_allocation: float = 1.0,
    ):
        """
        Initialize the Binance Futures trade executor.

        Args:
            client: Binance UMFutures client instance
            symbol: Trading pair symbol
            leverage: Trading leverage
            margin_type: Margin type ('ISOLATED' or 'CROSSED')
            risk_reward_ratio: Ratio of take profit to stop loss
            stop_loss_percent: Stop loss percentage from entry
            recv_window: Receive window for API calls
            dry_run: If True, don't execute actual trades (simulation)
        """
        self.client = client
        self.symbol = symbol
        self.leverage = leverage
        self.margin_type = margin_type
        self.risk_reward_ratio = risk_reward_ratio
        self.stop_loss_percent = stop_loss_percent
        self.recv_window = recv_window
        self.dry_run = dry_run
        self.max_trade_allocation = max_trade_allocation

        # Logger
        self.logger = logging.getLogger("BinanceFuturesExecutor")
        self.logger.setLevel(logging.INFO)

        # Position tracking
        self.position_open = False
        self.position_direction = 0  # 1 for long, -1 for short, 0 for no position
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_time = None

        # Order tracking
        self.entry_order_id = None
        self.stop_loss_order_id = None
        self.take_profit_order_id = None

        # Price precision info
        self.price_precision = None
        self.qty_precision = None

        # Initialize
        if not dry_run:
            self._update_trading_precision()
            self._initialize_account_settings()
            self._check_existing_positions()

    def _update_trading_precision(self) -> None:
        """Get symbol precision settings from exchange."""
        try:
            exchange_info = self.client.exchange_info()
            for symbol_info in exchange_info["symbols"]:
                if symbol_info["symbol"] == self.symbol:
                    self.price_precision = symbol_info["pricePrecision"]
                    self.qty_precision = symbol_info["quantityPrecision"]
                    break

            if self.price_precision is None:
                self.price_precision = 2
                self.qty_precision = 5
                self.logger.warning(
                    f"Could not find precision info for {self.symbol}, using defaults"
                )

        except Exception as e:
            self.logger.error(f"Error getting trading parameters: {e}")
            self.price_precision = 2
            self.qty_precision = 5

    def _initialize_account_settings(self) -> None:
        """Initialize account leverage and margin settings."""
        try:
            # Set leverage
            self.client.change_leverage(
                symbol=self.symbol, leverage=self.leverage, recvWindow=self.recv_window
            )

            # Set margin type
            self.client.change_margin_type(
                symbol=self.symbol,
                marginType=self.margin_type,
                recvWindow=self.recv_window,
            )

            self.logger.info(
                f"Account initialized for {self.symbol} with {self.leverage}x leverage, {self.margin_type} margin"
            )

        except Exception as e:
            if "No need to change margin type" in str(e):
                self.logger.info(f"Margin type already set to {self.margin_type}")
            else:
                self.logger.error(f"Error initializing account settings: {e}")

    def _check_existing_positions(self) -> None:
        """Check if there are already open positions for the symbol."""
        try:
            positions = self.client.get_position_risk(symbol=self.symbol)

            for position in positions:
                if position["symbol"] == self.symbol:
                    position_amt = float(position["positionAmt"])

                    if position_amt != 0:
                        self.position_open = True
                        self.position_size = abs(position_amt)
                        self.position_direction = 1 if position_amt > 0 else -1
                        self.entry_price = float(position["entryPrice"])
                        self.logger.info(
                            f"Found existing {self.symbol} position: "
                            f"{'Long' if position_amt > 0 else 'Short'} {self.position_size} "
                            f"at {self.entry_price}"
                        )
                        return

            # If we get here, no open position
            self.position_open = False
            self.position_direction = 0
            self.position_size = 0
            self.entry_price = 0

        except Exception as e:
            self.logger.error(f"Error checking existing positions: {e}")

    def _check_open_orders(self) -> List[Dict]:
        """Check for open orders for the symbol."""
        try:
            open_orders = self.client.get_open_orders(symbol=self.symbol)
            return open_orders
        except Exception as e:
            self.logger.error(f"Error checking open orders: {e}")
            return []

    def get_current_price(self) -> float:
        """Get current price of the symbol."""
        try:
            ticker = self.client.ticker_price(symbol=self.symbol)
            return float(ticker["price"])
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return 0.0

    def calculate_quantity(self, usdt_amount: float) -> float:
        """
        Calculate the quantity of the asset to buy based on USDT amount.

        Args:
            usdt_amount: Amount in USDT to allocate

        Returns:
            Quantity of the asset
        """
        current_price = self.get_current_price()

        if current_price == 0:
            self.logger.error("Cannot calculate quantity with price = 0")
            return 0.0

        # Binance minimum order value of 20 USD
        MIN_ORDER_VALUE = 20.0

        max_allocation_amount = usdt_amount * self.max_trade_allocation

        # Calculate quantity based on allocation
        quantity = min(
            usdt_amount / current_price, max_allocation_amount / current_price
        )

        # Check if order value meets minimum requirement
        order_value = quantity * current_price
        if order_value < MIN_ORDER_VALUE:
            # Adjust quantity to meet minimum order value
            quantity = round((MIN_ORDER_VALUE * 1.01) / current_price, self.qty_precision)
            self.logger.info(f"Adjusted order size to meet minimum value requirement of {MIN_ORDER_VALUE} USD")


        # Double-check after rounding
        if quantity * current_price < MIN_ORDER_VALUE:
            # If after rounding it's still below minimum, adjust again
            min_quantity = MIN_ORDER_VALUE / current_price
            # Round up to the allowed precision
            quantity = math.ceil(min_quantity * (10 ** self.qty_precision)) / (10 ** self.qty_precision)

        return quantity

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders for the symbol."""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Cancelling all orders for {self.symbol}")
            return True

        try:
            result = self.client.cancel_open_orders(
                symbol=self.symbol, recvWindow=self.recv_window
            )
            self.logger.info(f"Cancelled all orders for {self.symbol}")

            # Reset order IDs
            self.stop_loss_order_id = None
            self.take_profit_order_id = None

            return True

        except Exception as e:
            self.logger.error(f"Error cancelling orders: {e}")
            return False

    def close_position(self) -> bool:
        """Close any open position for the symbol."""
        if not self.position_open:
            return True

        if self.dry_run:
            self.logger.info(f"[DRY RUN] Closing position for {self.symbol}")
            self.position_open = False
            self.position_direction = 0
            self.position_size = 0
            self.entry_price = 0
            return True

        try:
            # Cancel all open orders first
            self.cancel_all_orders()

            # Close position with market order
            side = "SELL" if self.position_direction > 0 else "BUY"

            self.client.new_order(
                symbol=self.symbol,
                side=side,
                type="MARKET",
                quantity=self.position_size,
                recvWindow=self.recv_window,
            )

            self.logger.info(
                f"Closed {self.symbol} position: {side} {self.position_size}"
            )

            # Reset position tracking
            self.position_open = False
            self.position_direction = 0
            self.position_size = 0
            self.entry_price = 0

            return True

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False

    def execute_trade(
        self,
        action: int,  # 0: hold, 1: buy/long, 2: sell/short
        quantity: float = None,
        usdt_amount: float = None,
        custom_sl_percent: float = None,
        custom_tp_ratio: float = None,
        scale_in: bool = False,  # New parameter for scaling into positions
        scale_out: bool = False, # New parameter for scaling out of positions
        scale_percentage: float = 0.5, # Percentage to scale in/out (default 50%)
    ) -> Dict[str, Any]:
        """
        Execute a trade with proper stop-loss and take-profit orders.
        Supports scaling into winning positions and scaling out of losing ones.

        Args:
            action: Trading action (0: hold, 1: buy/long, 2: sell/short)
            quantity: Quantity to trade (override calculated quantity)
            usdt_amount: USDT amount to use for calculating quantity
            custom_sl_percent: Custom stop loss percentage
            custom_tp_ratio: Custom take profit ratio
            scale_in: Whether to scale into an existing position (add to it)
            scale_out: Whether to scale out of an existing position (reduce it)
            scale_percentage: Percentage to scale in/out (0.5 = 50%)

        Returns:
            Dictionary with trade information
        """
        # Handle hold action
        if action == 0:
            return {
                "action": "hold",
                "position_open": self.position_open,
                "position_direction": self.position_direction,
                "position_size": self.position_size,
                "entry_price": self.entry_price,
                "success": True,
            }

        # Handle scaling logic when a position is already open
        if self.position_open:
            # Check if we're trying to scale in/out
            if scale_in or scale_out:
                return self._handle_position_scaling(
                    action,
                    quantity,
                    usdt_amount,
                    custom_sl_percent,
                    custom_tp_ratio,
                    scale_in,
                    scale_out,
                    scale_percentage
                )
            else:
                # Normal case - position already open but not scaling
                return {
                    "action": "hold",
                    "position_open": self.position_open,
                    "position_direction": self.position_direction,
                    "position_size": self.position_size,
                    "entry_price": self.entry_price,
                    "success": True,
                }

        # Get current price and calculate quantity if not provided
        current_price = self.get_current_price()

        if current_price == 0:
            return {
                "action": "error",
                "message": "Could not get current price",
                "success": False,
            }

        # If quantity not provided, calculate it from USDT amount
        if quantity is None:
            if usdt_amount is None:
                return {
                    "action": "error",
                    "message": "Either quantity or usdt_amount must be provided",
                    "success": False,
                }

            quantity = self.calculate_quantity(usdt_amount)

        # Use provided stop loss or default
        sl_percent = (
            custom_sl_percent
            if custom_sl_percent is not None
            else self.stop_loss_percent
        )
        tp_ratio = (
            custom_tp_ratio if custom_tp_ratio is not None else self.risk_reward_ratio
        )

        # Determine side and order parameters
        if action == 1:  # Buy/Long
            side = "BUY"
            sl_price = round(current_price * (1 - sl_percent), self.price_precision)
            tp_price = round(
                current_price * (1 + sl_percent * tp_ratio), self.price_precision
            )

        elif action == 2:  # Sell/Short
            side = "SELL"
            sl_price = round(current_price * (1 + sl_percent), self.price_precision)
            tp_price = round(
                current_price * (1 - sl_percent * tp_ratio), self.price_precision
            )

        else:
            return {
                "action": "error",
                "message": f"Invalid action: {action}",
                "success": False,
            }

        # Execute in dry run mode
        if self.dry_run:
            self.logger.info(
                f"[DRY RUN] Opening {side} position: {quantity} {self.symbol} @ {current_price} "
                f"with SL @ {sl_price}, TP @ {tp_price}"
            )

            self.position_open = True
            self.position_direction = 1 if side == "BUY" else -1
            self.position_size = quantity
            self.entry_price = current_price
            self.entry_time = datetime.now()

            return {
                "action": side.lower(),
                "quantity": quantity,
                "price": current_price,
                "stop_loss": sl_price,
                "take_profit": tp_price,
                "position_open": True,
                "position_direction": self.position_direction,
                "success": True,
            }

        # Execute real trade
        try:
            # Step 1: Place the main market order
            entry_order = self.client.new_order(
                symbol=self.symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
                recvWindow=self.recv_window,
            )

            # Wait a moment for the order to be executed
            time.sleep(1)

            # Step 2: Get the executed price (important for accurate SL/TP)
            execution_price = None
            try:
                # Get fill price from order info
                order_info = self.client.query_order(  # Changed from get_order to query_order
                    symbol=self.symbol,
                    orderId=entry_order["orderId"],
                    recvWindow=self.recv_window,
                )

                if order_info.get("status") == "FILLED":
                    execution_price = float(order_info.get("avgPrice", current_price))
                else:
                    execution_price = current_price

            except Exception as e:
                self.logger.warning(
                    f"Could not get fill price, using current price: {e}"
                )
                execution_price = current_price

            # Recalculate SL/TP based on actual execution price
            if action == 1:  # Buy/Long
                sl_price = round(
                    execution_price * (1 - sl_percent), self.price_precision
                )
                tp_price = round(
                    execution_price * (1 + sl_percent * tp_ratio), self.price_precision
                )
            else:  # Sell/Short
                sl_price = round(
                    execution_price * (1 + sl_percent), self.price_precision
                )
                tp_price = round(
                    execution_price * (1 - sl_percent * tp_ratio), self.price_precision
                )

            # Step 3: Place stop loss order
            opposite_side = "SELL" if side == "BUY" else "BUY"

            sl_order = self.client.new_order(
                symbol=self.symbol,
                side=opposite_side,
                type="STOP_MARKET",
                timeInForce="GTC",
                quantity=quantity,
                stopPrice=sl_price,
                stopPx=sl_price,
                recvWindow=self.recv_window,
                closePosition=True,
            )

            # Step 4: Place take profit order
            tp_order = self.client.new_order(
                symbol=self.symbol,
                side=opposite_side,
                type="TAKE_PROFIT_MARKET",
                timeInForce="GTC",
                quantity=quantity,
                stopPrice=tp_price,
                recvWindow=self.recv_window,
                closePosition=True,
            )

            # Update position tracking
            self.position_open = True
            self.position_direction = 1 if side == "BUY" else -1
            self.position_size = quantity
            self.entry_price = execution_price
            self.entry_time = datetime.now()

            # Update order tracking
            self.entry_order_id = entry_order["orderId"]
            self.stop_loss_order_id = sl_order["orderId"]
            self.take_profit_order_id = tp_order["orderId"]

            self.logger.info(
                f"Opened {side} position: {quantity} {self.symbol} @ {execution_price} "
                f"with SL @ {sl_price}, TP @ {tp_price}"
            )

            return {
                "action": side.lower(),
                "quantity": quantity,
                "price": execution_price,
                "stop_loss": sl_price,
                "take_profit": tp_price,
                "position_open": True,
                "position_direction": self.position_direction,
                "entry_order_id": self.entry_order_id,
                "sl_order_id": self.stop_loss_order_id,
                "tp_order_id": self.take_profit_order_id,
                "success": True,
            }

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")

            return {"action": "error", "message": str(e), "success": False}

    def _handle_position_scaling(
        self,
        action: int,
        quantity: float = None,
        usdt_amount: float = None,
        custom_sl_percent: float = None,
        custom_tp_ratio: float = None,
        scale_in: bool = False,
        scale_out: bool = False,
        scale_percentage: float = 0.5
    ) -> Dict[str, Any]:
        """
        Handle scaling into or out of existing positions.

        Args:
            action: Trading action (1: buy/long, 2: sell/short)
            quantity: Quantity to trade (override calculated quantity)
            usdt_amount: USDT amount to use for calculating quantity
            custom_sl_percent: Custom stop loss percentage
            custom_tp_ratio: Custom take profit ratio
            scale_in: Whether to scale into an existing position
            scale_out: Whether to scale out of an existing position
            scale_percentage: Percentage to scale in/out

        Returns:
            Dictionary with trade information
        """
        # Get the current price
        current_price = self.get_current_price()
        if current_price == 0:
            return {
                "action": "error",
                "message": "Could not get current price for scaling",
                "success": False,
            }

        # Check if the action matches the current position direction
        position_is_long = self.position_direction > 0
        action_is_long = action == 1

        # For scaling in, the action must match the position direction
        if scale_in and position_is_long != action_is_long:
            return {
                "action": "error",
                "message": f"Cannot scale into {'long' if position_is_long else 'short'} position with {'long' if action_is_long else 'short'} action",
                "success": False,
            }

        # For scaling out, we need to reduce the position size
        if scale_out:
            # Calculate the reduction amount
            if quantity is None:
                quantity = self.position_size * scale_percentage

            # Make sure we don't try to reduce more than we have
            quantity = min(quantity, self.position_size)

            # Round to the appropriate precision for this symbol
            if self.qty_precision is not None:
                quantity = round(quantity, self.qty_precision)

            # Ensure quantity is not zero after rounding and meets minimum order requirements
            if quantity <= 0:
                # Use the minimum quantity allowed for this symbol
                quantity = float(f"{{:.{self.qty_precision}f}}".format(10 ** (-self.qty_precision)))

            # Check if it's too small for the exchange
            MIN_ORDER_VALUE = 20.0  # Binance minimum order value (20 USDT)
            order_value = quantity * current_price
            if order_value < MIN_ORDER_VALUE:
                # Adjust quantity to meet minimum order value
                min_quantity = MIN_ORDER_VALUE / current_price
                # Round to the precision allowed by the exchange
                quantity = math.ceil(min_quantity * (10 ** self.qty_precision)) / (10 ** self.qty_precision)
                self.logger.info(f"Adjusted scaling quantity to meet minimum order value of {MIN_ORDER_VALUE} USDT")

            # For scaling out, we need to take the opposite action of our position
            side = "SELL" if position_is_long else "BUY"

            # Execute in dry run mode
            if self.dry_run:
                self.logger.info(
                    f"[DRY RUN] Scaling out of {'long' if position_is_long else 'short'} position: "
                    f"{quantity} {self.symbol} @ {current_price} ({scale_percentage*100:.0f}%)"
                )

                # Update position tracking
                self.position_size -= quantity
                if self.position_size <= 0:
                    self.position_open = False
                    self.position_direction = 0
                    self.position_size = 0
                    self.entry_price = 0

                return {
                    "action": "scale_out",
                    "quantity": quantity,
                    "price": current_price,
                    "position_open": self.position_open,
                    "position_direction": self.position_direction,
                    "position_size": self.position_size,
                    "success": True,
                }

            # Execute real partial close
            try:
                # Close part of the position with a market order
                order = self.client.new_order(
                    symbol=self.symbol,
                    side=side,
                    type="MARKET",
                    quantity=quantity,
                    recvWindow=self.recv_window,
                )

                # Update position tracking
                self.position_size -= quantity

                # If we've closed the entire position
                if self.position_size <= 0:
                    self.position_open = False
                    self.position_direction = 0
                    self.position_size = 0

                    # Cancel any remaining orders
                    self.cancel_all_orders()
                # else:
                #     # We need to update SL/TP orders for the new position size
                #     self.cancel_all_orders()

                #     # Recalculate SL/TP with current values but new position size
                #     opposite_side = "SELL" if position_is_long else "BUY"

                #     # Get current SL and TP levels (estimated)
                #     sl_percent = custom_sl_percent if custom_sl_percent is not None else self.stop_loss_percent
                #     tp_ratio = custom_tp_ratio if custom_tp_ratio is not None else self.risk_reward_ratio

                #     sl_price = self.entry_price * (1 - sl_percent if position_is_long else 1 + sl_percent)
                #     tp_price = self.entry_price * (1 + sl_percent * tp_ratio if position_is_long else 1 - sl_percent * tp_ratio)

                #     # Round to the appropriate price precision
                #     if self.price_precision is not None:
                #         sl_price = round(sl_price, self.price_precision)
                #         tp_price = round(tp_price, self.price_precision)

                #     # Place new SL order for remaining position
                #     sl_order = self.client.new_order(
                #         symbol=self.symbol,
                #         side=opposite_side,
                #         type="STOP_MARKET",
                #         timeInForce="GTC",
                #         quantity=self.position_size,
                #         stopPrice=sl_price,
                #         stopPx=sl_price,
                #         recvWindow=self.recv_window,
                #         closePosition=True,
                #     )

                #     # Place new TP order for remaining position
                #     tp_order = self.client.new_order(
                #         symbol=self.symbol,
                #         side=opposite_side,
                #         type="TAKE_PROFIT_MARKET",
                #         timeInForce="GTC",
                #         quantity=self.position_size,
                #         stopPrice=tp_price,
                #         recvWindow=self.recv_window,
                #         closePosition=True,
                #     )
                #  scaling without changing SL/TP orders for now

                #     # Update order tracking
                #     self.stop_loss_order_id = sl_order["orderId"]
                #     self.take_profit_order_id = tp_order["orderId"]

                self.logger.info(
                    f"Scaled out of {'long' if position_is_long else 'short'} position: "
                    f"{quantity} {self.symbol} @ {current_price} ({scale_percentage*100:.0f}%)"
                )

                return {
                    "action": "scale_out",
                    "quantity": quantity,
                    "price": current_price,
                    "position_open": self.position_open,
                    "position_direction": self.position_direction,
                    "position_size": self.position_size,
                    "success": True,
                }

            except Exception as e:
                self.logger.error(f"Error scaling out of position: {e}")
                return {"action": "error", "message": str(e), "success": False}

        # Handle scaling into position
        if scale_in:
            # Calculate the additional size
            if quantity is None and usdt_amount is None:
                # Default to a percentage of current position
                current_value = self.position_size * current_price
                usdt_amount = current_value * scale_percentage

            # If quantity not provided, calculate it from USDT amount
            if quantity is None:
                quantity = self.calculate_quantity(usdt_amount)

            # Round to the appropriate precision for this symbol
            if self.qty_precision is not None:
                quantity = round(quantity, self.qty_precision)

            # Ensure quantity is not zero after rounding and meets minimum order requirements
            if quantity <= 0:
                # Use the minimum quantity allowed for this symbol
                quantity = float(f"{{:.{self.qty_precision}f}}".format(10 ** (-self.qty_precision)))

            # Check if it's too small for the exchange
            MIN_ORDER_VALUE = 20.0  # Binance minimum order value (20 USDT)
            order_value = quantity * current_price
            if order_value < MIN_ORDER_VALUE:
                # Adjust quantity to meet minimum order value
                min_quantity = MIN_ORDER_VALUE / current_price
                # Round to the precision allowed by the exchange
                quantity = math.ceil(min_quantity * (10 ** self.qty_precision)) / (10 ** self.qty_precision)
                self.logger.info(f"Adjusted scaling quantity to meet minimum order value of {MIN_ORDER_VALUE} USDT")

            side = "BUY" if position_is_long else "SELL"

            # Execute in dry run mode
            if self.dry_run:
                self.logger.info(
                    f"[DRY RUN] Scaling into {'long' if position_is_long else 'short'} position: "
                    f"Adding {quantity} {self.symbol} @ {current_price}"
                )

                # Update position tracking - recalculate average entry price
                old_value = self.position_size * self.entry_price
                new_value = quantity * current_price
                total_value = old_value + new_value
                total_size = self.position_size + quantity

                # Calculate new average entry price
                self.entry_price = total_value / total_size
                self.position_size = total_size

                return {
                    "action": "scale_in",
                    "quantity": quantity,
                    "price": current_price,
                    "position_open": True,
                    "position_direction": self.position_direction,
                    "position_size": self.position_size,
                    "entry_price": self.entry_price,
                    "success": True,
                }

            # Execute real scale-in with market order
            try:
                # First cancel existing SL/TP orders
                # self.cancel_all_orders()

                # Place the market order to add to position
                order = self.client.new_order(
                    symbol=self.symbol,
                    side=side,
                    type="MARKET",
                    quantity=quantity,
                    recvWindow=self.recv_window,
                )

                # Wait a moment for the order to execute
                time.sleep(1)

                # Update position tracking - calculate new average entry price and total size
                old_value = self.position_size * self.entry_price
                new_value = quantity * current_price
                total_value = old_value + new_value
                total_size = self.position_size + quantity

                # Set new position details
                self.position_size = total_size
                self.entry_price = total_value / total_size

                # Use provided stop loss or default
                sl_percent = custom_sl_percent if custom_sl_percent is not None else self.stop_loss_percent
                tp_ratio = custom_tp_ratio if custom_tp_ratio is not None else self.risk_reward_ratio

                # Calculate new SL/TP levels based on new average entry price
                opposite_side = "SELL" if side == "BUY" else "BUY"

                if side == "BUY":  # Long position
                    sl_price = round(self.entry_price * (1 - sl_percent), self.price_precision)
                    tp_price = round(self.entry_price * (1 + sl_percent * tp_ratio), self.price_precision)
                else:  # Short position
                    sl_price = round(self.entry_price * (1 + sl_percent), self.price_precision)
                    tp_price = round(self.entry_price * (1 - sl_percent * tp_ratio), self.price_precision)

                # # Place new SL order for entire position
                # sl_order = self.client.new_order(
                #     symbol=self.symbol,
                #     side=opposite_side,
                #     type="STOP_MARKET",
                #     timeInForce="GTC",
                #     quantity=self.position_size,
                #     stopPrice=sl_price,
                #     stopPx=sl_price,
                #     recvWindow=self.recv_window,
                #     closePosition=True,
                # )

                # # Place new TP order for entire position
                # tp_order = self.client.new_order(
                #     symbol=self.symbol,
                #     side=opposite_side,
                #     type="TAKE_PROFIT_MARKET",
                #     timeInForce="GTC",
                #     quantity=self.position_size,
                #     stopPrice=tp_price,
                #     recvWindow=self.recv_window,
                #     closePosition=True,
                # )

                # # Update order tracking
                # self.stop_loss_order_id = sl_order["orderId"]
                # self.take_profit_order_id = tp_order["orderId"]

                self.logger.info(
                    f"Scaled into {'long' if position_is_long else 'short'} position: "
                    f"Added {quantity} {self.symbol} @ {current_price}, "
                    f"New size: {self.position_size}, Avg entry: {self.entry_price}, "
                    f"SL @ {sl_price}, TP @ {tp_price}"
                )

                return {
                    "action": "scale_in",
                    "quantity": quantity,
                    "price": current_price,
                    "position_open": True,
                    "position_direction": self.position_direction,
                    "position_size": self.position_size,
                    "entry_price": self.entry_price,
                    "stop_loss": sl_price,
                    "take_profit": tp_price,
                    "success": True,
                }

            except Exception as e:
                self.logger.error(f"Error scaling into position: {e}")
                return {"action": "error", "message": str(e), "success": False}

        # Should never reach here
        return {"action": "error", "message": "Scaling logic error", "success": False}

    def check_position_status(self) -> Dict[str, Any]:
        """
        Check the status of the current position.

        Returns:
            Dictionary with position information
        """
        if not self.position_open:
            return {"position_open": False, "trigger_type": None}

        if self.dry_run:
            # In dry run, just return the current position info
            return {
                "position_open": self.position_open,
                "position_direction": self.position_direction,
                "position_size": self.position_size,
                "entry_price": self.entry_price,
                "trigger_type": None,
            }

        try:
            # Check if position still exists
            positions = self.client.get_position_risk(symbol=self.symbol)
            position_exists = False

            for position in positions:
                if position["symbol"] == self.symbol:
                    position_amt = float(position["positionAmt"])

                    if position_amt != 0:
                        position_exists = True
                        break

            # If position no longer exists but we thought it did, determine how it was closed
            if not position_exists and self.position_open:
                # Check if SL or TP orders are still open
                open_orders = self._check_open_orders()

                # If SL still exists, it was closed by TP
                sl_exists = any(
                    order["orderId"] == self.stop_loss_order_id for order in open_orders
                )

                # If TP still exists, it was closed by SL
                tp_exists = any(
                    order["orderId"] == self.take_profit_order_id
                    for order in open_orders
                )

                # Cancel any remaining orders
                if sl_exists or tp_exists:
                    self.cancel_all_orders()

                # Determine trigger type
                if sl_exists and not tp_exists:
                    trigger_type = "take_profit"
                elif tp_exists and not sl_exists:
                    trigger_type = "stop_loss"
                else:
                    trigger_type = (
                        "manual"  # Closed manually or both SL and TP were cancelled
                    )

                # Update position tracking
                old_direction = self.position_direction
                self.position_open = False
                self.position_direction = 0
                self.position_size = 0

                self.logger.info(
                    f"Position closed by {trigger_type}: "
                    f"{'Long' if old_direction > 0 else 'Short'} position on {self.symbol}"
                )

                return {"position_open": False, "trigger_type": trigger_type}

            # Position still exists
            return {
                "position_open": position_exists,
                "position_direction": self.position_direction,
                "position_size": self.position_size,
                "entry_price": self.entry_price,
                "trigger_type": None,
            }

        except Exception as e:
            self.logger.error(f"Error checking position status: {e}")

            return {
                "position_open": self.position_open,
                "position_direction": self.position_direction,
                "position_size": self.position_size,
                "entry_price": self.entry_price,
                "trigger_type": None,
                "error": str(e),
            }
