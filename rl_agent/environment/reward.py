"""
Reward functions for the trading environment.

This module contains different reward functions that can be used to train
reinforcement learning agents with different objectives.
"""

from abc import ABC, abstractmethod
import numpy as np


class RewardFunction(ABC):
    """Base class for reward functions."""

    @abstractmethod
    def calculate_reward(self, state, action, next_state, info):
        """
        Calculate the reward based on the state transition and action.
        Args:
            state: The current state
            action: The action taken
            next_state: The resulting state
            info: Additional information dict
        Returns:
            float: The calculated reward
        """
        pass


class RiskAdjustedReward(RewardFunction):
    """
    Advanced reward function that considers returns and various risk factors.
    Penalizes excessive trading, high leverage, drawdowns, and liquidation risk.
    """

    def __init__(
        self,
        trade_penalty=0.0005,
        leverage_penalty=0.02,
        drawdown_penalty=0.1,
        liquidation_penalty=1.0,
        funding_rate_penalty=0.01,
    ):
        self.trade_penalty = trade_penalty
        self.leverage_penalty = leverage_penalty
        self.drawdown_penalty = drawdown_penalty
        self.liquidation_penalty = liquidation_penalty
        self.funding_rate_penalty = funding_rate_penalty

    def calculate_reward(self, state, action, next_state, info):
        realized_pnl = info.get("realized_pnl", 0)
        unrealized_pnl = info.get("unrealized_pnl", 0)
        transaction_cost = info.get("transaction_cost", 0)
        reward = realized_pnl + 0.1 * unrealized_pnl - transaction_cost

        is_trade = info.get("is_trade", False)
        leverage = info.get("leverage", 1.0)
        drawdown = info.get("drawdown", 0.0)
        # Estimate liquidation risk if not provided: if liquidation_price is given,
        # check if current_price is within 10% of liquidation_price.
        current_price = info.get("current_price", None)
        liquidation_price = info.get("liquidation_price", None)
        if "liquidation_risk" in info:
            liquidation_risk = info.get("liquidation_risk", False)
        elif liquidation_price is not None and current_price is not None:
            distance = abs(current_price - liquidation_price) / current_price
            liquidation_risk = distance < 0.1
        else:
            liquidation_risk = False

        funding_rate = info.get("funding_rate", 0.0)

        if is_trade:
            reward -= self.trade_penalty

        reward -= self.leverage_penalty * (leverage**2)
        reward -= self.drawdown_penalty * drawdown
        if liquidation_risk:
            reward -= self.liquidation_penalty
        reward -= self.funding_rate_penalty * funding_rate

        return reward

class FuturesRiskAdjustedReward(RiskAdjustedReward):
    """
    Specialized reward function for futures trading that emphasizes leverage and liquidation risk.
    """

    def __init__(
        self,
        trade_penalty=0.0005,
        leverage_penalty=0.05,
        drawdown_penalty=0.2,
        liquidation_penalty=2.0,
        funding_rate_penalty=0.05,
        liquidation_distance_factor=1.0,
        max_leverage=5.0,
        loss_leverage_factor: float = 0.1, # Factor to amplify losses based on leverage
    ):
        super().__init__(
            trade_penalty=trade_penalty,
            leverage_penalty=leverage_penalty,
            drawdown_penalty=drawdown_penalty,
            liquidation_penalty=liquidation_penalty,
            funding_rate_penalty=funding_rate_penalty,
        )
        self.liquidation_distance_factor = liquidation_distance_factor
        self.max_leverage = max_leverage
        self.loss_leverage_factor = loss_leverage_factor

    def calculate_reward(self, state, action, next_state, info):
        # Extract base information
        current_balance = info.get("balance", 1000.0)
        new_balance = info.get("new_balance", current_balance)
        raw_return = new_balance - current_balance
        return_pct = raw_return / current_balance if current_balance > 0 else 0

        reward = raw_return  # Base reward from profit/loss

        is_trade = info.get("is_trade", False)
        leverage = info.get("leverage", 1.0)
        drawdown = info.get("drawdown", 0.0)
        funding_rate = info.get("funding_rate", 0.0)
        liquidation_price = info.get("liquidation_price", None)
        current_price = info.get("current_price", None)
        position_direction = info.get("position_direction", 0)

        # --- Adjust base reward based on PnL and leverage ---
        if raw_return > 0:
            # Optional: Amplify positive returns (consider if this encourages too much risk)
            # risk_multiplier = 1 + max(0, return_pct) # Original multiplier
            # reward = (raw_return * (1 + leverage / 2)) * risk_multiplier
            # Simpler: just use raw return for now
            reward = raw_return
        elif raw_return < 0:
            # Amplify losses based on leverage, ensuring reward stays negative
            # Factor increases penalty for losses with higher leverage
            loss_amplification = 1 + (leverage * self.loss_leverage_factor)
            reward = raw_return * loss_amplification # raw_return is negative, amplification > 1
        else: # raw_return == 0
            reward = 0.0
        # --- End reward adjustment ---

        # Apply penalties
        if is_trade:
            reward -= self.trade_penalty

        # Note: The original leverage penalty was divided by risk_multiplier.
        # Keeping it simple now by just applying the penalty directly.
        # Consider if risk_multiplier logic should be applied elsewhere if desired.
        leverage_penalty = self.leverage_penalty * (leverage**2) # / risk_multiplier
        reward -= leverage_penalty

        drawdown_penalty = self.drawdown_penalty * drawdown
        reward -= drawdown_penalty

        # Estimate liquidation proximity penalty if possible
        liquidation_proximity_penalty = 0
        if (
            liquidation_price is not None
            and current_price is not None
            and position_direction != 0
        ):
            if position_direction > 0:
                distance_to_liquidation = (
                    current_price - liquidation_price
                ) / current_price
            else:
                distance_to_liquidation = (
                    liquidation_price - current_price
                ) / current_price
            if distance_to_liquidation > 0:
                liquidation_proximity_penalty = self.liquidation_distance_factor / (
                    distance_to_liquidation + 0.01
                )
                liquidation_proximity_penalty = min(
                    liquidation_proximity_penalty, self.liquidation_penalty
                )
        reward -= liquidation_proximity_penalty

        funding_penalty = 0
        if position_direction != 0 and funding_rate != 0:
            if (position_direction > 0 and funding_rate > 0) or (
                position_direction < 0 and funding_rate < 0
            ):
                funding_penalty = self.funding_rate_penalty * abs(
                    funding_rate * leverage
                )
        reward -= funding_penalty

        reward *= 1  # Reward scaling if needed
        return reward
