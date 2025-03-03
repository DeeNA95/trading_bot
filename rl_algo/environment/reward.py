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


class SimpleReward(RewardFunction):
    """Simple profit and loss based reward function."""

    def calculate_reward(self, state, action, next_state, info):
        # Extract PnL information
        realized_pnl = info.get("realized_pnl", 0)
        unrealized_pnl = info.get("unrealized_pnl", 0)
        transaction_cost = info.get("transaction_cost", 0)
        # Reward is the total PnL minus transaction costs
        reward = realized_pnl + 0.1 * unrealized_pnl - transaction_cost
        return reward


class SharpeReward(RewardFunction):
    """Sharpe ratio based reward function that balances returns and risk."""

    def __init__(self, window_size=20, risk_free_rate=0.0, epsilon=1e-8):
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.epsilon = epsilon
        self.returns_history = []

    def calculate_reward(self, state, action, next_state, info):
        # Get the current return (must be provided by environment)
        current_return = info.get("return", 0)
        self.returns_history.append(current_return)
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)
        if len(self.returns_history) < 2:
            return current_return
        returns_array = np.array(self.returns_history)
        returns_mean = np.mean(returns_array)
        returns_std = np.std(returns_array)
        sharpe = (returns_mean - self.risk_free_rate) / (returns_std + self.epsilon)
        return sharpe


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


class TunableReward(RewardFunction):
    """
    Highly configurable reward function with tunable weights.
    """

    def __init__(
        self,
        realized_pnl_weight=1.0,
        unrealized_pnl_weight=0.1,
        transaction_cost_weight=1.0,
        trade_penalty=0.0005,
        leverage_penalty=0.01,
        drawdown_penalty=0.1,
        volatility_penalty=0.0,
        holding_time_bonus=0.0,
        trend_alignment_bonus=0.0,
        risk_adjusted=True,
        sharpe_window=20,
        sharpe_weight=0.0,
        reward_scale=1.0,
        track_components=False,
    ):
        self.realized_pnl_weight = realized_pnl_weight
        self.unrealized_pnl_weight = unrealized_pnl_weight
        self.transaction_cost_weight = transaction_cost_weight
        self.trade_penalty = trade_penalty
        self.leverage_penalty = leverage_penalty
        self.drawdown_penalty = drawdown_penalty
        self.volatility_penalty = volatility_penalty
        self.holding_time_bonus = holding_time_bonus
        self.trend_alignment_bonus = trend_alignment_bonus
        self.risk_adjusted = risk_adjusted
        self.sharpe_weight = sharpe_weight
        self.reward_scale = reward_scale
        self.track_components = track_components
        self.sharpe_window = sharpe_window
        self.returns_history = []
        self.reward_components = {
            "realized_pnl": 0,
            "unrealized_pnl": 0,
            "transaction_cost": 0,
            "trade_penalty": 0,
            "leverage_penalty": 0,
            "drawdown_penalty": 0,
            "volatility_penalty": 0,
            "holding_time_bonus": 0,
            "trend_alignment_bonus": 0,
            "sharpe_ratio": 0,
            "total": 0,
        }

    def calculate_reward(self, state, action, next_state, info):
        if self.track_components:
            for key in self.reward_components:
                self.reward_components[key] = 0

        realized_pnl = info.get("realized_pnl", 0)
        unrealized_pnl = info.get("unrealized_pnl", 0)
        transaction_cost = info.get("transaction_cost", 0)

        base_reward = (
            self.realized_pnl_weight * realized_pnl
            + self.unrealized_pnl_weight * unrealized_pnl
            - self.transaction_cost_weight * transaction_cost
        )
        reward = base_reward

        if self.track_components:
            self.reward_components["realized_pnl"] = (
                self.realized_pnl_weight * realized_pnl
            )
            self.reward_components["unrealized_pnl"] = (
                self.unrealized_pnl_weight * unrealized_pnl
            )
            self.reward_components["transaction_cost"] = (
                -self.transaction_cost_weight * transaction_cost
            )

        if self.risk_adjusted:
            is_trade = info.get("is_trade", False)
            leverage = info.get("leverage", 1.0)
            drawdown = info.get("drawdown", 0.0)
            holding_time = info.get("holding_time", 0)

            if is_trade:
                reward -= self.trade_penalty
                if self.track_components:
                    self.reward_components["trade_penalty"] = -self.trade_penalty

            leverage_penalty = self.leverage_penalty * (leverage**2)
            reward -= leverage_penalty
            if self.track_components:
                self.reward_components["leverage_penalty"] = -leverage_penalty

            drawdown_penalty = self.drawdown_penalty * drawdown
            reward -= drawdown_penalty
            if self.track_components:
                self.reward_components["drawdown_penalty"] = -drawdown_penalty

            if holding_time > 0:
                holding_bonus = self.holding_time_bonus * np.log1p(holding_time)
                reward += holding_bonus
                if self.track_components:
                    self.reward_components["holding_time_bonus"] = holding_bonus

            if "trend_alignment" in info:
                trend_bonus = self.trend_alignment_bonus * info.get(
                    "trend_alignment", 0
                )
                reward += trend_bonus
                if self.track_components:
                    self.reward_components["trend_alignment_bonus"] = trend_bonus

        if self.sharpe_weight > 0:
            current_return = info.get("return", 0)
            self.returns_history.append(current_return)
            if len(self.returns_history) > self.sharpe_window:
                self.returns_history.pop(0)
            if len(self.returns_history) >= 2:
                returns_array = np.array(self.returns_history)
                returns_mean = np.mean(returns_array)
                returns_std = np.std(returns_array) + 1e-8
                sharpe = returns_mean / returns_std
                sharpe_reward = self.sharpe_weight * sharpe
                reward += sharpe_reward
                if self.track_components:
                    self.reward_components["sharpe_ratio"] = sharpe_reward

        if self.volatility_penalty > 0 and len(self.returns_history) >= 2:
            volatility = np.std(self.returns_history)
            volatility_penalty = self.volatility_penalty * volatility
            reward -= volatility_penalty
            if self.track_components:
                self.reward_components["volatility_penalty"] = -volatility_penalty

        reward *= self.reward_scale
        if self.track_components:
            self.reward_components["total"] = reward

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

        # Adjust reward based on performance and risk
        risk_multiplier = 1 + max(0, return_pct)
        if raw_return > 0:
            reward = (raw_return * (1 + leverage / 2)) * risk_multiplier
        else:
            reward = raw_return * (1 - leverage / 3)

        if is_trade:
            reward -= self.trade_penalty

        leverage_penalty = self.leverage_penalty * (leverage**2) / risk_multiplier
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
