"""
OpenAI Gym compatible environment for cryptocurrency trading.
"""

from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from rl_algo.environment.reward import RewardFunction, RiskAdjustedReward


class TradingEnvironment(gym.Env):
    def __init__(
        self,
        data: pd.DataFrame,
        reward_function: RewardFunction = None,
        initial_balance: float = 10000.0,
        window_size: int = 30,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_leverage: float = 5.0,
        liquidation_threshold: float = 0.8,
    ):
        """Trading environment for reinforcement learning."""
        super().__init__()

        # Validate data
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")

        recommended_columns = ["rsi", "macd", "macd_signal", "atr"]
        missing_recommended = [
            col for col in recommended_columns if col not in data.columns
        ]
        if missing_recommended:
            print(f"Warning: Data missing recommended columns: {missing_recommended}")

        self.data = data
        self.reward_function = reward_function or RiskAdjustedReward()
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.commission = commission
        self.slippage = slippage
        self.max_leverage = max_leverage
        self.liquidation_threshold = liquidation_threshold

        # Action space: [direction, size, take_profit, stop_loss]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.5, 0.5]),
            high=np.array([1.0, 1.0, 5.0, 5.0]),
            dtype=np.float32,
        )

        # Initialize state variables
        self.balance = initial_balance
        self.max_balance = initial_balance
        self.position_size = 0
        self.position_direction = 0  # 1 for long, -1 for short
        self.position_price = 0.0
        self.position_step = 0  # Step when the position was opened

        # Observation space setup
        self.feature_list = self._get_observation_features()
        self.feature_count = len(self.feature_list)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.feature_count,), dtype=np.float32
        )

        # Reset environment to initial state
        self.reset()

    def _get_observation_features(self) -> List[str]:
        """Return the list of features for the observation."""
        base_features = [
            "open_norm",
            "high_norm",
            "low_norm",
            "close_norm",
            "volume_norm",
        ]
        optional_features = [
            "rsi",
            "macd",
            "macd_signal",
            "atr",
            "bb_upper",
            "bb_lower",
            "stoch_k",
            "stoch_d",
            "adx",
            "plus_di",
            "minus_di",
            "atr_pct",
        ]
        features = base_features.copy()
        for feature in optional_features:
            if feature in self.data.columns:
                features.append(feature)
        # Include account state features
        features.extend(
            ["balance_ratio", "position_value_ratio", "unrealized_pnl_ratio"]
        )
        return features

    def _get_current_price(self) -> float:
        """Return the current closing price."""
        return self.data.iloc[self.current_step]["close"]

    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL for the current position."""
        if self.position_size == 0:
            return 0.0
        current_price = self._get_current_price()
        if self.position_direction > 0:
            return self.position_size * (current_price / self.position_price - 1)
        else:
            return self.position_size * (1 - current_price / self.position_price)

    def _calculate_liquidation_price(
        self, position_direction, position_price, leverage
    ):
        """
        Calculate the liquidation price based on position direction, entry price, and leverage.
        """
        if position_direction == 0 or leverage <= 0:
            return None
        maintenance_margin = (1 / leverage) * (1 - self.liquidation_threshold)
        if position_direction > 0:  # Long
            return position_price * (1 - maintenance_margin)
        else:  # Short
            return position_price * (1 + maintenance_margin)

    def _get_observation(self) -> np.ndarray:
        """Build and return the current observation."""
        balance_ratio = self.balance / self.initial_balance
        current_price = self._get_current_price()
        position_value = abs(self.position_size) * current_price
        position_value_ratio = position_value / self.balance if self.balance > 0 else 0
        unrealized_pnl = self._calculate_unrealized_pnl()
        unrealized_pnl_ratio = unrealized_pnl / self.balance if self.balance > 0 else 0

        features = []
        for feature in self.feature_list:
            if feature in self.data.columns:
                value = self.data.iloc[self.current_step][feature]
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                features.append(value)
            elif feature == "balance_ratio":
                features.append(balance_ratio)
            elif feature == "position_value_ratio":
                features.append(position_value_ratio)
            elif feature == "unrealized_pnl_ratio":
                features.append(unrealized_pnl_ratio)
            else:
                features.append(0.0)
        return np.array(features, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        current_price = self.data.iloc[self.current_step]["close"]
        prev_price = self.data.iloc[self.current_step - 1]["close"]

        prev_balance = self.balance
        prev_position_size = self.position_size

        # Decode action parameters
        direction, size, take_profit, stop_loss = action
        leverage = size * self.max_leverage if direction != 0 else 0
        position_value = self.balance * size

        transaction_cost = 0.0
        unrealized_pnl = 0.0
        holding_time = 0

        trend = np.sign(current_price - prev_price)

        if self.position_size != 0:
            holding_time = self.current_step - self.position_step
            price_diff = (current_price - self.position_price) if self.position_direction > 0 else (self.position_price - current_price)
            unrealized_pnl = price_diff * self.position_size

        liquidation_price = None
        if self.position_size > 0:
            liquidation_price = self._calculate_liquidation_price(self.position_direction, self.position_price, leverage)
            if liquidation_price is not None:
                if (self.position_direction > 0 and current_price <= liquidation_price) or \
                   (self.position_direction < 0 and current_price >= liquidation_price):
                    self.balance -= self.position_value
                    self.position_size = 0
                    self.position_direction = 0
                    self.position_price = 0.0
                    transaction_cost = 0.0
                    unrealized_pnl = 0.0

        if self.position_size != 0 and (direction * self.position_direction <= 0 or size == 0):
            price_diff = (current_price - self.position_price) if self.position_direction > 0 else (self.position_price - current_price)
            position_pnl = price_diff * self.position_size
            slippage_cost = current_price * self.slippage * self.position_size
            commission_cost = current_price * self.commission * self.position_size
            self.balance += position_pnl - slippage_cost - commission_cost
            transaction_cost = slippage_cost + commission_cost
            self.position_size = 0
            self.position_direction = 0
            self.position_price = 0.0
            self.position_value = 0.0
            self.position_step = 0
            unrealized_pnl = 0.0

        if direction != 0 and self.position_size == 0:
            self.position_value = position_value
            self.position_size = (position_value * leverage / current_price) if current_price > 0 else 0.0
            self.position_direction = 1 if direction > 0 else -1
            slippage_cost = current_price * self.slippage * self.position_size
            commission_cost = current_price * self.commission * self.position_size
            self.position_price = current_price * (1 + self.position_direction * self.slippage)
            transaction_cost = slippage_cost + commission_cost
            self.position_step = self.current_step
            liquidation_price = self._calculate_liquidation_price(self.position_direction, self.position_price, leverage)

        funding_rate = 0.0
        if "fundingRate" in self.data.columns:
            funding_rate = self.data.iloc[self.current_step]["fundingRate"]
            if self.position_size > 0:
                funding_payment = -funding_rate * self.position_value * self.position_direction
                self.balance += funding_payment

        # New info dict includes additional keys for reward functions.
        info = {
            "realized_pnl": self.balance - prev_balance,
            "unrealized_pnl": unrealized_pnl,
            "transaction_cost": transaction_cost,
            "holding_time": holding_time,
            "current_price": current_price,
            "liquidation_price": liquidation_price,
            "position_direction": self.position_direction,
            "leverage": leverage,
            "drawdown": (self.max_balance - self.balance) / self.max_balance if self.max_balance > 0 else 0,
            "is_trade": True if (direction != 0 and prev_position_size == 0) else False,
            "balance": prev_balance,
            "return": (self.balance - prev_balance) / prev_balance if prev_balance > 0 else 0,
            "funding_rate": funding_rate,
        }
        observation = self._get_observation()
        reward = self.reward_function.calculate_reward(None, action, observation, info)
        self.max_balance = max(self.max_balance, self.balance + unrealized_pnl)
        done = self.current_step >= len(self.data) - 1 or self.balance <= 0
        return observation, reward, done, False, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.position_size = 0
        self.position_direction = 0
        self.position_price = 0.0
        self.position_step = 0
        self.take_profit = 0
        self.stop_loss = 0
        self.current_step = self.window_size if self.window_size < len(self.data) else 0
        observation = self._get_observation()
        return observation, {}
