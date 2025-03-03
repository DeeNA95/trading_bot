"""
Reward functions for the trading environment.

This module contains different reward functions that can be used to train
reinforcement learning agents with different objectives.
"""

import numpy as np
from abc import ABC, abstractmethod


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
        """
        Calculate reward based on realized and unrealized PnL.

        Args:
            state: The current state
            action: The action taken
            next_state: The resulting state
            info: Additional information dict containing:
                - realized_pnl: PnL from closed positions
                - unrealized_pnl: PnL from open positions
                - transaction_cost: Cost of executing the trade

        Returns:
            float: The calculated reward
        """
        # Extract PnL information
        realized_pnl = info.get('realized_pnl', 0)
        unrealized_pnl = info.get('unrealized_pnl', 0)
        transaction_cost = info.get('transaction_cost', 0)

        # Reward is the total PnL minus transaction costs
        reward = realized_pnl + 0.1 * unrealized_pnl - transaction_cost

        return reward


class SharpeReward(RewardFunction):
    """Sharpe ratio based reward function that balances returns and risk."""

    def __init__(self, window_size=20, risk_free_rate=0.0, epsilon=1e-8):
        """
        Initialize the Sharpe reward function.

        Args:
            window_size: Number of steps to calculate rolling Sharpe ratio
            risk_free_rate: Risk-free rate for Sharpe calculation
            epsilon: Small value to avoid division by zero
        """
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.epsilon = epsilon
        self.returns_history = []

    def calculate_reward(self, state, action, next_state, info):
        """
        Calculate reward based on Sharpe ratio of recent returns.

        Args:
            state: The current state
            action: The action taken
            next_state: The resulting state
            info: Additional information dict containing:
                - return: The return for the current step

        Returns:
            float: The calculated reward (Sharpe ratio)
        """
        # Get the current return
        current_return = info.get('return', 0)

        # Add to returns history
        self.returns_history.append(current_return)

        # Keep only the last window_size returns
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)

        # If we don't have enough history, return the current return
        if len(self.returns_history) < 2:
            return current_return

        # Calculate Sharpe ratio
        returns_array = np.array(self.returns_history)
        returns_mean = np.mean(returns_array)
        returns_std = np.std(returns_array)

        # Calculate the Sharpe ratio (add epsilon to avoid division by zero)
        sharpe = (returns_mean - self.risk_free_rate) / (returns_std + self.epsilon)

        # Scale the Sharpe ratio to a reasonable reward range
        reward = sharpe

        return reward


class RiskAdjustedReward(RewardFunction):
    """
    Advanced reward function that considers both returns and various risk factors.
    Penalizes excessive trading, high leverage, large drawdowns, and liquidation risk.
    """

    def __init__(self, trade_penalty=0.0005, leverage_penalty=0.02, drawdown_penalty=0.1, liquidation_penalty=1.0, funding_rate_penalty=0.01):
        """
        Initialize the risk-adjusted reward function.

        Args:
            trade_penalty: Penalty for making a trade (encourages less frequent trading)
            leverage_penalty: Penalty factor for high leverage
            drawdown_penalty: Penalty factor for drawdowns
            liquidation_penalty: Penalty factor for liquidation risk
            funding_rate_penalty: Penalty factor for funding rates
        """
        self.trade_penalty = trade_penalty
        self.leverage_penalty = leverage_penalty
        self.drawdown_penalty = drawdown_penalty
        self.liquidation_penalty = liquidation_penalty
        self.funding_rate_penalty = funding_rate_penalty

    def calculate_reward(self, state, action, next_state, info):
        """
        Calculate a risk-adjusted reward.

        Args:
            state: The current state
            action: The action taken
            next_state: The resulting state
            info: Additional information dict containing:
                - realized_pnl: PnL from closed positions
                - unrealized_pnl: PnL from open positions
                - transaction_cost: Cost of executing the trade
                - is_trade: Whether a new trade was executed
                - leverage: Current leverage used
                - drawdown: Current drawdown percentage
                - liquidation_risk: Whether liquidation risk is present
                - funding_rate: Current funding rate

        Returns:
            float: The calculated reward
        """
        # Extract PnL information
        realized_pnl = info.get('realized_pnl', 0)
        unrealized_pnl = info.get('unrealized_pnl', 0)
        transaction_cost = info.get('transaction_cost', 0)

        # Extract risk information
        is_trade = info.get('is_trade', False)
        leverage = info.get('leverage', 1.0)
        drawdown = info.get('drawdown', 0.0)
        liquidation_risk = info.get('liquidation_risk', False)
        funding_rate = info.get('funding_rate', 0.0)

        # Base reward is PnL
        reward = realized_pnl + 0.1 * unrealized_pnl - transaction_cost

        # Apply trade penalty if a new trade was made
        if is_trade:
            reward -= self.trade_penalty

        # Apply leverage penalty (quadratic to penalize high leverage more)
        reward -= self.leverage_penalty * (leverage ** 2)

        # Apply drawdown penalty
        reward -= self.drawdown_penalty * drawdown

        # Apply liquidation penalty if liquidation risk is present
        if liquidation_risk:
            reward -= self.liquidation_penalty

        # Apply funding rate penalty
        reward -= self.funding_rate_penalty * funding_rate

        return reward


class FuturesRiskAdjustedReward(RiskAdjustedReward):
    """
    Specialized reward function for futures trading that puts greater emphasis
    on leverage management and liquidation risk avoidance.
    """

    def __init__(self, 
                 trade_penalty=0.0005, 
                 leverage_penalty=0.05,  # Increased from 0.02
                 drawdown_penalty=0.2,   # Increased from 0.1
                 liquidation_penalty=2.0, # Increased from 1.0
                 funding_rate_penalty=0.05, # Increased from 0.01
                 liquidation_distance_factor=1.0,
                 max_leverage=5.0):
        """
        Initialize the futures-specific risk-adjusted reward function.

        Args:
            trade_penalty: Penalty for making a trade (encourages less frequent trading)
            leverage_penalty: Penalty factor for high leverage (higher in futures)
            drawdown_penalty: Penalty factor for drawdowns (higher in futures)
            liquidation_penalty: Penalty factor for liquidation risk (higher in futures)
            funding_rate_penalty: Penalty factor for funding rates
            liquidation_distance_factor: Factor to penalize proximity to liquidation price
            max_leverage: Maximum allowed leverage
        """
        super().__init__(
            trade_penalty=trade_penalty,
            leverage_penalty=leverage_penalty,
            drawdown_penalty=drawdown_penalty,
            liquidation_penalty=liquidation_penalty,
            funding_rate_penalty=funding_rate_penalty
        )
        self.liquidation_distance_factor = liquidation_distance_factor
        self.max_leverage = max_leverage

    def calculate_reward(self, state, action, next_state, info):
        """
        Calculate a futures-specific risk-adjusted reward.

        Args:
            state: The current state
            action: The action taken
            next_state: The resulting state
            info: Additional information dict containing futures-specific metrics

        Returns:
            float: The calculated reward
        """
        # Extract PnL information
        realized_pnl = info.get('realized_pnl', 0)
        unrealized_pnl = info.get('unrealized_pnl', 0)
        transaction_cost = info.get('transaction_cost', 0)

        # Extract risk information
        is_trade = info.get('is_trade', False)
        leverage = info.get('leverage', 1.0)
        drawdown = info.get('drawdown', 0.0)
        liquidation_risk = info.get('liquidation_risk', False)
        
        # Extract futures-specific information with safe defaults
        funding_rate = info.get('fundingRate', 0.0)
        liquidation_price = info.get('liquidation_price', None)
        current_price = info.get('current_price', None)
        position_direction = info.get('position_direction', 0)
        
        # Base reward is PnL
        reward = realized_pnl + 0.1 * unrealized_pnl - transaction_cost

        # Apply trade penalty if a new trade was made
        if is_trade:
            reward -= self.trade_penalty

        # Apply progressive leverage penalty (exponential to heavily penalize high leverage)
        # This creates a stronger disincentive as leverage approaches max_leverage
        leverage_ratio = leverage / self.max_leverage
        leverage_penalty = self.leverage_penalty * (np.exp(2 * leverage_ratio) - 1)
        reward -= leverage_penalty

        # Apply drawdown penalty (more sensitive in futures)
        reward -= self.drawdown_penalty * drawdown

        # Apply liquidation risk penalty
        if liquidation_risk:
            reward -= self.liquidation_penalty
        
        # Calculate distance to liquidation if we have the necessary information
        if liquidation_price is not None and current_price is not None and position_direction != 0:
            # Calculate percentage distance to liquidation
            if position_direction > 0:  # Long position
                distance_to_liquidation = (current_price - liquidation_price) / current_price
            else:  # Short position
                distance_to_liquidation = (liquidation_price - current_price) / current_price
                
            # Penalize proximity to liquidation (higher penalty as we get closer)
            # Uses an inverse relationship: penalty increases as distance decreases
            if distance_to_liquidation > 0:
                liquidation_proximity_penalty = self.liquidation_distance_factor / (distance_to_liquidation + 0.01)
                reward -= min(liquidation_proximity_penalty, self.liquidation_penalty)  # Cap at liquidation_penalty
        
        # Apply funding rate penalty (more important in futures)
        # Penalize when funding rate works against position direction
        if position_direction != 0 and funding_rate != 0:
            # For long positions, negative funding rates are good (we receive payment)
            # For short positions, positive funding rates are good (we receive payment)
            # Penalize when the sign of funding rate is the same as position direction
            if (position_direction > 0 and funding_rate > 0) or (position_direction < 0 and funding_rate < 0):
                funding_penalty = self.funding_rate_penalty * abs(funding_rate) * abs(position_direction)
                reward -= funding_penalty

        return reward


class TunableReward(RewardFunction):
    """
    Highly configurable reward function with tunable weights for different components.
    Allows for experimentation with different reward formulations without changing code.
    """

    def __init__(self,
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
                 track_components=False):
        """
        Initialize the tunable reward function with configurable weights.

        Args:
            realized_pnl_weight: Weight for realized profit/loss
            unrealized_pnl_weight: Weight for unrealized profit/loss
            transaction_cost_weight: Weight for transaction costs
            trade_penalty: Penalty for making a trade (encourages less frequent trading)
            leverage_penalty: Penalty factor for high leverage
            drawdown_penalty: Penalty factor for drawdowns
            volatility_penalty: Penalty factor for high return volatility
            holding_time_bonus: Bonus for holding positions longer (reduces overtrading)
            trend_alignment_bonus: Bonus for aligning with market trend
            risk_adjusted: Whether to include risk adjustment factors
            sharpe_window: Window size for Sharpe ratio calculation
            sharpe_weight: Weight for Sharpe ratio component
            reward_scale: Global scaling factor for the final reward
            track_components: Whether to track individual reward components for analysis
        """
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

        # For Sharpe ratio calculation
        self.sharpe_window = sharpe_window
        self.returns_history = []

        # For tracking reward components
        self.reward_components = {
            'realized_pnl': 0,
            'unrealized_pnl': 0,
            'transaction_cost': 0,
            'trade_penalty': 0,
            'leverage_penalty': 0,
            'drawdown_penalty': 0,
            'volatility_penalty': 0,
            'holding_time_bonus': 0,
            'trend_alignment_bonus': 0,
            'sharpe_ratio': 0,
            'total': 0
        }

    def calculate_reward(self, state, action, next_state, info):
        """
        Calculate a highly configurable reward based on multiple components.

        Args:
            state: The current state
            action: The action taken
            next_state: The resulting state
            info: Additional information dict containing various metrics

        Returns:
            float: The calculated reward
        """
        # Reset reward components if tracking
        if self.track_components:
            for key in self.reward_components:
                self.reward_components[key] = 0

        # Extract PnL information
        realized_pnl = info.get('realized_pnl', 0)
        unrealized_pnl = info.get('unrealized_pnl', 0)
        transaction_cost = info.get('transaction_cost', 0)

        # Calculate base reward components
        realized_pnl_reward = self.realized_pnl_weight * realized_pnl
        unrealized_pnl_reward = self.unrealized_pnl_weight * unrealized_pnl
        transaction_cost_penalty = self.transaction_cost_weight * transaction_cost

        # Initialize total reward
        reward = realized_pnl_reward + unrealized_pnl_reward - transaction_cost_penalty

        # Track base components
        if self.track_components:
            self.reward_components['realized_pnl'] = realized_pnl_reward
            self.reward_components['unrealized_pnl'] = unrealized_pnl_reward
            self.reward_components['transaction_cost'] = -transaction_cost_penalty

        # Apply risk adjustments if enabled
        if self.risk_adjusted:
            # Extract risk information
            is_trade = info.get('is_trade', False)
            leverage = info.get('leverage', 1.0)
            drawdown = info.get('drawdown', 0.0)
            holding_time = info.get('holding_time', 0)

            # Apply trade penalty if a new trade was made
            if is_trade:
                trade_penalty = self.trade_penalty
                reward -= trade_penalty
                if self.track_components:
                    self.reward_components['trade_penalty'] = -trade_penalty

            # Apply leverage penalty (quadratic to penalize high leverage more)
            leverage_penalty = self.leverage_penalty * (leverage ** 2)
            reward -= leverage_penalty
            if self.track_components:
                self.reward_components['leverage_penalty'] = -leverage_penalty

            # Apply drawdown penalty
            drawdown_penalty = self.drawdown_penalty * drawdown
            reward -= drawdown_penalty
            if self.track_components:
                self.reward_components['drawdown_penalty'] = -drawdown_penalty

            # Apply holding time bonus (encourages longer-term positions)
            if holding_time > 0:
                holding_bonus = self.holding_time_bonus * np.log1p(holding_time)
                reward += holding_bonus
                if self.track_components:
                    self.reward_components['holding_time_bonus'] = holding_bonus

            # Apply trend alignment bonus if market trend info is available
            if 'trend_alignment' in info:
                trend_alignment = info.get('trend_alignment', 0)
                trend_bonus = self.trend_alignment_bonus * trend_alignment
                reward += trend_bonus
                if self.track_components:
                    self.reward_components['trend_alignment_bonus'] = trend_bonus

        # Calculate Sharpe ratio component if weight > 0
        if self.sharpe_weight > 0:
            # Get the current return
            current_return = info.get('return', realized_pnl / info.get('balance', 1.0))

            # Add to returns history
            self.returns_history.append(current_return)

            # Keep only the last window_size returns
            if len(self.returns_history) > self.sharpe_window:
                self.returns_history.pop(0)

            # Calculate Sharpe ratio if we have enough history
            if len(self.returns_history) >= 2:
                returns_array = np.array(self.returns_history)
                returns_mean = np.mean(returns_array)
                returns_std = np.std(returns_array) + 1e-8  # Avoid division by zero

                # Calculate the Sharpe ratio
                sharpe = returns_mean / returns_std

                # Add Sharpe component to reward
                sharpe_reward = self.sharpe_weight * sharpe
                reward += sharpe_reward

                if self.track_components:
                    self.reward_components['sharpe_ratio'] = sharpe_reward

        # Apply volatility penalty if returns history is available
        if self.volatility_penalty > 0 and len(self.returns_history) >= 2:
            volatility = np.std(self.returns_history)
            volatility_penalty = self.volatility_penalty * volatility
            reward -= volatility_penalty
            if self.track_components:
                self.reward_components['volatility_penalty'] = -volatility_penalty

        # Apply global reward scaling
        reward *= self.reward_scale

        # Track total reward
        if self.track_components:
            self.reward_components['total'] = reward

        return reward

    def get_reward_components(self):
        """Get the tracked reward components."""
        if not self.track_components:
            raise ValueError("Reward component tracking is not enabled")
        return self.reward_components.copy()

if __name__ == "__main__":
    import argparse

    # Example usage
    parser = argparse.ArgumentParser(description='Reward functions for trading environments')
    parser.add_argument('--reward_type', type=str, default='simple', help='Type of reward function to use')
    args = parser.parse_args()

    if args.reward_type == 'simple':
        reward_fn = SimpleReward()
    elif args.reward_type == 'sharpe':
        reward_fn = SharpeReward()
    elif args.reward_type == 'risk_adjusted':
        reward_fn = RiskAdjustedReward()
    elif args.reward_type == 'tunable':
        reward_fn = TunableReward()
    elif args.reward_type == 'futures_risk_adjusted':
        reward_fn = FuturesRiskAdjustedReward()
    else:
        raise ValueError("Invalid reward type")

    # Example usage of the reward function
    state = None
    action = None
    next_state = None
    info = {
        'realized_pnl': 100,
        'unrealized_pnl': 50,
        'transaction_cost': 10,
        'is_trade': True,
        'leverage': 2.0,
        'drawdown': 0.1,
        'liquidation_risk': False,
        'funding_rate': 0.01
    }

    reward = reward_fn.calculate_reward(state, action, next_state, info)
    print("Reward:", reward)
