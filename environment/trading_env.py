"""
OpenAI Gym compatible environment for cryptocurrency trading.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from environment.reward import RewardFunction, RiskAdjustedReward
import matplotlib.pyplot as plt

class TradingEnvironment(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 data: pd.DataFrame,
                 reward_function: RewardFunction = None,
                 initial_balance: float = 10000.0,
                 window_size: int = 30,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 max_leverage: float = 5.0,
                 liquidation_threshold: float = 0.8,  
                 render_mode: Optional[str] = None):
        """Trading environment for reinforcement learning."""
        super().__init__()
        
        # Validate data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")
            
        # Check for recommended columns but don't require them
        recommended_columns = ['rsi', 'macd', 'macd_signal', 'atr']
        missing_recommended = [col for col in recommended_columns if col not in data.columns]
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
        self.render_mode = render_mode
        
        # Action space: [direction, size, take_profit, stop_loss]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.5, 0.5]),
            high=np.array([1.0, 1.0, 5.0, 5.0]),
            dtype=np.float32
        )
        
        # Initialize state variables to set up observation feature list
        self.balance = initial_balance
        self.max_balance = initial_balance
        self.position_size = 0
        self.position_direction = 0
        self.position_price = 0
        
        # Observation space
        self.feature_list = self._get_observation_features()
        self.feature_count = len(self.feature_list)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.feature_count,),
            dtype=np.float32
        )
        
        # For rendering
        self.render_fig = None
        self.render_ax = None
        
        # Reset environment
        self.reset()
        
    def _get_observation_features(self) -> List[str]:
        """Get feature list for the current observation."""
        base_features = [
            'open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm',
        ]
        
        # Add features if they exist in the data
        optional_features = [
            'rsi', 'macd', 'macd_signal', 'atr', 'bb_upper', 'bb_lower', 'stoch_k', 'stoch_d', 'adx', 
            'plus_di', 'minus_di', 'atr_pct'
        ]
        
        features = base_features.copy()
        if hasattr(self, 'data'):
            for feature in optional_features:
                if feature in self.data.columns:
                    features.append(feature)
        
        # Add account state features
        features.extend(['balance_ratio', 'position_value_ratio', 'unrealized_pnl_ratio'])
        
        return features
        
    def _get_current_price(self) -> float:
        """Get current close price."""
        return self.data.iloc[self.current_step]['close']
        
    def _get_observation(self) -> np.ndarray:
        """Get the current state observation."""
        # Market features - handled separately to deal with missing values
        market_features = []
        
        # Calculate account features
        balance_ratio = self.balance / self.initial_balance
        current_price = self._get_current_price()
        position_value = abs(self.position_size) * current_price
        position_value_ratio = position_value / self.balance if self.balance > 0 else 0
        unrealized_pnl = self._calculate_unrealized_pnl()
        unrealized_pnl_ratio = unrealized_pnl / self.balance if self.balance > 0 else 0
        
        # Combine features
        features = []
        for feature in self.feature_list:
            if feature in self.data.columns:
                value = self.data.iloc[self.current_step][feature]
                # Handle NaN or inf values
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                features.append(value)
            elif feature == 'balance_ratio':
                features.append(balance_ratio)
            elif feature == 'position_value_ratio':
                features.append(position_value_ratio)
            elif feature == 'unrealized_pnl_ratio':
                features.append(unrealized_pnl_ratio)
            else:
                # For missing features, add a placeholder
                features.append(0.0)
                
        return np.array(features, dtype=np.float32)
        
    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL for current position."""
        if self.position_size == 0:
            return 0.0
            
        current_price = self._get_current_price()
        position_value = abs(self.position_size)
        
        if self.position_direction > 0:  # Long
            return position_value * (current_price / self.position_price - 1)
        else:  # Short
            return position_value * (1 - current_price / self.position_price)
            
    def _apply_take_profit_stop_loss(self) -> Tuple[bool, float]:
        """Check and apply TP/SL if triggered."""
        if self.position_size == 0:
            return False, 0.0
            
        current_price = self._get_current_price()
        price_change = (current_price - self.position_price) / self.position_price
        
        # Get ATR for scaling if available
        atr_scale = self.data.iloc[self.current_step]['atr'] if 'atr' in self.data.columns else self.position_price * 0.01
        
        # Scale TP/SL based on ATR if not zero
        tp_threshold = self.take_profit * atr_scale / self.position_price if atr_scale > 0 else self.take_profit
        sl_threshold = self.stop_loss * atr_scale / self.position_price if atr_scale > 0 else self.stop_loss
        
        # Check take profit
        if (self.position_direction > 0 and price_change >= tp_threshold) or \
           (self.position_direction < 0 and -price_change >= tp_threshold):
            return True, self.position_direction * tp_threshold
            
        # Check stop loss
        if (self.position_direction > 0 and price_change <= -sl_threshold) or \
           (self.position_direction < 0 and -price_change <= -sl_threshold):
            return True, -self.position_direction * sl_threshold
            
        return False, 0.0
        
    def _calculate_liquidation_price(self, position_direction, position_price, leverage):
        """
        Calculate the liquidation price based on position direction, entry price, and leverage.
        
        For long positions: liquidation_price = entry_price * (1 - (1 / leverage) * (1 - liquidation_threshold))
        For short positions: liquidation_price = entry_price * (1 + (1 / leverage) * (1 - liquidation_threshold))
        
        Args:
            position_direction: Direction of position (1 for long, -1 for short)
            position_price: Entry price of the position
            leverage: Leverage used for the position
            
        Returns:
            float: The liquidation price
        """
        if position_direction == 0 or leverage <= 0:
            return None
            
        # Calculate the maintenance margin requirement
        maintenance_margin = (1 / leverage) * (1 - self.liquidation_threshold)
        
        if position_direction > 0:  # Long position
            liquidation_price = position_price * (1 - maintenance_margin)
        else:  # Short position
            liquidation_price = position_price * (1 + maintenance_margin)
            
        return liquidation_price
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Get current market data
        current_price = self.data.iloc[self.current_step]['close']
        prev_price = self.data.iloc[self.current_step - 1]['close']
        
        # Store previous state for reward calculation
        prev_balance = self.balance
        prev_position_size = self.position_size
        prev_position_direction = self.position_direction
        
        # Decode the action
        direction = action[0]  # -1 (short), 0 (hold), 1 (long)
        size = action[1]       # 0.0 to 1.0 (percentage of balance to use)
        take_profit = action[2]  # Take profit multiplier (relative to ATR)
        stop_loss = action[3]    # Stop loss multiplier (relative to ATR)
        
        # Calculate position size based on leverage
        leverage = size * self.max_leverage if direction != 0 else 0
        position_value = self.balance * size
        
        # Check if we need to close existing position
        transaction_cost = 0
        unrealized_pnl = 0
        holding_time = 0
        
        # Calculate market trend for trend alignment
        trend = np.sign(current_price - prev_price)
        trend_alignment = trend * self.position_direction if self.position_direction != 0 else 0
        
        # Calculate unrealized PnL for existing position
        if self.position_size > 0:
            holding_time = self.current_step - self.position_step
            price_diff = current_price - self.position_price if self.position_direction > 0 else self.position_price - current_price
            unrealized_pnl = price_diff * self.position_size
            
        # Check for liquidation before allowing new trades
        liquidation_price = None
        liquidation_risk = False
        
        if self.position_size > 0:
            liquidation_price = self._calculate_liquidation_price(
                self.position_direction, 
                self.position_price, 
                self.position_size * self.max_leverage / self.position_value if self.position_value > 0 else 0
            )
            
            # Check if current price has crossed the liquidation price
            if liquidation_price is not None:
                if (self.position_direction > 0 and current_price <= liquidation_price) or \
                   (self.position_direction < 0 and current_price >= liquidation_price):
                    # Position is liquidated
                    self.balance = self.balance - self.position_value  # Lose the margin
                    self.position_size = 0
                    self.position_direction = 0
                    self.position_price = 0
                    self.position_value = 0
                    self.position_step = 0
                    transaction_cost = 0  # No transaction cost on liquidation
                    unrealized_pnl = 0
                    
                # Check if price is getting close to liquidation price (within 10%)
                elif liquidation_price is not None:
                    if self.position_direction > 0:
                        distance_to_liquidation = (current_price - liquidation_price) / current_price
                    else:
                        distance_to_liquidation = (liquidation_price - current_price) / current_price
                        
                    liquidation_risk = distance_to_liquidation < 0.1  # Within 10% of liquidation
        
        # Close existing position if direction changes or size becomes zero
        if self.position_size > 0 and (direction * self.position_direction <= 0 or size == 0):
            # Calculate PnL
            price_diff = current_price - self.position_price if self.position_direction > 0 else self.position_price - current_price
            position_pnl = price_diff * self.position_size
            
            # Apply slippage to closing price
            slippage_cost = current_price * self.slippage * self.position_size
            
            # Apply commission
            commission_cost = current_price * self.commission * self.position_size
            
            # Update balance
            self.balance += position_pnl - slippage_cost - commission_cost
            transaction_cost = slippage_cost + commission_cost
            
            # Reset position
            self.position_size = 0
            self.position_direction = 0
            self.position_price = 0
            self.position_value = 0
            self.position_step = 0
            unrealized_pnl = 0
        
        # Open new position if direction is not zero and we don't have a position
        if direction != 0 and self.position_size == 0:
            # Calculate position size with leverage
            self.position_value = position_value
            self.position_size = position_value * leverage / current_price if current_price > 0 else 0
            self.position_direction = 1 if direction > 0 else -1
            
            # Apply slippage to opening price
            slippage_cost = current_price * self.slippage * self.position_size
            
            # Apply commission
            commission_cost = current_price * self.commission * self.position_size
            
            # Update position price with slippage
            self.position_price = current_price * (1 + self.position_direction * self.slippage)
            
            # Record transaction cost
            transaction_cost = slippage_cost + commission_cost
            
            # Record step when position was opened
            self.position_step = self.current_step
            
            # Calculate liquidation price for the new position
            liquidation_price = self._calculate_liquidation_price(
                self.position_direction, 
                self.position_price, 
                leverage
            )
        
        # Get current observation
        observation = self._get_observation()
        
        # Calculate funding rate if available in the data
        funding_rate = 0.0
        if 'fundingRate' in self.data.columns:
            funding_rate = self.data.iloc[self.current_step]['fundingRate']
            
            # Apply funding rate to position if we have one
            if self.position_size > 0:
                # For long positions: negative funding rate means we receive payment
                # For short positions: positive funding rate means we receive payment
                funding_payment = -funding_rate * self.position_value * self.position_direction
                self.balance += funding_payment
        
        info = {
            'realized_pnl': self.balance - prev_balance,
            'unrealized_pnl': unrealized_pnl,
            'transaction_cost': transaction_cost,
            'is_trade': direction != 0 and prev_position_size == 0,
            'leverage': leverage,
            'drawdown': (self.max_balance - self.balance) / self.max_balance if self.max_balance > 0 else 0,
            'holding_time': holding_time,
            'trend_alignment': trend_alignment,
            'return': (self.balance - prev_balance) / prev_balance if prev_balance > 0 else 0,
            # Futures-specific information
            'funding_rate': funding_rate,
            'liquidation_price': liquidation_price,
            'current_price': current_price,
            'position_direction': self.position_direction,
            'liquidation_risk': liquidation_risk
        }
        reward = self.reward_function.calculate_reward(None, action, observation, info)
        
        # Update maximum balance (including unrealized PnL)
        self.max_balance = max(self.max_balance, self.balance + unrealized_pnl)
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1 or self.balance <= 0
        
        return observation, reward, done, False, info
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.position_size = 0
        self.position_direction = 0
        self.position_price = 0
        self.take_profit = 0
        self.stop_loss = 0
        
        # Start after the window_size for sufficient history
        self.current_step = self.window_size
        if self.current_step >= len(self.data):
            self.current_step = 0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
        
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return
            
        if self.render_fig is None:
            self.render_fig, self.render_ax = plt.subplots(figsize=(10, 6))
            plt.ion()
            
        self.render_ax.clear()
        
        # Plot price
        start_idx = max(0, self.current_step - 100)
        end_idx = self.current_step
        
        price_data = self.data.iloc[start_idx:end_idx+1]
        self.render_ax.plot(price_data.index, price_data['close'], label='Price')
        
        # Highlight current position
        if self.position_size > 0:
            color = 'green' if self.position_direction > 0 else 'red'
            self.render_ax.axhline(y=self.position_price, color=color, linestyle='--', 
                                   label=f"{'Long' if self.position_direction > 0 else 'Short'} @ {self.position_price:.2f}")
            
            # Draw take profit and stop loss levels
            if self.take_profit > 0:
                tp_price = self.position_price * (1 + self.take_profit * self.position_direction)
                self.render_ax.axhline(y=tp_price, color='blue', linestyle=':', label=f"TP @ {tp_price:.2f}")
                
            if self.stop_loss > 0:
                sl_price = self.position_price * (1 - self.stop_loss * self.position_direction)
                self.render_ax.axhline(y=sl_price, color='purple', linestyle=':', label=f"SL @ {sl_price:.2f}")
        
        # Add account info
        title = f"Balance: ${self.balance:.2f} | "
        title += f"Position: {'Long' if self.position_direction > 0 else 'Short' if self.position_direction < 0 else 'None'} | "
        title += f"P&L: ${self._calculate_unrealized_pnl():.2f}"
        self.render_ax.set_title(title)
        
        self.render_ax.set_xlabel('Time')
        self.render_ax.set_ylabel('Price')
        self.render_ax.legend()
        
        plt.draw()
        plt.pause(0.01)
        
        if self.render_mode == 'rgb_array':
            # Convert the figure to an RGB array
            self.render_fig.canvas.draw()
            img = np.frombuffer(self.render_fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.render_fig.canvas.get_width_height()[::-1] + (3,))
            return img
