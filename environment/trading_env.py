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
                 render_mode: Optional[str] = None):
        """Trading environment for reinforcement learning."""
        super().__init__()
        
        # Validate data
        required_columns = ['open', 'high', 'low', 'close', 'volume', 
                            'rsi', 'macd', 'macd_signal', 'atr']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")
            
        self.data = data
        self.reward_function = reward_function or RiskAdjustedReward()
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.commission = commission
        self.slippage = slippage
        self.max_leverage = max_leverage
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
            'rsi', 'macd', 'macd_signal'
        ]
        
        # Add features if they exist in the data
        optional_features = [
            'bb_upper', 'bb_lower', 'stoch_k', 'stoch_d', 'adx', 
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
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Parse action
        direction = 1 if action[0] > 0.33 else (-1 if action[0] < -0.33 else 0)
        size = action[1] * self.max_leverage
        self.take_profit = action[2]
        self.stop_loss = action[3]
        
        # Store the previous state
        prev_balance = self.balance
        transaction_cost = 0.0
        
        # Close existing position if direction changed
        if self.position_direction != direction and self.position_size > 0:
            unrealized_pnl = self._calculate_unrealized_pnl()
            self.balance += unrealized_pnl
            
            # Apply transaction costs for closing
            transaction_cost += abs(self.position_size) * self._get_current_price() * self.commission
            self.balance -= transaction_cost
            
            self.position_size = 0
            self.position_direction = 0
            self.position_price = 0
        
        # Open new position
        if direction != 0 and self.position_size == 0:
            current_price = self._get_current_price()
            self.position_direction = direction
            
            # Calculate position size based on available balance
            max_position_value = self.balance * size
            self.position_size = max_position_value / current_price
            self.position_price = current_price
            
            # Apply transaction costs for opening
            new_transaction_cost = max_position_value * self.commission
            transaction_cost += new_transaction_cost
            self.balance -= new_transaction_cost
        
        # Check take profit/stop loss
        triggered, pnl_multiplier = self._apply_take_profit_stop_loss()
        if triggered:
            # Calculate PnL based on position value and price change percentage
            position_value = abs(self.position_size) * self._get_current_price()
            realized_pnl = position_value * pnl_multiplier
            self.balance += realized_pnl
            
            # Apply transaction costs for closing on TP/SL
            tp_sl_transaction_cost = position_value * self.commission
            transaction_cost += tp_sl_transaction_cost
            self.balance -= tp_sl_transaction_cost
            
            self.position_size = 0
            self.position_direction = 0
            self.position_price = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        observation = self._get_observation()
        
        unrealized_pnl = self._calculate_unrealized_pnl()
        
        # Apply slippage to unrealized PnL if we have a position
        if self.position_size > 0:
            slippage_cost = abs(self.position_size) * self._get_current_price() * self.slippage
            unrealized_pnl -= slippage_cost
        
        info = {
            'realized_pnl': self.balance - prev_balance,
            'unrealized_pnl': unrealized_pnl,
            'transaction_cost': transaction_cost,
            'is_trade': direction != 0 and self.position_size == 0,
            'leverage': size if direction != 0 else 0,
            'drawdown': (self.max_balance - self.balance) / self.max_balance if self.max_balance > 0 else 0
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
