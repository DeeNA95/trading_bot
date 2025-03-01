"""
Backtesting simulator for the trading bot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque

from environment.trading_env import TradingEnvironment


class BacktestSimulator:
    """Simulator for backtesting trading strategies."""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        """Initialize the backtest simulator.
        
        Args:
            data: Historical price data
            initial_balance: Initial account balance
            commission: Commission rate for trading
            slippage: Slippage rate for trading
        """
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        
        # Performance metrics
        self.reset_metrics()
        
    def reset_metrics(self):
        """Reset performance metrics."""
        self.balance_history = []
        self.position_history = []
        self.trade_history = []
        self.equity_curve = []
        self.returns = []
        
    def run_backtest(self, 
                     agent,
                     start_idx: int = 0,
                     end_idx: Optional[int] = None,
                     window_size: int = 30,
                     render: bool = False) -> Dict[str, Any]:
        """Run a backtest simulation.
        
        Args:
            agent: The trading agent to evaluate
            start_idx: Starting index in the data
            end_idx: Ending index in the data (optional)
            window_size: Lookback window size
            render: Whether to render the backtest visually
            
        Returns:
            Performance metrics
        """
        # Create trading environment
        env = TradingEnvironment(
            data=self.data,
            initial_balance=self.initial_balance,
            commission=self.commission,
            slippage=self.slippage,
            window_size=window_size,
            render_mode='human' if render else None
        )
        
        # Set up backtest
        if end_idx is None:
            end_idx = len(self.data) - 1
            
        env.current_step = start_idx + window_size
        
        # Reset metrics
        self.reset_metrics()
        
        # Initial observation
        observation, _ = env.reset()
        done = False
        
        # Run simulation
        while not done and env.current_step < end_idx:
            # Get action from agent
            action, _, _ = agent.choose_action(observation)
            
            # Take action in environment
            next_observation, reward, done, _, info = env.step(action)
            
            # Record metrics
            self.balance_history.append(env.balance)
            position_value = 0
            if env.position_size > 0:
                position_value = env.position_size * env.data.iloc[env.current_step]['close']
            self.equity_curve.append(env.balance + position_value)
            
            self.position_history.append({
                'timestamp': env.data.iloc[env.current_step].name if hasattr(env.data.iloc[env.current_step], 'name') else env.current_step,
                'position_size': env.position_size,
                'position_direction': env.position_direction,
                'position_price': env.position_price
            })
            
            # Record trade if completed
            if info.get('realized_pnl', 0) != 0:
                self.trade_history.append({
                    'timestamp': env.data.iloc[env.current_step].name if hasattr(env.data.iloc[env.current_step], 'name') else env.current_step,
                    'pnl': info['realized_pnl'],
                    'direction': 'long' if env.position_direction > 0 else 'short',
                    'entry_price': env.position_price,
                    'exit_price': env.data.iloc[env.current_step]['close'],
                    'quantity': env.position_size
                })
            
            # Calculate returns if we have enough history
            if len(self.equity_curve) > 1:
                daily_return = (self.equity_curve[-1] / self.equity_curve[-2]) - 1
                self.returns.append(daily_return)
            
            # Update observation
            observation = next_observation
            
            # Render if requested
            if render:
                env.render()
        
        # Calculate performance metrics
        metrics = self.calculate_metrics()
        
        return metrics
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from backtest results.
        
        Returns:
            Dictionary of performance metrics
        """
        # Convert to numpy arrays for calculations
        equity = np.array(self.equity_curve)
        returns = np.array(self.returns)
        
        # Basic metrics
        total_return = (equity[-1] / self.initial_balance) - 1
        win_trades = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
        loss_trades = [t['pnl'] for t in self.trade_history if t['pnl'] <= 0]
        
        # Win rate
        total_trades = len(self.trade_history)
        win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
        
        # Risk metrics
        if len(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            sortino_ratio = np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252) if len(returns[returns < 0]) > 0 and np.std(returns[returns < 0]) > 0 else 0
            
            # Max drawdown
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_drawdown = drawdown.max()
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
        
        # Average profit/loss
        avg_profit = np.mean(win_trades) if win_trades else 0
        avg_loss = np.mean(loss_trades) if loss_trades else 0
        
        # Risk-to-reward ratio
        risk_reward = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        
        # Profit factor
        total_profit = sum(win_trades) if win_trades else 0
        total_loss = abs(sum(loss_trades)) if loss_trades else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'risk_reward': risk_reward,
            'profit_factor': profit_factor,
            'final_balance': self.equity_curve[-1] if self.equity_curve else self.initial_balance
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot backtest results.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.equity_curve:
            print("No backtest results to plot")
            return
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot equity curve
        ax1.plot(self.equity_curve, label='Account Equity')
        ax1.set_title('Backtest Results')
        ax1.set_ylabel('Equity')
        ax1.legend()
        
        # Plot drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100  # Convert to percentage
        ax2.fill_between(range(len(drawdown)), 0, drawdown, alpha=0.3, color='red')
        ax2.set_ylabel('Drawdown %')
        
        # Plot trades
        for trade in self.trade_history:
            idx = trade['timestamp'] if isinstance(trade['timestamp'], int) else self.data.index.get_loc(trade['timestamp'])
            color = 'green' if trade['pnl'] > 0 else 'red'
            marker = '^' if trade['direction'] == 'long' else 'v'
            ax1.scatter(idx, self.equity_curve[idx], color=color, marker=marker, s=50)
        
        # Plot position
        positions = np.zeros(len(self.equity_curve))
        for i, pos in enumerate(self.position_history):
            if i < len(positions):
                positions[i] = pos['position_size'] * pos['position_direction']
        
        ax3.plot(positions, color='blue', label='Position Size')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_ylabel('Position')
        ax3.set_xlabel('Time')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
        
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a text report of backtest results.
        
        Args:
            metrics: Performance metrics from calculate_metrics()
            
        Returns:
            Formatted text report
        """
        report = "=== BACKTEST RESULTS ===\n\n"
        
        report += f"Initial Balance: ${self.initial_balance:.2f}\n"
        report += f"Final Balance: ${metrics['final_balance']:.2f}\n"
        report += f"Total Return: {metrics['total_return']*100:.2f}%\n\n"
        
        report += f"Total Trades: {metrics['total_trades']}\n"
        report += f"Win Rate: {metrics['win_rate']*100:.2f}%\n"
        report += f"Average Profit: ${metrics['avg_profit']:.2f}\n"
        report += f"Average Loss: ${abs(metrics['avg_loss']):.2f}\n"
        report += f"Risk/Reward Ratio: {metrics['risk_reward']:.2f}\n"
        report += f"Profit Factor: {metrics['profit_factor']:.2f}\n\n"
        
        report += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        report += f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n"
        report += f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%\n"
        
        return report
