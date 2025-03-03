"""
Backtesting simulator for the trading bot.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment.trading_env import TradingEnvironment


class BacktestSimulator:
    """Simulator for backtesting trading strategies."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        """
        Initialize the backtest simulator.
        Args:
            data: Historical price data.
            initial_balance: Starting account balance.
            commission: Commission rate for trading.
            slippage: Slippage rate for trading.
        """
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.reset_metrics()

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.balance_history: List[float] = []
        self.position_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.returns: List[float] = []

    def run_backtest(
        self,
        agent,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        window_size: int = 30,
        render: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a backtest simulation.
        Args:
            agent: The trading agent to evaluate.
            start_idx: Starting index in the data.
            end_idx: Ending index in the data (optional).
            window_size: Lookback window size.
            render: Whether to render the backtest visually.
        Returns:
            A dictionary with performance metrics.
        """
        # Initialize the trading environment
        env = TradingEnvironment(
            data=self.data,
            initial_balance=self.initial_balance,
            commission=self.commission,
            slippage=self.slippage,
            window_size=window_size,
            # render_mode="human" if render else None,
        )

        if end_idx is None:
            end_idx = len(self.data) - 1

        # Set the starting step (ensure we have enough history for the window)
        env.current_step = start_idx + window_size

        # Reset metrics and environment
        self.reset_metrics()
        observation, _ = env.reset()
        done = False

        while not done and env.current_step < end_idx:
            # Agent chooses an action
            action, _, _ = agent.choose_action(observation)
            # Execute the action in the environment
            next_observation, reward, done, _, info = env.step(action)

            # Record balance and equity (balance + current position value)
            self.balance_history.append(env.balance)
            current_close = env.data.iloc[env.current_step]["close"]
            position_value = (
                env.position_size * current_close if env.position_size != 0 else 0
            )
            self.equity_curve.append(env.balance + position_value)

            # Record position info (using timestamp if available)
            timestamp = (
                env.data.iloc[env.current_step].name
                if hasattr(env.data.iloc[env.current_step], "name")
                else env.current_step
            )
            self.position_history.append(
                {
                    "timestamp": timestamp,
                    "position_size": env.position_size,
                    "position_direction": env.position_direction,
                    "position_price": env.position_price,
                }
            )

            # Record trade details if a trade was closed (based on realized pnl)
            if info.get("realized_pnl", 0) != 0:
                self.trade_history.append(
                    {
                        "timestamp": timestamp,
                        "pnl": info["realized_pnl"],
                        "direction": "long" if env.position_direction > 0 else "short",
                        "entry_price": env.position_price,
                        "exit_price": current_close,
                        "quantity": env.position_size,
                    }
                )

            # Compute simple return and store it if possible
            if len(self.equity_curve) > 1:
                step_return = (self.equity_curve[-1] / self.equity_curve[-2]) - 1
                self.returns.append(step_return)

            observation = next_observation
            if render:
                env.render()

        return self.calculate_metrics()

    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics from backtest results.
        Returns:
            Dictionary of performance metrics.
        """
        if not self.equity_curve:
            return {}

        equity = np.array(self.equity_curve)
        returns = np.array(self.returns)
        total_return = (equity[-1] / self.initial_balance) - 1

        total_trades = len(self.trade_history)
        win_trades = [trade["pnl"] for trade in self.trade_history if trade["pnl"] > 0]
        loss_trades = [
            trade["pnl"] for trade in self.trade_history if trade["pnl"] <= 0
        ]
        win_rate = len(win_trades) / total_trades if total_trades > 0 else 0

        sharpe_ratio = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if np.std(returns) > 0
            else 0
        )
        sortino_returns = returns[returns < 0]
        sortino_ratio = (
            np.mean(returns) / np.std(sortino_returns) * np.sqrt(252)
            if (len(sortino_returns) > 0 and np.std(sortino_returns) > 0)
            else 0
        )

        # Calculate maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()

        avg_profit = np.mean(win_trades) if win_trades else 0
        avg_loss = np.mean(loss_trades) if loss_trades else 0
        risk_reward = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        total_profit = sum(win_trades) if win_trades else 0
        total_loss = abs(sum(loss_trades)) if loss_trades else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        return {
            "total_return": total_return,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "risk_reward": risk_reward,
            "profit_factor": profit_factor,
            "final_balance": equity[-1],
        }

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot backtest results.
        Args:
            save_path: Optional path to save the plot.
        """
        if not self.equity_curve:
            print("No backtest results to plot.")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(12, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1]},
        )
        ax1.plot(self.equity_curve, label="Equity Curve")
        ax1.set_title("Backtest Equity Curve")
        ax1.set_ylabel("Equity")
        ax1.legend()

        # Drawdown plot
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        ax2.fill_between(range(len(drawdown)), 0, drawdown, color="red", alpha=0.3)
        ax2.set_ylabel("Drawdown (%)")

        # Plot trade markers on equity curve
        for trade in self.trade_history:
            idx = (
                trade["timestamp"]
                if isinstance(trade["timestamp"], int)
                else self.data.index.get_loc(trade["timestamp"])
            )
            color = "green" if trade["pnl"] > 0 else "red"
            marker = "^" if trade["direction"] == "long" else "v"
            ax1.scatter(idx, self.equity_curve[idx], color=color, marker=marker, s=50)

        # Position history (if available)
        positions = np.zeros(len(self.equity_curve))
        for i, pos in enumerate(self.position_history):
            if i < len(positions):
                positions[i] = pos.get("position_size", 0) * pos.get(
                    "position_direction", 0
                )
        ax3.plot(positions, color="blue", label="Position")
        ax3.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax3.set_ylabel("Position Size")
        ax3.set_xlabel("Time")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a text report of backtest results.
        Args:
            metrics: Performance metrics dictionary.
        Returns:
            A formatted text report.
        """
        report = "=== BACKTEST REPORT ===\n\n"
        report += f"Initial Balance: ${self.initial_balance:.2f}\n"
        report += f"Final Balance: ${metrics.get('final_balance', self.initial_balance):.2f}\n"
        report += f"Total Return: {metrics.get('total_return', 0)*100:.2f}%\n\n"
        report += f"Total Trades: {metrics.get('total_trades', 0)}\n"
        report += f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}%\n"
        report += f"Avg Profit: ${metrics.get('avg_profit', 0):.2f}\n"
        report += f"Avg Loss: ${abs(metrics.get('avg_loss', 0)):.2f}\n"
        report += f"Risk/Reward Ratio: {metrics.get('risk_reward', 0):.2f}\n"
        report += f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n\n"
        report += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
        report += f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}\n"
        report += f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%\n"
        return report
