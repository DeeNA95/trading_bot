"""
Mock classes for the backtest module.
"""

class MockBacktestSimulator:
    """Mock backtest simulator for tests."""
    
    def __init__(self, initial_balance=10000.0, commission=0.001):
        """Initialize with balance and commission rate."""
        self.initial_balance = initial_balance
        self.commission = commission
        self.position_history = []
        
    def run_backtest(self, agent, start_idx=0, end_idx=None, window_size=10):
        """Mock method for running a backtest."""
        # Mock state for the agent to use
        class MockState:
            def __init__(self):
                self.price = 100.0
                self.features = [0.5, 0.2, 0.3, 0.1]
                
        # Call agent's choose_action at least once
        state = MockState()
        action, _, _ = agent.choose_action(state)
        
        # Record a simple trade
        self.position_history.append({
            'action': action,
            'position_size': 1000.0,
            'price': 100.0
        })
        
        # Return some mock metrics
        return {
            'total_return': 0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.1,
            'total_trades': 10
        }
    
    def reset_metrics(self):
        """Reset simulator metrics."""
        self.position_history = []
