"""
Mock classes for the agent module.
"""

import numpy as np


class MockNetwork:
    """Mock neural network for tests."""

    def __init__(self, input_dim=10, output_dim=4):
        """Initialize with input and output dimensions."""
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, state):
        """Forward pass through network."""
        return np.random.random(self.output_dim)

    def __call__(self, state):
        """Call method to use instance as function."""
        return self.forward(state)

    def parameters(self):
        """Return dummy parameters."""
        return [np.random.random((10, 10)), np.random.random(10)]

    def to(self, device):
        """Device placement mock."""
        return self


class MockMemory:
    """Mock memory for storing agent experiences."""

    def __init__(self, batch_size=32):
        """Initialize memory."""
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store_memory(self, state, action, probs, vals, reward, done):
        """Store transition in memory."""
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """Clear stored memory."""
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        """Generate batches from memory."""
        return [0], [0], [0], [0], [0]


class MockAgent:
    """Mock reinforcement learning agent for tests."""

    def __init__(self, input_dim=10, output_dim=4):
        """Initialize agent with dimensions."""
        self.actor = MockNetwork(input_dim, output_dim)
        self.critic = MockNetwork(input_dim, 1)
        self.memory = MockMemory()

    def choose_action(self, observation):
        """Choose an action based on observation."""
        action = np.random.uniform(-1, 1, 4)
        log_probs = np.array([0.1])
        value = np.array([0.5])
        return action, log_probs, value

    def learn(self):
        """Update agent parameters."""
        self.memory.clear_memory()
        return {"actor_loss": 0.1, "critic_loss": 0.2, "total_loss": 0.3}

    def save_models(self, path):
        """Save models to path."""
        pass

    def load_models(self, path):
        """Load models from path."""
        pass
