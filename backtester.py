#!/usr/bin/env python3
"""
Backtesting script for evaluating saved models.
This script:
  - Loads testing data from a parquet file.
  - Instantiates a PPOAgent with the correct state and action dimensions.
  - Loads the saved models (actor, critic, and target networks) from disk.
  - Runs a backtest on the testing data.
  - Prints a performance report.
"""

import os
import torch
import pandas as pd

from agent.rl_agent import PPOAgent
from backtest.simulator import BacktestSimulator
from environment.trading_env import TradingEnvironment
from environment.reward import (
    FuturesRiskAdjustedReward,
)  # or your chosen reward function

# Path to testing data
TEST_DATA_PATH = "data/test_data.parquet"

# Directory where saved models are stored
SAVE_DIR = "saved_models"

# Load testing data
test_data = pd.read_parquet(TEST_DATA_PATH)

# Create a dummy environment to determine observation dimensions
dummy_env = TradingEnvironment(
    data=test_data,
    initial_balance=10000.0,
    window_size=30,
    commission=0.001,
    slippage=0.0005,
    # render_mode=None,
)
state_dim = dummy_env.observation_space.shape[0]
action_dim = 4  # [direction, size, take_profit, stop_loss]

print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

# Initialize the reward function and PPO agent
reward_fn = FuturesRiskAdjustedReward()
agent = PPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dims=(256, 128),
    lr_actor=3e-4,
    lr_critic=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    policy_clip=0.2,
    batch_size=64,
    n_epochs=10,
    entropy_coef=0.01,
    value_coef=0.5,
    target_update_freq=10,
    target_update_tau=0.005,
    lr_scheduler_type="step",
    lr_scheduler_step_size=100,
    lr_scheduler_gamma=0.9,
    grad_clip_value=0.5,
    memory_capacity=10000,
    prioritized_replay_alpha=0.6,
    prioritized_replay_beta=0.4,
    prioritized_replay_beta_increment=0.001,
)

# Load saved models into the agent's networks
actor_path = os.path.join(SAVE_DIR, "actor_model.pt")
critic_path = os.path.join(SAVE_DIR, "critic_model.pt")
target_actor_path = os.path.join(SAVE_DIR, "target_actor_model.pt")
target_critic_path = os.path.join(SAVE_DIR, "target_critic_model.pt")

agent.actor.load_state_dict(torch.load(actor_path, map_location=agent.device))
agent.critic.load_state_dict(torch.load(critic_path, map_location=agent.device))
agent.target_actor.load_state_dict(
    torch.load(target_actor_path, map_location=agent.device)
)
agent.target_critic.load_state_dict(
    torch.load(target_critic_path, map_location=agent.device)
)
print("Saved models loaded successfully.")

# Initialize backtest simulator with testing data
backtester = BacktestSimulator(
    data=test_data,
    initial_balance=10000.0,
    commission=0.001,
    slippage=0.0005,
)

# Run backtest on testing data
metrics = backtester.run_backtest(
    agent=agent, start_idx=0, end_idx=len(test_data) - 1, window_size=30, render=False
)

# Generate and print performance report
report = backtester.generate_report(metrics)
print(report)
