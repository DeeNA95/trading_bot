#!/usr/bin/env python3
"""
Training and testing script for the PPO agent using the preprocessed training and testing data.
This script:
  - Loads training and testing data (assumed to be stored as parquet files).
  - Initializes the PPOAgent with the appropriate state and action dimensions.
  - Trains the agent on the training data.
  - Saves each model (actor, critic, target_actor, target_critic) under its own filename.
  - Runs backtesting on the testing data using the trained agent.
"""

import os
import numpy as np
import pandas as pd
import torch

from agent.rl_agent import PPOAgent
from backtest.simulator import BacktestSimulator
from environment.trading_env import TradingEnvironment
from environment.reward import (
    FuturesRiskAdjustedReward,
)  # or another reward function of your choice

# Paths to your preprocessed data
TRAIN_DATA_PATH = "data/train_data.parquet"
TEST_DATA_PATH = "data/test_data.parquet"

# Load training and testing data
train_data = pd.read_parquet(TRAIN_DATA_PATH)
test_data = pd.read_parquet(TEST_DATA_PATH)

# Determine state dimension from training data
# (Assuming that the trading environment uses all columns from its observation feature list)
# We create a dummy environment to get the feature count.
dummy_env = TradingEnvironment(
    data=train_data,
    initial_balance=10000.0,
    window_size=30,
    commission=0.001,
    slippage=0.0005,
    # render_mode=None,
)
state_dim = dummy_env.observation_space.shape[0]
action_dim = 4  # [direction, size, take_profit, stop_loss]

print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

# Initialize reward function and PPO agent
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

# Training loop on training data
# For simplicity, we simulate episodes by iterating over the training data with a sliding window.
num_episodes = 50  # Adjust number of episodes as needed

for episode in range(num_episodes):
    # Reset the environment (which resets internal balance, position, and current_step)
    state, _ = dummy_env.reset()
    done = False
    episode_reward = 0.0

    while not done and dummy_env.current_step < len(train_data) - 1:
        action, log_prob, value = agent.choose_action(state)
        next_state, reward, done, _, info = dummy_env.step(action)
        # Store transition in agent's memory (if you are using a replay buffer approach)
        agent.store_transition(state, action, log_prob, value, reward, done)
        state = next_state
        episode_reward += reward

    # After each episode, update the agent
    # Here we assume next_value=0 for terminal state; adjust as needed.
    agent.learn(next_value=0, episode_reward=episode_reward)
    print(
        f"Episode {episode+1}/{num_episodes} finished with reward: {episode_reward:.2f}"
    )

# Save each model under its own name
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
torch.save(agent.actor.state_dict(), os.path.join(save_dir, "actor_model.pt"))
torch.save(agent.critic.state_dict(), os.path.join(save_dir, "critic_model.pt"))
torch.save(
    agent.target_actor.state_dict(), os.path.join(save_dir, "target_actor_model.pt")
)
torch.save(
    agent.target_critic.state_dict(), os.path.join(save_dir, "target_critic_model.pt")
)
print("Models saved individually under 'saved_models/' directory.")

# Backtesting on testing data
backtester = BacktestSimulator(
    data=test_data,
    initial_balance=10000.0,
    commission=0.001,
    slippage=0.0005,
)
# Run backtest with the trained agent on the testing dataset.
metrics = backtester.run_backtest(
    agent=agent, start_idx=0, end_idx=len(test_data) - 1, window_size=30, render=False
)
report = backtester.generate_report(metrics)
print(report)
