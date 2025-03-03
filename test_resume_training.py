"""
Test script to verify resuming training from a checkpoint.
"""

import argparse
import os

import numpy as np
import torch

from agent.rl_agent import PPOAgent
from environment.trading_env import TradingEnvironment
from train import create_reward_function, get_best_device, prepare_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test resuming training from a checkpoint"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/checkpoint_2_checkpoint.pt",
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/BTC_USD_hourly_with_metrics.csv",
        help="Path to the data file",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=2,
        help="Number of additional episodes to train for",
    )

    return parser.parse_args()


def main():
    """Main function to test resuming training."""
    args = parse_args()

    # Select the best available device
    device = get_best_device()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint file {args.checkpoint_path} does not exist.")
        return

    # Load checkpoint data
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint_data = torch.load(args.checkpoint_path, weights_only=False)

    # Extract hyperparameters
    hyperparameters = checkpoint_data.get("hyperparameters", {})
    episode_count = checkpoint_data.get("episode_count", 0)

    print(f"Loaded checkpoint from episode {episode_count}")
    print(f"Hyperparameters: {hyperparameters}")

    # Prepare data
    data = prepare_data(args.data_path)

    # Create reward function
    reward_type = hyperparameters.get("reward_type", "risk_adjusted")
    reward_function = create_reward_function(reward_type)

    # Create environment
    env = TradingEnvironment(
        data=data,
        reward_function=reward_function,
        initial_balance=hyperparameters.get("initial_balance", 100000),
        commission=hyperparameters.get("commission", 0.01),
        max_leverage=hyperparameters.get("max_leverage", 2.0),
    )

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create agent with hyperparameters from checkpoint
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hyperparameters.get("hidden_dims", (256, 128)),
        lr_actor=hyperparameters.get("lr_actor", 3e-4),
        lr_critic=hyperparameters.get("lr_critic", 3e-4),
        gamma=hyperparameters.get("gamma", 0.99),
        gae_lambda=hyperparameters.get("gae_lambda", 0.95),
        policy_clip=hyperparameters.get("policy_clip", 0.2),
        batch_size=hyperparameters.get("batch_size", 64),
        n_epochs=hyperparameters.get("n_epochs", 10),
        entropy_coef=hyperparameters.get("entropy_coef", 0.01),
        device=device,
    )

    # Load the model
    agent.load_models(
        args.checkpoint_path.replace("_checkpoint.pt", ""), load_optimizer=True
    )

    # Train for additional episodes
    print(f"\nResuming training for {args.n_episodes} more episodes...")

    for episode in range(args.n_episodes):
        observation, _ = env.reset()

        episode_reward = 0
        done = False

        while not done:
            # Choose action
            action, log_prob, value = agent.choose_action(observation)

            # Take action
            next_observation, reward, done, truncated, info = env.step(action)

            # Store transition
            agent.store_transition(observation, action, log_prob, value, reward, done)

            # Update observation
            observation = next_observation

            # Update counters
            episode_reward += reward

            if done or truncated:
                break

        # Learn at the end of the episode
        if len(agent.memory.states) > 0:
            _, _, next_value = agent.choose_action(next_observation)
            agent.learn(next_value[0])

        # Print episode summary
        current_episode = episode_count + episode + 1
        print(
            f"Episode {current_episode}, Reward: {episode_reward:.2f}, "
            f"Final Balance: ${env.balance:.2f}"
        )

    # Save final model
    save_path = f"models/resumed_training_checkpoint.pt"
    agent.save_models(
        save_path.replace("_checkpoint.pt", ""), save_optimizer=True, save_memory=True
    )
    print(f"\nTraining completed. Model saved to {save_path}")


if __name__ == "__main__":
    main()
