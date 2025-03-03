"""
Training script for the reinforcement learning trading bot.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import gymnasium as gym
import multiprocessing
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import datetime
import subprocess

from environment.trading_env import TradingEnvironment
from environment.reward import SimpleReward, SharpeReward, RiskAdjustedReward, TunableReward, FuturesRiskAdjustedReward
from agent.rl_agent import PPOAgent


def get_best_device():
    """Determine the best available device for PyTorch.

    Checks for CUDA (NVIDIA GPU), then MPS (Apple Silicon),
    and falls back to CPU if neither is available.

    Returns:
        str: Device string for PyTorch ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        print("Using CPU")
    return device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a trading agent')

    parser.add_argument('--data_path', type=str, default='data/BTCUSDT_1m_with_metrics.csv',
                        help='Path to the data file')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model to train (required)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Specific checkpoint to resume from (e.g., "checkpoint_100", "best", "final")')
    parser.add_argument('--reward_type', type=str, default='futures',
                        choices=['simple', 'sharpe', 'risk_adjusted', 'futures', 'tunable'],
                        help='Type of reward function to use')
    parser.add_argument('--initial_balance', type=float, default=100000,
                        help='Initial balance for the trading environment')
    parser.add_argument('--commission', type=float, default=0.0004,
                        help='Commission rate for trading')
    parser.add_argument('--max_leverage', type=float, default=5.0,
                        help='Maximum leverage allowed')

    # Training parameters
    parser.add_argument('--n_episodes', type=int, default=1000,
                        help='Number of episodes to train for')
    parser.add_argument('--max_timesteps', type=int, default=1000,
                        help='Maximum timesteps per episode')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for PPO training')
    parser.add_argument('--update_interval', type=int, default=2048,
                        help='Number of timesteps between PPO updates')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Number of episodes between checkpoints')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes for parallel training (0 for auto)')

    # Parallel training parameters
    parser.add_argument('--no_parallel', action='store_true',
                        help='Disable parallel training of multiple models')
    parser.add_argument('--config_file', type=str, default='configs/default_futures_configs.json',
                        help='JSON file containing configurations for parallel training')
    parser.add_argument('--max_parallel_jobs', type=int, default=None,
                        help='Maximum number of parallel jobs (default: number of CPU cores)')
    parser.add_argument('--output_dir', type=str, default='parallel_training_results',
                        help='Directory to save parallel training results')

    # Model parameters
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='Hidden dimensions for the networks')
    parser.add_argument('--lr_actor', type=float, default=3e-4,
                        help='Learning rate for the actor network')
    parser.add_argument('--lr_critic', type=float, default=3e-4,
                        help='Learning rate for the critic network')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda parameter')
    parser.add_argument('--policy_clip', type=float, default=0.2,
                        help='Clipping parameter for PPO')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs to train on each batch of data')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient for exploration')

    # Tunable reward parameters
    parser.add_argument('--trade_penalty', type=float, default=0.0005,
                        help='Penalty for trading in the reward function')
    parser.add_argument('--leverage_penalty', type=float, default=0.05,
                        help='Penalty for leverage in the reward function')
    parser.add_argument('--drawdown_penalty', type=float, default=0.2,
                        help='Penalty for drawdown in the reward function')
    parser.add_argument('--liquidation_penalty', type=float, default=2.0,
                        help='Penalty for liquidation risk in the futures reward function')
    parser.add_argument('--funding_rate_penalty', type=float, default=0.05,
                        help='Penalty for funding rates in the futures reward function')
    parser.add_argument('--liquidation_distance_factor', type=float, default=1.0,
                        help='Factor to penalize proximity to liquidation price')
    parser.add_argument('--reward_scale', type=float, default=1.0,
                        help='Scaling factor for the reward in the tunable reward function')
    parser.add_argument('--realized_pnl_weight', type=float, default=1.0,
                        help='Weight for realized PnL in the tunable reward function')
    parser.add_argument('--unrealized_pnl_weight', type=float, default=0.1,
                        help='Weight for unrealized PnL in the tunable reward function')
    parser.add_argument('--transaction_cost_weight', type=float, default=1.0,
                        help='Weight for transaction costs in the tunable reward function')
    parser.add_argument('--volatility_penalty', type=float, default=0.0,
                        help='Penalty for return volatility in the tunable reward function')
    parser.add_argument('--holding_time_bonus', type=float, default=0.0,
                        help='Bonus for holding positions longer in the tunable reward function')
    parser.add_argument('--trend_alignment_bonus', type=float, default=0.0,
                        help='Bonus for aligning with market trend in the tunable reward function')
    parser.add_argument('--sharpe_weight', type=float, default=0.0,
                        help='Weight for Sharpe ratio component in the tunable reward function')
    parser.add_argument('--sharpe_window', type=int, default=20,
                        help='Window size for Sharpe ratio calculation in the tunable reward function')
    parser.add_argument('--track_reward_components', action='store_true', default=True,
                        help='Track individual reward components for analysis')

    # Paths
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')

    return parser.parse_args()


def prepare_data(data_path):
    """Prepare data for training."""
    # Load data
    data = pd.read_csv(data_path)

    # Normalize price and volume data
    for col in ['open', 'high', 'low', 'close']:
        rolling_mean = data[col].rolling(window=20).mean().bfill()
        rolling_std = data[col].rolling(window=20).std().bfill().replace(0, 1)
        data[f'{col}_norm'] = (data[col] - rolling_mean) / rolling_std

    # Normalize volume
    volume_mean = data['volume'].rolling(window=20).mean().bfill()
    volume_std = data['volume'].rolling(window=20).std().bfill().replace(0, 1)
    data['volume_norm'] = (data['volume'] - volume_mean) / volume_std

    # Add futures-specific data
    # Check if futures-specific columns exist and add them if they don't
    futures_columns = ['fundingRate', 'funding_rate', 'open_interest', 'liquidations', 'basis', 
                      'sumOpenInterest', 'liquidation_value']
    
    for col in futures_columns:
        if col not in data.columns:
            data[col] = 0.0
    
    # Ensure we have a funding_rate column (could be either fundingRate or funding_rate)
    if 'funding_rate' not in data.columns and 'fundingRate' in data.columns:
        data['funding_rate'] = data['fundingRate']
    elif 'funding_rate' not in data.columns:
        data['funding_rate'] = 0.0
        
    # Fill any remaining NaN values
    data = data.fillna(0)

    return data


def create_reward_function(reward_type, args):
    """Create a reward function based on the given type."""
    if reward_type == 'simple':
        return SimpleReward()
    elif reward_type == 'sharpe':
        return SharpeReward(window_size=50)
    elif reward_type == 'risk_adjusted':
        return RiskAdjustedReward(
            trade_penalty=args.trade_penalty,
            leverage_penalty=args.leverage_penalty,
            drawdown_penalty=args.drawdown_penalty
        )
    elif reward_type == 'futures':
        return FuturesRiskAdjustedReward(
            trade_penalty=args.trade_penalty,
            leverage_penalty=args.leverage_penalty,
            drawdown_penalty=args.drawdown_penalty,
            liquidation_penalty=args.liquidation_penalty if hasattr(args, 'liquidation_penalty') else 2.0,
            funding_rate_penalty=args.funding_rate_penalty if hasattr(args, 'funding_rate_penalty') else 0.05,
            liquidation_distance_factor=args.liquidation_distance_factor if hasattr(args, 'liquidation_distance_factor') else 1.0,
            max_leverage=args.max_leverage
        )
    elif reward_type == 'tunable':
        return TunableReward(
            realized_pnl_weight=args.realized_pnl_weight,
            unrealized_pnl_weight=args.unrealized_pnl_weight,
            transaction_cost_weight=args.transaction_cost_weight,
            trade_penalty=args.trade_penalty,
            leverage_penalty=args.leverage_penalty,
            drawdown_penalty=args.drawdown_penalty,
            volatility_penalty=args.volatility_penalty,
            holding_time_bonus=args.holding_time_bonus,
            trend_alignment_bonus=args.trend_alignment_bonus,
            sharpe_weight=args.sharpe_weight,
            sharpe_window=args.sharpe_window,
            reward_scale=args.reward_scale,
            track_components=args.track_reward_components
        )
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def load_configurations(config_file):
    """Load configurations from a JSON file."""
    with open(config_file, 'r') as f:
        configs = json.load(f)
    return configs


def analyze_results(output_dir):
    """Analyze results from parallel training."""
    # Load rewards history for each model
    rewards_history = {}
    for model_dir in os.listdir(output_dir):
        model_path = os.path.join(output_dir, model_dir)
        if os.path.isdir(model_path):
            rewards_file = os.path.join(model_path, 'rewards_history.npy')
            if os.path.exists(rewards_file):
                rewards_history[model_dir] = np.load(rewards_file)

    # Plot learning curves for each model
    plt.figure(figsize=(10, 5))
    for model, rewards in rewards_history.items():
        plt.plot(rewards, label=model)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()

    # Plot average rewards for each model
    plt.figure(figsize=(10, 5))
    for model, rewards in rewards_history.items():
        plt.bar(model, np.mean(rewards))
    plt.title('Average Training Rewards')
    plt.xlabel('Model')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(output_dir, 'average_rewards.png'))
    plt.close()


def create_visualizations(output_dir):
    """Create visualizations for parallel training."""
    # Load reward components for each model
    reward_components = {}
    for model_dir in os.listdir(output_dir):
        model_path = os.path.join(output_dir, model_dir)
        if os.path.isdir(model_path):
            components_dir = os.path.join(model_path, 'reward_components')
            if os.path.exists(components_dir):
                components = []
                for file in os.listdir(components_dir):
                    if file.endswith('_components.npy'):
                        components.append(np.load(os.path.join(components_dir, file), allow_pickle=True).item())
                reward_components[model_dir] = components

    # Plot reward components for each model
    for model, components in reward_components.items():
        plt.figure(figsize=(10, 5))
        for component in components:
            plt.plot(component['total'], label='Total')
            plt.plot(component['realized_pnl'], label='Realized PnL')
            plt.plot(component['unrealized_pnl'], label='Unrealized PnL')
            plt.plot(component['transaction_cost'], label='Transaction Cost')
            plt.plot(component['trade_penalty'], label='Trade Penalty')
            plt.plot(component['leverage_penalty'], label='Leverage Penalty')
            plt.plot(component['drawdown_penalty'], label='Drawdown Penalty')
        plt.title('Reward Components')
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{model}_reward_components.png'))
        plt.close()


def run_training_job(args):
    """Run a single training job."""
    # Select the best available device
    device = get_best_device()

    # Set up multiprocessing
    if args.num_workers == 0:
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = args.num_workers

    # Configure PyTorch to use multiple cores
    torch.set_num_threads(num_workers)
    torch.device(device)
    print(f"Using {num_workers} CPU cores for parallel processing")

    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create model-specific directory
    model_dir = os.path.join(args.model_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Create log directory for this model
    model_log_dir = os.path.join(args.log_dir, args.model_name)
    os.makedirs(model_log_dir, exist_ok=True)

    # Prepare data
    data = prepare_data(args.data_path)

    # Create reward function
    reward_function = create_reward_function(args.reward_type, args)

    # Create environment
    env = TradingEnvironment(
        data=data,
        reward_function=reward_function,
        initial_balance=args.initial_balance,
        commission=args.commission,
        max_leverage=args.max_leverage
    )

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=tuple(args.hidden_dims),
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        policy_clip=args.policy_clip,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        entropy_coef=args.entropy_coef,
        device=device  # Pass the detected device
    )

    # Load checkpoint if resuming training
    start_episode = 0
    best_reward = -float('inf')
    rewards_history = []

    if args.resume or args.resume_checkpoint:
        checkpoint_path = None

        if args.resume_checkpoint:
            # Use the specified checkpoint
            checkpoint_path = os.path.join(model_dir, args.resume_checkpoint)
        elif args.resume:
            # Find the latest checkpoint
            checkpoints = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_') and f.endswith('_checkpoint.pt')]
            if checkpoints:
                # Extract episode numbers and find the latest
                episodes = [int(f.split('_')[1].split('_checkpoint.pt')[0]) for f in checkpoints]
                latest_episode = max(episodes)
                checkpoint_path = os.path.join(model_dir, f'checkpoint_{latest_episode}')

        if checkpoint_path:
            print(f"Resuming training from {checkpoint_path}")
            if agent.load_models(checkpoint_path, load_optimizer=True, load_memory=True):
                # Set the starting episode to the loaded episode count
                start_episode = agent.episode_count
                print(f"Resuming from episode {start_episode}")

                # Try to load rewards history if it exists
                rewards_history_path = os.path.join(model_log_dir, 'rewards_history.npy')
                if os.path.exists(rewards_history_path):
                    try:
                        rewards_history = np.load(rewards_history_path).tolist()
                        if len(rewards_history) > 0:
                            best_reward = max(rewards_history)
                        print(f"Loaded rewards history with {len(rewards_history)} episodes")
                    except Exception as e:
                        print(f"Error loading rewards history: {str(e)}")
                        rewards_history = []
            else:
                print("Failed to load checkpoint, starting from scratch")

    # Training loop
    if not rewards_history:
        rewards_history = []

    total_timesteps = 0
    update_timesteps = args.update_interval

    for episode in tqdm(range(start_episode, args.n_episodes), desc="Training"):
        observation, _ = env.reset()

        episode_reward = 0
        done = False

        for t in range(args.max_timesteps):
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
            total_timesteps += 1

            # Update agent if enough steps have been taken
            if total_timesteps % update_timesteps == 0:
                # Get value estimate for the current state
                _, _, next_value = agent.choose_action(next_observation)
                agent.learn(next_value[0])

            if done or truncated:
                break

        # If we didn't update at the end of the episode, update now
        if len(agent.memory.states) > 0:
            _, _, next_value = agent.choose_action(next_observation)
            agent.learn(next_value[0])

        # Track rewards
        rewards_history.append(episode_reward)

        # Print episode summary
        print(f"Episode {episode+1}/{args.n_episodes}, Reward: {episode_reward:.2f}, "
              f"Final Balance: ${env.balance:.2f}")

        # Update agent's episode count
        agent.episode_count = episode + 1

        # Track reward components if using TunableReward with tracking enabled
        if args.reward_type == 'tunable' and args.track_reward_components:
            try:
                # Get the reward components from the environment's reward function
                reward_components = env.reward_function.get_reward_components()

                # Create reward components directory if it doesn't exist
                reward_components_dir = os.path.join(model_log_dir, 'reward_components')
                os.makedirs(reward_components_dir, exist_ok=True)

                # Save reward components for this episode
                np.save(
                    os.path.join(reward_components_dir, f'episode_{episode+1}_components.npy'),
                    reward_components
                )

                # Every 10 episodes, create a visualization of the reward components
                if (episode + 1) % 10 == 0:
                    # Collect all saved component files
                    component_files = [f for f in os.listdir(reward_components_dir) if f.endswith('_components.npy')]
                    if component_files:
                        # Sort by episode number
                        component_files.sort(key=lambda x: int(x.split('_')[1]))

                        # Load all components
                        all_components = []
                        for file in component_files:
                            try:
                                components = np.load(os.path.join(reward_components_dir, file), allow_pickle=True).item()
                                all_components.append(components)
                            except Exception as e:
                                print(f"Error loading reward components from {file}: {str(e)}")

                        if all_components:
                            # Create a plot for each component
                            component_keys = list(all_components[0].keys())

                            plt.figure(figsize=(15, 10))
                            for i, key in enumerate(component_keys):
                                if key != 'total':  # Skip total since we already plot it separately
                                    plt.subplot(3, 4, i % 12 + 1)
                                    values = [comp.get(key, 0) for comp in all_components]
                                    plt.plot(values)
                                    plt.title(key)
                                    plt.xlabel('Episode')
                                    plt.ylabel('Value')

                                    # If we've plotted 12 components or this is the last one, save the figure
                                    if (i + 1) % 12 == 0 or i == len(component_keys) - 1:
                                        plt.tight_layout()
                                        plt.savefig(os.path.join(reward_components_dir, f'components_plot_{i//12}.png'))
                                        plt.close()
                                        if i < len(component_keys) - 1:  # If there are more components to plot
                                            plt.figure(figsize=(15, 10))
            except Exception as e:
                print(f"Error tracking reward components: {str(e)}")

        # Save models if this is the best performance so far
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_models(os.path.join(model_dir, 'best'))

        # Save checkpoint
        if (episode + 1) % args.checkpoint_interval == 0:
            agent.save_models(os.path.join(model_dir, f'checkpoint_{episode+1}'))

    # Save final model
    agent.save_models(os.path.join(model_dir, 'final'))

    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(model_log_dir, 'learning_curve.png'))
    plt.close()

    # Save rewards history
    np.save(os.path.join(model_log_dir, 'rewards_history.npy'), np.array(rewards_history))

    print(f"Training completed. Best reward: {best_reward:.2f}")


def main():
    """Main training function."""
    args = parse_args()

    if not args.no_parallel:
        # Load configurations from JSON file
        configs = load_configurations(args.config_file)

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Run parallel training jobs
        with ProcessPoolExecutor(max_workers=args.max_parallel_jobs) as executor:
            futures = []
            for config in configs:
                # Create a copy of the arguments
                config_args = argparse.Namespace(**vars(args))

                # Override with configuration values, but keep original model_name if not in config
                for key, value in config.items():
                    if key != 'model_name' or 'model_name' not in vars(config_args):
                        setattr(config_args, key, value)

                futures.append(executor.submit(run_training_job, config_args))

            # Wait for all jobs to complete
            for future in as_completed(futures):
                future.result()

        # Analyze results
        analyze_results(args.output_dir)

        # Create visualizations
        create_visualizations(args.output_dir)
    else:
        run_training_job(args)


if __name__ == "__main__":
    main()
