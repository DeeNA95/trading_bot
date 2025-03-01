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

from environment.trading_env import TradingEnvironment
from environment.reward import SimpleReward, SharpeReward, RiskAdjustedReward
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
    
    parser.add_argument('--data_path', type=str, default='data/BTC_USD_complete.csv',
                        help='Path to the data file')
    parser.add_argument('--reward_type', type=str, default='risk_adjusted',
                        choices=['simple', 'sharpe', 'risk_adjusted'],
                        help='Type of reward function to use')
    parser.add_argument('--initial_balance', type=float, default=10000,
                        help='Initial balance for the trading environment')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate for trading')
    parser.add_argument('--max_leverage', type=float, default=3.0,
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
        rolling_mean = data[col].rolling(window=20).mean().fillna(method='bfill')
        rolling_std = data[col].rolling(window=20).std().fillna(method='bfill').replace(0, 1)
        data[f'{col}_norm'] = (data[col] - rolling_mean) / rolling_std
    
    # Normalize volume
    volume_mean = data['volume'].rolling(window=20).mean().fillna(method='bfill')
    volume_std = data['volume'].rolling(window=20).std().fillna(method='bfill').replace(0, 1)
    data['volume_norm'] = (data['volume'] - volume_mean) / volume_std
    
    # Fill any remaining NaN values
    data = data.fillna(0)
    
    return data


def create_reward_function(reward_type):
    """Create a reward function based on the given type."""
    if reward_type == 'simple':
        return SimpleReward()
    elif reward_type == 'sharpe':
        return SharpeReward(window_size=50)
    elif reward_type == 'risk_adjusted':
        return RiskAdjustedReward(
            trade_penalty=0.0002,
            leverage_penalty=0.001,
            drawdown_penalty=0.1
        )
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Select the best available device
    device = get_best_device()
    
    # Set up multiprocessing
    if args.num_workers == 0:
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = args.num_workers
    
    # Configure PyTorch to use multiple cores
    torch.set_num_threads(num_workers)
    print(f"Using {num_workers} CPU cores for parallel processing")
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Prepare data
    data = prepare_data(args.data_path)
    
    # Create reward function
    reward_function = create_reward_function(args.reward_type)
    
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
    
    # Training loop
    best_reward = -float('inf')
    rewards_history = []
    
    total_timesteps = 0
    update_timesteps = args.update_interval
    
    for episode in tqdm(range(args.n_episodes), desc="Training"):
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
        
        # Save models if this is the best performance so far
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_models(os.path.join(args.model_dir, 'best'))
        
        # Save checkpoint
        if (episode + 1) % args.checkpoint_interval == 0:
            agent.save_models(os.path.join(args.model_dir, f'checkpoint_{episode+1}'))
    
    # Save final model
    agent.save_models(os.path.join(args.model_dir, 'final'))
    
    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(args.log_dir, 'learning_curve.png'))
    plt.close()
    
    # Save rewards history
    np.save(os.path.join(args.log_dir, 'rewards_history.npy'), np.array(rewards_history))
    
    print(f"Training completed. Best reward: {best_reward:.2f}")


if __name__ == "__main__":
    main()
