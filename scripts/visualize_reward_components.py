#!/usr/bin/env python3
"""
Script to visualize reward components from a trained model.
This script loads reward component data saved during training and creates visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize reward components from a trained model')
    
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model to analyze')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory containing logs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations (defaults to model log dir)')
    parser.add_argument('--moving_avg', type=int, default=5,
                        help='Window size for moving average smoothing')
    
    return parser.parse_args()


def load_reward_components(model_name, log_dir):
    """Load reward component data for the specified model."""
    # Path to reward components directory
    components_dir = os.path.join(log_dir, model_name, 'reward_components')
    
    if not os.path.exists(components_dir):
        raise FileNotFoundError(f"Reward components directory not found: {components_dir}")
    
    # Find all component files
    component_files = glob(os.path.join(components_dir, 'episode_*_components.npy'))
    
    if not component_files:
        raise FileNotFoundError(f"No reward component files found in {components_dir}")
    
    # Sort by episode number
    component_files.sort(key=lambda x: int(x.split('episode_')[1].split('_')[0]))
    
    # Load all components
    all_components = []
    episodes = []
    
    for file in component_files:
        try:
            # Extract episode number
            episode = int(file.split('episode_')[1].split('_')[0])
            episodes.append(episode)
            
            # Load components
            components = np.load(file, allow_pickle=True).item()
            all_components.append(components)
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    return episodes, all_components


def moving_average(data, window_size):
    """Calculate moving average of data."""
    if window_size <= 1:
        return data
    
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def visualize_components(episodes, all_components, model_name, output_dir, moving_avg_window):
    """Create visualizations of reward components."""
    if not all_components:
        print("No components to visualize.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all component keys
    component_keys = list(all_components[0].keys())
    
    # Create a plot for the total reward
    plt.figure(figsize=(12, 6))
    total_rewards = [comp.get('total', 0) for comp in all_components]
    plt.plot(episodes, total_rewards, label='Raw', alpha=0.5)
    
    # Add moving average if we have enough data
    if len(total_rewards) >= moving_avg_window:
        ma_episodes = episodes[moving_avg_window-1:]
        ma_rewards = moving_average(total_rewards, moving_avg_window)
        plt.plot(ma_episodes, ma_rewards, label=f'Moving Avg (window={moving_avg_window})', linewidth=2)
    
    plt.title(f'Total Reward Over Time - {model_name}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'total_reward.png'))
    plt.close()
    
    # Create a plot for each component (except total)
    for key in component_keys:
        if key != 'total':
            plt.figure(figsize=(12, 6))
            values = [comp.get(key, 0) for comp in all_components]
            plt.plot(episodes, values, label='Raw', alpha=0.5)
            
            # Add moving average if we have enough data
            if len(values) >= moving_avg_window:
                ma_episodes = episodes[moving_avg_window-1:]
                ma_values = moving_average(values, moving_avg_window)
                plt.plot(ma_episodes, ma_values, label=f'Moving Avg (window={moving_avg_window})', linewidth=2)
            
            plt.title(f'{key} Over Time - {model_name}')
            plt.xlabel('Episode')
            plt.ylabel(key)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'{key}.png'))
            plt.close()
    
    # Create a stacked area chart of all components (except total)
    plt.figure(figsize=(14, 8))
    
    # Separate positive and negative components
    pos_components = {}
    neg_components = {}
    
    for key in component_keys:
        if key != 'total':
            values = [comp.get(key, 0) for comp in all_components]
            if np.mean(values) >= 0:
                pos_components[key] = values
            else:
                neg_components[key] = values
    
    # Plot positive components
    if pos_components:
        pos_keys = list(pos_components.keys())
        pos_values = np.array([pos_components[key] for key in pos_keys])
        plt.stackplot(episodes, pos_values, labels=pos_keys, alpha=0.7)
    
    # Plot negative components
    if neg_components:
        neg_keys = list(neg_components.keys())
        neg_values = np.array([neg_components[key] for key in neg_keys])
        plt.stackplot(episodes, neg_values, labels=neg_keys, alpha=0.7)
    
    plt.title(f'Reward Components Over Time - {model_name}')
    plt.xlabel('Episode')
    plt.ylabel('Component Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stacked_components.png'))
    plt.close()
    
    # Create a correlation heatmap of components
    plt.figure(figsize=(12, 10))
    
    # Create a dictionary of component values
    component_dict = {key: [comp.get(key, 0) for comp in all_components] for key in component_keys}
    
    # Calculate correlation matrix
    corr_matrix = np.zeros((len(component_keys), len(component_keys)))
    for i, key1 in enumerate(component_keys):
        for j, key2 in enumerate(component_keys):
            corr = np.corrcoef(component_dict[key1], component_dict[key2])[0, 1]
            corr_matrix[i, j] = corr
    
    # Plot heatmap
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(component_keys)), component_keys, rotation=45)
    plt.yticks(range(len(component_keys)), component_keys)
    
    # Add correlation values
    for i in range(len(component_keys)):
        for j in range(len(component_keys)):
            plt.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", 
                     color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
    
    plt.title(f'Reward Component Correlation - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_correlation.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    """Main function."""
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(args.log_dir, args.model_name, 'visualizations')
    else:
        output_dir = args.output_dir
    
    try:
        # Load reward components
        episodes, all_components = load_reward_components(args.model_name, args.log_dir)
        
        # Create visualizations
        visualize_components(episodes, all_components, args.model_name, output_dir, args.moving_avg)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
