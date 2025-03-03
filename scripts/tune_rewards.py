#!/usr/bin/env python3
"""
Script for running reward function tuning experiments.
This script automates the process of testing different reward function configurations.
"""

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json
from itertools import product

def parse_args():
    """Parse command line arguments for the reward tuning script."""
    parser = argparse.ArgumentParser(description='Run reward function tuning experiments')
    
    parser.add_argument('--base_model_name', type=str, required=True,
                        help='Base name for the model experiments')
    parser.add_argument('--data_path', type=str, default='data/BTC_USD_hourly_with_metrics.csv',
                        help='Path to the data file')
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='Number of episodes to train for each experiment')
    parser.add_argument('--experiments_file', type=str, default=None,
                        help='JSON file containing experiment configurations')
    parser.add_argument('--output_dir', type=str, default='tuning_results',
                        help='Directory to save tuning results')
    
    # Parameter ranges for grid search
    parser.add_argument('--realized_pnl_weights', type=float, nargs='+', default=[1.0],
                        help='List of realized PnL weights to try')
    parser.add_argument('--unrealized_pnl_weights', type=float, nargs='+', default=[0.1, 0.2, 0.5],
                        help='List of unrealized PnL weights to try')
    parser.add_argument('--trade_penalties', type=float, nargs='+', default=[0.0001, 0.0005, 0.001],
                        help='List of trade penalties to try')
    parser.add_argument('--leverage_penalties', type=float, nargs='+', default=[0.001, 0.01, 0.05],
                        help='List of leverage penalties to try')
    parser.add_argument('--drawdown_penalties', type=float, nargs='+', default=[0.05, 0.1, 0.2],
                        help='List of drawdown penalties to try')
    
    return parser.parse_args()

def load_experiments(file_path):
    """Load experiment configurations from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Experiments file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        experiments = json.load(f)
    
    return experiments

def generate_experiments(args):
    """Generate experiment configurations based on parameter ranges."""
    # Define the parameter grid
    param_grid = {
        'realized_pnl_weight': args.realized_pnl_weights,
        'unrealized_pnl_weight': args.unrealized_pnl_weights,
        'trade_penalty': args.trade_penalties,
        'leverage_penalty': args.leverage_penalties,
        'drawdown_penalty': args.drawdown_penalties
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = list(param_grid.values())
    experiments = []
    
    for combination in product(*values):
        experiment = dict(zip(keys, combination))
        experiments.append(experiment)
    
    return experiments

def run_experiment(base_model_name, experiment_id, params, data_path, n_episodes, output_dir):
    """Run a single experiment with the given parameters."""
    # Create a unique model name for this experiment
    model_name = f"{base_model_name}_exp{experiment_id}"
    
    # Construct the command with the experiment parameters
    cmd = [
        "python", "train.py",
        "--model_name", model_name,
        "--data_path", data_path,
        "--n_episodes", str(n_episodes),
        "--reward_type", "tunable",
        "--track_reward_components"
    ]
    
    # Add all parameters from the experiment
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    # Run the experiment
    print(f"Running experiment {experiment_id}: {model_name}")
    print(f"Parameters: {params}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment {experiment_id} failed: {e}")
        return False

def analyze_results(base_model_name, experiments, output_dir):
    """Analyze the results of all experiments."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect results
    results = []
    for i, params in enumerate(experiments):
        model_name = f"{base_model_name}_exp{i+1}"
        model_log_dir = os.path.join("logs", model_name)
        
        # Check if rewards history exists
        rewards_path = os.path.join(model_log_dir, "rewards_history.npy")
        if os.path.exists(rewards_path):
            rewards = np.load(rewards_path)
            
            # Calculate metrics
            mean_reward = np.mean(rewards)
            max_reward = np.max(rewards)
            final_reward = rewards[-1]
            
            # Add to results
            result = {
                "experiment_id": i+1,
                "model_name": model_name,
                "mean_reward": mean_reward,
                "max_reward": max_reward,
                "final_reward": final_reward
            }
            result.update(params)
            results.append(result)
    
    # Create results dataframe
    if results:
        df = pd.DataFrame(results)
        
        # Save results to CSV
        results_path = os.path.join(output_dir, f"{base_model_name}_results.csv")
        df.to_csv(results_path, index=False)
        
        # Create visualizations
        create_visualizations(df, base_model_name, output_dir)
        
        # Print top 5 experiments by mean reward
        print("\nTop 5 experiments by mean reward:")
        top_by_mean = df.sort_values("mean_reward", ascending=False).head(5)
        print(top_by_mean[["experiment_id", "model_name", "mean_reward", "max_reward", "final_reward"]])
        
        # Print top 5 experiments by max reward
        print("\nTop 5 experiments by max reward:")
        top_by_max = df.sort_values("max_reward", ascending=False).head(5)
        print(top_by_max[["experiment_id", "model_name", "mean_reward", "max_reward", "final_reward"]])
        
        return df
    else:
        print("No results found.")
        return None

def create_visualizations(df, base_model_name, output_dir):
    """Create visualizations of the experiment results."""
    # Create a directory for visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot distribution of mean rewards
    plt.figure(figsize=(10, 6))
    plt.hist(df["mean_reward"], bins=20)
    plt.title("Distribution of Mean Rewards")
    plt.xlabel("Mean Reward")
    plt.ylabel("Count")
    plt.savefig(os.path.join(viz_dir, f"{base_model_name}_mean_reward_dist.png"))
    plt.close()
    
    # Plot parameter impact on mean reward
    param_columns = ["realized_pnl_weight", "unrealized_pnl_weight", "trade_penalty", 
                     "leverage_penalty", "drawdown_penalty"]
    
    for param in param_columns:
        if len(df[param].unique()) > 1:  # Only plot if there are multiple values
            plt.figure(figsize=(10, 6))
            
            # Group by parameter and calculate mean of mean_reward
            grouped = df.groupby(param)["mean_reward"].mean().reset_index()
            plt.bar(grouped[param].astype(str), grouped["mean_reward"])
            
            plt.title(f"Impact of {param} on Mean Reward")
            plt.xlabel(param)
            plt.ylabel("Mean Reward")
            plt.savefig(os.path.join(viz_dir, f"{base_model_name}_{param}_impact.png"))
            plt.close()
    
    # Create a correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = df[param_columns + ["mean_reward", "max_reward", "final_reward"]].corr()
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    
    # Add correlation values
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", 
                     color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    
    plt.title("Parameter Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{base_model_name}_correlation_heatmap.png"))
    plt.close()

def main():
    """Main function for running reward tuning experiments."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or generate experiments
    if args.experiments_file:
        experiments = load_experiments(args.experiments_file)
    else:
        experiments = generate_experiments(args)
    
    # Save experiment configurations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiments_path = os.path.join(args.output_dir, f"{args.base_model_name}_{timestamp}_experiments.json")
    with open(experiments_path, 'w') as f:
        json.dump(experiments, f, indent=2)
    
    print(f"Running {len(experiments)} experiments")
    
    # Run experiments
    for i, params in enumerate(experiments):
        success = run_experiment(
            args.base_model_name, 
            i+1, 
            params, 
            args.data_path, 
            args.n_episodes, 
            args.output_dir
        )
        
        if not success:
            print(f"Experiment {i+1} failed, continuing with next experiment")
    
    # Analyze results
    results = analyze_results(args.base_model_name, experiments, args.output_dir)
    
    if results is not None:
        print(f"\nExperiments completed. Results saved to {args.output_dir}")
    else:
        print("\nNo results were generated from the experiments.")

if __name__ == "__main__":
    main()
