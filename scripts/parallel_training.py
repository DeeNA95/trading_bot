#!/usr/bin/env python3
"""
Script for running multiple training jobs in parallel.
This allows for faster experimentation with different hyperparameters and reward functions.
"""

import os
import argparse
import json
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run multiple training jobs in parallel')
    
    parser.add_argument('--config_file', type=str, required=True,
                        help='JSON file containing training configurations')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of parallel workers (default: number of CPU cores)')
    parser.add_argument('--output_dir', type=str, default='parallel_training_results',
                        help='Directory to save training results')
    parser.add_argument('--log_dir', type=str, default='parallel_logs',
                        help='Directory to save training logs')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Maximum time in seconds for each job (None for no timeout)')
    
    return parser.parse_args()


def load_configurations(config_file):
    """Load training configurations from a JSON file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        configs = json.load(f)
    
    return configs


def run_training_job(job_id, config, log_dir, timeout=None):
    """Run a single training job with the given configuration."""
    # Create a unique model name based on job ID
    model_name = config.get('model_name', f'parallel_job_{job_id}')
    
    # Create log file path
    log_file = os.path.join(log_dir, f"{model_name}.log")
    
    # Build command from configuration
    cmd = ["python", "train.py"]
    
    # Add all parameters from the configuration
    for key, value in config.items():
        if key == 'model_name':
            cmd.extend([f"--{key}", model_name])
        elif isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    # Start time
    start_time = time.time()
    
    # Run the training process
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Wait for the process to complete or timeout
            try:
                return_code = process.wait(timeout=timeout)
                success = return_code == 0
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"Job {job_id} ({model_name}) timed out after {timeout} seconds")
                success = False
                return_code = -1
    
    except Exception as e:
        print(f"Error running job {job_id} ({model_name}): {str(e)}")
        success = False
        return_code = -1
    
    # End time
    end_time = time.time()
    duration = end_time - start_time
    
    # Return job results
    return {
        'job_id': job_id,
        'model_name': model_name,
        'success': success,
        'return_code': return_code,
        'duration': duration,
        'log_file': log_file,
        'config': config
    }


def analyze_results(results, output_dir):
    """Analyze the results of all training jobs."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results to CSV
    results_file = os.path.join(output_dir, 'training_results.csv')
    df.to_csv(results_file, index=False)
    
    # Create summary visualizations
    create_summary_visualizations(df, output_dir)
    
    # Collect and compare training rewards
    compare_training_rewards(results, output_dir)
    
    return df


def create_summary_visualizations(df, output_dir):
    """Create summary visualizations of training results."""
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot job durations
    plt.figure(figsize=(12, 6))
    plt.bar(df['model_name'], df['duration'] / 60)  # Convert to minutes
    plt.title('Training Job Durations')
    plt.xlabel('Model')
    plt.ylabel('Duration (minutes)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'job_durations.png'))
    plt.close()
    
    # Plot success rate
    success_count = df['success'].sum()
    failure_count = len(df) - success_count
    plt.figure(figsize=(8, 8))
    plt.pie([success_count, failure_count], 
            labels=['Success', 'Failure'],
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'])
    plt.title('Training Job Success Rate')
    plt.savefig(os.path.join(viz_dir, 'success_rate.png'))
    plt.close()


def compare_training_rewards(results, output_dir):
    """Compare the training rewards of all successful jobs."""
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Collect rewards for all successful jobs
    rewards_data = []
    
    for result in results:
        if result['success']:
            model_name = result['model_name']
            model_log_dir = os.path.join('logs', model_name)
            rewards_file = os.path.join(model_log_dir, 'rewards_history.npy')
            
            if os.path.exists(rewards_file):
                try:
                    rewards = np.load(rewards_file)
                    rewards_data.append({
                        'model_name': model_name,
                        'rewards': rewards,
                        'mean_reward': np.mean(rewards),
                        'max_reward': np.max(rewards),
                        'final_reward': rewards[-1] if len(rewards) > 0 else 0
                    })
                except Exception as e:
                    print(f"Error loading rewards for {model_name}: {str(e)}")
    
    if not rewards_data:
        print("No reward data found for successful jobs")
        return
    
    # Create rewards comparison DataFrame
    rewards_df = pd.DataFrame([
        {
            'model_name': data['model_name'],
            'mean_reward': data['mean_reward'],
            'max_reward': data['max_reward'],
            'final_reward': data['final_reward']
        }
        for data in rewards_data
    ])
    
    # Save rewards comparison to CSV
    rewards_file = os.path.join(output_dir, 'rewards_comparison.csv')
    rewards_df.to_csv(rewards_file, index=False)
    
    # Plot rewards comparison
    plt.figure(figsize=(14, 8))
    
    # Plot learning curves for all models
    for data in rewards_data:
        plt.plot(data['rewards'], label=data['model_name'])
    
    plt.title('Learning Curves Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'learning_curves_comparison.png'))
    plt.close()
    
    # Plot mean rewards comparison
    plt.figure(figsize=(12, 6))
    plt.bar(rewards_df['model_name'], rewards_df['mean_reward'])
    plt.title('Mean Rewards Comparison')
    plt.xlabel('Model')
    plt.ylabel('Mean Reward')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'mean_rewards_comparison.png'))
    plt.close()
    
    # Plot max rewards comparison
    plt.figure(figsize=(12, 6))
    plt.bar(rewards_df['model_name'], rewards_df['max_reward'])
    plt.title('Max Rewards Comparison')
    plt.xlabel('Model')
    plt.ylabel('Max Reward')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'max_rewards_comparison.png'))
    plt.close()
    
    # Print top models by mean reward
    print("\nTop models by mean reward:")
    top_by_mean = rewards_df.sort_values('mean_reward', ascending=False)
    print(top_by_mean)


def main():
    """Main function for running parallel training jobs."""
    args = parse_args()
    
    # Load configurations
    configs = load_configurations(args.config_file)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Determine number of workers
    if args.max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(configs))
    else:
        max_workers = min(args.max_workers, len(configs))
    
    print(f"Running {len(configs)} training jobs with {max_workers} parallel workers")
    
    # Run training jobs in parallel
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_job = {
            executor.submit(run_training_job, i, config, args.log_dir, args.timeout): (i, config)
            for i, config in enumerate(configs)
        }
        
        # Process results as they complete
        for future in as_completed(future_to_job):
            job_id, config = future_to_job[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print progress
                print(f"Completed job {job_id+1}/{len(configs)}: {result['model_name']} "
                      f"(Success: {result['success']}, Duration: {result['duration']:.2f}s)")
                
            except Exception as e:
                print(f"Job {job_id} generated an exception: {str(e)}")
                results.append({
                    'job_id': job_id,
                    'model_name': config.get('model_name', f'parallel_job_{job_id}'),
                    'success': False,
                    'return_code': -1,
                    'duration': 0,
                    'log_file': None,
                    'config': config,
                    'error': str(e)
                })
    
    # Analyze results
    df = analyze_results(results, args.output_dir)
    
    # Print summary
    success_count = df['success'].sum()
    print(f"\nTraining complete: {success_count}/{len(configs)} jobs successful")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
