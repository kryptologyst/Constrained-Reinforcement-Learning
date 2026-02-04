"""Modernized constrained reinforcement learning example.

This script demonstrates the modernized constrained RL implementation
with proper error handling, type hints, and comprehensive evaluation.
"""

import argparse
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from src.algorithms.constrained_q_learning import ConstrainedQLearningAgent
from src.envs.constrained_wrappers import ConstrainedEnvWrapper
from src.utils import (
    ConstraintMonitor,
    evaluate_policy,
    get_device,
    set_seed,
)


def main():
    """Main function demonstrating constrained RL."""
    parser = argparse.ArgumentParser(description="Constrained RL Example")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--constraint_threshold", type=float, default=0.5, help="Constraint threshold")
    parser.add_argument("--penalty_weight", type=float, default=2.0, help="Penalty weight")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create environment with constraint wrapper
    env = gym.make(args.env)
    env = ConstrainedEnvWrapper(
        env,
        constraint_threshold=args.constraint_threshold,
        constraint_type="position",
        constraint_penalty=args.penalty_weight,
    )
    
    # Create constrained Q-learning agent
    device = get_device()
    agent = ConstrainedQLearningAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        constraint_threshold=args.constraint_threshold,
        penalty_weight=args.penalty_weight,
        constraint_type="position",
        device=device,
    )
    
    # Training loop
    print(f"Training constrained Q-learning agent on {args.env}")
    print(f"Constraint: position < {args.constraint_threshold}")
    print(f"Penalty weight: {args.penalty_weight}")
    print(f"Device: {device}")
    print("-" * 50)
    
    constraint_monitor = ConstraintMonitor()
    episode_rewards = []
    episode_lengths = []
    
    for episode in tqdm(range(args.episodes), desc="Training"):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.update_q_value(obs, action, reward, next_obs, done)
            
            # Track constraint violations
            if 'constraint_violation' in info:
                constraint_monitor.update(info['constraint_violation'])
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Logging
        if episode % 100 == 0:
            constraint_stats = constraint_monitor.get_stats()
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Length: {avg_length:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Violations: {constraint_stats['total_violations']:3d} | "
                  f"Violation Rate: {constraint_stats['violation_rate']:.3f}")
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    
    eval_results = evaluate_policy(
        env, agent, num_episodes=10, render=args.render, seed=args.seed
    )
    
    print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Mean Length: {eval_results['mean_length']:.2f} ± {eval_results['std_length']:.2f}")
    print(f"Mean Violations: {eval_results['mean_violations']:.2f} ± {eval_results['std_violations']:.2f}")
    
    # Constraint satisfaction analysis
    constraint_stats = constraint_monitor.get_stats()
    print(f"\nConstraint Statistics:")
    print(f"Total Violations: {constraint_stats['total_violations']}")
    print(f"Violation Rate: {constraint_stats['violation_rate']:.1%}")
    print(f"Mean Violation: {constraint_stats['mean_violation']:.3f}")
    print(f"Max Violation: {constraint_stats['max_violation']:.3f}")
    
    # Performance analysis
    final_rewards = episode_rewards[-100:]
    print(f"\nPerformance Analysis (Last 100 episodes):")
    print(f"Average Reward: {np.mean(final_rewards):.2f}")
    print(f"Reward Std: {np.std(final_rewards):.2f}")
    print(f"Best Episode: {np.max(final_rewards):.2f}")
    print(f"Worst Episode: {np.min(final_rewards):.2f}")
    
    # Learning curve analysis
    window_size = 50
    if len(episode_rewards) >= window_size:
        learning_curve = []
        for i in range(window_size, len(episode_rewards)):
            learning_curve.append(np.mean(episode_rewards[i-window_size:i]))
        
        print(f"\nLearning Progress:")
        print(f"Initial Performance (first {window_size} episodes): {np.mean(episode_rewards[:window_size]):.2f}")
        print(f"Final Performance (last {window_size} episodes): {np.mean(episode_rewards[-window_size:]):.2f}")
        print(f"Improvement: {np.mean(episode_rewards[-window_size:]) - np.mean(episode_rewards[:window_size]):.2f}")
    
    env.close()
    
    print("\nTraining completed successfully!")
    print("⚠️  DISCLAIMER: This is a research/educational example.")
    print("   Not for production control of real systems.")


if __name__ == "__main__":
    main()
