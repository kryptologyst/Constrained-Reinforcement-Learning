"""Training script for constrained reinforcement learning."""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from src.algorithms.constrained_q_learning import ConstrainedQLearningAgent
from src.algorithms.lagrangian_ppo import LagrangianPPO
from src.envs.constrained_wrappers import ConstrainedEnvWrapper, SafetyConstraintWrapper
from src.utils import (
    ConstraintMonitor,
    evaluate_policy,
    get_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train constrained RL agents")
    
    # Environment arguments
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of evaluation episodes")
    
    # Algorithm arguments
    parser.add_argument("--algorithm", type=str, default="constrained_q", 
                       choices=["constrained_q", "lagrangian_ppo"], help="Algorithm to use")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="Discount factor")
    
    # Constraint arguments
    parser.add_argument("--constraint_threshold", type=float, default=0.5, 
                       help="Constraint violation threshold")
    parser.add_argument("--constraint_type", type=str, default="position",
                       choices=["position", "velocity", "combined"], help="Type of constraint")
    parser.add_argument("--penalty_weight", type=float, default=2.0, 
                       help="Weight for constraint penalty")
    
    # Training arguments
    parser.add_argument("--save_freq", type=int, default=100, help="Save frequency")
    parser.add_argument("--eval_freq", type=int, default=50, help="Evaluation frequency")
    parser.add_argument("--log_freq", type=int, default=10, help="Logging frequency")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from file.
    
    Args:
        config_path: Path to config file.
        
    Returns:
        Configuration dictionary.
    """
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
        return OmegaConf.load(config_path)
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")


def create_environment(env_name: str, constraint_config: Dict) -> gym.Env:
    """Create and configure the environment.
    
    Args:
        env_name: Name of the environment.
        constraint_config: Constraint configuration.
        
    Returns:
        Configured environment.
    """
    env = gym.make(env_name)
    
    # Apply constraint wrapper
    if constraint_config.get("use_constraint_wrapper", True):
        env = ConstrainedEnvWrapper(
            env,
            constraint_threshold=constraint_config["threshold"],
            constraint_type=constraint_config["type"],
            constraint_penalty=constraint_config["penalty"],
        )
    
    # Apply safety wrapper if specified
    if constraint_config.get("use_safety_wrapper", False):
        env = SafetyConstraintWrapper(
            env,
            safety_threshold=constraint_config["threshold"],
            safety_type=constraint_config["type"],
            violation_penalty=constraint_config["penalty"],
        )
    
    return env


def create_agent(env: gym.Env, algorithm: str, config: Dict) -> object:
    """Create the RL agent.
    
    Args:
        env: Environment.
        algorithm: Algorithm name.
        config: Agent configuration.
        
    Returns:
        RL agent.
    """
    device = get_device()
    
    if algorithm == "constrained_q":
        return ConstrainedQLearningAgent(
            state_space=env.observation_space,
            action_space=env.action_space,
            learning_rate=config["learning_rate"],
            discount_factor=config["discount_factor"],
            constraint_threshold=config["constraint_threshold"],
            penalty_weight=config["penalty_weight"],
            constraint_type=config["constraint_type"],
            device=device,
        )
    elif algorithm == "lagrangian_ppo":
        return LagrangianPPO(
            state_space=env.observation_space,
            action_space=env.action_space,
            learning_rate=config["learning_rate"],
            constraint_threshold=config["constraint_threshold"],
            constraint_type=config["constraint_type"],
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def train_agent(
    agent: object,
    env: gym.Env,
    num_episodes: int,
    eval_freq: int,
    log_freq: int,
    save_freq: int,
    output_dir: str,
    algorithm: str,
) -> List[Dict]:
    """Train the RL agent.
    
    Args:
        agent: RL agent.
        env: Environment.
        num_episodes: Number of training episodes.
        eval_freq: Evaluation frequency.
        log_freq: Logging frequency.
        save_freq: Save frequency.
        output_dir: Output directory.
        algorithm: Algorithm name.
        
    Returns:
        List of training metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    training_metrics = []
    constraint_monitor = ConstraintMonitor()
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Store data for PPO
        if algorithm == "lagrangian_ppo":
            states, actions, rewards, values, log_probs, dones, next_values = [], [], [], [], [], [], []
        
        while not done:
            if algorithm == "constrained_q":
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
                
            elif algorithm == "lagrangian_ppo":
                action, log_prob = agent.get_action(obs)
                value = agent.get_value(obs)
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                next_value = agent.get_value(next_obs) if not done else 0
                
                # Store experience
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                dones.append(done)
                next_values.append(next_value)
                
                # Track constraint violations
                if 'constraint_violation' in info:
                    constraint_monitor.update(info['constraint_violation'])
                
                obs = next_obs
                episode_reward += reward
                episode_length += 1
        
        # Update PPO agent
        if algorithm == "lagrangian_ppo" and len(states) > 0:
            update_metrics = agent.update(states, actions, rewards, values, log_probs, dones, next_values)
            training_metrics.append(update_metrics)
        
        # Logging
        if episode % log_freq == 0:
            constraint_stats = constraint_monitor.get_stats()
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Length={episode_length}, "
                  f"Violations={constraint_stats['total_violations']}, "
                  f"Violation Rate={constraint_stats['violation_rate']:.3f}")
        
        # Evaluation
        if episode % eval_freq == 0:
            eval_metrics = evaluate_policy(env, agent, num_episodes=5)
            print(f"Evaluation - Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}, "
                  f"Mean Violations: {eval_metrics['mean_violations']:.2f}")
        
        # Save model
        if episode % save_freq == 0:
            model_path = os.path.join(output_dir, f"{algorithm}_episode_{episode}.pt")
            agent.save(model_path)
    
    return training_metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override with command line arguments
    config.update({
        "env": args.env,
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "eval_episodes": args.eval_episodes,
        "algorithm": args.algorithm,
        "learning_rate": args.learning_rate,
        "discount_factor": args.discount_factor,
        "constraint_threshold": args.constraint_threshold,
        "constraint_type": args.constraint_type,
        "penalty_weight": args.penalty_weight,
        "save_freq": args.save_freq,
        "eval_freq": args.eval_freq,
        "log_freq": args.log_freq,
        "output_dir": args.output_dir,
    })
    
    # Set random seed
    set_seed(config["seed"])
    
    # Create environment
    constraint_config = {
        "threshold": config["constraint_threshold"],
        "type": config["constraint_type"],
        "penalty": config["penalty_weight"],
        "use_constraint_wrapper": True,
        "use_safety_wrapper": False,
    }
    
    env = create_environment(config["env"], constraint_config)
    
    # Create agent
    agent = create_agent(env, config["algorithm"], config)
    
    # Train agent
    print(f"Training {config['algorithm']} agent on {config['env']}")
    print(f"Constraint: {config['constraint_type']} < {config['constraint_threshold']}")
    print(f"Penalty weight: {config['penalty_weight']}")
    
    training_metrics = train_agent(
        agent=agent,
        env=env,
        num_episodes=config["num_episodes"],
        eval_freq=config["eval_freq"],
        log_freq=config["log_freq"],
        save_freq=config["save_freq"],
        output_dir=config["output_dir"],
        algorithm=config["algorithm"],
    )
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_eval = evaluate_policy(env, agent, num_episodes=config["eval_episodes"], render=args.render)
    print(f"Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"Mean Length: {final_eval['mean_length']:.2f} ± {final_eval['std_length']:.2f}")
    print(f"Mean Violations: {final_eval['mean_violations']:.2f} ± {final_eval['std_violations']:.2f}")
    
    # Save final model
    final_model_path = os.path.join(config["output_dir"], f"{config['algorithm']}_final.pt")
    agent.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    env.close()


if __name__ == "__main__":
    main()
