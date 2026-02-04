"""Utility functions for constrained reinforcement learning."""

import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import gymnasium as gym


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_env(env_name: str, seed: Optional[int] = None) -> gym.Env:
    """Create and configure a Gymnasium environment.
    
    Args:
        env_name: Name of the environment.
        seed: Random seed for the environment.
        
    Returns:
        Configured Gymnasium environment.
    """
    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=seed)
    return env


def evaluate_policy(
    env: gym.Env,
    policy,
    num_episodes: int = 10,
    render: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate a policy on an environment.
    
    Args:
        env: Gymnasium environment.
        policy: Policy to evaluate.
        num_episodes: Number of episodes to run.
        render: Whether to render the environment.
        seed: Random seed for evaluation.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    if seed is not None:
        env.reset(seed=seed)
    
    episode_rewards = []
    episode_lengths = []
    constraint_violations = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        violations = 0
        
        done = False
        while not done:
            if hasattr(policy, 'predict'):
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = policy.select_action(obs, training=False)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Check for constraint violations
            if 'constraint_violation' in info:
                violations += info['constraint_violation']
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        constraint_violations.append(violations)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_violations': np.mean(constraint_violations),
        'std_violations': np.std(constraint_violations),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'constraint_violations': constraint_violations,
    }


def compute_constraint_violation(
    state: np.ndarray,
    constraint_threshold: float,
    constraint_type: str = "position",
) -> float:
    """Compute constraint violation for a given state.
    
    Args:
        state: Current state.
        constraint_threshold: Threshold for constraint violation.
        constraint_type: Type of constraint to check.
        
    Returns:
        Constraint violation value (0 if satisfied, >0 if violated).
    """
    if constraint_type == "position":
        return max(0, abs(state[0]) - constraint_threshold)
    elif constraint_type == "velocity":
        return max(0, abs(state[1]) - constraint_threshold)
    elif constraint_type == "combined":
        return max(0, abs(state[0]) + abs(state[1]) - constraint_threshold)
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")


def safe_reward_shaping(
    reward: float,
    constraint_violation: float,
    penalty_weight: float = 1.0,
) -> float:
    """Apply safe reward shaping based on constraint violations.
    
    Args:
        reward: Original reward.
        constraint_violation: Constraint violation value.
        penalty_weight: Weight for penalty.
        
    Returns:
        Shaped reward.
    """
    return reward - penalty_weight * constraint_violation


class ConstraintMonitor:
    """Monitor constraint violations during training."""
    
    def __init__(self, window_size: int = 100):
        """Initialize constraint monitor.
        
        Args:
            window_size: Size of the sliding window for statistics.
        """
        self.window_size = window_size
        self.violations = []
        self.total_steps = 0
        self.violation_steps = 0
    
    def update(self, violation: float) -> None:
        """Update monitor with new violation.
        
        Args:
            violation: Constraint violation value.
        """
        self.violations.append(violation)
        self.total_steps += 1
        
        if violation > 0:
            self.violation_steps += 1
        
        # Keep only recent violations
        if len(self.violations) > self.window_size:
            self.violations.pop(0)
    
    def get_stats(self) -> Dict[str, float]:
        """Get constraint violation statistics.
        
        Returns:
            Dictionary containing violation statistics.
        """
        if not self.violations:
            return {
                'violation_rate': 0.0,
                'mean_violation': 0.0,
                'max_violation': 0.0,
                'total_violations': 0,
            }
        
        return {
            'violation_rate': self.violation_steps / self.total_steps,
            'mean_violation': np.mean(self.violations),
            'max_violation': np.max(self.violations),
            'total_violations': self.violation_steps,
        }
    
    def reset(self) -> None:
        """Reset monitor statistics."""
        self.violations = []
        self.total_steps = 0
        self.violation_steps = 0
