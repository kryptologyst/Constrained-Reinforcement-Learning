"""Constrained environment wrapper for safety constraints."""

from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ConstrainedEnvWrapper(gym.Wrapper):
    """Wrapper that adds constraint monitoring to any Gymnasium environment.
    
    This wrapper tracks constraint violations and provides constraint-related
    information in the info dictionary.
    """
    
    def __init__(
        self,
        env: gym.Env,
        constraint_threshold: float = 0.5,
        constraint_type: str = "position",
        constraint_penalty: float = 1.0,
        max_constraint_violations: int = 100,
    ):
        """Initialize the constrained environment wrapper.
        
        Args:
            env: Base Gymnasium environment.
            constraint_threshold: Threshold for constraint violation.
            constraint_type: Type of constraint to monitor.
            constraint_penalty: Penalty for constraint violations.
            max_constraint_violations: Maximum violations before episode termination.
        """
        super().__init__(env)
        self.constraint_threshold = constraint_threshold
        self.constraint_type = constraint_type
        self.constraint_penalty = constraint_penalty
        self.max_constraint_violations = max_constraint_violations
        
        # Constraint tracking
        self.constraint_violations = 0
        self.total_constraint_cost = 0.0
        self.episode_constraint_violations = 0
        
        # Modify info dict to include constraint information
        self._modify_info_space()
    
    def _modify_info_space(self) -> None:
        """Modify the info space to include constraint information."""
        if hasattr(self.env, 'info_space'):
            # Add constraint fields to info space
            constraint_info = {
                'constraint_violation': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
                'constraint_cost': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
                'total_violations': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int32),
                'constraint_satisfied': spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
            }
            
            if isinstance(self.env.info_space, spaces.Dict):
                self.info_space = spaces.Dict({
                    **self.env.info_space.spaces,
                    **constraint_info,
                })
            else:
                self.info_space = spaces.Dict(constraint_info)
    
    def _compute_constraint_violation(self, state: np.ndarray) -> float:
        """Compute constraint violation for the current state.
        
        Args:
            state: Current state.
            
        Returns:
            Constraint violation value.
        """
        if self.constraint_type == "position":
            return max(0, abs(state[0]) - self.constraint_threshold)
        elif self.constraint_type == "velocity":
            return max(0, abs(state[1]) - self.constraint_threshold)
        elif self.constraint_type == "combined":
            return max(0, abs(state[0]) + abs(state[1]) - self.constraint_threshold)
        elif self.constraint_type == "energy":
            # For systems with energy-like quantities
            return max(0, np.sum(state**2) - self.constraint_threshold)
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and constraint tracking.
        
        Args:
            **kwargs: Additional arguments for reset.
            
        Returns:
            Tuple of (observation, info).
        """
        obs, info = self.env.reset(**kwargs)
        
        # Reset constraint tracking
        self.constraint_violations = 0
        self.total_constraint_cost = 0.0
        self.episode_constraint_violations = 0
        
        # Add constraint information to info
        info.update(self._get_constraint_info(obs))
        
        return obs, info
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment and update constraint tracking.
        
        Args:
            action: Action to take.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Compute constraint violation
        constraint_violation = self._compute_constraint_violation(obs)
        
        # Update constraint tracking
        if constraint_violation > 0:
            self.constraint_violations += 1
            self.episode_constraint_violations += 1
        
        self.total_constraint_cost += constraint_violation
        
        # Apply constraint penalty to reward
        if constraint_violation > 0:
            reward -= self.constraint_penalty * constraint_violation
        
        # Check if maximum violations exceeded
        if self.episode_constraint_violations >= self.max_constraint_violations:
            terminated = True
        
        # Add constraint information to info
        info.update(self._get_constraint_info(obs))
        
        return obs, reward, terminated, truncated, info
    
    def _get_constraint_info(self, state: np.ndarray) -> Dict[str, Any]:
        """Get constraint-related information.
        
        Args:
            state: Current state.
            
        Returns:
            Dictionary containing constraint information.
        """
        constraint_violation = self._compute_constraint_violation(state)
        
        return {
            'constraint_violation': constraint_violation,
            'constraint_cost': self.total_constraint_cost,
            'total_violations': self.episode_constraint_violations,
            'constraint_satisfied': constraint_violation == 0,
        }
    
    def get_constraint_stats(self) -> Dict[str, float]:
        """Get constraint violation statistics.
        
        Returns:
            Dictionary containing constraint statistics.
        """
        return {
            'total_violations': self.episode_constraint_violations,
            'total_cost': self.total_constraint_cost,
            'violation_rate': self.episode_constraint_violations / max(1, self.env.unwrapped.steps),
        }


class SafetyConstraintWrapper(gym.Wrapper):
    """Wrapper that enforces safety constraints by terminating episodes.
    
    This wrapper terminates episodes when safety constraints are violated,
    providing a hard constraint enforcement mechanism.
    """
    
    def __init__(
        self,
        env: gym.Env,
        safety_threshold: float = 0.5,
        safety_type: str = "position",
        violation_penalty: float = -10.0,
    ):
        """Initialize the safety constraint wrapper.
        
        Args:
            env: Base Gymnasium environment.
            safety_threshold: Threshold for safety violation.
            safety_type: Type of safety constraint.
            violation_penalty: Penalty for safety violations.
        """
        super().__init__(env)
        self.safety_threshold = safety_threshold
        self.safety_type = safety_type
        self.violation_penalty = violation_penalty
    
    def _check_safety_violation(self, state: np.ndarray) -> bool:
        """Check if safety constraint is violated.
        
        Args:
            state: Current state.
            
        Returns:
            True if safety constraint is violated.
        """
        if self.safety_type == "position":
            return abs(state[0]) > self.safety_threshold
        elif self.safety_type == "velocity":
            return abs(state[1]) > self.safety_threshold
        elif self.safety_type == "combined":
            return abs(state[0]) + abs(state[1]) > self.safety_threshold
        else:
            raise ValueError(f"Unknown safety type: {self.safety_type}")
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment with safety constraint enforcement.
        
        Args:
            action: Action to take.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check safety constraint
        safety_violated = self._check_safety_violation(obs)
        
        if safety_violated:
            terminated = True
            reward += self.violation_penalty
            info['safety_violation'] = True
        else:
            info['safety_violation'] = False
        
        return obs, reward, terminated, truncated, info


class ConstraintCurriculumWrapper(gym.Wrapper):
    """Wrapper that implements curriculum learning for constraints.
    
    This wrapper gradually increases constraint difficulty during training,
    helping agents learn to satisfy constraints progressively.
    """
    
    def __init__(
        self,
        env: gym.Env,
        initial_threshold: float = 1.0,
        final_threshold: float = 0.5,
        curriculum_steps: int = 10000,
        constraint_type: str = "position",
    ):
        """Initialize the constraint curriculum wrapper.
        
        Args:
            env: Base Gymnasium environment.
            initial_threshold: Initial constraint threshold.
            final_threshold: Final constraint threshold.
            curriculum_steps: Number of steps to complete curriculum.
            constraint_type: Type of constraint.
        """
        super().__init__(env)
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.curriculum_steps = curriculum_steps
        self.constraint_type = constraint_type
        self.current_step = 0
        
        # Initialize constraint threshold
        self.constraint_threshold = initial_threshold
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment and update curriculum.
        
        Args:
            action: Action to take.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update curriculum
        self.current_step += 1
        progress = min(1.0, self.current_step / self.curriculum_steps)
        
        # Linearly interpolate threshold
        self.constraint_threshold = (
            self.initial_threshold * (1 - progress) + self.final_threshold * progress
        )
        
        # Add curriculum information to info
        info['curriculum_progress'] = progress
        info['constraint_threshold'] = self.constraint_threshold
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.
        
        Args:
            **kwargs: Additional arguments for reset.
            
        Returns:
            Tuple of (observation, info).
        """
        obs, info = self.env.reset(**kwargs)
        info['constraint_threshold'] = self.constraint_threshold
        return obs, info
