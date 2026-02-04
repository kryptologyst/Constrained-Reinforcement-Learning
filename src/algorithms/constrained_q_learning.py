"""Constrained Q-Learning agent implementation."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

from ..utils import compute_constraint_violation, safe_reward_shaping


class ConstrainedQLearningAgent:
    """Q-Learning agent with constraint handling via reward penalties.
    
    This agent implements constrained reinforcement learning by applying
    penalty-based reward shaping when constraints are violated.
    """
    
    def __init__(
        self,
        state_space: spaces.Space,
        action_space: spaces.Space,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        constraint_threshold: float = 0.5,
        penalty_weight: float = 2.0,
        constraint_type: str = "position",
        device: Optional[torch.device] = None,
    ):
        """Initialize the constrained Q-learning agent.
        
        Args:
            state_space: State space of the environment.
            action_space: Action space of the environment.
            learning_rate: Learning rate for Q-value updates.
            discount_factor: Discount factor for future rewards.
            epsilon: Initial exploration rate.
            epsilon_decay: Decay rate for exploration.
            epsilon_min: Minimum exploration rate.
            constraint_threshold: Threshold for constraint violation.
            penalty_weight: Weight for constraint penalty.
            constraint_type: Type of constraint to enforce.
            device: PyTorch device for computations.
        """
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.constraint_threshold = constraint_threshold
        self.penalty_weight = penalty_weight
        self.constraint_type = constraint_type
        self.device = device or torch.device("cpu")
        
        # Q-table for discrete state-action spaces
        if isinstance(state_space, spaces.Discrete) and isinstance(action_space, spaces.Discrete):
            self.q_table = np.zeros((state_space.n, action_space.n))
            self.use_neural_network = False
        else:
            # Use neural network for continuous or large discrete spaces
            self.use_neural_network = True
            self.q_network = self._build_q_network()
            self.target_network = self._build_q_network()
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.update_target_network()
    
    def _build_q_network(self) -> nn.Module:
        """Build the Q-network for continuous state spaces.
        
        Returns:
            PyTorch neural network.
        """
        if isinstance(self.state_space, spaces.Box):
            input_dim = self.state_space.shape[0]
        elif isinstance(self.state_space, spaces.Discrete):
            input_dim = self.state_space.n
        else:
            raise ValueError(f"Unsupported state space: {self.state_space}")
        
        if isinstance(self.action_space, spaces.Discrete):
            output_dim = self.action_space.n
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}")
        
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        ).to(self.device)
    
    def update_target_network(self) -> None:
        """Update the target network with current network weights."""
        if self.use_neural_network:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state.
            training: Whether the agent is in training mode.
            
        Returns:
            Selected action.
        """
        if not training:
            # Exploitation only
            if self.use_neural_network:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    return q_values.argmax().item()
            else:
                # For discrete states, convert to index
                if isinstance(self.state_space, spaces.Discrete):
                    state_idx = state
                else:
                    # For continuous states, discretize or use neural network
                    raise NotImplementedError("Continuous states require neural network")
                
                return np.argmax(self.q_table[state_idx])
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            if self.use_neural_network:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    return q_values.argmax().item()
            else:
                if isinstance(self.state_space, spaces.Discrete):
                    state_idx = state
                else:
                    raise NotImplementedError("Continuous states require neural network")
                
                return np.argmax(self.q_table[state_idx])
    
    def update_q_value(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Update Q-values using constrained Q-learning.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.
        """
        # Compute constraint violation
        constraint_violation = compute_constraint_violation(
            state, self.constraint_threshold, self.constraint_type
        )
        
        # Apply safe reward shaping
        shaped_reward = safe_reward_shaping(reward, constraint_violation, self.penalty_weight)
        
        if self.use_neural_network:
            self._update_neural_network(state, action, shaped_reward, next_state, done)
        else:
            self._update_q_table(state, action, shaped_reward, next_state, done)
        
        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _update_q_table(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Update Q-table for discrete state-action spaces.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Shaped reward.
            next_state: Next state.
            done: Whether episode is done.
        """
        if isinstance(self.state_space, spaces.Discrete):
            state_idx = state
            next_state_idx = next_state
        else:
            raise NotImplementedError("Continuous states require neural network")
        
        # Q-learning update
        current_q = self.q_table[state_idx, action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_idx])
            target_q = reward + self.discount_factor * max_next_q
        
        self.q_table[state_idx, action] = current_q + self.learning_rate * (target_q - current_q)
    
    def _update_neural_network(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Update neural network using Q-learning.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Shaped reward.
            next_state: Next state.
            done: Whether episode is done.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        # Current Q-value
        current_q = self.q_network(state_tensor)[0, action]
        
        # Target Q-value
        with torch.no_grad():
            if done:
                target_q = torch.tensor(reward, device=self.device)
            else:
                next_q_values = self.target_network(next_state_tensor)
                target_q = reward + self.discount_factor * next_q_values.max()
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save(self, filepath: str) -> None:
        """Save the agent's parameters.
        
        Args:
            filepath: Path to save the model.
        """
        if self.use_neural_network:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
            }, filepath)
        else:
            np.save(filepath, self.q_table)
    
    def load(self, filepath: str) -> None:
        """Load the agent's parameters.
        
        Args:
            filepath: Path to load the model from.
        """
        if self.use_neural_network:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
        else:
            self.q_table = np.load(filepath)
