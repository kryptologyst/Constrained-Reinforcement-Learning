"""Lagrangian-based constrained reinforcement learning algorithms."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

from ..utils import compute_constraint_violation, get_device


class LagrangianPPO:
    """Proximal Policy Optimization with Lagrangian constraints.
    
    This implementation uses the Lagrangian method to handle constraints
    by adding constraint violations to the objective function.
    """
    
    def __init__(
        self,
        state_space: spaces.Space,
        action_space: spaces.Space,
        learning_rate: float = 3e-4,
        constraint_threshold: float = 0.5,
        constraint_type: str = "position",
        lagrangian_multiplier_lr: float = 1e-3,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        """Initialize Lagrangian PPO agent.
        
        Args:
            state_space: State space of the environment.
            action_space: Action space of the environment.
            learning_rate: Learning rate for policy and value networks.
            constraint_threshold: Threshold for constraint violation.
            constraint_type: Type of constraint to enforce.
            lagrangian_multiplier_lr: Learning rate for Lagrangian multiplier.
            clip_ratio: PPO clipping ratio.
            value_loss_coef: Coefficient for value function loss.
            entropy_coef: Coefficient for entropy bonus.
            max_grad_norm: Maximum gradient norm for clipping.
            device: PyTorch device for computations.
        """
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.constraint_threshold = constraint_threshold
        self.constraint_type = constraint_type
        self.lagrangian_multiplier_lr = lagrangian_multiplier_lr
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or get_device()
        
        # Initialize Lagrangian multiplier
        self.lagrangian_multiplier = torch.tensor(1.0, device=self.device, requires_grad=True)
        
        # Build networks
        self.policy_network = self._build_policy_network()
        self.value_network = self._build_value_network()
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        self.lagrangian_optimizer = optim.Adam([self.lagrangian_multiplier], lr=lagrangian_multiplier_lr)
    
    def _build_policy_network(self) -> nn.Module:
        """Build the policy network.
        
        Returns:
            Policy network.
        """
        if isinstance(self.state_space, spaces.Box):
            input_dim = self.state_space.shape[0]
        else:
            raise ValueError("Only continuous state spaces supported")
        
        if isinstance(self.action_space, spaces.Box):
            output_dim = self.action_space.shape[0]
        else:
            raise ValueError("Only continuous action spaces supported")
        
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh(),  # Assuming actions are in [-1, 1]
        ).to(self.device)
    
    def _build_value_network(self) -> nn.Module:
        """Build the value network.
        
        Returns:
            Value network.
        """
        if isinstance(self.state_space, spaces.Box):
            input_dim = self.state_space.shape[0]
        else:
            raise ValueError("Only continuous state spaces supported")
        
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Get action from policy.
        
        Args:
            state: Current state.
            deterministic: Whether to use deterministic policy.
            
        Returns:
            Tuple of (action, log_probability).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean = self.policy_network(state_tensor)
            
            if deterministic:
                action = action_mean
            else:
                # Add noise for exploration (simplified)
                action_std = torch.ones_like(action_mean) * 0.1
                action = action_mean + action_std * torch.randn_like(action_mean)
            
            # Compute log probability (simplified)
            log_prob = -0.5 * ((action - action_mean) / 0.1).pow(2).sum(dim=-1)
        
        return action.squeeze(0).cpu().numpy(), log_prob.item()
    
    def get_value(self, state: np.ndarray) -> float:
        """Get state value.
        
        Args:
            state: Current state.
            
        Returns:
            State value.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.value_network(state_tensor)
        
        return value.item()
    
    def update(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        values: List[float],
        log_probs: List[float],
        dones: List[bool],
        next_values: List[float],
    ) -> Dict[str, float]:
        """Update the policy and value networks.
        
        Args:
            states: List of states.
            actions: List of actions.
            rewards: List of rewards.
            values: List of state values.
            log_probs: List of log probabilities.
            dones: List of done flags.
            next_values: List of next state values.
            
        Returns:
            Dictionary containing loss information.
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        values_tensor = torch.FloatTensor(values).to(self.device)
        log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        next_values_tensor = torch.FloatTensor(next_values).to(self.device)
        
        # Compute advantages and returns
        advantages = self._compute_advantages(
            rewards_tensor, values_tensor, next_values_tensor, dones_tensor
        )
        returns = advantages + values_tensor
        
        # Compute constraint violations
        constraint_violations = []
        for state in states:
            violation = compute_constraint_violation(
                state, self.constraint_threshold, self.constraint_type
            )
            constraint_violations.append(violation)
        
        constraint_violations_tensor = torch.FloatTensor(constraint_violations).to(self.device)
        
        # Update policy network
        policy_loss = self._compute_policy_loss(
            states_tensor, actions_tensor, log_probs_tensor, advantages, constraint_violations_tensor
        )
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        
        # Update value network
        value_loss = self._compute_value_loss(states_tensor, returns)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        
        # Update Lagrangian multiplier
        lagrangian_loss = self._compute_lagrangian_loss(constraint_violations_tensor)
        
        self.lagrangian_optimizer.zero_grad()
        lagrangian_loss.backward()
        self.lagrangian_optimizer.step()
        
        # Ensure Lagrangian multiplier is non-negative
        with torch.no_grad():
            self.lagrangian_multiplier.clamp_(min=0.0)
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'lagrangian_loss': lagrangian_loss.item(),
            'lagrangian_multiplier': self.lagrangian_multiplier.item(),
            'mean_constraint_violation': constraint_violations_tensor.mean().item(),
        }
    
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> torch.Tensor:
        """Compute generalized advantage estimation.
        
        Args:
            rewards: Reward tensor.
            values: Value tensor.
            next_values: Next value tensor.
            dones: Done flag tensor.
            gamma: Discount factor.
            lam: GAE lambda parameter.
            
        Returns:
            Advantage tensor.
        """
        advantages = []
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t] if not dones[t] else 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantage = delta + gamma * lam * advantage * (1 - dones[t])
            advantages.insert(0, advantage)
        
        return torch.stack(advantages)
    
    def _compute_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        constraint_violations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute policy loss with Lagrangian constraints.
        
        Args:
            states: State tensor.
            actions: Action tensor.
            old_log_probs: Old log probability tensor.
            advantages: Advantage tensor.
            constraint_violations: Constraint violation tensor.
            
        Returns:
            Policy loss.
        """
        # Get current policy log probabilities
        action_means = self.policy_network(states)
        action_stds = torch.ones_like(action_means) * 0.1
        
        # Compute log probabilities
        log_probs = -0.5 * ((actions - action_means) / action_stds).pow(2).sum(dim=-1)
        
        # Compute probability ratios
        ratios = torch.exp(log_probs - old_log_probs)
        
        # Compute clipped surrogate loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Add Lagrangian constraint term
        constraint_term = self.lagrangian_multiplier * constraint_violations.mean()
        
        return policy_loss + constraint_term
    
    def _compute_value_loss(
        self,
        states: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """Compute value function loss.
        
        Args:
            states: State tensor.
            returns: Return tensor.
            
        Returns:
            Value loss.
        """
        values = self.value_network(states).squeeze()
        return nn.MSELoss()(values, returns)
    
    def _compute_lagrangian_loss(self, constraint_violations: torch.Tensor) -> torch.Tensor:
        """Compute Lagrangian multiplier loss.
        
        Args:
            constraint_violations: Constraint violation tensor.
            
        Returns:
            Lagrangian loss.
        """
        return -self.lagrangian_multiplier * constraint_violations.mean()
    
    def save(self, filepath: str) -> None:
        """Save the agent's parameters.
        
        Args:
            filepath: Path to save the model.
        """
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'lagrangian_optimizer_state_dict': self.lagrangian_optimizer.state_dict(),
            'lagrangian_multiplier': self.lagrangian_multiplier.item(),
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's parameters.
        
        Args:
            filepath: Path to load the model from.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.lagrangian_optimizer.load_state_dict(checkpoint['lagrangian_optimizer_state_dict'])
        self.lagrangian_multiplier = torch.tensor(
            checkpoint['lagrangian_multiplier'], device=self.device, requires_grad=True
        )
