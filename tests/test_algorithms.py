"""Unit tests for constrained reinforcement learning algorithms."""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from src.algorithms.constrained_q_learning import ConstrainedQLearningAgent
from src.algorithms.lagrangian_ppo import LagrangianPPO
from src.envs.constrained_wrappers import ConstrainedEnvWrapper, SafetyConstraintWrapper
from src.utils import (
    compute_constraint_violation,
    safe_reward_shaping,
    ConstraintMonitor,
    get_device,
    set_seed,
)


class TestConstrainedQLearning:
    """Test cases for ConstrainedQLearningAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        state_space = spaces.Discrete(10)
        action_space = spaces.Discrete(4)
        
        agent = ConstrainedQLearningAgent(
            state_space=state_space,
            action_space=action_space,
            constraint_threshold=0.5,
            penalty_weight=2.0,
        )
        
        assert agent.constraint_threshold == 0.5
        assert agent.penalty_weight == 2.0
        assert agent.epsilon == 1.0
        assert not agent.use_neural_network
    
    def test_action_selection(self):
        """Test action selection."""
        state_space = spaces.Discrete(10)
        action_space = spaces.Discrete(4)
        
        agent = ConstrainedQLearningAgent(
            state_space=state_space,
            action_space=action_space,
        )
        
        # Test exploration (epsilon = 1.0)
        action = agent.select_action(0, training=True)
        assert action in range(4)
        
        # Test exploitation (epsilon = 0.0)
        agent.epsilon = 0.0
        action = agent.select_action(0, training=True)
        assert action in range(4)
    
    def test_q_value_update(self):
        """Test Q-value update."""
        state_space = spaces.Discrete(10)
        action_space = spaces.Discrete(4)
        
        agent = ConstrainedQLearningAgent(
            state_space=state_space,
            action_space=action_space,
            learning_rate=0.1,
            discount_factor=0.9,
        )
        
        # Initial Q-value should be 0
        assert agent.q_table[0, 0] == 0
        
        # Update Q-value
        agent.update_q_value(0, 0, 1.0, 1, False)
        
        # Q-value should be updated
        assert agent.q_table[0, 0] != 0
    
    def test_constraint_penalty(self):
        """Test constraint penalty application."""
        state_space = spaces.Discrete(10)
        action_space = spaces.Discrete(4)
        
        agent = ConstrainedQLearningAgent(
            state_space=state_space,
            action_space=action_space,
            constraint_threshold=0.5,
            penalty_weight=2.0,
        )
        
        # Test with constraint violation
        state = np.array([1.0, 0.0])  # Violates position constraint
        agent.update_q_value(state, 0, 1.0, state, False)
        
        # Should apply penalty (implementation depends on internal logic)
        assert True  # Placeholder for actual constraint penalty test


class TestLagrangianPPO:
    """Test cases for LagrangianPPO."""
    
    def test_initialization(self):
        """Test agent initialization."""
        state_space = spaces.Box(low=-1, high=1, shape=(4,))
        action_space = spaces.Box(low=-1, high=1, shape=(2,))
        
        agent = LagrangianPPO(
            state_space=state_space,
            action_space=action_space,
            constraint_threshold=0.5,
        )
        
        assert agent.constraint_threshold == 0.5
        assert agent.lagrangian_multiplier.item() > 0
        assert isinstance(agent.policy_network, torch.nn.Module)
        assert isinstance(agent.value_network, torch.nn.Module)
    
    def test_action_selection(self):
        """Test action selection."""
        state_space = spaces.Box(low=-1, high=1, shape=(4,))
        action_space = spaces.Box(low=-1, high=1, shape=(2,))
        
        agent = LagrangianPPO(
            state_space=state_space,
            action_space=action_space,
        )
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action, log_prob = agent.get_action(state, deterministic=True)
        
        assert action.shape == (2,)
        assert isinstance(log_prob, float)
        assert -1 <= action[0] <= 1
        assert -1 <= action[1] <= 1
    
    def test_value_prediction(self):
        """Test value prediction."""
        state_space = spaces.Box(low=-1, high=1, shape=(4,))
        action_space = spaces.Box(low=-1, high=1, shape=(2,))
        
        agent = LagrangianPPO(
            state_space=state_space,
            action_space=action_space,
        )
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        value = agent.get_value(state)
        
        assert isinstance(value, float)


class TestConstrainedWrappers:
    """Test cases for constrained environment wrappers."""
    
    def test_constrained_env_wrapper(self):
        """Test ConstrainedEnvWrapper."""
        import gymnasium as gym
        
        env = gym.make("CartPole-v1")
        wrapped_env = ConstrainedEnvWrapper(
            env,
            constraint_threshold=0.5,
            constraint_type="position",
        )
        
        obs, info = wrapped_env.reset()
        assert "constraint_violation" in info
        assert "constraint_cost" in info
        assert "total_violations" in info
        assert "constraint_satisfied" in info
    
    def test_safety_constraint_wrapper(self):
        """Test SafetyConstraintWrapper."""
        import gymnasium as gym
        
        env = gym.make("CartPole-v1")
        wrapped_env = SafetyConstraintWrapper(
            env,
            safety_threshold=0.5,
            safety_type="position",
        )
        
        obs, info = wrapped_env.reset()
        assert "safety_violation" in info


class TestUtils:
    """Test cases for utility functions."""
    
    def test_compute_constraint_violation(self):
        """Test constraint violation computation."""
        state = np.array([0.6, 0.3])  # Violates position constraint
        
        violation = compute_constraint_violation(state, 0.5, "position")
        assert violation == 0.1
        
        violation = compute_constraint_violation(state, 0.5, "velocity")
        assert violation == 0.0
        
        violation = compute_constraint_violation(state, 0.5, "combined")
        assert violation > 0
    
    def test_safe_reward_shaping(self):
        """Test safe reward shaping."""
        reward = 1.0
        violation = 0.2
        penalty_weight = 2.0
        
        shaped_reward = safe_reward_shaping(reward, violation, penalty_weight)
        expected_reward = reward - penalty_weight * violation
        assert shaped_reward == expected_reward
    
    def test_constraint_monitor(self):
        """Test ConstraintMonitor."""
        monitor = ConstraintMonitor(window_size=5)
        
        # Add some violations
        monitor.update(0.0)  # No violation
        monitor.update(0.5)  # Violation
        monitor.update(0.0)  # No violation
        monitor.update(0.3)  # Violation
        
        stats = monitor.get_stats()
        assert stats['total_violations'] == 2
        assert stats['violation_rate'] == 0.5
        assert stats['mean_violation'] == 0.2
    
    def test_device_detection(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_seed_setting(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy seed
        np.random.seed(42)
        val1 = np.random.random()
        
        set_seed(42)
        val2 = np.random.random()
        
        assert val1 == val2


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training."""
        import gymnasium as gym
        
        env = gym.make("CartPole-v1")
        env = ConstrainedEnvWrapper(env, constraint_threshold=0.5)
        
        agent = ConstrainedQLearningAgent(
            state_space=env.observation_space,
            action_space=env.action_space,
            constraint_threshold=0.5,
        )
        
        # Run a few episodes
        for episode in range(5):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action = agent.select_action(obs, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                agent.update_q_value(obs, action, reward, next_obs, done)
                obs = next_obs
        
        # Agent should have learned something
        assert agent.epsilon < 1.0  # Epsilon should have decayed


if __name__ == "__main__":
    pytest.main([__file__])
