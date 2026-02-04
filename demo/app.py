"""Streamlit demo for constrained reinforcement learning."""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.algorithms.constrained_q_learning import ConstrainedQLearningAgent
from src.algorithms.lagrangian_ppo import LagrangianPPO
from src.envs.constrained_wrappers import ConstrainedEnvWrapper
from src.utils import evaluate_policy, get_device, set_seed


def load_agent(algorithm: str, model_path: str, env) -> object:
    """Load a trained agent.
    
    Args:
        algorithm: Algorithm name.
        model_path: Path to model file.
        env: Environment.
        
    Returns:
        Loaded agent.
    """
    device = get_device()
    
    if algorithm == "constrained_q":
        agent = ConstrainedQLearningAgent(
            state_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
        agent.load(model_path)
        return agent
    elif algorithm == "lagrangian_ppo":
        agent = LagrangianPPO(
            state_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
        agent.load(model_path)
        return agent
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def create_environment(env_name: str, constraint_config: Dict):
    """Create constrained environment.
    
    Args:
        env_name: Environment name.
        constraint_config: Constraint configuration.
        
    Returns:
        Environment.
    """
    import gymnasium as gym
    
    env = gym.make(env_name)
    env = ConstrainedEnvWrapper(
        env,
        constraint_threshold=constraint_config["threshold"],
        constraint_type=constraint_config["type"],
        constraint_penalty=constraint_config["penalty"],
    )
    return env


def run_episode(env, agent, algorithm: str, max_steps: int = 500) -> Dict:
    """Run a single episode and collect data.
    
    Args:
        env: Environment.
        agent: RL agent.
        algorithm: Algorithm name.
        max_steps: Maximum steps per episode.
        
    Returns:
        Episode data.
    """
    obs, _ = env.reset()
    done = False
    steps = 0
    
    states = []
    actions = []
    rewards = []
    constraint_violations = []
    values = []
    
    while not done and steps < max_steps:
        if algorithm == "constrained_q":
            action = agent.select_action(obs, training=False)
        elif algorithm == "lagrangian_ppo":
            action, _ = agent.get_action(obs, deterministic=True)
            value = agent.get_value(obs)
            values.append(value)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        states.append(obs.copy())
        actions.append(action)
        rewards.append(reward)
        
        if 'constraint_violation' in info:
            constraint_violations.append(info['constraint_violation'])
        else:
            constraint_violations.append(0)
        
        obs = next_obs
        steps += 1
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'constraint_violations': constraint_violations,
        'values': values,
        'total_reward': sum(rewards),
        'total_violations': sum(constraint_violations),
        'episode_length': steps,
    }


def plot_trajectory(episode_data: Dict, constraint_threshold: float) -> go.Figure:
    """Plot episode trajectory.
    
    Args:
        episode_data: Episode data.
        constraint_threshold: Constraint threshold.
        
    Returns:
        Plotly figure.
    """
    states = np.array(episode_data['states'])
    
    fig = go.Figure()
    
    # Plot trajectory
    fig.add_trace(go.Scatter(
        x=states[:, 0],
        y=states[:, 1] if states.shape[1] > 1 else np.zeros(len(states)),
        mode='lines+markers',
        name='Trajectory',
        line=dict(color='blue', width=2),
        marker=dict(size=4),
    ))
    
    # Plot constraint boundaries
    if states.shape[1] > 0:
        x_min, x_max = states[:, 0].min(), states[:, 0].max()
        fig.add_vline(x=constraint_threshold, line_dash="dash", line_color="red", 
                     annotation_text="Constraint Boundary")
        fig.add_vline(x=-constraint_threshold, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="Episode Trajectory",
        xaxis_title="Position",
        yaxis_title="Velocity" if states.shape[1] > 1 else "State",
        showlegend=True,
    )
    
    return fig


def plot_constraint_violations(episode_data: Dict) -> go.Figure:
    """Plot constraint violations over time.
    
    Args:
        episode_data: Episode data.
        
    Returns:
        Plotly figure.
    """
    violations = episode_data['constraint_violations']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(violations))),
        y=violations,
        mode='lines',
        name='Constraint Violations',
        line=dict(color='red', width=2),
    ))
    
    fig.update_layout(
        title="Constraint Violations Over Time",
        xaxis_title="Time Step",
        yaxis_title="Violation Magnitude",
        showlegend=True,
    )
    
    return fig


def plot_rewards(episode_data: Dict) -> go.Figure:
    """Plot rewards over time.
    
    Args:
        episode_data: Episode data.
        
    Returns:
        Plotly figure.
    """
    rewards = episode_data['rewards']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(rewards))),
        y=rewards,
        mode='lines',
        name='Rewards',
        line=dict(color='green', width=2),
    ))
    
    fig.update_layout(
        title="Rewards Over Time",
        xaxis_title="Time Step",
        yaxis_title="Reward",
        showlegend=True,
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Constrained RL Demo",
        page_icon="🤖",
        layout="wide",
    )
    
    st.title("Constrained Reinforcement Learning Demo")
    st.markdown("""
    This demo showcases constrained reinforcement learning algorithms that learn to maximize rewards 
    while satisfying safety constraints. The agents are trained to avoid violating position constraints 
    in the CartPole environment.
    
    **⚠️ DISCLAIMER: This is a research/educational demo. Not for production control of real systems.**
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Environment settings
    env_name = st.sidebar.selectbox(
        "Environment",
        ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"],
        index=0,
    )
    
    constraint_type = st.sidebar.selectbox(
        "Constraint Type",
        ["position", "velocity", "combined"],
        index=0,
    )
    
    constraint_threshold = st.sidebar.slider(
        "Constraint Threshold",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
    )
    
    penalty_weight = st.sidebar.slider(
        "Penalty Weight",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
    )
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["constrained_q", "lagrangian_ppo"],
        index=0,
    )
    
    # Model loading
    st.sidebar.header("Model")
    
    # Check if models exist
    output_dir = "./outputs"
    model_files = []
    if os.path.exists(output_dir):
        model_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
    
    if model_files:
        model_file = st.sidebar.selectbox(
            "Load Model",
            ["None"] + model_files,
            index=0,
        )
    else:
        model_file = "None"
        st.sidebar.warning("No trained models found. Train a model first using the training script.")
    
    # Create environment
    constraint_config = {
        "threshold": constraint_threshold,
        "type": constraint_type,
        "penalty": penalty_weight,
    }
    
    env = create_environment(env_name, constraint_config)
    
    # Load agent if model is selected
    agent = None
    if model_file != "None":
        try:
            model_path = os.path.join(output_dir, model_file)
            agent = load_agent(algorithm, model_path, env)
            st.sidebar.success(f"Loaded model: {model_file}")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
            agent = None
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Episode Visualization")
        
        if agent is not None:
            if st.button("Run Episode", type="primary"):
                with st.spinner("Running episode..."):
                    episode_data = run_episode(env, agent, algorithm)
                
                # Display episode statistics
                col1_stats, col2_stats, col3_stats = st.columns(3)
                with col1_stats:
                    st.metric("Total Reward", f"{episode_data['total_reward']:.2f}")
                with col2_stats:
                    st.metric("Episode Length", episode_data['episode_length'])
                with col3_stats:
                    st.metric("Constraint Violations", episode_data['total_violations'])
                
                # Plot trajectory
                trajectory_fig = plot_trajectory(episode_data, constraint_threshold)
                st.plotly_chart(trajectory_fig, use_container_width=True)
                
                # Plot constraint violations
                violations_fig = plot_constraint_violations(episode_data)
                st.plotly_chart(violations_fig, use_container_width=True)
                
                # Plot rewards
                rewards_fig = plot_rewards(episode_data)
                st.plotly_chart(rewards_fig, use_container_width=True)
        else:
            st.info("Please load a trained model to run episodes.")
    
    with col2:
        st.header("Evaluation")
        
        if agent is not None:
            if st.button("Evaluate Policy", type="secondary"):
                with st.spinner("Evaluating policy..."):
                    eval_results = evaluate_policy(env, agent, num_episodes=10)
                
                st.metric("Mean Reward", f"{eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
                st.metric("Mean Length", f"{eval_results['mean_length']:.2f} ± {eval_results['std_length']:.2f}")
                st.metric("Mean Violations", f"{eval_results['mean_violations']:.2f} ± {eval_results['std_violations']:.2f}")
                
                # Constraint satisfaction rate
                satisfaction_rate = 1 - (eval_results['mean_violations'] / eval_results['mean_length'])
                st.metric("Constraint Satisfaction Rate", f"{satisfaction_rate:.1%}")
        else:
            st.info("Please load a trained model to evaluate.")
        
        st.header("Environment Info")
        st.write(f"**Environment:** {env_name}")
        st.write(f"**Constraint:** {constraint_type} < {constraint_threshold}")
        st.write(f"**Penalty Weight:** {penalty_weight}")
        st.write(f"**Algorithm:** {algorithm}")
        
        # Environment details
        st.subheader("Environment Details")
        st.write(f"**Observation Space:** {env.observation_space}")
        st.write(f"**Action Space:** {env.action_space}")
        
        if hasattr(env, 'get_constraint_stats'):
            constraint_stats = env.get_constraint_stats()
            st.write("**Constraint Statistics:**")
            for key, value in constraint_stats.items():
                st.write(f"- {key}: {value:.3f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### How to Use This Demo
    
    1. **Configure Parameters**: Use the sidebar to set environment, constraint type, threshold, and penalty weight.
    2. **Load Model**: Select a trained model from the dropdown (if available).
    3. **Run Episode**: Click "Run Episode" to see the agent's behavior with visualizations.
    4. **Evaluate**: Click "Evaluate Policy" to get performance statistics.
    
    ### Training a Model
    
    To train your own model, use the training script:
    ```bash
    python scripts/train.py --algorithm constrained_q --num_episodes 1000
    ```
    
    ### Constraint Types
    
    - **Position**: Constrains the position of the cart/pole
    - **Velocity**: Constrains the velocity of the cart/pole  
    - **Combined**: Constrains both position and velocity
    
    ### Algorithms
    
    - **Constrained Q-Learning**: Uses reward penalties to discourage constraint violations
    - **Lagrangian PPO**: Uses Lagrangian multipliers to handle constraints in policy optimization
    """)


if __name__ == "__main__":
    main()
