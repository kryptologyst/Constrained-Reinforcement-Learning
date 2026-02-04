# Constrained Reinforcement Learning

A research-ready implementation of constrained reinforcement learning algorithms that learn to maximize rewards while satisfying safety constraints.

## ⚠️ Safety Disclaimer

**This project is for research and educational purposes only. It is NOT intended for production control of real-world systems, especially in safety-critical domains such as robotics, autonomous vehicles, healthcare, or energy systems. Always consult domain experts and conduct thorough safety testing before deploying RL algorithms in real-world applications.**

## Overview

Constrained Reinforcement Learning (CRL) extends traditional RL by incorporating safety constraints that must be satisfied during learning and deployment. This project implements several state-of-the-art CRL algorithms:

- **Constrained Q-Learning**: Uses reward penalties to discourage constraint violations
- **Lagrangian PPO**: Implements Lagrangian multipliers for constraint handling in policy optimization
- **Constraint Monitoring**: Comprehensive tracking of constraint violations and safety metrics

## Features

- **Modern Tech Stack**: Built with Gymnasium, PyTorch 2.x, and Python 3.10+
- **Multiple Algorithms**: Both value-based and policy-based constrained RL methods
- **Comprehensive Evaluation**: Constraint violation metrics, safety statistics, and performance analysis
- **Interactive Demo**: Streamlit-based visualization of agent behavior and constraint satisfaction
- **Reproducible**: Deterministic seeding and comprehensive logging
- **Extensible**: Modular design for easy addition of new algorithms and environments

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)
- MPS support (optional, for Apple Silicon)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Constrained-Reinforcement-Learning.git
cd Constrained-Reinforcement-Learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e ".[dev]"
```

3. Verify installation:
```bash
python -c "import gymnasium; import torch; print('Installation successful!')"
```

## Quick Start

### Training an Agent

Train a constrained Q-learning agent on CartPole:

```bash
python scripts/train.py --algorithm constrained_q --num_episodes 1000
```

Train a Lagrangian PPO agent with custom constraints:

```bash
python scripts/train.py --algorithm lagrangian_ppo --constraint_threshold 0.3 --penalty_weight 5.0
```

### Using Configuration Files

```bash
python scripts/train.py --config configs/lagrangian_ppo.yaml
```

### Running the Interactive Demo

```bash
streamlit run demo/app.py
```

## Algorithms

### Constrained Q-Learning

A tabular and neural network-based Q-learning algorithm that applies penalty-based reward shaping when constraints are violated.

**Key Features:**
- Supports both discrete and continuous state spaces
- Configurable constraint types (position, velocity, combined)
- Adaptive penalty weighting
- Epsilon-greedy exploration with decay

**Usage:**
```python
from src.algorithms.constrained_q_learning import ConstrainedQLearningAgent

agent = ConstrainedQLearningAgent(
    state_space=env.observation_space,
    action_space=env.action_space,
    constraint_threshold=0.5,
    penalty_weight=2.0,
    constraint_type="position"
)
```

### Lagrangian PPO

A policy gradient method that uses Lagrangian multipliers to handle constraints during policy optimization.

**Key Features:**
- Proximal Policy Optimization with constraint handling
- Adaptive Lagrangian multiplier updates
- Generalized Advantage Estimation (GAE)
- Gradient clipping for stability

**Usage:**
```python
from src.algorithms.lagrangian_ppo import LagrangianPPO

agent = LagrangianPPO(
    state_space=env.observation_space,
    action_space=env.action_space,
    constraint_threshold=0.5,
    lagrangian_multiplier_lr=1e-3
)
```

## Environments

### Supported Environments

- **CartPole-v1**: Classic control task with position constraints
- **MountainCar-v0**: Continuous control with energy constraints
- **Acrobot-v1**: Underactuated system with joint limits

### Constraint Types

1. **Position Constraints**: Limit the position of the cart/pole
2. **Velocity Constraints**: Limit the velocity of the system
3. **Combined Constraints**: Simultaneous position and velocity limits
4. **Energy Constraints**: Limit total system energy

### Environment Wrappers

The project includes several environment wrappers for constraint handling:

- **ConstrainedEnvWrapper**: Monitors constraint violations and applies penalties
- **SafetyConstraintWrapper**: Terminates episodes on constraint violations
- **ConstraintCurriculumWrapper**: Gradually increases constraint difficulty

## Evaluation Metrics

### Performance Metrics

- **Average Return**: Mean episode reward with confidence intervals
- **Success Rate**: Percentage of episodes that reach the goal
- **Sample Efficiency**: Steps required to reach performance threshold
- **Episode Length**: Average episode duration

### Safety Metrics

- **Constraint Violation Rate**: Percentage of steps with violations
- **Average Constraint Cost**: Mean constraint violation magnitude
- **Constraint Satisfaction Rate**: Percentage of constraint-satisfying steps
- **Safety Violations**: Number of episodes terminated due to safety violations

### Risk Metrics

- **CVaR Returns**: Conditional Value at Risk for tail events
- **Violation Frequency**: Rate of constraint violations over time
- **Constraint Recovery**: Ability to return to safe states after violations

## Configuration

### Training Configuration

```yaml
# configs/training.yaml
env:
  name: "CartPole-v1"
  seed: 42

training:
  num_episodes: 1000
  eval_episodes: 10
  eval_freq: 50
  log_freq: 10
  save_freq: 100

algorithm:
  name: "constrained_q"
  learning_rate: 0.001
  discount_factor: 0.99

constraints:
  threshold: 0.5
  type: "position"
  penalty_weight: 2.0
```

### Command Line Arguments

```bash
python scripts/train.py \
    --env CartPole-v1 \
    --algorithm constrained_q \
    --num_episodes 1000 \
    --constraint_threshold 0.5 \
    --constraint_type position \
    --penalty_weight 2.0 \
    --output_dir ./outputs
```

## Interactive Demo

The Streamlit demo provides an interactive interface for:

- **Visualizing Agent Behavior**: Real-time episode trajectories
- **Constraint Monitoring**: Constraint violation tracking over time
- **Performance Evaluation**: Statistical analysis of agent performance
- **Parameter Tuning**: Interactive adjustment of constraint parameters

### Demo Features

- Real-time episode visualization with constraint boundaries
- Constraint violation plots and statistics
- Performance metrics and evaluation results
- Model loading and comparison capabilities

## Project Structure

```
constrained-reinforcement-learning/
├── src/
│   ├── algorithms/          # RL algorithm implementations
│   │   ├── constrained_q_learning.py
│   │   └── lagrangian_ppo.py
│   ├── envs/                # Environment wrappers
│   │   └── constrained_wrappers.py
│   └── utils/               # Utility functions
│       └── __init__.py
├── configs/                 # Configuration files
│   ├── default.yaml
│   └── lagrangian_ppo.yaml
├── scripts/                 # Training and evaluation scripts
│   └── train.py
├── demo/                    # Interactive demo
│   └── app.py
├── tests/                   # Unit tests
├── assets/                  # Generated plots and videos
├── outputs/                 # Model checkpoints and logs
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Development

### Code Quality

The project uses modern Python development practices:

- **Type Hints**: Full type annotation coverage
- **Documentation**: Google-style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff for code quality checks
- **Testing**: Pytest for unit and integration tests

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_algorithms.py
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Format code: `black src tests`
6. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{constrained_rl,
  title={Constrained Reinforcement Learning: A Modern Implementation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Constrained-Reinforcement-Learning}
}
```

## References

1. Achiam, J., et al. "Constrained policy optimization." ICML 2017.
2. Tessler, C., et al. "Reward constrained policy optimization." ICLR 2019.
3. Ray, A., et al. "Benchmarking safe exploration in deep reinforcement learning." NeurIPS 2019.
4. Chow, Y., et al. "Risk-constrained reinforcement learning with percentile risk criteria." JMLR 2017.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym/Gymnasium for the environment framework
- Stable Baselines3 for algorithm inspiration
- The RL research community for foundational work in constrained RL
# Constrained-Reinforcement-Learning
