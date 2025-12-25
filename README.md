# Life Game - Advanced Predator-Prey Simulation

An advanced ecosystem simulation featuring Actor-Critic PPO agents with pheromones, energy systems, age mechanics, and multi-head attention.

## Features

- **Actor-Critic PPO Algorithm**: Advanced reinforcement learning with Proximal Policy Optimization
- **Pheromone System**: 4 pheromone types (prey, predator, food, danger) for complex communication
- **Energy System**: Animals need energy to survive, reproduce (prey: 100, predator: energy from hunting)
- **Age System**: Animals age and die naturally (max age: 1000 steps)
- **8-Direction Movement**: Full directional control with toroidal vision
- **Multi-Head Attention**: Neural network with 4 attention heads for processing visible animals
- **77,001 Parameters**: Deep network architecture for complex behavior learning

## Project Structure

```
ChatGPTLifeGame/
├── train_advanced.py          # PPO training script
├── demo.py                    # Visual demo of trained agents
├── src/
│   ├── actor_critic_network.py   # Deep neural network with attention
│   ├── replay_buffer.py          # PPO memory with GAE
│   ├── pheromone_system.py       # Pheromone map implementation
│   ├── animal.py                 # Animal entity with energy/age
│   └── config.py                 # All configuration parameters
├── models/
│   ├── model_A_ppo.pth           # Best prey model
│   ├── model_B_ppo.pth           # Best predator model
│   ├── model_A_ppo_ep10.pth      # Checkpoint
│   └── model_B_ppo_ep10.pth      # Checkpoint
├── ADVANCED_FEATURES.md       # Detailed feature documentation
├── GPU_SETUP.md              # GPU setup guide
└── README.md                 # This file
```

## Quick Start

### Install Dependencies

```bash
# Using Python 3.11 (recommended)
py -3.11 -m pip install torch matplotlib numpy
```

### Train Models

```bash
py train_advanced.py
```

Training parameters (configurable in `train_advanced.py`):
- Episodes: 100
- Steps per episode: 200
- Batch size: 64
- PPO epochs: 4
- Learning rate: 3e-4

Models automatically save when performance improves.

### Run Visual Demo

```bash
py demo.py
```

Watch trained agents interact in real-time with animated visualization.

## Configuration

All parameters are in `src/config.py`:

```python
# Population
INITIAL_PREY_COUNT = 100
INITIAL_PREDATOR_COUNT = 20

# Energy System
INITIAL_ENERGY = 100.0
MAX_ENERGY = 200.0
ENERGY_DECAY_RATE = 0.5
ENERGY_FROM_FOOD = 50.0

# Age System
MAX_AGE = 1000

# Pheromones
PHEROMONE_DECAY = 0.95
PHEROMONE_STRENGTH = 1.0

# PPO Hyperparameters
PPO_BATCH_SIZE = 64
PPO_EPOCHS = 4
PPO_CLIP_EPSILON = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
```

## Training Output

```
Episode 1/100
  Final: Prey=334, Predators=0
  Births=217, Deaths=3, Meals=3
  Exhaustion=0, Old Age=0
  Rewards: Prey=13150.9, Predator=226.8
  Losses: Policy(P=-0.000/Pr=0.005), Value(P=0.190/Pr=10.702)
  ✓ New best! Saved models
```

Statistics tracked:
- Population dynamics (births, deaths, meals)
- Energy exhaustion and old age deaths
- Total rewards accumulated
- Policy and value losses for both species

## Network Architecture

```
ActorCriticNetwork (77,001 parameters)
├── Animal Encoder (12 inputs → 64)
├── Visible Animals Processor (8 animals × 12 features)
│   └── Transform: 12 → 32 → 32
├── Multi-Head Attention (4 heads, 32D)
├── Actor Head (→ 9 actions: 8 directions + stay)
└── Critic Head (→ 1 value estimate)
```

## GPU Support

For CPU training (current setup):
```bash
py train_advanced.py
```

For AMD GPU (experimental):
See `GPU_SETUP.md` for DirectML setup instructions.

For NVIDIA GPU:
```bash
# Install CUDA PyTorch from pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
py train_advanced.py  # Auto-detects CUDA
```

## Advanced Features Details

See `ADVANCED_FEATURES.md` for in-depth documentation on:
- PPO algorithm implementation
- Pheromone system mechanics
- Energy and reproduction dynamics
- Age system and death conditions
- Multi-head attention mechanism
- Training tips and hyperparameter tuning

## Performance

- CPU Training: ~1-2 episodes/minute (Python 3.11, 8 threads)
- Model Size: 77K parameters (~300KB per model file)
- Inference: Real-time on any CPU

## License

MIT License - Free to use, modify, and distribute.
