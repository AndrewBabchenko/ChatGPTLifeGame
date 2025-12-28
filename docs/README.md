# Documentation Index

Complete guide to the Life Game predator-prey simulation system.

## Getting Started

- **[QUICKSTART](QUICKSTART.md)** - Setup and first training run
- **[SAFE_TRAINING_GUIDE](SAFE_TRAINING_GUIDE.md)** - Best practices and troubleshooting

## Architecture & Design

- **[ARCHITECTURE](ARCHITECTURE.md)** - System design, neural network, reward structure ⭐ **NEW**
- **[MODEL_ARCHITECTURE](MODEL_ARCHITECTURE.md)** - Neural network layer details
- **[MODEL_PARAMETERS](MODEL_PARAMETERS.md)** - Parameter reference and tuning

## Training & Configuration

- **[CURRICULUM_LEARNING](CURRICULUM_LEARNING.md)** - Staged training strategy (2 stages)
- **[DISTANCE_BASED_REWARDS](DISTANCE_BASED_REWARDS.md)** - Reward shaping for faster learning

## Features

- **[ADVANCED_FEATURES](ADVANCED_FEATURES.md)** - Pheromones, energy, age systems
- **[HIERARCHICAL_PPO_USAGE](HIERARCHICAL_PPO_USAGE.md)** - Turn→move action hierarchy

## Tools & Monitoring

- **[DASHBOARD_USAGE](DASHBOARD_USAGE.md)** - Training monitoring GUI
- **[UI_OVERHAUL](UI_OVERHAUL.md)** - Interactive demo features

## Platform Setup

- **[WSL2_ROCM_SETUP](WSL2_ROCM_SETUP.md)** - AMD GPU setup (Linux/WSL2)
- **[WINDOWS_ROCM_ISSUE](WINDOWS_ROCM_ISSUE.md)** - Windows ROCm troubleshooting

## Reference

- **[FOLDER_STRUCTURE](FOLDER_STRUCTURE.md)** - Project organization
- **[PROJECT_CRITIQUE](PROJECT_CRITIQUE.md)** - Design decisions
- **[SPECIES_AWARE_OPTIMIZATION](SPECIES_AWARE_OPTIMIZATION.md)** - Species tuning
- **[life_game_requirements](life_game_requirements.md)** - Original specs

## Quick Reference

| Task | Command | Documentation |
|------|---------|---------------|
| Train models | `py scripts\train.py` | [QUICKSTART](QUICKSTART.md) |
| Monitor training | `py scripts\dashboard_app.py` | [DASHBOARD_USAGE](DASHBOARD_USAGE.md) |
| Run demo | `py scripts\simulation_demo.py` | [UI_OVERHAUL](UI_OVERHAUL.md) |
| Run tests | `py -m pytest tests/` | [SAFE_TRAINING_GUIDE](SAFE_TRAINING_GUIDE.md) |
| Configure | Edit `src/config.py` | [ARCHITECTURE](ARCHITECTURE.md) |

## System Overview

### Neural Network
- **Parameters**: 2.9M
- **Architecture**: Actor-Critic with cross-attention
- **Layers**: ~20 (embeddings, attention, dual heads, critic)
- **Attention**: 8 heads, 256 dimensions

### Training
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Duration**: 100 episodes × 200 steps (~100 minutes on GPU)
- **Batch size**: 4096 experiences
- **Curriculum**: 2 stages (easy → hard)

### Agents
- **Prey**: Learn escape, mating, survival (100 initial)
- **Predators**: Learn hunting, cooperation (30 initial)

### Environment
- **Grid**: 100×100 toroidal
- **Mechanics**: Energy, age, reproduction
- **Communication**: 4 pheromone types
- **Rewards**: Distance-based shaping
