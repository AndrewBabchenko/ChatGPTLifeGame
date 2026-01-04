# Documentation Index

Complete guide to the Life Game predator-prey simulation system.

## Getting Started

- **[Game Requirements](Life_Game_Requirements.md)** - Project specifications
- **[How Training Works](How_Learning_Works.md)** - Beginner's guide to RL in this project
- **[Running Training Guide](Running_Training.md)** - How to run training

## Architecture & Design

- **[Architecture](System_Architecture.md)** - System design, neural network details, reward structure, file organization
- **[Model Parameters](Model_Parameters.md)** - explanation of PPO reinforcement learning parameters used in the Life Game simulation for model tuning 

## Training & Optimization


## Reference

- **[Eval Checkpoint Usage](Eval_Checkpoint_Usage.md)** - Checkpoint evaluation scrtipt guide - used to check how actual models perform their tasks

## Quick Reference

| Task | Command |
|------|---------|
| Train models | `python scripts/train.py` |
| Run dashboard | `python scripts/run_dashboard.py` or double-click `Dashboard.vbs` |
| Run demo | `python scripts/run_demo.py` or double-click `Demo.vbs` |
| Run tests | `python -m pytest tests/` |
| Phase training | `python scripts/run_phase.py --phase 1` |
| Configure | Edit `src/config.py` |

## System Overview

### Neural Network
- **Parameters**: ~2.5M (2,506,636 total)
- **Architecture**: Actor-Critic with cross-attention
- **Self-state input**: 323 features (34 base + 289 grass) × 10 frames = 3,230 dims
- **Visible animals**: 24 slots × 9 features each
- **Attention**: 8 heads, 256 dimensions
- **Dual action heads**: Turn (3 actions) + Move (8 directions)

### Training
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Duration**: 150 episodes × 300 steps per episode
- **Batch size**: 2048 experiences
- **PPO epochs**: 6 per update
- **Curriculum**: 4 phases (Hunt/Evade → Starvation → Reproduction → Full)

### Agents
- **Prey**: Learn escape, foraging, mating (40 initial)
- **Predators**: Learn hunting (20 initial)
- **Learning rates**: Prey=0.00008, Predator=0.0001

### Environment
- **Grid**: 100×100 toroidal
- **Mechanics**: Energy, age, reproduction, grass foraging
- **Communication**: 4 pheromone types (danger, mating, food, territory)
- **Rewards**: Distance-based shaping with directional supervision
