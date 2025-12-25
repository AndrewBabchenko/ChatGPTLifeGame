# Quick Start Guide

## Installation

1. **Install Python 3.11** (recommended)
   ```bash
   # Download from python.org
   ```

2. **Install Dependencies**
   ```bash
   py -3.11 -m pip install torch matplotlib numpy
   ```

3. **Verify Installation**
   ```bash
   py -c "import torch, matplotlib, numpy; print('✓ All dependencies installed')"
   ```

## Usage

### 1. Train Models (Recommended First Step)

```bash
py train_advanced.py
```

**Training Options** (edit `train_advanced.py`):
- Line 366: `num_episodes = 100` - Number of training episodes
- Line 367: `steps_per_episode = 200` - Steps per episode
- Line 348-352: CPU thread optimization

**Expected Output**:
```
Episode 1/100
  Final: Prey=334, Predators=0
  Births=217, Deaths=3, Meals=3
  ✓ New best! Saved models
```

**Training Time**: ~50-100 minutes on CPU for 100 episodes

**Models Saved To**: `models/model_A_ppo.pth`, `models/model_B_ppo.pth`

### 2. Watch Visual Demo

```bash
py demo.py
```

**What You'll See**:
- Green circles = Prey animals
- Red circles = Predator animals
- Real-time population counter
- Animated agent behavior

**Controls**: Close window to exit

### 3. Adjust Configuration

Edit `src/config.py` to change behavior:

```python
# Population sizes
INITIAL_PREY_COUNT = 100      # Starting prey
INITIAL_PREDATOR_COUNT = 20   # Starting predators

# Energy system
INITIAL_ENERGY = 100.0        # Starting energy
ENERGY_DECAY_RATE = 0.5       # Energy lost per step
ENERGY_FROM_FOOD = 50.0       # Energy from eating

# Vision
VISION_RANGE = 10             # How far animals can see
MAX_VISIBLE_ANIMALS = 8       # Max animals processed

# PPO training
PPO_BATCH_SIZE = 64          # Batch size for updates
LEARNING_RATE = 3e-4         # Learning rate
PPO_CLIP_EPSILON = 0.2       # PPO clipping parameter
```

## Common Tasks

### Resume Training from Checkpoint

Models automatically load if they exist:
```bash
py train_advanced.py
```

### Train for More Episodes

Edit `train_advanced.py` line 366:
```python
num_episodes = 200  # Increase from 100
```

### Monitor Training

Watch the output for:
- Population dynamics (births, deaths)
- Reward trends (should increase)
- Loss values (should decrease)
- "✓ New best!" indicates improvement

### Use Trained Models in Demo

Demo automatically loads models from `models/` directory:
```bash
py demo.py
```

## Troubleshooting

### Import Errors
```bash
# Reinstall dependencies
py -3.11 -m pip install --upgrade torch matplotlib numpy
```

### Model Not Found
```bash
# Check models exist
dir models
# Should show: model_A_ppo.pth, model_B_ppo.pth
```

### Slow Training
- Reduce `num_episodes` in `train_advanced.py`
- Reduce `steps_per_episode` 
- Increase `PPO_BATCH_SIZE` in `src/config.py`

### Out of Memory
- Reduce `PPO_BATCH_SIZE` in `src/config.py`
- Reduce `MAX_VISIBLE_ANIMALS` in `src/config.py`

## File Overview

| File | Purpose | Edit? |
|------|---------|-------|
| `train_advanced.py` | Train models | Yes - episodes, steps |
| `demo.py` | Visual demo | No |
| `src/config.py` | All parameters | Yes - tune behavior |
| `src/actor_critic_network.py` | Neural network | No (advanced) |
| `src/replay_buffer.py` | PPO memory | No (advanced) |
| `src/pheromone_system.py` | Pheromones | No (advanced) |
| `src/animal.py` | Animal entity | No (advanced) |
| `models/` | Trained models | Auto-generated |

## Next Steps

1. ✅ Install dependencies
2. ✅ Train models (100 episodes)
3. ✅ Watch demo
4. Experiment with `src/config.py` parameters
5. Retrain and compare results
6. Read `ADVANCED_FEATURES.md` for details

## Learning Path

**Beginner**:
- Run training and demo as-is
- Watch population dynamics
- Read README.md

**Intermediate**:
- Adjust config.py parameters
- Retrain with different settings
- Compare model performance

**Advanced**:
- Study actor_critic_network.py
- Modify PPO hyperparameters
- Implement custom reward functions
- Read ADVANCED_FEATURES.md

## Support

- See `README.md` for overview
- See `ADVANCED_FEATURES.md` for technical details
- See `GPU_SETUP.md` for GPU acceleration
- See `CLEANUP_SUMMARY.md` for project structure

---

**Quick Command Reference**:
```bash
# Install
py -3.11 -m pip install torch matplotlib numpy

# Train
py train_advanced.py

# Demo
py demo.py

# Check imports
py -c "import torch, matplotlib; print('OK')"
```
