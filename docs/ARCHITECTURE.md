# System Architecture

## Overview

Life Game is a predator-prey ecosystem simulation using deep reinforcement learning (PPO) to train intelligent agents. The system features hierarchical decision-making, pheromone communication, and realistic energy/age mechanics.

## Core Components

### Neural Network Architecture

**File**: [`src/models/actor_critic_network.py`](../src/models/actor_critic_network.py)

- **Type**: Actor-Critic with dual action heads
- **Parameters**: ~2.9M
- **Layers**: ~20 linear layers
- **Architecture**:
  - Self-state embedding (34 → 256)
  - Visible animals embedding (8 → 256)
  - Cross-attention (8 heads, 256 dims)
  - Feature fusion (512 → 512)
  - Turn head (512 → 3 actions: left/straight/right)
  - Move head (512 → 8 actions: N/NE/E/SE/S/SW/W/NW)
  - Critic head (512 → 1 value)

**Key Features**:
- Cross-attention: Self-state queries visible animals
- Hierarchical actions: Turn → Re-observe → Move
- Dropout (0.2) for regularization
- Temperature scaling for exploration

### Observation Space (OBS_VERSION=2)

**Self-state** (34 features):
- Position: x, y, normalized_x, normalized_y
- Energy: normalized, rate, max
- Age: normalized, maturity status
- Species: is_predator, is_prey flags
- Heading: cos(θ), sin(θ)
- Reproduction: energy_ready, cooldown, fertility
- Hunger: steps_since_meal, normalized
- Pheromone sensors: 4 types × (gradient_x, gradient_y, magnitude)

**Visible animals** (8 features per animal, max 15):
- Relative position: dx, dy (signed, normalized)
- Distance: normalized
- Type flags: is_predator, is_prey, same_species, same_type
- Padding flag: is_present (1.0=real, 0.0=padding)

### Reward System

**File**: [`scripts/train.py`](../scripts/train.py) (process_animal_hierarchical)

**Prey rewards**:
- Survival: +0.05/step
- Escape predator: +3.0 (distance-based shaping)
- Approach mate: +2.0 (when safe)
- Reproduction: +2.0
- Death: -10.0 (eaten) + -5.0 (penalty)

**Predator rewards**:
- Survival: +0.05/step
- Chase prey: +5.0 (distance-based shaping)
- Hunt success: +10.0
- Starvation: -10.0 (death) + -8.0 (penalty)

### Training Algorithm

**Algorithm**: Proximal Policy Optimization (PPO)

**Hyperparameters** (in [`src/config.py`](../src/config.py)):
- Episodes: 100
- Steps per episode: 200
- PPO epochs: 2
- Clip epsilon: 0.1
- Batch size: 4096
- Learning rates: Prey 5e-5, Predator 1e-4
- GAE lambda: 0.95
- Discount (γ): 0.99

**Training features**:
- Hierarchical policy (turn → move)
- Temporal difference (TD) bootstrapping
- Gradient accumulation (batch size 4096)
- KL divergence early stopping
- Curriculum learning (2 stages)

### Pheromone System

**File**: [`src/core/pheromone_system.py`](../src/core/pheromone_system.py)

**Types**:
1. Food pheromones (0.7 strength when seeing prey, 0.9 after hunt)
2. Danger pheromones (0.8 strength when fleeing)
3. Mating pheromones (0.6 strength when seeking mate)
4. Generic species markers

**Mechanics**:
- Decay: 0.95/step (40% intensity after 10 steps)
- Diffusion: 0.1 to adjacent 8 cells
- Sensing range: 5 cells
- Predators deposit toward prey location (cooperative hunting)
- Food pheromones hidden from prey observations

### Curriculum Learning

**Stage 1** (Episodes 1-10): Prey Mating Curriculum
- Reduced predators: 20 (vs 30 normal)
- Slower predators: 2 moves/step (vs 3)
- Better prey vision: 9 range (vs 7)
- Goal: Learn mating without constant pressure

**Stage 2** (Episodes 11-100): Normal Configuration
- Full predator pressure restored
- Test learned mating behavior under threat

## File Organization

```
src/
├── config.py                    # All parameters (training, environment, rewards)
├── models/
│   ├── actor_critic_network.py # Neural network architecture
│   └── replay_buffer.py         # PPO memory with GAE
└── core/
    ├── animal.py                # Entity logic (movement, energy, aging)
    └── pheromone_system.py      # Communication system

scripts/
├── train.py                     # Main training loop
├── demo.py                      # Visual simulation demo
└── dashboard_app.py             # Training monitoring GUI
```

## Key Design Decisions

### 1. Hierarchical Actions
- **Why**: Separate turn and move prevents diagonal movement bias
- **How**: Turn → Re-observe → Move (2 transitions per step)
- **Benefit**: More precise directional control

### 2. Cross-Attention vs Self-Attention
- **Why**: What matters depends on MY state (hungry predator vs satiated)
- **How**: Self-state queries visible animals
- **Benefit**: Context-dependent processing of environment

### 3. Distance-Based Reward Shaping
- **Why**: Dense rewards accelerate learning
- **How**: Reward incremental progress (getting closer/farther)
- **Benefit**: Faster convergence than sparse terminal rewards

### 4. Visibility Cache Validation
- **Why**: Prevent stale visibility data from previous steps
- **How**: Track CURRENT_STEP counter, validate before pheromone deposits
- **Benefit**: Accurate cooperative hunting signals

### 5. Curriculum Learning
- **Why**: Complex behaviors need staged introduction
- **How**: Reduce pressure → learn basics → increase difficulty
- **Benefit**: Prey learn mating before facing full predator threat

## Performance

### Training Speed
- **GPU (ROCm)**: ~60s/episode (env=30s, GPU=30s)
- **CPU**: ~300s/episode (5x slower)
- **Recommendation**: GPU required for 100-episode runs

### Model Size
- **Parameters**: 2,900,000
- **Memory**: ~12MB (model) + ~2GB (training batch)
- **GPU Memory**: ~4GB typical usage

## Debugging & Analysis

### Attention Visualization
```python
# After forward pass with batch_size=1:
attention_weights = model.last_attention_weights  # (1, 8, N)
# Shape: (batch, num_heads, num_visible_animals)
```

### Training Diagnostics
- **KL divergence**: Monitor policy stability (<0.05 good)
- **Clip fraction**: % of samples clipped (20-30% optimal)
- **Entropy**: Exploration level (should decrease gradually)
- **Action distribution**: Check for bias (no action >30%)

### Common Issues
1. **Policy collapse**: KL divergence spike → early stopping engaged
2. **Value divergence**: Large value loss → reward scale issue
3. **Action bias**: One direction >30% → exploration insufficient
4. **Stale visibility**: Cache not validated → wrong pheromone signals
