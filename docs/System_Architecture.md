# System Architecture

## Overview

Life Game is a predator-prey ecosystem simulation using deep reinforcement learning (PPO) to train intelligent agents. The system features hierarchical decision-making, pheromone communication, and realistic energy/age mechanics.

## Core Components

### Neural Network Architecture

**File**: [`src/models/actor_critic_network.py`](../src/models/actor_critic_network.py)

- **Type**: Actor-Critic with dual action heads
- **Parameters**: ~2.5M (2,506,636 total)
- **Architecture**:
  - Self-state embedding: Linear(3230 → 256)
    - Input: 323 features × 10 history frames = 3,230 dims
    - 323 = 34 base features + 289 grass FOV map
  - Visible animals embedding: Linear(9 → 256)
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

### Observation Space

**Self-state** (34 base features + 289 grass = 323 per frame, 10 frames = 3,230 total):
- Position: x, y (normalized to [0,1])
- Species: is_species_A, is_species_B, is_predator
- Energy: hunger_level, mating_cooldown (normalized)
- Nearest threats: predator_dist, predator_dx, predator_dy
- Nearest targets: prey_dist, prey_dx, prey_dy
- Visible counts: predator_count, prey_count (normalized)
- Age, energy level (normalized)
- Pheromone sensors: danger, mating, food intensities
- Heading direction: dx, dy
- Pheromone gradients: 6 values (danger/mating/food × x/y)
- Gradient magnitudes: 3 values
- Danger memory, population ratio, previous turn action
- Grass FOV map: 17×17 = 289 cells (prey only, zeros for predators)

**Visible animals** (9 features per animal, max 24):
- Relative position: dx, dy (signed, normalized)
- Distance: normalized
- Type flags: is_predator, is_prey, same_species, same_type
- Reserved: placeholder (always 0)
- Padding flag: is_present (1.0=real, 0.0=padding)

### Reward System

**File**: [`scripts/train.py`](../scripts/train.py) (process_animal_hierarchical)

**Prey rewards**:
- Survival: +0.2/step
- Escape predator: +2.5 × distance_increase
- Grass eaten: +1.0
- Threat penalty: -0.15 × closeness^1.5
- Blocked while threatened: -0.5
- Being eaten: -25.0
- Exhaustion death: -12.5
- Reproduction: +0.2

**Predator rewards**:
- Eating prey: +30.0
- Approaching prey: +5.0 × distance_decrease (capped at 0.8)
- Prey detection bonus: +0.05
- Prey visible per step: +0.01
- New territory explored: +0.01
- Starvation death: -10.0

### Training Algorithm

**Algorithm**: Proximal Policy Optimization (PPO)

**Hyperparameters** (in [`src/config.py`](../src/config.py)):
| Parameter | Value |
|-----------|-------|
| NUM_EPISODES | 150 |
| STEPS_PER_EPISODE | 300 |
| PPO_EPOCHS | 6 |
| CLIP_EPSILON | 0.15 |
| BATCH_SIZE | 2048 |
| LEARNING_RATE_PREY | 0.00008 |
| LEARNING_RATE_PREDATOR | 0.0001 |
| GAE_LAMBDA | 0.95 |
| GAMMA | 0.99 |
| VALUE_LOSS_COEF | 0.25 |
| ENTROPY_COEF | 0.04 |
| MAX_GRAD_NORM | 0.3 |

**Training features**:
- Hierarchical policy (turn → move)
- Temporal difference (TD(0)) bootstrapping
- KL divergence early stopping (0.03 threshold)
- Directional supervision loss (auxiliary task)
- 4-phase curriculum learning

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

**4-Phase System** (see [PHASED_TRAINING_GUIDE](PHASED_TRAINING_GUIDE.md)):

**Phase 1: Hunt/Evade** (Episodes 1-150)
- Focus: Basic predator-prey interactions
- Prey learn to flee, predators learn to chase
- Config: `src/config_phase1.py`

**Phase 2: Starvation** (Episodes 1-150)
- Focus: Energy management
- Animals must eat to survive
- Loads Phase 1 checkpoints
- Config: `src/config_phase2.py`

**Phase 3: Reproduction** (Episodes 1-150)
- Focus: Mating behavior
- Learn to find mates while avoiding threats
- Loads Phase 2 checkpoints
- Config: `src/config_phase3.py`

**Phase 4: Full Ecosystem** (Episodes 1-150)
- Focus: All mechanics combined
- Complete predator-prey ecosystem
- Loads Phase 3 checkpoints
- Config: `src/config.py`

## File Organization

```
Root/
├── Dashboard.vbs                # Double-click launcher for Dashboard GUI
├── Demo.vbs                     # Double-click launcher for Demo simulation

src/
├── config.py                    # Main config (Phase 4)
├── config_phase1.py             # Phase 1 config
├── config_phase2.py             # Phase 2 config
├── config_phase3.py             # Phase 3 config
├── models/
│   ├── actor_critic_network.py  # Neural network architecture
│   └── replay_buffer.py         # PPO memory
└── core/
    ├── animal.py                # Entity logic (movement, energy, aging)
    ├── grass_field.py           # Grass foraging system
    └── pheromone_system.py      # Communication system

scripts/
├── train.py                     # Main training loop
├── run_demo.py                  # Visual simulation demo
├── run_dashboard.py             # Training monitoring GUI
├── run_phase.py                 # Phase-based training runner
├── eval_checkpoints.py          # Checkpoint evaluation
├── rank_eval_results.py         # Utility to rank evaluation results
├── tee_output.py                # Output logging utility
├── dashboard/                   # Dashboard GUI modules
│   ├── app.py                   # Main dashboard application
│   └── tabs/
│       ├── base_tab.py          # Base class for all tabs
│       ├── behaviors_tab.py     # Animal behavior analysis
│       ├── config_tab.py        # Configuration display
│       ├── environment_tab.py   # Environment visualization
│       ├── evaluation_tab.py    # Checkpoint evaluation controls
│       ├── log_tab.py           # Training log viewer
│       ├── stability_tab.py     # Training stability metrics
│       ├── training_control.py  # Start/stop training controls
│       └── trends_tab.py        # Training trends over time
└── demo/                        # Demo GUI modules
    ├── app.py                   # Main demo application
    └── tabs/
        ├── chart_tab.py         # Real-time population charts
        ├── config_tab.py        # Configuration display
        ├── evaluation_tab.py    # Model evaluation controls
        └── simulation_tab.py    # Live simulation view
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
- **Parameters**: 2,500,000
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
