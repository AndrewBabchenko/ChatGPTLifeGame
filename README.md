# Life Game - Predator-Prey Ecosystem with Deep RL

An advanced ecosystem simulation where predators and prey learn intelligent behaviors through deep reinforcement learning (PPO with Actor-Critic architecture).

## Key Features

- **Deep RL Training**: Proximal Policy Optimization (PPO) with 2.9M parameter neural network
- **Hierarchical Actions**: Turn → Re-observe → Move for precise directional control
- **Cross-Attention**: Self-state queries visible animals (8-head attention mechanism)
- **Pheromone Communication**: 4 types for cooperative hunting and danger signaling
- **Energy & Age Systems**: Realistic survival mechanics with reproduction requirements
- **Curriculum Learning**: Staged training to teach complex behaviors progressively
- **GPU Accelerated**: ROCm/CUDA support for fast training (~60s per episode)

## Project Structure

```
ChatGPTLifeGame/
├── scripts/
│   ├── train.py                  # Main PPO training loop
│   ├── demo.py                   # Visual simulation demo (Pygame)
│   ├── simulation_demo.py        # Interactive GUI with live charts (Tkinter)
│   ├── dashboard_app.py          # Training monitoring dashboard
│   └── run_training_safe.ps1    # Safe training wrapper with logging
├── src/
│   ├── config.py                 # All configuration parameters
│   ├── models/
│   │   ├── actor_critic_network.py  # Neural network (~2.9M parameters)
│   │   └── replay_buffer.py         # PPO memory with GAE
│   └── core/
│       ├── animal.py             # Entity logic (movement, energy, aging)
│       └── pheromone_system.py   # Communication system
├── docs/
│   ├── ARCHITECTURE.md           # Detailed system design
│   ├── QUICKSTART.md             # Setup and first run
│   ├── MODEL_ARCHITECTURE.md     # Neural network details
│   └── CURRICULUM_LEARNING.md    # Training strategy
└── tests/                        # Unit tests for critical components
```

## Quick Start

### Prerequisites

- Python 3.11+
- GPU (NVIDIA/AMD) recommended for training
- 4GB+ GPU memory

### Installation

```bash
# Using Python 3.11
py -3.11 -m pip install torch matplotlib numpy

# For AMD GPU (Windows):
pip install torch-directml

# For NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Training

```bash
# Recommended: Use safe wrapper with logging
.\scripts\run_training_safe.ps1

# Or run directly
py scripts\train.py
```

**Training takes ~100 minutes on GPU** (100 episodes × 60s each)

Models save to `outputs/checkpoints/` automatically.

### Demo

```bash
# Interactive GUI with live charts
py scripts\simulation_demo.py

# Classic Pygame visualization  
py scripts\demo.py
```

## Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design, neural network, reward system
- **[Quick Start](docs/QUICKSTART.md)** - Setup and first training run
- **[Model Architecture](docs/MODEL_ARCHITECTURE.md)** - Neural network details
- **[Curriculum Learning](docs/CURRICULUM_LEARNING.md)** - Training strategy
- **[Dashboard Usage](docs/DASHBOARD_USAGE.md)** - Monitoring training progress

## Configuration

Key parameters in [`src/config.py`](src/config.py):

```python
# Training
NUM_EPISODES = 100
STEPS_PER_EPISODE = 200
PPO_BATCH_SIZE = 4096
LEARNING_RATE_PREY = 5e-5
LEARNING_RATE_PREDATOR = 1e-4

# Population
INITIAL_PREY_COUNT = 100
INITIAL_PREDATOR_COUNT = 30

# Energy System
INITIAL_ENERGY = 100.0
ENERGY_DECAY_RATE = 0.5
REPRODUCTION_COST_PREY = 100.0

# Vision
PREY_VISION_RANGE = 7
PREDATOR_VISION_RANGE = 10

# Pheromones
PHEROMONE_DECAY = 0.95
PHEROMONE_DIFFUSION = 0.1
```

## Training Output

Training provides detailed metrics every episode:

```
[23:45:12] Episode 50/100
[23:45:42] Final: Prey=85, Predators=28
[23:45:42] Births=45, Deaths=38, Meals=32
[23:45:42] Exhaustion=2, Old Age=4, Starvation=2
[23:45:42] Rewards: Prey=982.5, Predator=1245.8
[23:45:42] Losses: Policy(P=0.015/Pr=0.018), Value(P=4.4/Pr=6.2), Entropy(P=1.05/Pr=2.01)
[23:45:42] [PPO Diagnostics] KL: 0.008, ClipFrac: 0.25
[23:45:42] Prey Actions: N:18%, NE:15%, E:12%, SE:14%, S:16%
[23:45:42] Checkpoint saved (episode 50)
[23:45:42] Timing: Env=28s (47%), GPU=32s (53%), Total=60s
```

**Key metrics**:
- **Population**: Current alive counts
- **Events**: Births, deaths (by cause)
- **Rewards**: Cumulative per species
- **Losses**: Policy, value, entropy (should decrease/stabilize)
- **KL/ClipFrac**: Training stability indicators
- **Actions**: Distribution check for bias
- **Timing**: Performance breakdown

## How It Works

### 1. Neural Network
Animals use a deep neural network to make decisions:
- **Input**: Self-state (34 features) + visible animals (15 slots × 8 features)
- **Processing**: Cross-attention queries environment based on animal's state
- **Output**: Turn action (3) + Move action (8) + Value estimate (1)

### 2. Learning Process
Agents improve through Proximal Policy Optimization:
- **Collect**: Run 200 steps, gather 4096+ experiences
- **Learn**: Update policy using PPO for 2 epochs
- **Explore**: Entropy bonus encourages trying new strategies
- **Stabilize**: KL divergence limits prevent catastrophic updates

### 3. Reward Structure
Agents learn from consequences:
- **Prey**: Survive (+0.05/step), escape predators (+3.0), mate (+2.0), avoid death (-15.0)
- **Predators**: Hunt successfully (+10.0), chase prey (+5.0), avoid starvation (-18.0)

### 4. Curriculum Learning
Training uses 2 stages:
- **Stage 1 (1-10)**: Reduced predator pressure → prey learn mating
- **Stage 2 (11-100)**: Full difficulty → test under real conditions

## Advanced Features

### Pheromone Communication
- **Food trails**: Predators mark prey locations for cooperation
- **Danger signals**: Prey warn others of predator presence
- **Mating markers**: Signal reproductive readiness
- **Decay & diffusion**: Realistic gradient-based sensing

### Energy System
- Prey start with 100 energy, decay 0.5/step
- Reproduction costs 100 energy (prey must save)
- Predators gain energy from hunting, starve if unsuccessful
- Movement consumes extra energy

### Age & Lifecycle
- Animals age over 1000 steps (5 episodes)
- Maturity at age 10 (can reproduce)
- Natural death from old age
- Experience accumulated over lifetime

## GPU Support

**GPU is required for training** (CPU is 5x slower).

### AMD GPU (Windows)
```bash
pip install torch-directml
py scripts\train.py  # Auto-detects DirectML
```

### NVIDIA GPU
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
py scripts\train.py  # Auto-detects CUDA
```

### Linux (ROCm/CUDA)
See [WSL2 ROCm Setup](docs/WSL2_ROCM_SETUP.md) for AMD GPU configuration.

## Monitoring Training

Use the dashboard to monitor training in real-time:

```bash
py scripts\dashboard_app.py
```

**Features**:
- Live training logs
- Population graphs
- Loss curves
- Episode statistics
- Start/stop training
- View checkpoints

See [Dashboard Usage](docs/DASHBOARD_USAGE.md) for details.

## Testing

Run unit tests to verify critical components:

```bash
# Test specific component
py -m pytest tests/test_directional_loss_finite.py -v

# Test all
py -m pytest tests/ -v
```

**Test coverage**:
- Directional loss computation
- Action mapping contracts
- Visibility cache validation
- GPU smoke tests
- PPO policy updates

## Performance Tips

1. **Use GPU**: 5x faster than CPU
2. **Batch size**: Increase to 8192 if GPU memory allows
3. **Accumulation steps**: Set to 1 for maximum GPU utilization
4. **Mixed precision**: Disabled by default (ROCm compatibility)
5. **Monitoring**: Use dashboard instead of terminal for less I/O overhead

## Project Status

- ✅ Core training loop with PPO
- ✅ Hierarchical turn→move actions
- ✅ Cross-attention architecture
- ✅ Pheromone communication system
- ✅ Curriculum learning (2 stages)
- ✅ Energy & age mechanics
- ✅ Distance-based reward shaping
- ✅ Interactive demo with charts
- ✅ Training monitoring dashboard
- ✅ GPU acceleration (CUDA/ROCm/DirectML)

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Areas for improvement:
- Additional curriculum stages
- More sophisticated pheromone strategies
- Alternative network architectures
- Multi-species ecosystems
- Long-term population dynamics analysis
3. `prey_safe` - Tests if prey conserve energy when safe
4. `predator_well_fed` - Tests if well-fed predators avoid unnecessary hunting
5. `overcrowded` - Tests population density awareness

**Output includes:**
- Top action probabilities with visual bars
- Feature importance ranking (gradient-based saliency)
- Behavioral insights and learned patterns
- Value function estimates for situation assessment

The tool uses the best trained models (`model_A_ppo.pth`, `model_B_ppo.pth`) from your training runs.

## 📚 Documentation

**[Complete Documentation Hub](docs/README.md)** - All guides, architecture docs, and setup instructions

Quick links:
- **[Quick Start Guide](docs/QUICKSTART.md)** - Get running in 5 minutes
- **[Model Architecture](docs/MODEL_ARCHITECTURE.md)** - Neural network and PPO implementation
- **[Advanced Features](docs/ADVANCED_FEATURES.md)** - Energy, pheromones, multi-head attention
- **[Training Results](docs/SESSION_SUMMARY_DEC26.md)** - Latest optimizations and performance
- **[GPU Setup](docs/WINDOWS_ROCM_ISSUE.md)** - AMD ROCm configuration for Windows

## Performance

- **GPU Training** (AMD RX 9070 XT): ~1.5 episodes/minute with ROCm
- **Model Size**: 2.9M parameters (Actor-Critic with 4-head attention)
- **Predator Survival**: 7-17 predators per episode (vs 0-4 baseline)
- **Hunting Success**: 16-34 meals per episode (with 2× vision range optimization)

## License

MIT License - Free to use, modify, and distribute.
