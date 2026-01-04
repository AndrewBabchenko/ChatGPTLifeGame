# Life Game - Predator-Prey Ecosystem with Deep RL

An ecosystem simulation where predators and prey learn intelligent behaviors through deep reinforcement learning (PPO with Actor-Critic architecture).

## Key Features

- **Deep RL Training**: Proximal Policy Optimization (PPO) with ~2.5M parameter neural network
- **Hierarchical Actions**: Turn → Re-observe → Move for precise directional control
- **Cross-Attention**: Self-state queries visible animals (8-head attention mechanism)
- **Pheromone Communication**: 4 types for cooperative hunting and danger signaling
- **Energy & Age Systems**: Realistic survival mechanics with reproduction requirements
- **Curriculum Learning**: 4-phase training to teach complex behaviors progressively
- **GPU Accelerated**: DirectML/CUDA support for fast training

## Project Structure

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

## Quick Start

### Prerequisites

- Python 3.11+
- GPU (AMD/NVIDIA/Intel) recommended for training
- 4GB+ GPU memory

### Installation

```bash
# Create virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install torch matplotlib numpy pygame

# For AMD/Intel GPU (Windows) - recommended:
pip install torch-directml

# For NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Training

```bash
# Run training
python scripts/train.py

# Or use phase-based training
python scripts/run_phase.py --phase 1
```

Models save to `outputs/checkpoints/` automatically.

### Demo & Dashboard

```bash
# Interactive demo with live simulation
python scripts/run_demo.py
# Or double-click: Demo.vbs

# Training monitoring dashboard
python scripts/run_dashboard.py
# Or double-click: Dashboard.vbs
```

## Documentation

**[📚 Complete Documentation](docs/README.md)** - All guides and architecture docs

Quick links:
- **[System Architecture](docs/System_Architecture.md)** - System design, neural network, reward system, file organization
- **[Model Parameters](docs/Model_Parameters.md)** - PPO parameters and configuration reference
- **[How Learning Works](docs/How_Learning_Works.md)** - Beginner's guide to RL
- **[Running Training](docs/Running_Training.md)** - How to run training
- **[Game Requirements](docs/Life_Game_Requirements.md)** - Project specifications
- **[Eval Checkpoints](docs/Eval_Checkpoint_Usage.md)** - Checkpoint evaluation guide

## Configuration

Key parameters in [`src/config.py`](src/config.py):

```python
# Training
NUM_EPISODES = 150
STEPS_PER_EPISODE = 200
PPO_BATCH_SIZE = 4096
PPO_EPOCHS = 6
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

======================================================================
  STARTING PHASE 3: Add Reproduction
======================================================================
Loading config: src.config_phase3
Phase settings:
  PHASE_NUMBER: 3
  PHASE_NAME: Add Reproduction
  LOAD_PREY_CHECKPOINT: outputs/checkpoints/phase2_ep43_model_A.pth
  LOAD_PREDATOR_CHECKPOINT: outputs/checkpoints/phase2_ep37_model_B.pth
  SAVE_CHECKPOINT_PREFIX: phase3
  NUM_EPISODES: 50
======================================================================

[WARNING] failed to run amdgpu-arch: binary not found.

======================================================================
  ADVANCED LIFE GAME TRAINING (PPO + Pheromones + Energy)
======================================================================
Seed: 0
Device: cuda
Using GPU backend: CUDA/ROCm
GPU: AMD Radeon RX 9070 XT
CUDA Version: None
GPU Memory: 17.1 GB

======================================================================
  CURRICULUM PHASE 3: Add Reproduction
======================================================================
Prey checkpoint: outputs/checkpoints/phase2_ep43_model_A.pth
Predator checkpoint: outputs/checkpoints/phase2_ep37_model_B.pth
Checkpoint save prefix: phase3
Early stop patience: 15 episodes
[CHECKPOINT] Loaded prey model from: outputs/checkpoints/phase2_ep43_model_A.pth
[CHECKPOINT] Loaded predator model from: outputs/checkpoints/phase2_ep37_model_B.pth
[PHASE] Continuing from previous phase checkpoint(s)
======================================================================


Model Size: 2,506,636 parameters (2,506,636 trainable)

Training for 50 episodes
Steps per episode: 300
Using Actor-Critic with PPO algorithm
Advanced features: Energy, Age, Pheromones, Multi-Head Attention


[21:54:05.216] Episode 1/50
[21:54:19.080] Step 50/300: 48 animals (Prey=28, Pred=20)
[21:54:30.884] Step 100/300: 33 animals (Prey=11, Pred=22)
[21:54:40.960] Step 150/300: 27 animals (Prey=3, Pred=24)
[21:54:47.078] Step 200/300: 16 animals (Prey=0, Pred=16)
[21:54:51.357] Step 250/300: 9 animals (Prey=0, Pred=9)
[21:54:53.259] Step 300/300: 3 animals (Prey=0, Pred=3)
DEBUG: Training on all 6743 predator experiences (no filtering)
[21:54:53.282] Starting PPO update (Prey experiences=2699, Predator=6743)...
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.2%
  [STOP] KL spike 0.0567 > 0.05, skipping remaining minibatches
  [PPO Diagnostics] KL: 0.056694, ClipFrac: 0.050
  [Supervision] Target visible: 10.8%, Mean target dist: 0.7
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.1%
  [INFO] Extreme ratios: 0.2%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.2%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.1%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [INFO] Extreme ratios: 0.0%
  [PPO Diagnostics] KL: 0.010579, ClipFrac: 0.089
  [Supervision] Target visible: 8.1%, Mean target dist: 0.6
[21:54:56.756] PPO update completed!
[21:54:56.756] Timing: Env=48.1s (93%), GPU=3.5s (7%), Total=51.5s
[21:54:56.756] GPU Memory: 0.23GB allocated, 0.95GB reserved
[21:54:56.756] Final: Prey=0, Predators=2
[21:54:56.756] Births=11, Deaths=47, Meals=47
[21:54:56.756] Exhaustion=0, Old Age=0, Starvation=22
[21:54:56.756] Rewards: Prey=521.4, Predator=2456.7
[21:54:56.756] Losses: Policy(P=0.000/Pr=0.002), Value(P=4.975/Pr=10.197), Entropy(P=1.738/Pr=2.197)
[21:54:56.756] Prey Actions: N:18%, NE:17%, E:10%, SE:3%, S:15%
[21:54:56.756] Predator Actions: N:12%, NE:11%, E:5%, SE:8%, S:7%
[21:54:56.776] Checkpoint saved: phase3_ep1
```

**Key metrics**:
- **Population**: Current alive counts
- **Events**: Births, deaths (by cause)
- **Rewards**: Cumulative per species
- **Losses**: Policy, value, entropy (should decrease/stabilize)
- **KL/ClipFrac**: Training stability indicators

## How It Works

### 1. Neural Network (~2.5M parameters)
Animals use a deep neural network to make decisions:
- **Input**: Self-state (323 features × 10 frames = 3,230) + visible animals (24 slots × 9 features)
- **Processing**: Cross-attention queries environment based on animal's state
- **Output**: Turn action (3) + Move action (8) + Value estimate (1)

### 2. Learning Process
Agents improve through Proximal Policy Optimization:
- **Collect**: Run 300 steps, gather 2048+ experiences
- **Learn**: Update policy using PPO for 6 epochs
- **Explore**: Entropy bonus encourages trying new strategies
- **Stabilize**: KL divergence limits prevent catastrophic updates

### 3. Reward Structure
Agents learn from consequences:
- **Prey**: Survive (+0.05/step), escape predators (+3.0), mate (+2.0), avoid death (-15.0)
- **Predators**: Hunt successfully (+10.0), chase prey (+5.0), avoid starvation (-18.0)

### 4. Curriculum Learning (4 Phases)
Training uses progressive difficulty:
- **Phase 1**: Hunt/Evade basics
- **Phase 2**: Starvation pressure
- **Phase 3**: Reproduction mechanics
- **Phase 4**: Full ecosystem simulation

## GPU Support

**GPU is required for training**.

### AMD/Intel GPU (Windows - Recommended)
```bash
pip install torch-directml
python scripts/train.py  # Auto-detects DirectML
```

### NVIDIA GPU
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python scripts/train.py  # Auto-detects CUDA
```

## Testing

Run unit tests to verify critical components:

```bash
# Test all
python -m pytest tests/ -v

# Test specific component
python -m pytest tests/test_directional_loss_finite.py -v
```

## Project Status

- ✅ Core training loop with PPO
- ✅ Hierarchical turn→move actions
- ✅ Cross-attention architecture
- ✅ Pheromone communication system
- ✅ 4-phase curriculum learning
- ✅ Energy & age mechanics
- ✅ Distance-based reward shaping
- ✅ Interactive demo with charts
- ✅ Training monitoring dashboard
- ✅ GPU acceleration (CUDA/DirectML)

## License

This project is provided for educational and personal use.
