# Improved Folder Structure

This document describes the reorganized project structure for better maintainability and clarity.

## New Structure

```
ChatGPTLifeGame/
├── README.md                    # Main project documentation
├── .gitignore                   # Git ignore patterns
│
├── docs/                        # All documentation
│   ├── QUICKSTART.md            # Quick start guide
│   ├── ADVANCED_FEATURES.md     # Advanced features documentation
│   ├── ENABLE_GPU.md            # GPU setup instructions
│   ├── GPU_SETUP.md             # Detailed GPU configuration
│   ├── GPU_OPTIMIZATION_REVIEW.md  # GPU optimization analysis
│   ├── SAFE_TRAINING_GUIDE.md   # Safe training recommendations
│   ├── MODEL_ARCHITECTURE.md    # Model architecture details
│   └── CLEANUP_SUMMARY.md       # Previous cleanup summary
│
├── scripts/                     # Executable scripts
│   ├── train_advanced.py        # Main training script (PPO)
│   ├── demo.py                  # Visual demo/simulation
│   └── run_training_safe.ps1    # Safe training wrapper
│
├── src/                         # Source code
│   ├── config.py                # Global configuration
│   ├── core/                    # Core game logic
│   │   ├── __init__.py
│   │   ├── animal.py            # Animal class and behavior
│   │   └── pheromone_system.py  # Pheromone map system
│   └── models/                  # Neural networks
│       ├── __init__.py
│       ├── actor_critic_network.py  # Actor-Critic PPO model
│       └── replay_buffer.py     # PPO memory buffer
│
├── outputs/                     # Training outputs
│   ├── logs/                    # Training logs
│   │   └── training_*.log
│   └── checkpoints/             # Model checkpoints
│       ├── model_A_ppo.pth      # Best prey model
│       ├── model_B_ppo.pth      # Best predator model
│       ├── model_A_ppo_ep*.pth  # Episode checkpoints
│       └── model_B_ppo_ep*.pth
│
├── tests/                       # Unit tests (future)
│
├── versions/                    # Version history/backups
│
└── .venv_rocm/                  # Python virtual environment (ROCm)
```

## Key Improvements

### 1. **Documentation Separation** (`docs/`)
All markdown documentation is now in one place, making it easier to find and maintain guides, architecture descriptions, and setup instructions.

### 2. **Script Organization** (`scripts/`)
All executable Python and PowerShell scripts are grouped together, clearly separated from library code.

### 3. **Source Code Structure** (`src/`)
- **`src/core/`**: Core game logic (animals, pheromones)
- **`src/models/`**: Neural network models and training utilities
- **`src/config.py`**: Centralized configuration

This separation makes it clear which code handles game mechanics vs. machine learning.

### 4. **Output Organization** (`outputs/`)
- **`outputs/logs/`**: All training logs with timestamps
- **`outputs/checkpoints/`**: Model checkpoints and best models

Keeps generated files separate from source code and makes cleanup easier.

### 5. **Future Growth** (`tests/`)
Empty directory prepared for unit tests as the project grows.

## Path Updates

All scripts and configuration have been updated to use the new paths:

### Import Paths
```python
# OLD
from src.animal import Animal
from src.actor_critic_network import ActorCriticNetwork

# NEW
from src.core.animal import Animal
from src.models.actor_critic_network import ActorCriticNetwork
```

### Model Checkpoints
```python
# OLD
torch.save(model.state_dict(), "models/model_A_ppo.pth")

# NEW
torch.save(model.state_dict(), "outputs/checkpoints/model_A_ppo.pth")
```

### Log Files
```powershell
# OLD
$logFile = "logs/training_$timestamp.log"

# NEW
$logFile = "outputs/logs/training_$timestamp.log"
```

### Script Execution
```powershell
# OLD (from project root)
python train_advanced.py

# NEW (from anywhere via run_training_safe.ps1)
python scripts/train_advanced.py
```

## Benefits

1. **Clarity**: Purpose of each directory is immediately clear
2. **Scalability**: Easy to add new scripts, models, or documentation
3. **Cleanliness**: Generated files (logs, models) separated from source
4. **Maintainability**: Related code grouped logically
5. **Professional**: Standard Python project structure
6. **Git-Friendly**: Easier to write `.gitignore` rules

## Running Training

The training script now uses the new structure automatically:

```powershell
# From project root
.\scripts\run_training_safe.ps1

# Or directly
cd scripts
python train_advanced.py
```

All paths are handled automatically by the updated scripts.
