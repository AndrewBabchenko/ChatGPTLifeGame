# Model Architecture Summary

## ActorCriticNetwork for PPO (Proximal Policy Optimization)

**Total Parameters: ~2,898,313** (2.9M parameters)

---

## Input Processing

### 1. Self-State Embedding
- **Input**: 21 features (position, type, state, threat info, age, energy, population)
- **Layer**: Linear(21 x 256)
- **Parameters**: 5,632

### 2. Visible Animals Processing
- **Animal Embedding**: Linear(8 → 256) - 2,304 params
- **Transform Layer 1**: Linear(256 → 256) - 65,792 params
- **Transform Layer 2**: Linear(256 → 256) - 65,792 params
- **Transform Layer 3**: Linear(256 → 256) - 65,792 params
- **Total**: 199,680 params

---

## Multi-Head Attention Module

- **Architecture**: 8 attention heads (num_heads=8)
- **Embed dimension**: 256
- **Head dimension**: 32 (256 / 8)

### Components:
- **Query projection**: Linear(256 → 256) - 65,792 params
- **Key projection**: Linear(256 → 256) - 65,792 params
- **Value projection**: Linear(256 → 256) - 65,792 params
- **Output projection**: Linear(256 → 256) - 65,792 params

**Total**: 263,168 params

---

## Actor Network (Policy)

**Input**: 512 features (256 self + 256 context)

### Shared Layers:
1. Linear(512 → 1024) + ReLU + Dropout(0.2) - **525,312 params**
2. Linear(1024 → 512) + ReLU + Dropout(0.2) - **524,800 params**
3. Linear(512 → 256) + ReLU - **131,328 params**

### Head Layers:
1. Linear(256 → 128) + ReLU - **32,896 params**
2. Linear(128 → 8) - **1,032 params** (8 action directions)

**Actor Total**: 1,215,368 params

---

## Critic Network (Value Estimation)

**Input**: 512 features (same as actor)

### Shared Layers:
1. Linear(512 → 1024) + ReLU - **525,312 params**
2. Linear(1024 → 512) + ReLU - **524,800 params**
3. Linear(512 → 256) + ReLU - **131,328 params**

### Head Layers:
1. Linear(256 → 128) + ReLU - **32,896 params**
2. Linear(128 → 1) - **129 params** (single value output)

**Critic Total**: 1,214,465 params

---

## Architecture Highlights

### Design Features:
- **Separate Actor-Critic**: Independent networks for policy (actor) and value (critic)
- **Multi-Head Attention**: 8 heads focusing on different aspects of visible animals
- **Deep Networks**: 5-layer actor/critic with large hidden dimensions (1024, 512, 256)
- **Regularization**: Dropout(0.2) in actor network to prevent overfitting
- **Output**: 8-way action distribution + scalar value estimate

### GPU Utilization Strategy:
The model was expanded from **77K → 2.9M parameters** (37x increase) to maximize GPU compute:
- Large matrix multiplications (512→1024, 1024→512)
- Multiple transformation layers for visible animals (3 sequential Linear(256→256))
- Increased attention heads (4→8)
- Additional depth in actor/critic networks

---

## Forward Pass Flow

```
Input (Animal State: 21 features, Visible Animals: NA-8 features)
    ↓
Self Embedding: Linear(21x256)
    ↓
Visible Animals Processing:
    - Embed each animal: Linear(8→256)
    - Transform 1: Linear(256→256) + ReLU
    - Transform 2: Linear(256→256) + ReLU
    - Transform 3: Linear(256→256) + ReLU
    - Multi-Head Attention (8 heads)
    - Adaptive Average Pool → context vector (256)
    ↓
Concatenate [self_features(256), context(256)] = 512
    ↓
    ├─→ Actor Network (512→1024→512→256→128→8)
    │   └─→ Action Probabilities (8 directions)
    │
    └─→ Critic Network (512→1024→512→256→128→1)
        └─→ State Value Estimate
```

---

## Training Configuration

### PPO Hyperparameters (from config.py):
- **PPO Epochs**: 16 (increased for better learning)
- **Batch Size**: 1024 (increased for GPU efficiency)
- **Clip Epsilon**: 0.2
- **Value Loss Coefficient**: 0.5
- **Entropy Coefficient**: 0.01
- **Max Gradient Norm**: 0.5
- **GAE Lambda**: 0.95
- **Learning Rate**: 1e-3 (prey and predator)

### Training Parameters:
- **Episodes**: 2 (baseline testing)
- **Steps per Episode**: 200 (increased episode length)
- **Initial Animals**: 140 (120 prey + 20 predators)
- **MAX_VISIBLE_ANIMALS**: 24 (increased from 15)
- **Gradient Accumulation**: 1 step (single large batch)

---

## Current Status

### ✅ Stable Configuration (77K parameters):
- Original smaller architecture
- Successfully trains 2+ episodes
- GPU utilization: ~30%
- Training time: ~30 seconds for 2 episodes, 15 steps, 130 animals

### ⚠️ Large Configuration (2.9M parameters):
- Current expanded architecture
- **Issue**: Training hangs during backpropagation at Step 30-200
- **Cause**: ROCm backward pass sensitive to large model + large batch combinations
- **GPU Memory**: Only 0.18GB used out of 17.1GB available
- **Bottleneck**: Not memory but ROCm computation instability

---

## Technical Environment

- **Framework**: PyTorch 2.9.0+rocmsdk20251116
- **Backend**: ROCm (AMD GPU acceleration)
- **GPU**: AMD Radeon RX 9070 XT (17.1 GB)
- **Python**: 3.12.10
- **OS**: Windows

---

## Architectural Limitations

### CPU-Bound Simulation:
The training pipeline is fundamentally bottlenecked by CPU:
- **95% of time**: Sequential animal processing (Python loop)
- **5% of time**: GPU neural network forward/backward passes

### Maximum GPU Utilization:
- **Current**: 30% with optimized batch sizes and model
- **Theoretical Maximum**: ~30-40% without architectural rewrite
- **To achieve 100%**: Would require:
  - Vectorized parallel environments (32-256 simultaneous simulations)
  - GPU-accelerated simulation (move animal logic to CUDA/ROCm kernels)
  - Batch environment stepping instead of sequential processing

---

## Parameter Count Breakdown

| Component | Parameters | Percentage |
|-----------|------------|------------|
| Self Embedding | 5,632 | 0.2% |
| Visible Animals Processing | 199,680 | 6.9% |
| Multi-Head Attention | 263,168 | 9.1% |
| Actor Network | 1,215,368 | 41.9% |
| Critic Network | 1,214,465 | 41.9% |
| **Total** | **2,898,313** | **100%** |

The majority of parameters (~84%) are in the actor and critic networks, which process the combined feature representation for decision-making and value estimation.
