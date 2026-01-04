# Training Findings: PPO Settings & Reward Shaping Improvements

This document describes how the training process was improved by implementing new PPO settings and updated reward shaping to fix learning problems.

---

## 1. Initial Problems Identified

### 1.1 Prey Behavior Issues
- **Entropy Collapse**: Prey policy quickly converged to a single deterministic action
- **No Evasion Learning**: Prey failed to flee from predators effectively
- **High Death Rates**: 200-350 prey deaths per episode with poor escape rates (70-80%)

### 1.2 Predator Behavior Issues
- **Mass Starvation**: 20-30 predator starvation deaths per episode
- **Inefficient Hunting**: Low capture rates (~19-28%) despite abundant prey
- **Energy Management Failure**: Predators couldn't balance hunting urgency with energy levels

### 1.3 Ecosystem Instability
- **Population Crashes**: Either prey or predators would go extinct
- **No Balanced Coexistence**: Could not achieve stable predator-prey dynamics
- **Reward Signal Noise**: Too many conflicting reward signals confused learning

---

## 2. Diagnostic Analysis

### 2.1 Evaluation Results (Phase 1 Training)

**Top Predator Performance** (by composite score):
| Episode | Score | Capture Rate | Meals/Pred | Starvations |
|---------|-------|--------------|------------|-------------|
| 33 | 0.826 | 28.4% | 3.37 | 21 |
| 38 | 0.824 | 30.8% | 3.13 | 24 |
| 49 | 0.824 | 26.7% | 3.48 | 25 |

**Top Prey Performance** (by composite score):
| Episode | Score | Escape Rate | Deaths | Final Pop |
|---------|-------|-------------|--------|-----------|
| 1 | 0.875 | 96.2% | 22 | 40.0 |
| 5 | 0.673 | 89.1% | 250 | 390.3 |
| 46 | 0.502 | 82.3% | 243 | 179.7 |

### 2.2 Key Observations
1. **Best prey checkpoint (ep1)** had nearly untrained policy - suggests over-training hurts prey
2. **Predator performance improved** over training but starvation remained high
3. **Capture vs escape tradeoff**: When predators got better, prey got worse
4. **Episode 5** showed best ecosystem balance (both species viable)

---

## 3. PPO Parameter Improvements

### 3.1 Learning Rates

**Before:**
```python
LEARNING_RATE_PREY = 0.0003
LEARNING_RATE_PREDATOR = 0.0003
```

**After:**
```python
LEARNING_RATE_PREY = 0.00008      # Slower to prevent entropy collapse
LEARNING_RATE_PREDATOR = 0.0001   # Slightly higher for faster hunting adaptation
```

**Rationale:**
- Prey needed slower learning to maintain exploration diversity
- Predators needed faster adaptation to learn hunting before starvation
- Asymmetric rates allow species to learn at appropriate speeds

### 3.2 Batch Size & Epochs

**Before:**
```python
PPO_BATCH_SIZE = 512
PPO_EPOCHS = 10
```

**After:**
```python
PPO_BATCH_SIZE = 2048    # Larger batches for stable gradients
PPO_EPOCHS = 6           # Fewer epochs to prevent overfitting
```

**Rationale:**
- Larger batches reduce gradient variance
- Fewer epochs prevent policy from changing too drastically per update
- Combined effect: more stable, incremental learning

### 3.3 Entropy Coefficient

**Before:**
```python
ENTROPY_COEF = 0.01
```

**After:**
```python
ENTROPY_COEF = 0.04      # 4x higher entropy bonus
```

**Rationale:**
- Higher entropy prevents premature policy convergence
- Maintains exploration throughout training
- Critical for prey to discover evasion strategies
- Prevents predators from fixating on suboptimal hunting patterns

### 3.4 PPO Clipping

**Before:**
```python
PPO_CLIP_EPSILON = 0.2
```

**After:**
```python
PPO_CLIP_EPSILON = 0.15  # Tighter clipping
```

**Rationale:**
- Smaller clip range = more conservative policy updates
- Prevents catastrophic policy changes
- Improves training stability

### 3.5 Discount Factor (Gamma)

**Before:**
```python
GAMMA = 0.99
```

**After:**
```python
GAMMA = 0.97
```

**Rationale:**
- At γ=0.99: reward 100 steps away valued at 36.6% of immediate
- At γ=0.97: reward 100 steps away valued at 4.8% of immediate
- Lower gamma reduces variance from long episode trajectories
- Makes immediate survival/hunting more important than distant futures

### 3.6 GAE Lambda

**Before:**
```python
GAE_LAMBDA = 0.95
```

**After:**
```python
GAE_LAMBDA = 0.92
```

**Rationale:**
- Lower lambda = more bias toward immediate rewards
- Reduces variance in advantage estimation
- Helps with credit assignment in sparse reward environments

---

## 4. Reward Shaping Improvements

### 4.1 Predator Rewards

**Eating Reward (Primary Goal):**
```python
PREDATOR_EAT_REWARD = 30.0    # Strong positive signal for successful hunts
```

**Approach Reward (Shaping):**
```python
PREDATOR_APPROACH_REWARD = 0.8  # Small reward for closing distance to prey
```

**Starvation Penalty:**
```python
STARVATION_PENALTY = -5.0      # Penalty for dying of hunger
STARVATION_THRESHOLD = 150     # Generous threshold for learning
```

### 4.2 Prey Rewards

**Evasion Rewards:**
```python
PREY_EVASION_REWARD = 2.5      # Reward for increasing distance from predators
PREY_EVASION_PENALTY = 1.0     # Penalty for getting closer to predators
PREY_EVASION_SCALE_CELLS = 4.0 # Distance scaling factor
```

**Threat Presence Penalty:**
```python
PREY_THREAT_PRESENCE_PENALTY = -0.15  # Constant pressure when predator visible
PREY_THREAT_PRESENCE_POWER = 1.5      # Exponential scaling by proximity
```

**Death Penalties:**
```python
EATEN_PENALTY = -20.0          # Strong penalty for being caught
DEATH_PENALTY = -5.0           # General death penalty
```

### 4.3 Reward Design Principles

1. **Sparse vs Dense**: Combined sparse terminal rewards (death/eat) with dense shaping (approach/evasion)
2. **Magnitude Balance**: Ensured no single reward dominates the signal
3. **Temporal Consistency**: Rewards scale appropriately with discount factor
4. **Clear Gradients**: Rewards provide clear directional signal (closer=better/worse)

---

## 5. Phased Training Curriculum

### 5.1 Four-Phase Approach

| Phase | Name | Duration | Features Enabled |
|-------|------|----------|------------------|
| 1 | Hunt & Evade | 50+ eps | Basic predator-prey mechanics only |
| 2 | Add Starvation | 50 eps | Energy system, starvation deaths |
| 3 | Add Reproduction | 50 eps | Mating, population dynamics |
| 4 | Full Ecosystem | Ongoing | All features including grass foraging |

### 5.2 Why Phased Training Works

**Problem with Full Complexity:**
- Too many objectives confuse the learning signal
- Animals can't learn basic skills (hunt/evade) while worrying about mating
- Reward signals cancel each other out

**Solution - Curriculum Learning:**
1. **Phase 1**: Master fundamental skills (hunting, evasion)
2. **Phase 2**: Add survival pressure (forces hunting to be necessary)
3. **Phase 3**: Add reproduction (balance survival with breeding)
4. **Phase 4**: Full ecosystem (integrate all behaviors)

### 5.3 Checkpoint Selection

Each phase loads the best-performing checkpoint from previous phase:
```python
# config_phase2.py
LOAD_PREY_CHECKPOINT = "outputs/checkpoints/phase1_ep192_model_A.pth"
LOAD_PREDATOR_CHECKPOINT = "outputs/checkpoints/phase1_ep181_model_B.pth"

# config_phase3.py  
LOAD_PREY_CHECKPOINT = "outputs/checkpoints/phase2_ep43_model_A.pth"
LOAD_PREDATOR_CHECKPOINT = "outputs/checkpoints/phase2_ep37_model_B.pth"
```

---

## 6. Observation Memory Stacking

### 6.1 Implementation

**Before:**
- Single-frame observations
- No temporal context
- Animals couldn't perceive movement/velocity

**After:**
```python
OBS_HISTORY_LEN = 10   # Stack 10 frames of self-state
BASE_SELF_FEATURE_DIM = 323   # 34 features + 289 grass patch
SELF_FEATURE_DIM = 3230       # 323 × 10 frames
```

### 6.2 Benefits

1. **Velocity Perception**: Can infer if targets are approaching or fleeing
2. **Pattern Recognition**: Can learn movement patterns over time
3. **Better Decision Making**: Context for "what was I doing?"
4. **Threat Assessment**: Distinguish stationary vs charging predators

### 6.3 Technical Details

- Temporal order: `[t, t-1, t-2, ..., t-9]` (current first)
- History updates only on move phase (not turn phase)
- Zero-padding for new animals
- Reset at episode boundaries to prevent information leakage

---

## 7. Bug Fixes in Behavior Evaluation

### 7.1 Coordinate Normalization Fix

**Bug**: Behavior tests were passing raw coordinates instead of normalized values

**Before (Wrong):**
```python
vis[:, i, 0] = dx        # Raw value (e.g., 5.0)
vis[:, i, 1] = dy
vis[:, i, 2] = distance
```

**After (Correct):**
```python
vis[:, i, 0] = dx / vision_range        # Normalized (e.g., 0.625)
vis[:, i, 1] = dy / vision_range
vis[:, i, 2] = distance / vision_range
```

**Impact**: Model was receiving inputs in completely wrong scale, making behavior analysis meaningless.

### 7.2 Dynamic Threshold Based on Entropy

**Problem**: Fixed threshold (dot > 0) gave 50% baseline even for random policy

**Solution**: Dynamic threshold linked to entropy coefficient:
```python
def get_intentional_threshold(config):
    entropy_coef = getattr(config, 'ENTROPY_COEF', 0.04)
    threshold = 0.65 - entropy_coef * 3.0
    return min(0.70, max(0.30, threshold))
```

**Result**:
- Higher entropy → lower threshold (more lenient)
- Lower entropy → higher threshold (stricter)
- Random baseline now ~31% instead of 50%

---

## 8. Expected Outcomes

### 8.1 Prey Improvements
- ✅ Maintain exploration (entropy > 1.0)
- ✅ Learn effective evasion (escape rate > 85%)
- ✅ Survive longer (fewer deaths per episode)
- ✅ Respond to threat proximity

### 8.2 Predator Improvements
- ✅ Successful hunting (capture rate > 25%)
- ✅ Reduced starvation (< 15 deaths per episode)
- ✅ Energy-aware behavior (hunt more when hungry)
- ✅ Selective targeting (prefer closer prey)

### 8.3 Ecosystem Stability
- ✅ Both species survive to episode end
- ✅ Population oscillations within bounds
- ✅ Balanced predator-prey ratio maintained
- ✅ Sustainable reproduction rates

---

## 9. Summary of Key Changes

| Parameter | Before | After | Impact |
|-----------|--------|-------|--------|
| LR (Prey) | 0.0003 | 0.00008 | Prevents entropy collapse |
| LR (Pred) | 0.0003 | 0.0001 | Faster hunting adaptation |
| Entropy | 0.01 | 0.04 | Maintains exploration |
| Batch Size | 512 | 2048 | Stable gradients |
| Epochs | 10 | 6 | Prevents overfitting |
| Clip ε | 0.2 | 0.15 | Conservative updates |
| Gamma | 0.99 | 0.97 | Reduced variance |
| GAE λ | 0.95 | 0.92 | Better credit assignment |
| Obs History | 1 | 10 | Temporal context |

---

## 10. Lessons Learned

1. **Slower is often better** - Aggressive learning rates cause policy collapse
2. **Exploration is critical** - High entropy coefficient prevents local optima
3. **Curriculum helps** - Phase training lets agents master skills incrementally  
4. **Observation matters** - Proper normalization and temporal context are essential
5. **Balance rewards carefully** - Dense shaping + sparse terminal rewards work best
6. **Test your tests** - Behavior evaluation bugs can hide real progress
7. **Save everything** - Per-episode checkpoints enable best-checkpoint selection

---

*Document created: January 4, 2026*
*Project: ChatGPTLifeGame - Predator-Prey RL Simulation*
