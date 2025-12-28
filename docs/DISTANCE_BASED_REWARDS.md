# Distance-Based Reward Shaping Fix

**Date**: December 26, 2024  
**Status**: Implemented  
**Problem**: Models achieved high value estimates but weren't learning directional movement

## Problem Analysis

### Discovery
After implementing direction correctness evaluation in checkpoint comparisons, we discovered a critical learning failure:

**Predator Hunting (prey at South):**
```
Episode 5:  Value +115.55, Action NE (away),          17.0% confidence
Episode 10: Value +230.50, Action E  (perpendicular), 13.4% confidence
Episode 15: Value +169.19, Action E  (perpendicular), 17.7% confidence
Episode 20: Value +156.90, Action NE (away),          17.0% confidence
```

**Key Finding**: Predators NEVER chose the correct direction (South) toward prey across all 4 checkpoints, despite high value estimates.

**Prey Escaping (predator at North):**
```
Episode 5:   Value +3.36,  Action E  (perpendicular),  13.7% confidence
Episode 10:  Value +2.48,  Action NE (toward - BAD!),  13.5% confidence
Episode 15:  Value +5.86,  Action W  (perpendicular),  15.8% confidence
Episode 20:  Value +0.55,  Action SE (away - GOOD!),   14.3% confidence
```

**Key Finding**: Only 1 out of 4 checkpoints chose correct escape direction.

### Root Cause

The reward structure created a **sparse reward problem**:

```python
PREDATOR_APPROACH_REWARD = 0.5   # Tiny reward for being near prey
PREDATOR_EAT_REWARD = 80.0       # Large reward for catching prey
# Ratio: 1:160 - approach reward is negligible!
```

**Why This Fails:**
- Predators only get significant reward when they successfully catch prey
- No feedback for intermediate behaviors (moving toward prey, closing distance)
- Models learn general survival patterns but NOT spatial directional movement
- This is a classic "needle in haystack" problem - rare lucky catches provide only signal

**What Models Were Learning:**
- Energy management
- Population dynamics
- General movement patterns
- BUT NOT: "move toward prey" or "move away from predator"

## Solution: Distance-Based Reward Shaping

### Concept
Provide **dense reward signal** by rewarding every step that closes distance to targets:

- **For Predators**: Reward for reducing distance to nearest prey
- **For Prey**: Reward for increasing distance from nearest predator

This gives immediate feedback: "getting warmer" vs "getting colder"

### Implementation Details

#### 1. Distance Tracking
Added `previous_distances` dictionary to track step-to-step distance changes:

```python
# Episode initialization (line ~237)
previous_distances = {}  # {animal_id: {target_id: distance}}
```

#### 2. Predator Distance Rewards (lines ~410-450)

```python
# Find closest prey
closest_prey = None
min_dist = float('inf')
for other in animals:
    if not other.predator:
        dx = abs(other.x - animal.x)
        dy = abs(other.y - animal.y)
        dx = min(dx, config.GRID_SIZE - dx)  # Handle toroidal wrap
        dy = min(dy, config.GRID_SIZE - dy)
        dist = (dx**2 + dy**2)**0.5
        if dist < min_dist:
            min_dist = dist
            closest_prey = other

# Reward for reducing distance
if closest_prey and moved:
    prey_id = id(closest_prey)
    current_dist = min_dist
    
    if animal_id in previous_distances and prey_id in previous_distances[animal_id]:
        prev_dist = previous_distances[animal_id][prey_id]
        distance_change = prev_dist - current_dist
        
        # Strong reward for closing distance
        if distance_change > 0:
            reward += 3.0 * min(distance_change / 5.0, 1.0)
            # Max: 3.0 per step (moving 5+ tiles closer)
            # Typical: 0.6-1.5 per step (moving 1-2.5 tiles closer)
        
        # Penalty for moving away when hungry
        elif distance_change < 0 and animal.steps_since_last_meal > config.HUNGER_THRESHOLD:
            reward -= 1.5 * min(abs(distance_change) / 5.0, 0.5)
            # Max: -0.75 per step
    
    # Store for next step
    if animal_id not in previous_distances:
        previous_distances[animal_id] = {}
    previous_distances[animal_id][prey_id] = current_dist
```

**Reward Scale:**
- Closing distance: +3.0 max per step (vs +80 for catch)
- Moving away (hungry): -1.5 max per step
- Ratio: 3.75% of terminal reward per step
- Over 20 steps: Can accumulate +60 reward approaching prey

#### 3. Prey Distance Rewards (lines ~330-370)

```python
# Find closest predator
closest_predator = None
min_dist = float('inf')
for other in animals:
    if other.predator:
        dx = abs(other.x - animal.x)
        dy = abs(other.y - animal.y)
        dx = min(dx, config.GRID_SIZE - dx)
        dy = min(dy, config.GRID_SIZE - dy)
        dist = (dx**2 + dy**2)**0.5
        if dist < min_dist:
            min_dist = dist
            closest_predator = other

# Reward for increasing distance (escaping)
if closest_predator and moved:
    predator_id = id(closest_predator)
    current_dist = min_dist
    
    if animal_id in previous_distances and predator_id in previous_distances[animal_id]:
        prev_dist = previous_distances[animal_id][predator_id]
        distance_change = current_dist - prev_dist  # Note: reversed for prey!
        
        # Strong reward for escaping (increasing distance)
        if distance_change > 0 and current_dist < 15:  # Only if predator nearby
            reward += 2.0 * min(distance_change / 5.0, 1.0)
            # Max: 2.0 per step when escaping
        
        # Penalty for moving toward predator
        elif distance_change < 0 and current_dist < 10:
            reward -= 1.0 * min(abs(distance_change) / 5.0, 0.5)
            # Max: -0.5 per step
    
    # Store for next step
    if animal_id not in previous_distances:
        previous_distances[animal_id] = {}
    previous_distances[animal_id][predator_id] = current_dist
```

**Reward Scale:**
- Escaping: +2.0 max per step (only when predator nearby)
- Moving toward: -1.0 max per step (only when predator close)

#### 4. Original Proximity Rewards (Reduced)

The original approach/evasion rewards are kept but scaled down by 50%:

```python
# Predators
reward += config.PREDATOR_APPROACH_REWARD * 0.5 * (1.0 - nearest_prey_dist)

# Prey
reward += config.PREY_EVASION_REWARD * 0.5 * (1.0 - nearest_pred_dist)
```

This prevents conflicting signals while maintaining some proximity awareness.

## Expected Results

### Short-Term (Episodes 1-20)
1. **Action Direction Correctness**:
   - Predators should start choosing "toward" actions (S, SW, SE when prey is South)
   - Prey should choose "away" actions (SE, S, SW when predator is North)
   - Expect 50-70% correctness by Episode 10, 70-90% by Episode 20

2. **Value Estimates**:
   - May be lower initially as models relearn
   - Should stabilize higher than before once directional learning works

3. **Action Probabilities**:
   - Should increase from 13-17% (near random) to 30-50%
   - Indicates model is becoming confident about correct actions

### Long-Term (Episodes 20-50)
1. **Hunting Success Rate**:
   - Predators should catch prey more efficiently
   - Starvation deaths should decrease

2. **Survival Time**:
   - Prey should survive longer on average
   - Population should be more stable

3. **Curriculum Learning Synergy**:
   - Stage 1 (abundant prey, no starvation): Fast directional learning
   - Stage 2-4: Refinement and generalization

## Validation Plan

### 1. Run Training
```powershell
python scripts/train_advanced.py
```

Training will:
- Apply curriculum learning stages automatically
- Save checkpoints at episodes 5, 10, 15, 20, etc.
- Use distance-based rewards from Episode 1

### 2. Compare Checkpoints
```powershell
python scripts/compare_checkpoints.py
```

**Success Criteria:**
- Episode 5: At least 1/2 scenarios showing "toward"/"away (good)" actions
- Episode 10: At least 3/4 scenarios showing correct directions
- Episode 15: All scenarios showing correct directions
- Episode 20: High confidence (30-50%) on correct directions

**Example Expected Output:**
```
Predator Hunting (Hungry, prey at South):
Episode 5:  +45.20   S   toward           25.3%
Episode 10: +78.50   S   toward           38.7%  [+] Improving
Episode 15: +92.10   SE  toward           45.2%  [+] Improving
Episode 20: +105.30  S   toward           51.8%  [+] Improving

Prey Escaping (predator at North):
Episode 5:  +12.40   SE  away (good)      22.1%
Episode 10: +18.90   S   away (good)      35.4%  [+] Improving
Episode 15: +24.60   SE  away (good)      42.8%  [+] Improving
Episode 20: +31.20   S   away (good)      48.3%  [+] Improving
```

### 3. Analyze Attention
```powershell
python scripts/analyze_attention.py
```

Check if:
- Direction features (nearest_prey_dx, nearest_prey_dy) have high importance
- Distance features show correlation with value estimates
- Feature usage is more focused than before

### 4. Monitor Training Logs
Watch for:
- Average episode rewards increasing
- Predator starvation rate decreasing
- Population stability improving
- No value explosions or collapses

## Tuning Parameters

If learning is too slow or unstable, adjust these:

### Predator Approach Reward
```python
# Current: 3.0 * min(distance_change / 5.0, 1.0)
# Increase if learning too slow:
reward += 5.0 * min(distance_change / 5.0, 1.0)  # Stronger signal
# Decrease if too noisy:
reward += 2.0 * min(distance_change / 5.0, 1.0)  # Gentler signal
```

### Prey Escape Reward
```python
# Current: 2.0 * min(distance_change / 5.0, 1.0)
# Adjust similarly to predator rewards
reward += 3.0 * min(distance_change / 5.0, 1.0)  # Stronger
```

### Distance Normalization
```python
# Current: / 5.0 (normalizes 5-tile movement to 1.0)
# For faster learning (more sensitivity):
reward += 3.0 * min(distance_change / 3.0, 1.0)  # 3 tiles = max reward
# For slower, more stable learning:
reward += 3.0 * min(distance_change / 8.0, 1.0)  # 8 tiles = max reward
```

### Penalty Strength
```python
# Current: -1.5 for predators, -1.0 for prey
# Increase if models still move wrong direction:
reward -= 3.0 * min(abs(distance_change) / 5.0, 0.5)  # Stronger penalty
```

## Technical Notes

### Why This Works (RL Theory)
1. **Sparse Rewards Problem**: When rewards only come at terminal states (catching prey), exploration is random and learning is slow
2. **Reward Shaping**: Adding intermediate rewards based on progress toward goal accelerates learning
3. **Potential-Based Shaping**: Our distance-based rewards approximate the "potential" of a state (how close to success)
4. **Dense Signal**: Every step provides feedback, enabling gradient descent to find policies faster

### Toroidal Grid Handling
```python
dx = min(dx, config.GRID_SIZE - dx)
dy = min(dy, config.GRID_SIZE - dy)
```
This ensures we calculate the shortest distance on a toroidal (wrap-around) grid.

### Memory Efficiency
- Only stores distances for animals that are actively tracking targets
- Clears previous_distances at episode end
- Uses Python's id() for fast animal identification

### Edge Cases
1. **No targets available**: If no prey/predators exist, distance tracking is skipped
2. **First step of episode**: No previous distance available, so no reward/penalty applied
3. **Target switches**: If closest prey/predator changes, previous distance is from different target (minor issue, self-corrects)

## Comparison with Original

### Before (Sparse Rewards)
```
Reward Structure:
- Catch prey: +80.0 (terminal)
- Near prey: +0.5 (negligible)
- Starvation: -100.0 (terminal)

Learning Signal:
- Only significant reward comes from lucky catches
- No gradient toward correct behavior
- Random exploration until lucky success
```

### After (Dense Rewards)
```
Reward Structure:
- Catch prey: +80.0 (terminal)
- Close distance 1 tile: +0.6 (immediate)
- Close distance 2 tiles: +1.2 (immediate)
- Move away when hungry: -0.3 to -0.75 (immediate)
- Starvation: -100.0 (terminal)

Learning Signal:
- Every step provides feedback
- Clear gradient toward closing distance
- Rapid learning of approach behavior
- Terminal rewards remain primary goal
```

### Reward Accumulation Example

**Scenario**: Predator starts 15 tiles from prey, takes 10 steps to catch

**Before:**
- Steps 1-9: +0.05 to +0.5 (negligible approach rewards)
- Step 10: +80.0 (catch)
- Total: ~82.5

**After:**
- Step 1: +0.6 (closed 1 tile) + survival
- Step 2: +0.9 (closed 1.5 tiles) + survival
- Step 3: +1.2 (closed 2 tiles) + survival
- ...
- Steps 1-9: ~15-25 accumulated distance rewards
- Step 10: +80.0 (catch)
- Total: ~95-105

The difference: **10-20% more reward for successful hunts**, but critically, **immediate feedback every step** that guides exploration toward success.

## Related Documentation

- [Curriculum Learning](CURRICULUM_LEARNING.md): Complementary system that provides easier initial learning
- [Checkpoint Analysis Dec 26](CHECKPOINT_ANALYSIS_DEC26.md): Original analysis that discovered the directional learning failure
- [scripts/compare_checkpoints.py](../scripts/compare_checkpoints.py): Tool for validating direction correctness

## References

**Reward Shaping in RL:**
- Ng, A. Y., Harada, D., & Russell, S. (1999). "Policy invariance under reward transformations: Theory and application to reward shaping"
- Laud, A. D. (2004). "Theory and application of reward shaping in reinforcement learning"

**Sparse vs Dense Rewards:**
- Pathak, D., et al. (2017). "Curiosity-driven exploration by self-supervised prediction"
- Burda, Y., et al. (2018). "Large-Scale Study of Curiosity-Driven Learning"
