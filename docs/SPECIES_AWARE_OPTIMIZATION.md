# Species-Aware Model Optimization Summary

## ✅ Model Features for Species Recognition

### Self-State Input (21 features)
The model receives information about itself:
- **Index 2**: Is species A (prey)
- **Index 4**: Is predator
- **Index 7-9**: Nearest predator (distance, dx, dy)
- **Index 10-12**: Nearest prey (distance, dx, dy)  
- **Index 13**: Count of visible predators
- **Index 14**: Count of visible prey
- **Index 20**: **Population pressure** (current_animals / usable_capacity)

### Visible Animals Input (8 features × 24 animals)
For each visible animal, the model sees:
- **Index 0-1**: Absolute position (x, y)
- **Index 2-3**: Relative direction (dx, dy)
- **Index 4**: Distance (normalized)
- **Index 5**: **Is predator?** (1.0 for predators, 0.0 for prey)
- **Index 6**: **Is prey?** (1.0 for prey, 0.0 for predators)
- **Index 7**: **Same species?** (1.0 if same, 0.0 if different)

## ✅ Reward Shaping for Species-Aware Behavior

### Prey Rewards (Avoid Predators)
```python
# Base survival
reward = config.SURVIVAL_REWARD  # +0.2

# Distance-based evasion reward
if predator_nearby and moved_away:
    reward += config.PREY_EVASION_REWARD * (1.0 - distance)  # Up to +5.0
    # Closer predator = bigger reward for evading

# Overcrowding penalty (respects OTHER_SPECIES_CAPACITY)
if same_species_count > usable_capacity:
    overcrowd_ratio = (count - capacity) / capacity
    reward += config.OVERPOPULATION_PENALTY * ratio  # Up to -10.0
```

### Predator Rewards (Chase Prey)
```python
# Base survival
reward = config.SURVIVAL_REWARD  # +0.2

# Distance-based approach reward
if prey_nearby and moved_closer:
    reward += config.PREDATOR_APPROACH_REWARD * (1.0 - distance)  # Up to +2.0
    # Closer prey = bigger reward for approaching

# Extra hungry predator bonus
if hungry and prey_nearby:
    reward += 1.0 * (1.0 - distance)  # Up to +1.0

# Eating reward (from config)
if ate_prey:
    reward += config.PREDATOR_EAT_REWARD  # +15.0

# Overcrowding penalty (respects OTHER_SPECIES_CAPACITY)
if same_species_count > usable_capacity:
    overcrowd_ratio = (count - capacity) / capacity
    reward += config.OVERPOPULATION_PENALTY * ratio  # Up to -10.0
```

## ✅ OTHER_SPECIES_CAPACITY Implementation

### In Config (src/config.py)
```python
MAX_ANIMALS = 400
OTHER_SPECIES_CAPACITY = 5  # Reserve 5 slots for other species
```

### In Population Pressure Feature (animal.py)
```python
max_current_animals = max(1, config.MAX_ANIMALS - config.OTHER_SPECIES_CAPACITY)
current_population_ratio = min(1.0, len(animals) / max_current_animals)
# = current_animals / 395

# This tells the model:
# - "We're at 90% capacity" (high pressure)
# - "We're at 20% capacity" (room to grow)
```

### In Overcrowding Penalty (train_advanced.py)
```python
same_species_count = sum(1 for a in animals if not a.predator)  # Count prey
usable_capacity = config.MAX_ANIMALS - config.OTHER_SPECIES_CAPACITY  # 395

if same_species_count > usable_capacity:
    # Species exceeded its allocated space
    overcrowd_ratio = (same_species_count - usable_capacity) / usable_capacity
    reward += config.OVERPOPULATION_PENALTY * overcrowd_ratio  # Negative
```

## 🎯 How It All Works Together

### 1. **Model Perception**
- Self-state tells animal what it is (prey/predator)
- Visible animals show species of neighbors
- Population pressure indicates capacity constraints

### 2. **Attention Mechanism**
- 8 attention heads process visible animals
- Can learn to focus on:
  - Predators (for prey to avoid)
  - Prey (for predators to chase)
  - Same species (for mating/cooperation)

### 3. **Reward Incentives**
- **Prey**: Get rewarded for increasing distance from predators
- **Predators**: Get rewarded for decreasing distance to prey
- **Both**: Get penalized for overpopulation beyond capacity

### 4. **Capacity Enforcement**
- Population pressure feature (input index 20) always visible
- Overcrowding penalty kicks in when exceeding usable_capacity
- Model learns to self-regulate population growth

## 📊 Expected Behaviors

### Prey Will Learn To:
- ✅ Detect nearby predators (features 7-9, 13)
- ✅ Move away from predators (+5.0 reward)
- ✅ Monitor population pressure (feature 20)
- ✅ Avoid overpopulation (-10.0 penalty)

### Predators Will Learn To:
- ✅ Detect nearby prey (features 10-12, 14)
- ✅ Move toward prey (+2.0 to +3.0 reward)
- ✅ Hunt more aggressively when hungry (extra +1.0)
- ✅ Catch prey (+15.0 reward)
- ✅ Avoid overpopulation (-10.0 penalty)

## 🔧 Training Configuration

Current settings maximize species interaction:
```python
MAX_VISIBLE_ANIMALS = 24      # See many neighbors
PPO_BATCH_SIZE = 1024          # Large batches
PPO_EPOCHS = 16                # Many training iterations
steps_per_episode = 200        # Long episodes
```

This gives the model:
- **More context** (24 visible animals vs 15)
- **More updates** (16 epochs vs 8)
- **More experience** (200 steps vs 100)

## 🧪 Testing Species Awareness

To verify the model is using species info:
1. Watch prey flee from predators
2. Watch predators chase prey
3. Monitor population stays under capacity
4. Check reward logs show evasion/approach bonuses

## 💡 Why This Works

The model has:
1. ✅ **Input**: Species information (self + visible animals)
2. ✅ **Processing**: Multi-head attention to focus on relevant species
3. ✅ **Output**: Actions that move toward/away from species
4. ✅ **Feedback**: Rewards that reinforce species-aware behavior
5. ✅ **Constraints**: Capacity penalties that enforce limits

Everything is in place for the model to learn proper predator-prey dynamics! 🎯
