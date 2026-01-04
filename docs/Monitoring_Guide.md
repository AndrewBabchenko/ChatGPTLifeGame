# Training Monitoring Guide

**Understanding the metrics displayed during training to diagnose issues and track progress**

---

## Table of Contents
- [Episode Statistics](#episode-statistics)
- [PPO Diagnostics](#ppo-diagnostics)
- [Loss Metrics](#loss-metrics)
- [Action Distribution](#action-distribution)
- [Environment Metrics](#environment-metrics)
- [Quick Diagnostic Table](#quick-diagnostic-table)

---

## Episode Statistics

These metrics show the outcome of each episode simulation.

### **Final: Prey=X, Predators=Y**

**Current population at episode end**

- **Healthy range**: Prey 20-150, Predators 5-35
- **Problem signs**:
  - Prey = 0: Predators too effective or prey not learning to flee
  - Prey > 180: Predators not hunting effectively
  - Predators = 0: Starvation threshold too low or predators too passive

### **Births, Deaths, Meals**

**Population dynamics events during episode**

| Metric | What it means | Healthy range |
|--------|---------------|---------------|
| Births | Successful mating events | 10-50 per episode |
| Deaths | Total deaths (all causes) | 10-60 per episode |
| Meals | Successful predator hunts | 5-30 per episode |

**Interpretation**:
- High births, low meals = Prey thriving, predators struggling
- High meals, low births = Predators dominant, prey declining
- Balanced = Ecosystem equilibrium achieved

### **Exhaustion, Old Age, Starvation**

**Breakdown of death causes**

| Death Type | Cause | If too high |
|------------|-------|-------------|
| Exhaustion | Energy dropped to 0 from movement | Reduce `ENERGY_DECAY_RATE` |
| Old Age | Reached `MAX_AGE` steps | Normal, indicates survival |
| Starvation | Predator didn't eat for `STARVATION_THRESHOLD` steps | Predators can't hunt |

**Ideal balance**: Mix of all three indicates a functioning ecosystem. Starvation-only deaths suggest hunting issues.

---

## PPO Diagnostics

These metrics indicate training stability and policy optimization health.

### **KL Divergence (KL: 0.00X)**

**How much the policy changed this update**

KL divergence measures the "distance" between the old policy and new policy.

| KL Value | Meaning | Action |
|----------|---------|--------|
| < 0.005 | Very small updates | Normal, conservative learning |
| 0.005-0.02 | Healthy learning | ✅ Ideal range |
| 0.02-0.05 | Moderate changes | Monitor for stability |
| > 0.05 | Large policy shift | ⚠️ May cause instability |

**Why it matters**:
- PPO uses KL to prevent catastrophic policy changes
- High KL means the network is changing too fast
- Training automatically stops epochs early if KL exceeds 0.03

**If KL is consistently high**:
- Reduce `LEARNING_RATE` by 50%
- Increase `PPO_CLIP_EPSILON` to 0.2
- Reduce `PPO_EPOCHS` to 4

### **Clip Fraction (ClipFrac: 0.XXX)**

**Proportion of experiences where PPO clipping was applied**

Clip fraction shows how often the policy tried to change more than `PPO_CLIP_EPSILON` allows.

| ClipFrac | Meaning | Action |
|----------|---------|--------|
| 0.00-0.10 | Minimal clipping | Policy changes are small |
| 0.10-0.25 | Moderate clipping | ✅ Normal, healthy learning |
| 0.25-0.40 | High clipping | Learning is aggressive |
| > 0.40 | Excessive clipping | ⚠️ May need tuning |

**Interpretation**:
- 0% = Policy barely changing (might be stuck)
- 20-25% = Good balance of learning and stability
- 50%+ = Policy trying to change too much, PPO is restraining it

**If ClipFrac is consistently high**:
- Reduce learning rate
- Increase `PPO_CLIP_EPSILON` (allows larger changes)

---

## Loss Metrics

### **Policy Loss (Policy: P=X.XXX/Pr=X.XXX)**

**How well actions match expected rewards**

- **P** = Prey policy loss
- **Pr** = Predator policy loss

| Value | Meaning |
|-------|---------|
| 0.001-0.05 | ✅ Healthy, stable learning |
| 0.05-0.20 | Active learning, adjusting policy |
| > 0.50 | Large policy corrections needed |
| Negative | Unusual but possible (advantage-weighted) |

**Note**: Policy loss can fluctuate significantly. Focus on trends over 10+ episodes rather than individual values.

### **Value Loss (Value: P=X.X/Pr=X.X)**

**How accurately the network predicts future rewards**

Value loss measures the error in the "critic" predicting cumulative rewards.

| Value | Meaning |
|-------|---------|
| 0.5-5.0 | ✅ Normal range |
| 5.0-15.0 | High variance in outcomes |
| > 20.0 | Prediction very inaccurate |
| Decreasing | ✅ Network learning to predict |

**If value loss is very high**:
- Rewards may have high variance
- Episode outcomes are unpredictable
- Consider reward scaling

### **Entropy (Entropy: P=X.XX/Pr=X.XX)**

**How random/exploratory the policy is**

Entropy measures the "spread" of action probabilities:
- High entropy = Many actions have similar probability (exploring)
- Low entropy = One action dominates (exploiting)

| Value | Meaning | Action |
|-------|---------|--------|
| > 2.0 | Very random | Early training, still exploring |
| 1.0-2.0 | ✅ Balanced | Good exploration/exploitation |
| 0.5-1.0 | Specializing | Policy becoming deterministic |
| < 0.5 | ⚠️ Collapsed | May be stuck, increase `ENTROPY_COEF` |

**Why entropy matters**:
- Too high = Never commits to learned behaviors
- Too low = Stuck in local optimum, won't try new strategies

---

## Action Distribution

### **Prey/Predator Actions: N:X%, NE:X%, E:X%...**

**Which directions animals are choosing**

Shows percentage breakdown of 8 movement directions (N, NE, E, SE, S, SW, W, NW).

**Healthy distribution**: Each direction roughly 10-15%, with slight variation.

**Warning signs**:
- One direction > 30% = **Bias detected** (animals stuck in pattern)
- N + S combined > 50% = Vertical movement bias
- One direction > 50% = Severe collapse, investigate rewards

**If bias detected**:
1. Check if directional loss is working
2. Verify rewards aren't accidentally favoring one direction
3. Increase `ENTROPY_COEF` to encourage exploration
4. Check for bugs in observation encoding

---

## Environment Metrics

### **Supervision Metrics**

```
[Supervision] Target visible: X.X%, Mean target dist: X.X
```

**Target visible**: Percentage of steps where predators see prey (or prey see predators)
- Low (< 20%) = Animals rarely encounter each other, consider smaller grid or more animals
- High (> 80%) = Very dense population

**Mean target dist**: Average distance to nearest target when visible
- Lower = Closer encounters, faster learning
- Higher = More travel required to reach targets

### **GPU Memory**

```
GPU Memory: X.XXB allocated, X.XXB reserved
```

- **Allocated**: Actually used by tensors
- **Reserved**: Total held by PyTorch (includes cache)

**If running out of memory**:
- Reduce `PPO_BATCH_SIZE`
- Reduce `MAX_VISIBLE_ANIMALS`
- Clear cache periodically

---

## Quick Diagnostic Table

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| KL > 0.05 consistently | Learning rate too high | Reduce `LEARNING_RATE` by 50% |
| ClipFrac > 0.4 | Policy changing too fast | Increase `PPO_CLIP_EPSILON` to 0.25 |
| Entropy < 0.5 | Policy collapsed | Increase `ENTROPY_COEF` to 0.06 |
| All prey dead | Predators too effective | Increase `PREY_EVASION_REWARD` |
| All predators dead | Can't hunt successfully | Reduce `STARVATION_THRESHOLD` |
| One action > 30% | Direction bias | Check directional loss, increase entropy |
| Value loss > 20 | Reward variance too high | Scale rewards, check reward logic |
| No births | Mating conditions too strict | Check energy thresholds, cooldowns |
| Meals = 0 | Predators not catching prey | Check hunting rewards, vision range |

---

## Healthy Training Example

```
[12:34:56] Episode 50/150
[12:35:30] Final: Prey=85, Predators=22
[12:35:30] Births=35, Deaths=28, Meals=18
[12:35:30] Exhaustion=3, Old Age=8, Starvation=2
[12:35:30] Rewards: Prey=1250.5, Predator=890.3
[12:35:30] Losses: Policy(P=0.015/Pr=0.022), Value(P=4.2/Pr=5.8), Entropy(P=1.45/Pr=1.82)
  [PPO Diagnostics] KL: 0.008, ClipFrac: 0.18
  [Supervision] Target visible: 45.2%, Mean target dist: 8.3
[12:35:30] Prey Actions: N:14%, E:12%, SE:13%, S:11%, W:15%
[12:35:30] Predator Actions: N:11%, NE:14%, S:13%, SW:12%, W:12%
```

**This is healthy because**:
- ✅ Population balanced (85 prey, 22 predators)
- ✅ Deaths from mixed causes (not just one)
- ✅ KL divergence in good range (0.008)
- ✅ ClipFrac moderate (0.18)
- ✅ Entropy healthy (1.45, 1.82)
- ✅ Action distribution even (no bias warnings)

---

## Unhealthy Training Example

```
[12:34:56] Episode 50/150
[12:35:30] Final: Prey=0, Predators=5
[12:35:30] Births=2, Deaths=138, Meals=95
[12:35:30] Exhaustion=0, Old Age=0, Starvation=43
[12:35:30] Rewards: Prey=-2500.0, Predator=3200.5
[12:35:30] Losses: Policy(P=0.450/Pr=0.008), Value(P=25.8/Pr=2.1), Entropy(P=0.35/Pr=1.95)
  [PPO Diagnostics] KL: 0.065, ClipFrac: 0.52
[12:35:30] Prey Actions: N:45%, NE:8%, E:5%, S:38%, W:4%
[12:35:30] WARNING: Prey bias detected - N = 45%
```

**Problems detected**:
- ❌ Prey extinct (Final: Prey=0)
- ❌ Very high meals (95) = Predators too dominant
- ❌ High KL (0.065) = Training unstable
- ❌ High ClipFrac (0.52) = Policy oscillating
- ❌ Low prey entropy (0.35) = Prey policy collapsed
- ❌ Prey bias (N = 45%) = Stuck moving north

**Fixes needed**:
1. Increase `PREY_EVASION_REWARD`
2. Reduce learning rates
3. Increase `ENTROPY_COEF`
4. Check directional loss supervision
