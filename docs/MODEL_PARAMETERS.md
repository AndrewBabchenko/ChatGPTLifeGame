# Neural Network & Training Parameters Guide

**Complete explanation of PPO reinforcement learning parameters used in the Life Game simulation**

---

## Table of Contents
- [Reinforcement Learning Settings](#reinforcement-learning-settings)
- [PPO Training Parameters](#ppo-training-parameters)
- [Summary Table](#summary-table)

---

## Reinforcement Learning Settings

### **LEARNING_RATE_PREY / LEARNING_RATE_PREDATOR = 0.001**

**How big of a "step" the neural network takes when learning**

Think of it like climbing down a hill to find the lowest point (best strategy):
- **Too high (0.01)**: Take huge steps → might jump over the best spot, unstable learning
- **Too low (0.0001)**: Take tiny steps → learn very slowly, takes forever to improve
- **0.001 (current)**: Goldilocks zone - steady, reliable learning

**Technical**: After calculating gradients, weights are updated by:
```python
weight = weight - 0.001 × gradient
```

---

### **GAMMA = 0.99**

**How much animals value future rewards vs immediate rewards**

**Example scenario**: A predator sees prey 50 steps away.
- Should it chase now (cost energy) for a meal later?
- Or rest now (save energy) and hope for closer prey?

**With GAMMA = 0.99:**
- Reward today: 10 points
- Reward in 10 steps: 10 × 0.99¹⁰ = **9.04 points** (still very valuable!)
- Reward in 50 steps: 10 × 0.99⁵⁰ = **6.05 points** (still worth pursuing)
- Reward in 200 steps: 10 × 0.99²⁰⁰ = **1.34 points** (getting less valuable)

**Why 0.99 for your simulation:**
- Episodes last 200 steps
- Animals need to plan ahead (don't just grab nearest food if it leads to trap)
- Encourages strategic thinking: "Will this action help me in 50 steps?"

**Comparison:**
- **If it were 0.50**: Animals would be short-sighted, only caring about next 5-10 steps
- **If it were 0.999**: Animals would value distant future almost equally to now (sometimes too patient)

---

## PPO Training Parameters

### **PPO_EPOCHS = 16**

**How many times to study the same episode before moving to the next one**

**Analogy**: Like reviewing the same exam multiple times to learn from mistakes.

**What happens after each episode:**
1. Collect all decisions animals made (30,000+ experiences)
2. Train on that data **16 times** (16 epochs)
3. Move to next episode with improved policy

**Trade-off:**
- **More epochs (16)**: Better learning from each episode, but slower training
- **Fewer epochs (4)**: Faster training, but might not extract all lessons
- **Too many (50)**: Overfit - memorize specific episode instead of learning general strategy

---

### **PPO_CLIP_EPSILON = 0.2**

**Prevents the policy from changing too drastically between episodes**

**The Problem PPO Solves:**
Without clipping, one bad episode could destroy 49 episodes of good learning.

**How it works:**
After an episode, PPO calculates: "How different is my new policy from my old policy?"

```python
ratio = new_policy_probability / old_policy_probability
```

**Example:**
- Old policy: 10% chance to run from predator
- New policy: 50% chance to run from predator
- Ratio = 50%/10% = 5.0 (massive change!)

**With CLIP_EPSILON = 0.2:**
- Ratio is clamped between **0.8 and 1.2**
- New policy can be at most **20% different** from old policy
- Change happens gradually over many episodes, not one big jump

**Why this matters:**
- **Stability**: Prevents catastrophic forgetting
- **Reliability**: Small, consistent improvements instead of wild swings
- **0.2 is standard**: Proven to work well across many RL problems

---

### **PPO_BATCH_SIZE = 1024**

**How many experiences to process at once during training**

**Your episode generates:**
- ~30,000 prey experiences
- ~6,000 predator experiences
- Total: ~36,000 experiences

**Training process:**
1. Shuffle all 36,000 experiences
2. Split into batches of 1,024
3. Process each batch through neural network
4. Update weights after each batch
5. Repeat for 16 epochs

**Math:** 36,000 experiences ÷ 1,024 batch size × 16 epochs = **~563 weight updates per episode**

**Why 1,024 is good:**
- **GPU efficiency**: Modern GPUs process 1,024 samples in parallel very fast
- **Gradient stability**: Larger batches = more stable gradients (less noisy)
- **Memory**: 1,024 × 2.9M parameters still fits in 17GB GPU RAM

**Trade-offs:**
- **Smaller (256)**: Noisier gradients, more updates, potentially better exploration
- **Larger (4096)**: Smoother gradients, fewer updates, but requires more VRAM

---

### **VALUE_LOSS_COEF = 0.5**

**How much to weight the "value prediction" error vs "action choice" error**

**The neural network learns two things:**
1. **Policy (Actor)**: "What action should I take?" → Policy Loss
2. **Value (Critic)**: "How good is my current situation?" → Value Loss

**Total loss formula:**
```python
total_loss = policy_loss + 0.5 × value_loss + 0.01 × entropy_loss
```

**Why value matters:**
The value function estimates: "If I'm at 50% energy with 5 predators nearby, how good is that?"
This helps calculate "advantage" = "Was this action better or worse than expected?"

**With VALUE_LOSS_COEF = 0.5:**
- Policy loss contributes 100% to learning
- Value loss contributes 50% to learning
- Model focuses more on "what to do" than "how to evaluate"

**Comparison:**
- **If it were 1.0**: Equal weight (sometimes leads to value network dominating training)
- **If it were 0.1**: Almost ignores value (advantage estimates become unreliable)

---

### **ENTROPY_COEF = 0.01**

**Encourages exploration by rewarding randomness in decisions**

**The Problem:**
Neural networks can get "stuck" choosing the same action repeatedly:
- Predator always chases straight toward prey (predictable, easy for prey to evade)
- Prey always runs directly away (sometimes runs into corner)

**Entropy measures randomness:**
- High entropy: Action probabilities spread out (30% left, 25% right, 25% up, 20% down)
- Low entropy: Action probabilities concentrated (95% straight ahead, 5% everything else)

**With ENTROPY_COEF = 0.01:**
```python
total_loss = policy_loss + 0.5*value_loss - 0.01*entropy
```
(Note the minus sign - we SUBTRACT entropy from loss, so higher entropy = lower loss = better)

**Effect:**
- Adds small bonus for keeping options open
- Prevents premature convergence to suboptimal strategies
- 0.01 is small enough not to make animals random, but enough to maintain diversity

**Analogy**: Like a teacher saying "try different approaches" instead of "only use method A"

---

### **MAX_GRAD_NORM = 0.5**

**Caps the size of gradient updates to prevent training explosions**

**The Problem:**
Sometimes gradients can become extremely large (e.g., when network makes terrible prediction):
- Normal gradient: 0.01 → small, reasonable update
- Exploding gradient: 1000 → massive update that breaks the network

**Gradient clipping:**
1. Calculate all gradients
2. Compute total magnitude (norm) = √(grad₁² + grad₂² + ... + grad_n²)
3. If norm > 0.5, scale all gradients down: `gradient = gradient × (0.5 / norm)`

**Example:**
- Gradient norm = 2.0 (too big!)
- Scale factor = 0.5 / 2.0 = 0.25
- All gradients multiplied by 0.25 to bring norm down to 0.5

**Why 0.5:**
- Conservative clipping (fairly aggressive)
- Prevents rare catastrophic updates from ruining 49 episodes of training
- Standard for PPO (prevents policy collapse)

---

### **GAE_LAMBDA = 0.95**

**Controls the bias-variance trade-off in advantage calculation**

**What is "advantage"?**
Advantage = "How much better was this action than average?"
- Positive advantage: Action was good, do it more
- Negative advantage: Action was bad, avoid it

**The Problem:**
How far ahead should we look to judge if an action was good?

**Two extremes:**
1. **1-step (λ=0)**: Only look at immediate next reward
   - Low variance (consistent) but high bias (might miss long-term consequences)
2. **Full episode (λ=1)**: Look at all remaining rewards until episode ends
   - Low bias (accurate) but high variance (noisy, influenced by later random events)

**GAE with λ=0.95 is a weighted average:**
```
Advantage = 0.05×(1-step) + 0.0475×(2-step) + 0.045×(3-step) + ... + small×(200-step)
```

**Effect:**
- Heavily weights nearby rewards (next 20-30 steps)
- Still considers distant future (next 100+ steps) but with less weight
- **95% of weight is on rewards within ~60 steps**

**Why 0.95 is standard:**
- Balances learning from immediate feedback AND long-term outcomes
- Works well for episodes of 100-1000 steps
- Less susceptible to random noise than λ=1.0

---

## Summary Table

| Parameter | Value | Primary Effect | If Lower | If Higher |
|-----------|-------|----------------|----------|-----------|
| **LEARNING_RATE** | 0.001 | Steady learning speed | Too slow progress | Unstable training |
| **GAMMA** | 0.99 | Long-term planning (200 steps) | Myopic (short-sighted) | Too patient (over-values distant future) |
| **PPO_EPOCHS** | 16 | Study each episode 16 times | Less learning per episode | Overfitting to specific episodes |
| **PPO_CLIP_EPSILON** | 0.2 | Max 20% policy change/update | Too conservative | Unstable (wild policy swings) |
| **PPO_BATCH_SIZE** | 1024 | GPU-efficient batching | Noisy gradients | More VRAM needed |
| **VALUE_LOSS_COEF** | 0.5 | Balance actor/critic learning | Unreliable value estimates | Value network dominates |
| **ENTROPY_COEF** | 0.01 | Slight exploration bonus | Premature convergence | Too random (excessive exploration) |
| **MAX_GRAD_NORM** | 0.5 | Prevent gradient explosions | Allows dangerous jumps | Too conservative (slow learning) |
| **GAE_LAMBDA** | 0.95 | ~60-step advantage lookahead | Only immediate rewards | Full episode (noisy estimates) |

---

## Quick Reference: Training Flow

```
Episode Loop (50 episodes):
  ├─ Simulation Step (200 steps)
  │   ├─ Animals observe environment
  │   ├─ Neural network decides actions
  │   ├─ Actions executed, rewards calculated
  │   └─ Experiences stored (~36,000 total)
  │
  └─ PPO Update Phase
      ├─ Shuffle all experiences
      ├─ Split into batches of 1024
      ├─ For each of 16 epochs:
      │   ├─ Process all batches (~35 batches)
      │   ├─ Calculate policy loss (actor)
      │   ├─ Calculate value loss (critic)
      │   ├─ Calculate entropy bonus
      │   ├─ Combine: loss = policy + 0.5*value - 0.01*entropy
      │   ├─ Backpropagate gradients
      │   ├─ Clip gradients (max_norm=0.5)
      │   └─ Update weights (learning_rate=0.001)
      │
      └─ Result: ~563 weight updates per episode
```

---

## Configuration Best Practices

**Your current settings are standard PPO best practices:**
- ✅ Learning rate: 0.001 (Adam optimizer default)
- ✅ Gamma: 0.99 (standard for medium-length episodes)
- ✅ Clip epsilon: 0.2 (proven PPO default)
- ✅ Batch size: 1024 (GPU-optimized)
- ✅ Value coef: 0.5 (standard actor-critic balance)
- ✅ Entropy coef: 0.01 (maintains exploration)
- ✅ Grad norm: 0.5 (prevents instability)
- ✅ GAE lambda: 0.95 (balanced advantage estimation)

**These parameters are well-tuned for stable, reliable learning in multi-agent predator-prey scenarios!** 🎯

---

## When to Adjust Parameters

**If training is unstable (losses jumping wildly):**
- Reduce `LEARNING_RATE` to 0.0005
- Reduce `MAX_GRAD_NORM` to 0.3
- Increase `PPO_CLIP_EPSILON` to 0.3

**If learning is too slow:**
- Increase `LEARNING_RATE` to 0.002
- Increase `PPO_EPOCHS` to 20
- Reduce `ENTROPY_COEF` to 0.005

**If animals are too random/chaotic:**
- Reduce `ENTROPY_COEF` to 0.005
- Increase `PPO_CLIP_EPSILON` to 0.3

**If animals converge to boring strategies:**
- Increase `ENTROPY_COEF` to 0.02
- Reduce `PPO_EPOCHS` to 10

**For shorter episodes (50-100 steps):**
- Reduce `GAMMA` to 0.95
- Reduce `GAE_LAMBDA` to 0.90

**For longer episodes (500+ steps):**
- Increase `GAMMA` to 0.995
- Increase `GAE_LAMBDA` to 0.97

---

*Last updated: December 26, 2025*
*Model: 2.9M parameters, Actor-Critic with Multi-Head Attention*
