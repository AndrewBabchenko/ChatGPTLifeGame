# Basic Description (100 level): How Learning Works

## Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [How the Learning Works: A Complete Beginner's Guide](#how-the-learning-works-a-complete-beginners-guide)
3. [How This Neural Network Architecture Helps Learning](#how-this-neural-network-architecture-helps-learning)
4. [Technical Deep Dive: How the Networks Work](#1-how-the-networks-work)
5. [How Animals Make Decisions](#2-how-animals-make-decisions)
6. [How Memory Is Used](#3-how-memory-is-used)
7. [Important Details](#4-important-details)
8. [Visual Map + Quick Start](#5-visual-map--quick-start)
9. [Glossary of Objects](#glossary-of-objects)
10. [Call Flow: One Training Step](#call-flow-one-training-step)

---

## High-Level Overview

This is a **2D predator-prey ecosystem simulation** using reinforcement learning (PPO/Proximal Policy Optimization). Two neural networks - one for prey (species A) and one for predators (species B) - control animal movement on a 100×100 toroidal grid.

**Architecture**: Each animal has an `ActorCriticNetwork` with:
- **Cross-attention** over visible neighbors (up to 24 animals)
- **Dual action heads**: Turn (3 choices: left/straight/right) and Move (8 cardinal directions)
- **Stacked temporal observations** (10 frames of 323 features each = 3,230 input dims)

**Training loop**: Animals move → eat/mate/die → rewards computed → PPO updates networks. The system uses **on-policy learning** with `PPOMemory` (no traditional replay buffer). Training occurs at episode end, not per-step.

**Curriculum learning** progresses through 4 phases: Hunt/Evade → Starvation → Reproduction → Full Ecosystem. Each phase loads checkpoints from the previous phase.

The simulation includes pheromone trails, energy/starvation mechanics, grass foraging for prey, and distance-based reward shaping for directional learning.

---

# How the Learning Works: A Complete Beginner's Guide

## The Big Picture: What Are We Trying to Do?

Imagine you're training a dog. The dog doesn't understand English - it just tries random things, and when it does something good, you give it a treat. Over time, the dog learns: "When I sit, I get a treat. I should sit more often."

This simulation works the same way:
- **The animals are like dogs** that don't know what to do
- **The neural network is the animal's brain** that decides what to do
- **Rewards are like treats** that tell the brain "that was good, do more of that"
- **Penalties are like scolding** that tell the brain "that was bad, don't do that"

---

## What Is a Neural Network?

A neural network is just a **mathematical function** that takes numbers in and spits numbers out.

```
INPUT (numbers describing the situation)
    ↓
  [BRAIN MATH]
    ↓
OUTPUT (numbers saying what to do)
```

In this project:
- **Input**: 3,230 numbers describing what the animal sees (where am I? where are predators? how hungry am I?)
- **Output**: 11 numbers representing probabilities for actions (3 for turning + 8 for moving)

The "brain math" is made of **weights** - thousands of numbers that get multiplied with the inputs. These weights start random and get adjusted during training.

---

## The Core Learning Loop (One Episode)

Think of training like a school day:

### Step 1: Morning - Animals Try Stuff (300 steps)

```
For 300 steps:
    Each animal:
        1. Looks around (builds input numbers)
        2. Brain calculates what to do (neural network forward pass)
        3. Does the action (moves, turns)
        4. Something happens (survives, dies, eats, gets eaten)
        5. Gets a score for that action (reward or penalty)
        6. Writes it all down in a notebook (memory)
```

During this phase, the brain is **NOT learning**. It's just trying things and taking notes.

### Step 2: Afternoon - Study the Notes (PPO Update)

After all 300 steps, the animal sits down with its notebook and thinks:

```
"Let me look at everything I did today..."

Page 1: I was at position (50,50), saw a predator nearby, 
        turned left, moved south. 
        Result: +2.5 points (got away from predator - good!)

Page 2: I was at position (51,49), predator still visible,
        turned right, moved north.
        Result: -1.0 points (got closer to predator - bad!)

Page 3: ...
```

Now the brain adjusts its weights:
- **Actions that got positive rewards** → make the brain MORE likely to do those again
- **Actions that got negative rewards** → make the brain LESS likely to do those again

This adjustment is called **backpropagation** - it's calculus that figures out "which weights should I change, and by how much, to get better scores?"

### Step 3: Tomorrow - Try Again

The next episode starts with:
- Fresh animals (old ones are gone)
- Same brain weights (learning carries over)
- Brain should make slightly better decisions now

Repeat for 150 episodes. Each day the brain gets a little smarter.

---

## What Are "Weights" and How Do They Change?

Imagine the simplest possible brain:

```
Input: Distance to predator = 0.3 (close!)

Brain calculation:
    output = input × weight
    
If weight = -10:
    output = 0.3 × -10 = -3 (very negative = "RUN AWAY!")
    
If weight = +10:
    output = 0.3 × +10 = +3 (very positive = "GO TOWARD IT!")
```

At the start, weights are random. The brain might output "go toward predator" which gets the prey killed (big negative reward).

**Learning adjusts the weight:**
```
Old weight: +10 (wrong! led to death)
Adjustment: -20 (change it a lot because the penalty was huge)
New weight: -10 (now it says "run away")
```

Real neural networks have millions of weights, but the principle is the same: **adjust weights to get more reward**.

---

## What Is PPO? (The Learning Algorithm)

PPO stands for "Proximal Policy Optimization." Here's what it actually does:

### The Problem PPO Solves

Imagine you're learning to ride a bike. If someone said:
- "You fell, so from now on ALWAYS turn left at maximum force"

That would be terrible advice - you'd overcorrect and crash the other way.

PPO says: **"Change your behavior, but not too much at once."**

### How PPO Works (Simplified)

```
1. Look at what you did:      "I moved North when predator was South"
2. Look at the outcome:       "I got +5 reward (escaped!)"
3. Calculate how much more likely to do this:
   - Outcome was good → increase probability of "move North when predator South"
   - BUT cap the increase to avoid overcorrecting
4. Actually adjust the weights (backpropagation)
```

The "cap" is called **clipping** - it prevents the brain from changing too drastically in one update.

---

## Concrete Example: One Prey Learning to Flee

### Before Training (Episode 1)

```
Situation: Predator is 3 cells to the North

Brain outputs (random weights):
  Move North:  12.5%  ← Toward predator (bad!)
  Move South:  12.5%  ← Away from predator (good!)
  Move East:   12.5%
  Move West:   12.5%
  ... (all roughly equal because weights are random)

Animal randomly picks: North (unlucky!)
Result: Gets eaten → Reward: -25 points

Notebook entry: "At position X, predator North, I went North → -25 points"
```

### After PPO Update

```
Brain thinks: "Going North when predator is North = terrible"

Weight adjustment:
  - Weights connecting "predator North" to "move North" → DECREASE
  - Weights connecting "predator North" to "move South" → INCREASE
```

### Episode 2 (After Learning)

```
Same situation: Predator is 3 cells to the North

Brain outputs (adjusted weights):
  Move North:   5%   ← Reduced! Brain learned this is bad
  Move South:  30%   ← Increased! Brain learned this is good
  Move East:   15%
  Move West:   15%
  ...

Animal picks: South (more likely now!)
Result: Escapes → Reward: +2.5 points

Brain gets even more confident that South is correct.
```

### After 150 Episodes

```
Same situation: Predator is North

Brain outputs (well-trained):
  Move North:   1%   ← Almost never
  Move South:  70%   ← Almost always
  ...
  
Animal consistently flees correctly.
```

---

## Why Two Separate Brains?

Prey and predators need to learn **opposite things**:

| Prey Brain | Predator Brain |
|------------|----------------|
| "Predator nearby → run away" | "Prey nearby → chase it" |
| Reward: +2.5 for increasing distance | Reward: +5.0 for decreasing distance |
| Penalty: -25 for being eaten | Reward: +30 for eating prey |

If they shared a brain, it would get confused: "Should I run toward or away from the other animal?"

---

## The Reward is the Teacher

The entire learning process is driven by **what gives rewards**:

```python
# Prey rewards (from config.py)
SURVIVAL_REWARD = 0.2        # +0.2 every step you're alive
PREY_EVASION_REWARD = 2.5    # +2.5 for getting away from predator
GRASS_EAT_REWARD = 1.0       # +1.0 for eating grass
EATEN_PENALTY = -20.0        # -20 for getting eaten
DEATH_PENALTY = -5.0         # -5 for dying (any cause)
```

The brain doesn't "know" that predators are dangerous. It just learns:
- "When I see a predator and run away → positive number"
- "When I see a predator and don't run → negative number"
- "Therefore, run away from predators"

---

## Summary: The Learning Cycle

```
┌─────────────────────────────────────────────────────────┐
│                    ONE EPISODE                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. EXPERIENCE COLLECTION (300 steps)                   │
│     ┌─────────────────────────────────────────────┐     │
│     │ Animal sees world → Brain picks action →    │     │
│     │ Action happens → Get reward → Write it down │     │
│     │         (repeat 300 times)                  │     │
│     └─────────────────────────────────────────────┘     │
│                         ↓                               │
│  2. LEARNING (PPO Update)                               │
│     ┌─────────────────────────────────────────────┐     │
│     │ Read all notes → Find patterns →            │     │
│     │ "These actions got good rewards" →          │     │
│     │ Adjust weights to do those more often       │     │
│     └─────────────────────────────────────────────┘     │
│                         ↓                               │
│  3. SAVE BRAIN (Checkpoint)                             │
│     ┌─────────────────────────────────────────────┐     │
│     │ Save the adjusted weights to a file         │     │
│     │ so we don't lose progress                   │     │
│     └─────────────────────────────────────────────┘     │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          ↓
              REPEAT FOR 150 EPISODES
                          ↓
        Brain gets progressively smarter at surviving
```

---

## Key Insight

The neural network is a math formula that:
1. Takes numbers in (what the animal sees)
2. Multiplies by weights
3. Outputs numbers (what to do)

Learning is **adjusting the weights** so the output numbers lead to more rewards over time.

It's trial and error at massive scale - thousands of animals making millions of decisions, keeping track of what worked, and gradually getting better.

---

# How This Neural Network Architecture Helps Learning

## The Core Problem the Architecture Solves

An animal needs to answer: **"Given everything I can see and remember, what should I do right now?"**

This is hard because:
1. The animal sees **many other animals** (up to 24) - how do you process a variable number of things?
2. The situation **changes over time** - a predator approaching is different from one standing still
3. Some visible animals **matter more** than others - the closest predator is more urgent than a distant one

The architecture has specific components to handle each of these challenges.

---

## Part 1: The Self-State Vector (34 numbers + 289 grass map)

### What It Contains

```
"Who am I and what's my situation?"

Position:        (0.5, 0.5)     → "I'm in the middle of the map"
Am I predator:   0              → "No, I'm prey"
Hunger:          0.3            → "I'm getting a bit hungry"
Nearest threat:  (0.2, -0.1, 0.15) → "Predator is close, slightly right and above"
My heading:      (1, 0)         → "I'm facing East"
Energy:          0.8            → "I have good energy"
Pheromones:      (0.5, 0.1, 0.0) → "Danger smell nearby, faint mating smell"
...
```

### Why It Helps Learning

The network doesn't have to **figure out** these things - they're given directly:

```
WITHOUT this info:
  Network sees: raw pixel grid
  Network must learn: "these pixels mean predator", "this pattern means close"
  This takes MILLIONS of examples

WITH this info:
  Network sees: nearest_predator_distance = 0.15
  Network can directly learn: "when this number < 0.3, run away"
  This takes THOUSANDS of examples
```

**The self-state gives the network pre-digested, useful information** so it can focus on learning *what to do* rather than *what things are*.

---

## Part 2: The 10-Frame History Stack

### The Problem It Solves

A single snapshot doesn't tell you **motion**:

```
Frame 5 only:
  Predator distance: 0.3
  
  Is the predator:
  - Standing still?
  - Running toward me?
  - Running away?
  
  I can't tell from one frame.
```

### How Stacking Helps

```
10 Frames stacked:

Frame 1 (oldest):  predator_dist = 0.5
Frame 2:           predator_dist = 0.48
Frame 3:           predator_dist = 0.45
Frame 4:           predator_dist = 0.42
Frame 5:           predator_dist = 0.38
Frame 6:           predator_dist = 0.35
Frame 7:           predator_dist = 0.32
Frame 8:           predator_dist = 0.30
Frame 9:           predator_dist = 0.28
Frame 10 (now):    predator_dist = 0.25

The network can see: "Distance is DECREASING over time"
Conclusion: "Predator is CHASING me - this is urgent!"
```

### What the Network Learns from History

```
Pattern A: distances decreasing rapidly
  → "Being chased" → Flee urgently → High reward for escaping
  
Pattern B: distances stable
  → "Predator not moving toward me" → Can be less urgent
  
Pattern C: distances increasing
  → "I'm successfully escaping" → Keep doing what I'm doing
  
Pattern D: predator appears suddenly (was absent, now close)
  → "Ambush!" → Turn and run immediately
```

**The network learns to recognize these PATTERNS across time** and respond appropriately.

---

## Part 3: The Visible Animals List (24 × 9 numbers)

### The Problem It Solves

The animal might see 0, 5, or 20 other animals. Neural networks need **fixed-size inputs**. How do we handle variable numbers?

### The Solution: Fixed Slots with Padding

```
Always 24 slots, each with 9 numbers:

Slot 1:  [0.1, -0.2, 0.3, 1, 0, 0, 0, 0, 1]  ← Real predator
         dx   dy   dist pred prey ... present

Slot 2:  [0.3, 0.1, 0.4, 0, 1, 1, 1, 0, 1]   ← Real prey (same species)

Slot 3:  [0, 0, 0, 0, 0, 0, 0, 0, 0]         ← Empty (padding)
...
Slot 24: [0, 0, 0, 0, 0, 0, 0, 0, 0]         ← Empty (padding)
```

The `present` flag (index 8) tells the network: "Ignore this slot, it's empty."

### Why 9 Numbers Per Animal?

Each number answers a question:

| Index | Value | Question Answered |
|-------|-------|-------------------|
| 0 | dx = 0.1 | "How far right/left?" |
| 1 | dy = -0.2 | "How far up/down?" |
| 2 | dist = 0.3 | "How far total?" |
| 3 | is_pred = 1 | "Is this a threat?" |
| 4 | is_prey = 0 | "Is this food?" |
| 5 | same_species = 0 | "Is this my kind?" |
| 6 | same_type = 0 | "Exactly my type?" |
| 7 | (reserved) | - |
| 8 | present = 1 | "Is this slot real?" |

---

## Part 4: Cross-Attention

### The Problem It Solves

With 24 visible animals, which ones should I pay attention to?

```
Scenario:
- Slot 1: Predator, distance 0.8 (far)
- Slot 2: Predator, distance 0.2 (CLOSE!)
- Slot 3: Prey (same species), distance 0.5
- Slot 4-24: Empty

Clearly slot 2 (close predator) is most important.
But how does the network know this?
```

### How Cross-Attention Works

Think of it as the animal **asking questions** about what it sees:

```
SELF-STATE (the "query"):
  "I am prey, low energy, heading East, predator nearby..."

VISIBLE ANIMALS (the "keys/values"):
  [predator far, PREDATOR CLOSE, friendly prey, empty, empty, ...]

ATTENTION MECHANISM:
  1. Compare self-state to each visible animal
  2. Compute "relevance score" for each:
     - Predator close: score = 0.85 (very relevant to prey!)
     - Predator far: score = 0.10
     - Friendly prey: score = 0.04
     - Empty slots: score = 0.01
  
  3. Weight each animal's info by its score:
     Output = 0.85 × (predator close info) 
            + 0.10 × (predator far info)
            + 0.04 × (friendly prey info)
            + ...
```

### What the Network Learns

The attention mechanism **learns** what to pay attention to based on rewards:

```
Early training (random attention):
  Prey pays equal attention to everything
  → Often ignores close predators
  → Gets eaten
  → Negative reward

After training:
  Prey learns: "When I'm prey, predators get high attention scores"
  Prey learns: "Closer animals get higher attention scores"
  → Focuses on nearest predator
  → Runs away correctly
  → Positive reward
```

**Cross-attention lets the network learn WHAT MATTERS in a situation**, not just see everything equally.

---

## Part 5: Dual Action Heads (Turn + Move)

### The Problem It Solves

Movement has two parts:
1. **Which direction am I facing?** (affects what I can see)
2. **Which direction do I move?** (affects where I go)

These are related but different decisions.

### How Two Heads Help

```
SINGLE HEAD (simpler but worse):
  One output: "Move Northeast"
  Problem: Can't express "Turn to look North, but move East"

DUAL HEADS (this architecture):
  Turn head: "Turn left" (now facing North)
  Move head: "Move East"
  Result: Animal faces North (can see predator) but moves East (escapes sideways)
```

### What Each Head Learns

```
TURN HEAD learns:
  "Turn toward threats to keep them in view"
  "Turn toward mates when safe"
  "Turn toward food smells"

MOVE HEAD learns:
  "Move away from threats"
  "Move toward food"
  "Move toward mates when ready"
```

This separation lets the animal **track threats while escaping** - like how you might look at a car while stepping backward onto the sidewalk.

---

## Part 6: The Critic (Value Estimation)

### The Problem It Solves

Not all rewards come immediately:

```
Step 1: Predator appears at distance 0.8
Step 2: I move away, distance now 0.9
Step 3: I keep moving, distance now 1.0 (predator out of sight)
Step 4: I survive!

Which step deserves credit for surviving?
ALL of them - each contributed to the final outcome.
```

### How the Critic Helps

The critic learns to predict **future rewards** from the current situation:

```
Critic sees: predator at distance 0.3
Critic outputs: value = -2.5

Meaning: "From this situation, I expect about -2.5 total reward"
         (Danger ahead, likely to get hurt)

Critic sees: no predators visible, high energy
Critic outputs: value = +5.0

Meaning: "From this situation, I expect about +5.0 total reward"
         (Safe situation, good survival odds)
```

### Why This Helps Learning

The critic enables **credit assignment**:

```
Step 5: I get eaten (reward = -25)

Without critic:
  Only step 5 gets blamed
  Steps 1-4 (which led to this!) get no feedback

With critic:
  Step 1: Action led to value dropping from +2 to -1 → bad action
  Step 2: Action led to value dropping from -1 to -3 → bad action
  ...
  Each step gets appropriate blame based on how it changed predicted future
```

---

## Putting It All Together: One Decision

```
SITUATION:
- Prey animal at position (50, 50)
- One predator visible at relative position (3, -2), distance 3.6
- Energy: 70%, not hungry
- Facing: South

STEP 1: Build self-state (34 features × 10 frames = 340... wait, + grass = 3230)
[0.5, 0.5, 1, 0, 0, 0.3, 0, 0.36, 0.08, -0.05, ..., (history), ...]

STEP 2: Build visible animals (24 × 9 = 216)
[[0.08, -0.05, 0.36, 1, 0, 0, 0, 0, 1],  ← predator
 [0, 0, 0, 0, 0, 0, 0, 0, 0],             ← empty
 ...]

STEP 3: Self-embedding
3230 numbers → Linear layer → 256 numbers
(Compresses "who am I" into a dense representation)

STEP 4: Animal embedding
Each of 24 animals: 9 numbers → Linear → 256 numbers
(Compresses "what do I see" into dense representations)

STEP 5: Cross-attention
Self (256) asks: "What in my surroundings matters?"
Animals (24 × 256) answer with relevance-weighted summary
Output: context (256 numbers) - "Here's what's important"

STEP 6: Fusion
Concatenate: [self (256), context (256)] → 512 numbers
Process through MLP → 512 numbers representing "full understanding"

STEP 7: Turn decision
512 → Turn head → [0.15, 0.70, 0.15] (left, straight, right)
"70% confidence: keep facing current direction"

STEP 8: Move decision  
512 → Move head → [0.05, 0.10, 0.05, 0.02, 0.50, 0.15, 0.08, 0.05]
                   N    NE    E    SE    S     SW    W    NW
"50% confidence: move South (away from predator which is North-ish)"

STEP 9: Value estimation
512 → Critic → value = +1.2
"I predict modest positive future reward from this situation"

STEP 10: Execute
Sample from distributions → Turn: Straight, Move: South
Animal moves to (50, 51), still facing South
```

---

## How Training Improves Each Component

| Component | What It Learns | How Rewards Shape It |
|-----------|----------------|----------------------|
| Self-embedding | Which personal features matter | "Energy matters when deciding to flee vs forage" |
| Animal embedding | How to represent other animals | "Distance + type are key features" |
| Cross-attention | What to focus on | "Closest predator gets most attention" |
| Turn head | When to change facing | "Turn toward threats to track them" |
| Move head | Where to go | "Move away from high-attention threats" |
| Critic | Predict future outcomes | "Close predator = bad future value" |

---

## The 10-Frame Memory: A Concrete Example

```
SCENARIO: Predator starts far, approaches, prey must detect and react

Frame 1:  no predator visible, value = +3.0
Frame 2:  no predator visible, value = +3.0
Frame 3:  no predator visible, value = +3.0
Frame 4:  predator appears at edge! dist=0.95, value drops to +1.0
Frame 5:  predator closer, dist=0.80, value = +0.5
Frame 6:  predator closer, dist=0.65, value = -0.5
Frame 7:  predator closer, dist=0.50, value = -1.5
Frame 8:  predator closer, dist=0.35, value = -3.0
Frame 9:  prey starts running, dist=0.40, value = -2.0
Frame 10: prey escaping, dist=0.50, value = -1.0

WHAT THE NETWORK SEES AT FRAME 10:
[frame10_features, frame9_features, frame8_features, ..., frame1_features]

PATTERNS THE NETWORK CAN DETECT:
1. "predator_dist went 0.95→0.35 then 0.35→0.50" 
   → "Was approaching, now I'm escaping - keep running!"

2. "value went +3→-3→-1"
   → "Situation got bad but is improving"

3. "my_heading changed at frame 9"
   → "I turned around, that's when escape started"
```

Without the 10-frame history, the network at frame 10 would only see:
- Predator at distance 0.5
- Can't tell if approaching or retreating
- Can't tell if it just appeared or has been there
- Can't tell if current strategy is working

**The history provides CONTEXT that makes intelligent decisions possible.**

---

## Architecture Summary: Why This Architecture Works

| Challenge | Solution | How It Helps Learning |
|-----------|----------|----------------------|
| Variable number of visible animals | Fixed 24 slots + padding flag | Network has consistent input size |
| Which animals matter most? | Cross-attention mechanism | Learns to focus on relevant threats/opportunities |
| Is situation getting better or worse? | 10-frame history stack | Can detect motion, trends, approach/retreat |
| Need to look and move independently | Dual turn/move heads | Can track threats while escaping |
| Delayed consequences | Critic value estimation | Credit assignment to earlier actions |
| Complex situation understanding | 512-dim fused representation | Rich internal model of "what's happening" |

The architecture doesn't **give** the animals intelligence—it gives them **the capacity to learn** intelligent behavior through trial, error, and reward feedback.

---

# Technical Deep Dive

## 1) HOW THE NETWORK(S) WORK

### Model Definition & Instantiation

**File**: `src/models/actor_critic_network.py`  
**Class**: `ActorCriticNetwork` (line 72)

**Instantiation** in `scripts/train.py` (lines 1786-1787):
```python
model_prey = ActorCriticNetwork(config).to(device)
model_predator = ActorCriticNetwork(config).to(device)
```

Both prey and predator use **identical network architectures** but are **trained separately** with different rewards.

### Exact Architecture

```
INPUT PIPELINE:
├── Self-state: (B, 3230) → Linear(3230, 256) → ReLU → self_features (B, 256)
│   [34 base features + 289 grass map] × 10 history frames = 3,230 dims
│
└── Visible animals: (B, 24, 9) → Linear(9, 256) → 2-layer MLP → animal_embeds (B, 24, 256)

CROSS-ATTENTION (self queries visible animals):
├── CrossAttention(query_dim=256, key_dim=256, num_heads=8)
└── Output: context (B, 256) - attention-weighted summary of neighbors

FUSION:
├── Concat: [self_features, context] → (B, 512)
└── feature_fusion: 2-layer MLP with Dropout(0.2) → fused (B, 512)

DUAL ACTOR HEADS:
├── turn_head: 512 → 256 → 128 → 3 (LEFT/STRAIGHT/RIGHT)
└── move_head: 512 → 256 → 128 → 8 (N/NE/E/SE/S/SW/W/NW)

CRITIC HEAD:
└── critic: 512 → 512 → 256 → 128 → 1 (state value V(s))
```

**Activations**: ReLU throughout  
**Normalization**: None (no BatchNorm/LayerNorm)  
**Output**: Softmax probabilities for actions, raw value for critic

### Model Inputs

**Self-state vector** (34 base features, defined in `Animal.get_enhanced_input()` at `src/core/animal.py` line 86):

| Index | Feature | Range |
|-------|---------|-------|
| 0-1 | Position (x, y) | [0,1] |
| 2-3 | Species (A, B) | binary |
| 4 | Is predator | binary |
| 5 | Hunger level | [0,1] |
| 6 | Mating cooldown | [0,1] |
| 7-12 | Nearest predator/prey (dist, dx, dy each) | [0,1] |
| 13-14 | Visible predator/prey counts | [0,1] |
| 15-16 | Age, energy | [0,1] |
| 17-19 | Pheromone intensities (danger, mating, food) | [0,1] |
| 20-21 | Heading direction (dx, dy) | [-1,1] |
| 22-27 | Pheromone gradients | [-1,1] |
| 28-30 | Gradient magnitudes | [0,1] |
| 31 | Danger memory (time since threat) | [0,1] |
| 32 | Population ratio | [0,1] |
| 33 | Previous turn action | [0,1] |
| 34-322 | Grass FOV map (prey only, 289 cells) | binary |

**Stacked history**: 10 frames concatenated = 3,230 total dimensions

**Visible animals tensor** (9 features per animal, max 24):

| Index | Feature |
|-------|---------|
| 0-1 | Relative dx, dy (normalized) |
| 2 | Distance (normalized) |
| 3-6 | is_predator, is_prey, same_species, same_type |
| 7 | Reserved (always 0) |
| 8 | is_present (1=real, 0=padding) |

### Model Outputs

**Turn probabilities** (3): Applied via `apply_turn_action()` to change heading ±1 step (8-direction compass)

**Move probabilities** (8): Converted to grid movement via `_apply_action()`

**State value** (1): V(s) estimate for TD(0) bootstrapping

### Loss Function

**PPO Loss** computed in `ppo_update()` at `scripts/train.py` line 500:

```python
# Policy loss (clipped surrogate objective)
ratio = exp(log_probs_new - log_probs_old)
surr1 = ratio * advantages
surr2 = clamp(ratio, 1-ε, 1+ε) * advantages
policy_loss = -min(surr1, surr2).mean()

# Value loss (MSE)
value_loss = MSE(values, returns)

# Entropy bonus (exploration)
entropy_loss = -entropy.mean()

# Directional supervision loss (auxiliary)
directional_loss = NLL(log_probs, target_direction)

# Combined loss
loss = policy_loss + 0.25*value_loss + 0.04*entropy_loss + 2.0*directional_loss
```

### Optimization

**Optimizer**: Adam  
**Learning rates**: 
- Prey: `0.00008` (slower to protect fragile flee policy)
- Predator: `0.0001`

**Gradient clipping**: `max_norm=0.3`  
**PPO epochs**: 6 per episode  
**Batch size**: 2048  
**Clip epsilon**: 0.15

### No Target Networks

This uses **on-policy PPO**, not off-policy DQN. There are:
- ❌ No separate target networks
- ❌ No epsilon-greedy exploration (uses entropy bonus instead)
- ❌ No soft updates
- ✅ KL divergence early stopping (prevents policy collapse)

### Prey vs Predator Differences

| Aspect | Prey | Predator |
|--------|------|----------|
| Learning rate | 0.00008 | 0.0001 |
| Directional loss | 2.0 (context-aware: flee + mate) | 2.0 (always toward prey) |
| Vision range | 5 cells | 8 cells |
| FOV | 240° (wide peripheral) | 180° (forward-focused) |
| Grass map input | Full 289-cell FOV | All zeros |

---

## 2) HOW ANIMALS MAKE DECISIONS

### Decision Pipeline

**Location**: `Animal.move_training()` for training, `Animal.move()` for inference (both in `src/core/animal.py`)

**Hierarchical two-phase action per step**:

```
PHASE 1: TURN
1. Build observation (get_enhanced_input + communicate)
2. Forward pass → turn_probs
3. Sample turn action via multinomial
4. Apply turn (changes heading/FOV)

PHASE 2: MOVE
5. Rebuild observation (FOV changed!)
6. Forward pass → move_probs
7. Sample move action via multinomial
8. Apply movement if not blocked
```

### Stochastic vs Deterministic

**Training** (`move_training`): Stochastic sampling via `torch.multinomial(probs, 1)`

**Inference** (`move`): Deterministic argmax via `deterministic=True`

**Exploration**: Temperature scaling (`ACTION_TEMPERATURE=1.0`) and entropy bonus in loss

### Action-to-Movement Mapping

**Turn actions** (0-2):
```python
TURN_LEFT = 0    # heading_idx -= 1
TURN_STRAIGHT = 1  # no change
TURN_RIGHT = 2   # heading_idx += 1
```

**Move actions** (0-7) → `_apply_action()`:
```python
0: North (y-1)     4: South (y+1)
1: NE (x+1, y-1)   5: SW (x-1, y+1)
2: East (x+1)      6: West (x-1)
3: SE (x+1, y+1)   7: NW (x-1, y-1)
```

### Constraints

**Collision**: `_position_occupied()` checks if target cell is occupied  
**Boundaries**: Toroidal wrapping via modulo (`(x + dx) % GRID_SIZE`)  
**Speed**: Prey=1 move/step, Predator=1-2 moves/step (more when hungry)

### Heuristics vs Learned

**During training**: 100% neural network control (no heuristics)  
**During inference**: Optional chase override for predators (`CHASE_OVERRIDE_IN_INFERENCE=False` by default)

### Worked Example

```python
# Sample state for prey:
animal_input = tensor([0.5, 0.5, 1, 0, 0, 0.3, 0, 0.2, -0.1, 0.1, ...])  # (1, 3230)
visible_animals = tensor([[[0.1, -0.2, 0.3, 1, 0, 0, 0, 0, 1], ...]])  # (1, 24, 9)
#                          ^ dx=0.1, dy=-0.2, dist=0.3, is_predator=1

# Forward pass
turn_probs, move_probs, value = model(animal_input, visible_animals)
# turn_probs = [0.1, 0.7, 0.2]  → likely TURN_STRAIGHT
# move_probs = [0.05, 0.05, 0.05, 0.05, 0.6, 0.1, 0.05, 0.05]  → likely SOUTH (away from predator)

# Sample
turn_action = torch.multinomial(turn_probs, 1)  # → 1 (STRAIGHT)
move_action = torch.multinomial(move_probs, 1)  # → 4 (SOUTH)

# Apply
animal.apply_turn_action(1)  # No heading change
new_x, new_y = animal._apply_action(4, config)  # y += 1
```

---

## 3) HOW MEMORY IS USED

### Memory Type: On-Policy PPO Buffer

**File**: `src/models/replay_buffer.py`  
**Class**: `PPOMemory` (NOT a replay buffer—it's cleared after each episode)

### Structure (Hierarchical Mode)

```python
class PPOMemory:
    # Turn observations
    obs_turn: List[Tensor]        # Pre-turn self-state (1, 3230)
    vis_turn: List[Tensor]        # Pre-turn visible animals (1, 24, 9)
    turn_actions: List[Tensor]    # Turn action taken (scalar)
    turn_log_probs_old: List[Tensor]  # Log prob at collection time
    
    # Move observations
    obs_move: List[Tensor]        # Post-turn self-state
    vis_move: List[Tensor]        # Post-turn visible animals
    move_actions: List[Tensor]
    move_log_probs_old: List[Tensor]
    
    # Shared
    values: List[Tensor]          # V(s_turn)
    next_values: List[Tensor]     # V(s_{t+1}) for TD(0)
    rewards: List[float]
    dones: List[bool]
    traj_ids: List[int]           # Animal ID for trajectory grouping
```

### Adding Experiences

**When**: Every micro-step during `move_training()`

```python
transition = {
    'obs_turn': animal_input_turn_cpu,
    'vis_turn': visible_input_turn_cpu,
    'turn_action': turn_action,
    'turn_logp_old': turn_log_prob,
    'obs_move': animal_input_move_cpu,
    'vis_move': visible_input_move_cpu,
    'move_action': move_action,
    'move_logp_old': move_log_prob,
    'value_old': state_value_turn,
    'value_next': torch.zeros_like(state_value_turn),  # Patched later
}
memory.add(transition=transition, reward=reward, done=False)
```

### Batch Sampling

**Method**: `get_batches()` with random shuffle

```python
indices = np.arange(n_samples)
np.random.shuffle(indices)
for start_idx in range(0, n_samples, batch_size):
    batch_indices = indices[start_idx:start_idx + batch_size]
    yield batch  # Dictionary with all fields sliced
```

**Sampling**: Uniform random (no prioritization in main training loop)

### Memory → Training

1. Episode runs, collecting transitions into `PPOMemory`
2. Episode ends → `compute_returns_and_advantages()`
3. TD(0) advantages: `δ = r + γ * V(s') * (1-done) - V(s)`
4. Advantages normalized per-minibatch
5. PPO update loops over shuffled batches for 6 epochs
6. Memory cleared → next episode

### Capacity & Lifecycle

- **No fixed capacity** (grows during episode)
- **Cleared after each episode** via `memory.clear()`
- **Done flag**: True for terminal states (death, episode end)
- **Episode boundary**: All surviving animals' last transitions marked `done=True`

### Alternative Buffers (Present but Unused)

The file also defines `ExperienceReplayBuffer` and `PrioritizedExperienceReplay`, but they are **not used** in the main training loop.

---

## 4) IMPORTANT DETAILS

### Reward Functions

**File**: `src/config.py` (values) + `scripts/train.py` line 1044 (logic)

**Prey rewards**:
| Event | Reward |
|-------|--------|
| Survival per step | +0.2 |
| Successful evasion (distance increased) | +2.5 × Δdist |
| Grass eaten | +1.0 |
| Threat presence (predator visible) | -0.15 × closeness^1.5 |
| Blocked while threatened | -0.5 |
| Being eaten | -25.0 |
| Exhaustion death | -12.5 |
| Reproduction | +0.2 |

**Predator rewards**:
| Event | Reward |
|-------|--------|
| Eating prey | +30.0 |
| Approaching prey | +5.0 × Δdist |
| Prey detection bonus | +0.05 |
| Prey visible per step | +0.01 |
| New territory explored | +0.01 |
| Starvation death | -10.0 |

### Episode Lifecycle

**Reset**: `create_population()` creates 40 prey + 20 predators  
**Termination**: Fixed 300 steps OR all animals extinct  
**Metrics**: Births, deaths, meals, exhaustion, old age, starvation

### Training Schedule

- Training occurs **once per episode** (after rollout)
- No warmup (starts learning immediately)
- 6 PPO epochs per update
- KL divergence early stopping (0.03 epoch average, 0.05 minibatch)

### Saving/Loading

**Checkpoints** saved every episode:
```
outputs/checkpoints/phase{N}_ep{M}_model_A.pth  # Prey
outputs/checkpoints/phase{N}_ep{M}_model_B.pth  # Predator
```

**Phase loading**: Specified in config files (`LOAD_PREY_CHECKPOINT`, `LOAD_PREDATOR_CHECKPOINT`)

**Ctrl+C handler**: Saves `model_A_interrupt.pth` on SIGINT

### Device Usage

**Priority**: DirectML (AMD/Intel) > CUDA (NVIDIA) > Error (no CPU fallback)  
**dtype**: float32 throughout  
**Memory optimization**: Observations stored on CPU, moved to GPU for forward pass

---

## 5) VISUAL MAP + QUICK START

### Architecture Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING LOOP (train.py)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐        │
│  │  create_pop()   │ ──► │  run_episode()  │ ──► │  ppo_update()   │        │
│  │  40 Prey        │     │  300 steps      │     │  6 epochs       │        │
│  │  20 Predators   │     │  PPOMemory      │     │  clip=0.15      │        │
│  └─────────────────┘     └────────┬────────┘     └─────────────────┘        │
│                                   │                                          │
│                    ┌──────────────┴──────────────┐                          │
│                    ▼                              ▼                          │
│  ┌─────────────────────────┐      ┌─────────────────────────┐               │
│  │  Prey (animal.py)       │      │  Predator (animal.py)   │               │
│  │  - get_enhanced_input() │      │  - get_enhanced_input() │               │
│  │  - move_training()      │      │  - move_training()      │               │
│  │  - deposit_pheromones() │      │  - perform_eat()        │               │
│  └──────────┬──────────────┘      └──────────┬──────────────┘               │
│             │                                 │                              │
│             ▼                                 ▼                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    ActorCriticNetwork (actor_critic_network.py)         ││
│  │  ┌───────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐││
│  │  │ self_embed    │  │ CrossAttention  │  │ Dual Heads                  │││
│  │  │ (3230→256)    │  │ (256q × 256kv)  │  │ turn: 512→3                 │││
│  │  │               │  │ 8 heads         │  │ move: 512→8                 │││
│  │  │ animal_embed  │  │                 │  │ value: 512→1                │││
│  │  │ (9→256)       │  │                 │  │                             │││
│  │  └───────────────┘  └─────────────────┘  └─────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ PheromoneMap    │  │ GrassField      │  │ PPOMemory       │              │
│  │ (danger/mating/ │  │ (binary grid,   │  │ (hierarchical   │              │
│  │  food/territory)│  │  regrows)       │  │  turn+move)     │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Quick Start

1. **Run training**:
   ```powershell
   .\Dashboard.vbs
   ```
   Or manually:
   ```powershell
   .venv_rocm\Scripts\python.exe scripts\train.py
   ```

2. **Run demo** (visualization):
   ```powershell
   .\Demo.vbs
   ```

3. **Key config values** (`src/config.py`):
   - `NUM_EPISODES = 150` - Total training episodes
   - `STEPS_PER_EPISODE = 300` - Simulation length
   - `PPO_EPOCHS = 6` - Optimization passes per update
   - `LEARNING_RATE_PREY = 0.00008` - Prey learning rate
   - `LOAD_PREY_CHECKPOINT` - Resume from checkpoint





