# Life Game RL System — Requirements & Expected Agent Behavior
*(Predator–prey ecosystem with PPO, dual-head control, energy/age, and pheromones)*

## 1. Purpose and Scope
This document describes **functional and non-functional requirements** for a grid-based “Life Game” ecosystem in which **Prey** and **Predator** agents learn behaviors via **Proximal Policy Optimization (PPO)**. The system combines:
- **Hierarchical control** (Turn → Re-observe → Move) with a **dual-head Actor** (3 turn actions + 8 move actions)
- **Critic value estimation** for PPO
- **Energy** and **age** systems
- **Pheromone maps** (danger, mating, food, territory)
- **Cross-attention** from self-state to visible neighbors
- **Curriculum learning** to shape early behavior (e.g., prey mating under reduced predator pressure)

The requirements are written to be used as a spec for implementation, validation, and future refactoring.

---

## 2. System Overview

### 2.1 Environment Model
- The world is a **toroidal 2D grid** of size `GRID_SIZE × GRID_SIZE`.
- Toroidal means moving beyond an edge wraps around to the opposite side.
- The simulation runs in discrete **steps**; each episode runs for `steps_per_episode` steps, unless extinction occurs.

### 2.2 Agent Types
- **Prey**: avoids predators, survives, and reproduces.
- **Predator**: hunts prey to avoid starvation, survives, and reproduces under fitness constraints.

### 2.3 Learning Objective
Agents learn **policies** that maximize expected discounted return under:
- survival rewards,
- reproduction rewards,
- hunting/evasion rewards,
- penalties for death/exhaustion/starvation,
- and auxiliary “directional supervision” loss that accelerates learning of spatial navigation.

---

## 3. Core Functional Requirements

### 3.1 World Geometry & Movement
**REQ-WORLD-1**: The grid MUST be toroidal. Distances and directions MUST use toroidal shortest-path deltas.

**REQ-MOVE-1**: Move actions MUST correspond to exactly 8 directions:
- `0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW`

**REQ-MOVE-2**: The movement mapping MUST match the environment’s movement function exactly (no diagonal bias).

**REQ-MOVE-3**: The system MUST prevent multiple animals occupying the same cell (collision/occupancy check). If a chosen move would collide, the move is rejected (agent stays).

### 3.2 Heading and Turn Control (Hierarchical Policy)
Agents have a heading direction `heading_idx ∈ {0..7}` which influences their **Field of View (FOV)**.

**REQ-TURN-1**: Turn actions MUST be limited to ±1 heading step per micro-step:
- `TURN_LEFT=0` → heading −1  
- `TURN_STRAIGHT=1` → heading +0  
- `TURN_RIGHT=2` → heading +1

**REQ-TURN-2**: A micro-step MUST follow this sequence:
1. Observe (pre-turn)
2. Sample **turn action**
3. Apply turn (updates heading)
4. Re-observe (post-turn; FOV may change)
5. Sample **move action**
6. Apply move

**REQ-TURN-3**: Heading MUST be controlled by the neural policy during training. Any “auto heading update from movement” MUST NOT override NN control.

### 3.3 Vision System (Range + Cone FOV)
**REQ-VISION-1**: Visibility MUST require:
- within circular range (`vision_range`), AND
- inside the cone defined by heading and FOV angle.

**REQ-VISION-2**: The FOV test SHOULD be implemented with a dot-product check against `cos(FOV/2)` to avoid expensive trigonometry per candidate.

**REQ-VISION-3**: Vision range and FOV MUST be type-specific:
- Prey: wider FOV (e.g., ~240°)
- Predator: narrower FOV (e.g., ~180°)

### 3.4 Communication / Visible Neighbors Encoding
Each agent produces a fixed-size list of visible neighbors, each with **8 features**:
1. `dx_norm` (signed, normalized)
2. `dy_norm` (signed, normalized)
3. `dist_norm` (0..1)
4. `is_predator` (0/1)
5. `is_prey` (0/1)
6. `same_species` (0/1)
7. `same_type` (0/1)  *(same class: prey vs predator)*
8. `is_present` (1 for real neighbor, 0 for padding)

**REQ-VIS-1**: The visible list MUST be padded to `MAX_VISIBLE_ANIMALS` using all-zero rows where `is_present=0`.

**REQ-VIS-2**: The selection strategy MUST reserve capacity:
- half slots for same-type,
- half for opposite-type,
- with deterministic backfill if one group is undersubscribed.

**REQ-VIS-3**: The visible neighbors SHOULD be distance-sorted for stable selection.

### 3.5 Observation Contract (Self-State Vector)
Each agent MUST build a self-state observation vector of fixed length.  
**OBS_VERSION=2** defines **34 features** (must remain stable unless the version increments).

**REQ-OBS-1**: The feature order MUST remain stable. If changed, `OBS_VERSION` MUST increment and the model architecture MUST update accordingly.

**REQ-OBS-2**: All normalized features MUST be clamped to safe ranges to prevent blow-ups.

**Self-state features (34)**:
- **[0–1]** position `(x,y)` normalized
- **[2–3]** species A/B one-hot
- **[4]** predator flag
- **[5–6]** hunger + mating cooldown (normalized)
- **[7–12]** nearest predator and nearest prey: dist + dx + dy (normalized)
- **[13–14]** counts of visible predators/prey (normalized)
- **[15–16]** age + energy (normalized)
- **[17–19]** pheromone max intensity: danger/mating/food
- **[20–21]** current heading dx/dy
- **[22–27]** pheromone gradients (danger/mating/food x/y)
- **[28–30]** pheromone gradient magnitudes (danger/mating/food)
- **[31]** “danger memory” (recent predator exposure)
- **[32]** population ratio (current species count vs cap)
- **[33]** previous turn action (normalized to [0,1])

---

## 4. Pheromone System Requirements

### 4.1 Pheromone Types
The environment MUST maintain separate pheromone maps for:
- `danger` (prey warning)
- `mating` (reproduction readiness)
- `food` (predator hunt success)
- `territory` (optional / currently not used in NN features)

### 4.2 Deposition Rules
**REQ-PHERO-DEP-1 (Prey)**: Prey MUST deposit **danger pheromone** when predators are visible.

**REQ-PHERO-DEP-2 (Prey)**: Prey MUST deposit **mating pheromone** when they are reproductively ready.

**REQ-PHERO-DEP-3 (Predator)**: Predators MUST deposit **food pheromone** after a successful hunt (e.g., when `steps_since_last_meal==0`).

**REQ-PHERO-DEP-4 (Predator)**: Predators MAY deposit mating pheromone when ready to reproduce.

### 4.3 Decay & Diffusion
**REQ-PHERO-DYN-1**: All pheromone maps MUST decay by factor `PHEROMONE_DECAY` each step.

**REQ-PHERO-DYN-2**: Pheromone diffusion SHOULD be applied each step (simple neighbor blur), controlled by `PHEROMONE_DIFFUSION`.

**REQ-PHERO-DYN-3**: Values MUST be clamped to `[0,1]` after update to avoid drift.

### 4.4 Sensing Model
**REQ-PHERO-SENSE-1**: Pheromone sensing MUST be omnidirectional and MUST NOT be limited by heading/FOV.

**REQ-PHERO-SENSE-2**: Sensing MUST be toroidal, consistent with world wrapping.

**REQ-PHERO-SENSE-3**: Sensory input SHOULD include:
- local averages + maxima within radius,
- gradient directions (unit vectors),
- gradient magnitudes (normalized).

**REQ-PHERO-SENSE-4**: The pheromone sensory computation SHOULD use caching keyed by `(x,y,radius,scale,version)` to avoid redundant calculation within a step.

---

## 5. Energy and Age Systems

### 5.1 Energy
**REQ-ENERGY-1**: Agents MUST lose energy every step due to metabolism (`ENERGY_DECAY_RATE`).

**REQ-ENERGY-2**: Movement MUST incur additional energy cost (`MOVE_ENERGY_COST`).

**REQ-ENERGY-3**: If an agent does not move, it SHOULD regain energy (`REST_ENERGY_GAIN`).

**REQ-ENERGY-4**: Energy MUST be clamped to `[0, MAX_ENERGY]`.

**REQ-ENERGY-5**: If energy reaches 0, the agent is **exhausted** and MUST die (or be removed) with penalties applied to learning memory.

### 5.2 Age
**REQ-AGE-1**: Agents MUST increment age each step.

**REQ-AGE-2**: If age ≥ `MAX_AGE`, the agent MUST die (old age) with appropriate penalty shaping.

---

## 6. Feeding / Hunting and Starvation

### 6.1 Eating Mechanics
**REQ-EAT-1**: Predators MUST be able to eat prey when adjacent (toroidal adjacency, including diagonals).

**REQ-EAT-2**: `perform_eat()` MUST return `(success, reward, eaten_agent)` but MUST NOT directly remove the prey from the global list (caller removes).

**REQ-EAT-3**: On successful eat:
- predator energy increases by `EATING_ENERGY_GAIN` (capped),
- predator hunger counter resets,
- predator receives `PREDATOR_EAT_REWARD` added to the correct transition reward in memory.

### 6.2 Starvation
**REQ-STARVE-1**: Predators MUST track `steps_since_last_meal`.

**REQ-STARVE-2**: If starvation is enabled and `steps_since_last_meal >= STARVATION_THRESHOLD`, the predator MUST die and receive penalties.

**REQ-STARVE-3**: Curriculum MAY disable starvation death early, but hunger penalties should still apply in late hunger states to preserve learning signal.

---

## 7. Reproduction (Mating) Requirements

### 7.1 Eligibility
**REQ-MATE-ELIG-1 (Prey)**: Prey can reproduce only if:
- age ≥ `MATURITY_AGE`
- energy ≥ `MATING_ENERGY_COST`
- mating cooldown == 0

**REQ-MATE-ELIG-2 (Predator)**: Predators can reproduce only if base requirements above are met AND they are “fit” (e.g., have eaten recently, `steps_since_last_meal < threshold`).

### 7.2 Pairing & Offspring
**REQ-MATE-PAIR-1**: Mating requires adjacency (toroidal) and same type/species. Parent–child incest rules MUST be enforced using parent IDs.

**REQ-MATE-PAIR-2**: Mating probability MUST be configurable (`MATING_PROBABILITY_PREY`, `MATING_PROBABILITY_PREDATOR`).

**REQ-MATE-CHILD-1**: Offspring MUST inherit species/type from parents and start with initial energy.

**REQ-MATE-POST-1**: On mating, both parents MUST:
- lose mating energy cost,
- gain cooldown,
- be nudged/moved away to avoid immediate repeated mating,
- record child count.

### 7.3 Reward Assignment
**REQ-MATE-REWARD-1**: Successful mating MUST apply `REPRODUCTION_REWARD` to **both parents’ last stored transition** (if not terminal). This is critical for learnability.

---

## 8. Expected Agent Behavior (Detailed)

This section describes what “good” learned behavior should look like at different levels of competence. These are **behavioral requirements** and acceptance targets.

### 8.1 Prey (Evasion + Social / Mating)

#### 8.1.1 Baseline survival behavior
**PREY-BEH-1**: If a predator is visible and close, prey SHOULD move in a direction that increases predator distance (flee response).  
- “Close” should be based on normalized distance threshold (e.g., `PREY_FLEE_SUPERVISION_DIST`).

**PREY-BEH-2**: Prey SHOULD exploit toroidal geometry: fleeing may wrap around edges if that increases distance faster.

**PREY-BEH-3**: Prey SHOULD avoid collisions and remain mobile unless resting is strategically beneficial (energy recovery or low threat).

#### 8.1.2 Threat-aware heading control
**PREY-BEH-4**: Prey SHOULD turn to keep predators within its FOV when threat is high (track threat).  
Expected pattern:
- When predator count > 0, prey turns to maintain predator direction near the forward cone to improve future reaction.

#### 8.1.3 Pheromone-informed evasion
**PREY-BEH-5**: Prey SHOULD interpret danger pheromone gradients as an early warning and bias movement away from stronger danger signal areas.  
- When predators are not directly visible but danger pheromone is high, prey should still avoid those zones.

#### 8.1.4 Mating-seeking behavior (context-aware)
Prey mating is allowed only when it’s reasonably safe and energy is sufficient.

**PREY-BEH-6**: If predators are not close AND the prey is “ready” (energy high, mature, cooldown complete), prey SHOULD seek mates:
- approach the nearest valid mate,
- follow mating pheromone gradients as indirect communication,
- avoid dead zones with strong danger pheromones.

**PREY-BEH-7**: Prey SHOULD stop mate-approach when predators become visible/close and revert to evasion.

**PREY-BEH-8**: Prey SHOULD demonstrate “opportunistic mating”:
- if safe and mate is nearby, move directly to mate-adjacent cells to trigger reproduction chances.

#### 8.1.5 Rest and energy management
**PREY-BEH-9**: When threats are low and energy is below mating thresholds, prey SHOULD occasionally rest (not moving) to regain energy, especially if resting is rewarded or prevents exhaustion.

---

### 8.2 Predator (Hunting + Resource Management)

#### 8.2.1 Basic hunting behavior
**PRED-BEH-1**: If prey is visible, predators SHOULD move toward the nearest prey (reduce distance).

**PRED-BEH-2**: Predators SHOULD exploit their vision advantage (longer range) by turning to keep prey within their forward cone and predicting likely prey escape paths.

#### 8.2.2 Hunger-driven speed and urgency
Predator move count increases when hungry.

**PRED-BEH-3**: When hunger is high (approaching starvation), predators SHOULD prioritize direct pursuit of prey, even if it risks entering danger-pheromone zones.

**PRED-BEH-4**: When well-fed, predators MAY patrol or follow food pheromone gradients to areas with previously successful hunts.

#### 8.2.3 Pheromone-informed hunting
**PRED-BEH-5**: Predators SHOULD learn to interpret **food pheromone** gradients as a sign of prey presence or hunting success and explore those regions when prey is not visible.

#### 8.2.4 Reproduction behavior
**PRED-BEH-6**: Predators SHOULD attempt to reproduce primarily when well-fed (fitness constraint) and in presence of another compatible predator.

---

## 9. Reinforcement Learning Requirements

### 9.1 Policy Architecture
**REQ-NET-1**: The model MUST implement:
- self-state embedding,
- visible-neighbor embedding,
- cross-attention (self queries neighbors),
- fused representation,
- dual actor heads (turn and move),
- critic head.

**REQ-NET-2**: The network MUST produce **probabilities** (softmax) for both actor heads. All training code that logs `log_probs` assumes this.

**REQ-NET-3**: Padding rows in visible inputs MUST be masked in attention (using `is_present`).

### 9.2 Rollout Collection (Training-time)
**REQ-ROLLOUT-1**: Rollout MUST be done in `no_grad()` mode and store transitions on CPU to avoid VRAM blowup.

**REQ-ROLLOUT-2**: In hierarchical mode, each micro-step MUST store:
- pre-turn observation/visible,
- turn action + log prob,
- post-turn observation/visible,
- move action + log prob,
- value estimate,
- and a placeholder for next value (TD bootstrapping).

### 9.3 Memory and Bootstrapping
**REQ-MEM-1**: The experience buffer MUST support hierarchical transitions and batching.

**REQ-MEM-2**: TD(0) bootstrapping MUST link:
- within-step micro-steps (`next_value[t] = value[t+1]`)
- across steps for each agent (patch previous step’s last transition with current first value).

**REQ-MEM-3**: Terminal transitions MUST have `next_value=0` and `done=True` to prevent leakage across episode boundaries.

### 9.4 PPO Update
**REQ-PPO-1**: PPO MUST compute returns/advantages (TD(0) deltas) and normalize advantages safely.

**REQ-PPO-2**: PPO update MUST run multiple epochs per episode as configured.

**REQ-PPO-3**: PPO MUST clip policy updates via `PPO_CLIP_EPSILON`.

**REQ-PPO-4**: PPO MUST include entropy bonus to reduce action collapse/bias.

**REQ-PPO-5**: Gradients MUST be clipped to `MAX_GRAD_NORM`.

**REQ-PPO-6**: Training MUST support early stopping if KL divergence spikes (policy collapse prevention).

### 9.5 Directional Supervision (Auxiliary Loss)
To accelerate spatial learning, an auxiliary supervised loss is applied to the move head.

**REQ-DIR-1**: Predators: supervised target direction is **toward nearest visible prey**.

**REQ-DIR-2**: Prey: supervision MUST be context-aware:
- if predator is close → flee (move away from predator)
- else if ready and mate visible → approach mate
- else → no supervision applied

**REQ-DIR-3**: Supervised targets MUST ignore padding rows and wrong-type animals using `is_present` and type flags.

**REQ-DIR-4**: The supervised loss SHOULD compute a hard best-direction label (argmax cosine similarity to 8 action directions) and apply NLL on valid rows only.

**REQ-DIR-5**: Loss computation MUST avoid NaNs by filtering minibatch rows with valid targets before reduction.

---

## 10. Curriculum Learning Requirements

**REQ-CUR-1**: The system MUST support curriculum stages defined by episode ranges with config overrides.

**REQ-CUR-2**: When a new stage is applied, base values MUST be restored first (no “leakage” from prior stages).

**REQ-CUR-3**: Optionally boost learning rate on stage change and decay it back over several episodes to help adaptation.

**Expected curriculum outcomes**:
- Early stage: prey should learn mating-seeking without constant predator pressure.
- Later stage: prey should retain mating behavior while improving evasion; predators should improve hunting under realistic pressure.

---

## 11. Non-Functional Requirements

### 11.1 Performance
**NFR-PERF-1**: Simulation SHOULD avoid O(N²) hotspots in critical loops:
- mating checks should use spatial hashing / neighborhood lookup,
- visibility selection should avoid full sorting when possible,
- pheromone sensing should cache per-step results.

**NFR-PERF-2**: Rollout SHOULD run on CPU-heavy logic (world scans) but keep NN forward passes on GPU.

### 11.2 Reproducibility
**NFR-REPRO-1**: The system MUST allow a deterministic seed for:
- Python random,
- Torch CPU RNG,
- Torch GPU RNG (if available).

### 11.3 Hardware / Backend Compatibility
**NFR-HW-1**: Training MUST require GPU acceleration (DirectML or CUDA/ROCm).

**NFR-HW-2**: If running on ROCm with known SDPA issues, attention SHOULD be forced to math backend (avoid flash kernels) where applicable.

### 11.4 Logging & Monitoring
**NFR-LOG-1**: The system SHOULD log per-episode:
- final populations,
- births/deaths/meals,
- rewards per species,
- policy/value/entropy losses,
- action distribution (bias detection),
- PPO diagnostics (approx KL, clip fraction),
- supervision coverage (target visible %, mean target distance),
- time breakdown (env vs GPU update).

### 11.5 Checkpointing & Safety
**NFR-CKPT-1**: The system MUST save checkpoints regularly (e.g., each episode) and on interrupt (SIGINT).

**NFR-CKPT-2**: Best-performing models SHOULD be tracked and saved (e.g., best prey survival).

---

## 12. Acceptance Criteria (What “works”)

### 12.1 Prey
- Prey survival increases over episodes under realistic predator pressure.
- Prey exhibits consistent flee responses when predators are close (distance increases more often than decreases).
- Prey mates when safe and ready, increasing births without instant extinction.
- Prey uses pheromone gradients: avoids danger hotspots and finds mates more often than random walking.

### 12.2 Predator
- Predators achieve stable meal counts sufficient to avoid mass starvation.
- Predators approach prey more reliably than random action.
- Predators use heading to keep prey in FOV and improve capture rates.

### 12.3 Training Stability
- PPO diagnostics remain in reasonable ranges (no persistent KL spikes, clip fraction not extreme).
- Action distribution does not collapse to a single direction (no “north bias”).

---

## 13. Recommended Tests & Validation

### 13.1 Unit Tests
- Toroidal delta correctness
- Move action mapping correctness
- FOV inclusion test (dot-product threshold)
- Pheromone deposit/sense correctness and cache invalidation
- Observation vector length and ordering (34 features)
- Padding masking correctness in attention

### 13.2 Integration Tests
- Episode runs without NaNs in loss
- PPO update completes without device fallback stalls
- Terminal handling prevents value leakage across episodes
- Reward attribution modifies the correct last transition for each agent (death/eat/reproduction).

### 13.3 Behavioral Tests (Scenario-based)
- Single prey + single predator chase scenario: prey flees when predator within threshold.
- Two ready prey with no predators: prey approach each other and mate with elevated probability.
- Predator hunger escalation: predator movement urgency rises and meal probability increases.

---

## 14. Notes on Expected Emergent Dynamics
If requirements are satisfied, expected emergent patterns include:
- Prey clustering and dispersal cycles driven by mating readiness and predator threats.
- Predator “search and pounce” behavior with heading control.
- Indirect communication via pheromones (danger zones avoided; mating hotspots form; food trails bias predator exploration).

---

