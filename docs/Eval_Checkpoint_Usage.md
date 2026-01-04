# Checkpoint Evaluation Module

## Overview
The `eval_checkpoints.py` script evaluates trained checkpoints by running real environment simulations and tracking event-based metrics to quantify what the agents have learned.

## Features

### Deterministic Evaluation
- Fixed random seeds for reproducible results
- Deterministic action selection (argmax) or stochastic (sampling)
- Uses same observation path as training for consistency

### Event Tracking

**Prey Detection Events** - When prey sees predator:
- Initial distance to predator
- Escape success rate (alive after T steps)
- Distance gain after 1 step
- Distance gain after 5 steps

**Predator Detection Events** - When predator sees prey:
- Initial distance to prey
- Capture success rate
- Time to capture (for successful hunts)

### Aggregate Metrics
- Population counts (final, births, deaths)
- Starvation deaths
- Meals per predator
- Event counts and success rates
- Distance-based escape metrics

## Usage

### Basic Evaluation
```bash
python scripts/eval_checkpoints.py \
    --checkpoint-dir outputs/checkpoints \
    --episodes 1 10 50 100 \
    --num-eval-episodes 3 \
    --steps 200
```

### Arguments
- `--checkpoint-dir`: Directory containing checkpoint pairs (default: `outputs/checkpoints`)
- `--episodes`: List of episode numbers to evaluate (e.g., `1 10 50 100`)
- `--num-eval-episodes`: Number of eval episodes per checkpoint (default: 3)
- `--steps`: Steps per eval episode (default: 200)
- `--output-dir`: Directory to save results (default: `outputs/eval_results`)
- `--device`: Device to use - cpu/cuda (default: cpu)

### Expected Checkpoint Naming
The script supports two naming formats:

**Old format:**
- Prey: `model_A_ppo_ep{N}.pth`
- Predator: `model_B_ppo_ep{N}.pth`

**New format:**
- Prey: `{prefix}_ep{N}_model_A.pth`
- Predator: `{prefix}_ep{N}_model_B.pth`

Where `{N}` is the episode number (e.g., `ep1`, `ep10`, `ep100`).

## Output Format

### Individual Checkpoint Results
Saved as `eval_ep{N}.json` with structure:
```json
{
  "checkpoint_episode": 1,
  "num_eval_episodes": 3,
  "steps_per_episode": 200,
  
  "final_prey_count_mean": 45.2,
  "final_predator_count_mean": 28.5,
  
  "prey_escape_rate_mean": 0.78,
  "prey_dist_gain_1_mean": 1.2,
  "prey_dist_gain_5_mean": 3.5,
  
  "predator_capture_rate_mean": 0.22,
  "predator_time_to_capture_median": 12.5,
  "predator_meals_per_alive_mean": 1.8,
  
  "prey_detection_events_total": 450,
  "predator_detection_events_total": 320,
  
  "episodes": [
    {
      "episode": 0,
      "final_prey_count": 48,
      ...
    },
    ...
  ]
}
```

### Combined Summary
Saved as `eval_summary.json`:
```json
{
  "checkpoints_evaluated": 4,
  "results": [
    { /* ep1 results */ },
    { /* ep10 results */ },
    { /* ep50 results */ },
    { /* ep100 results */ }
  ]
}
```

## Metrics Interpretation

### Prey Metrics
- **escape_rate**: Fraction of prey that survive after being detected by predator
  - 0.0 = never escape
  - 1.0 = always escape
  - Target: Should increase with training (better evasion)

- **dist_gain_1/5**: Average distance gained after 1/5 steps
  - Positive = increasing distance (fleeing)
  - Negative = decreasing distance (getting caught)
  - Target: Should be positive and increasing with training

### Predator Metrics
- **capture_rate**: Fraction of detected prey that are captured
  - 0.0 = never capture
  - 1.0 = always capture
  - Target: Should increase with training (better hunting)

- **time_to_capture_median**: Median steps to successful capture
  - Lower = faster hunting
  - Target: Should decrease with training (more efficient)

- **meals_per_alive**: Average meals per surviving predator
  - Higher = better sustained hunting
  - Target: Should increase with training

## Integration with Dashboard

The dashboard can read these JSON files to display:
- Learning curves (metrics vs checkpoint episode)
- Comparison across checkpoints
- Event distributions and success rates

## Implementation Details

### Observation Path
Uses identical observation path as training:
```python
visible = animal.communicate(animals, config)
obs = animal.get_enhanced_input(animals, config, pheromone_map, visible_animals=visible)
vis_tensor = torch.tensor([visible], dtype=torch.float32)
```

### Action Selection
Deterministic mode (default):
```python
turn_action = turn_logits.argmax(dim=1).item()
move_action = move_logits.argmax(dim=1).item()
```

Stochastic mode (set `deterministic=False`):
```python
turn_action = torch.multinomial(turn_probs, 1).item()
move_action = torch.multinomial(move_probs, 1).item()
```

### Movement Logic
Hierarchical turn→move:
1. Apply turn action to update heading
2. Move in heading direction for `num_moves` steps
3. Check position occupancy before each move

### Event Detection
Prey threat detection:
```python
visible = prey.communicate(animals, config)
vis_info = prey.summarize_visible(visible)
if vis_info["predator_count"] > 0:
    # Track new threat event
```

Predator hunt detection:
```python
visible = predator.communicate(animals, config)
vis_info = predator.summarize_visible(visible)
if vis_info["prey_count"] > 0:
    # Track new hunt event
```

## Example Workflow

1. **Train models** for multiple episodes:
   ```bash
   python scripts/train.py --episodes 100
   ```

2. **Evaluate key checkpoints**:
   ```bash
   python scripts/eval_checkpoints.py \
       --episodes 1 10 20 50 100 \
       --num-eval-episodes 5 \
       --steps 200
   ```

3. **View results** in dashboard:
   - Load `outputs/eval_results/eval_summary.json`
   - Plot learning curves
   - Compare metrics across checkpoints

## Future Enhancements

- [x] Track distance gains at t+1, t+5 during simulation (real-time tracking implemented)
- [ ] Add more detailed event timelines (step-by-step tracking)
- [ ] Compute statistical significance tests (t-tests across eval episodes)
- [ ] Add heatmap visualization (where events occur on grid)
- [ ] Track cooperation metrics (multi-predator hunts, prey herding)
- [ ] Add curriculum-aware evaluation (test at different difficulty levels)
