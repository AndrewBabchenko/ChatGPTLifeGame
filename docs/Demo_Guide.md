# Demo Application User Guide

**Interactive simulation viewer for trained predator-prey models**

---

## Overview

The Demo application lets you watch trained AI agents (predators and prey) interact in real-time. You can control simulation speed, change random seeds for different scenarios, and view live statistics charts.

**To launch:**
- Double-click `Demo.vbs` in the project root, or
- Run `python scripts/run_demo.py`

![Demo App Overview](screenshots/demo_overview.png)
*Screenshot: Demo application main window*

---

## Table of Contents
- [Getting Started](#getting-started)
- [Main Controls](#main-controls)
- [Simulation Tab](#simulation-tab)
- [Charts Tab](#charts-tab)
- [Evaluation Tab](#evaluation-tab)
- [Configuration Tab](#configuration-tab)
- [Tips & Tricks](#tips--tricks)

---

## Getting Started

### Prerequisites
- Trained model checkpoints in `outputs/checkpoints/`
- Python environment with required packages installed

### First Launch
1. Launch the demo application
2. Select a checkpoint from the dropdown (or use the default)
3. Click **▶ Play** to start the simulation

![Checkpoint Selection](screenshots/demo_checkpoint_select.png)
*Screenshot: Checkpoint dropdown selector*

---

## Main Controls

### Checkpoint Selector
Located at the top of the window:
- **Dropdown**: Lists all available checkpoints from `outputs/checkpoints/`
- **🔄 Refresh**: Rescan for new checkpoints

Checkpoints are named like:
- `phase1_ep50` - Phase 1, episode 50
- `model_A_ppo` - Latest best model

---

## Simulation Tab

The main visualization of the predator-prey ecosystem.

![Simulation Tab](screenshots/demo_simulation.png)
*Screenshot: Simulation tab showing animals and environment*

### Playback Controls

| Button | Action |
|--------|--------|
| **▶ Play** | Start continuous simulation |
| **⏸ Pause** | Pause simulation |
| **➡ Step** | Advance one step (when paused) |
| **🔄 Restart** | Reset simulation with current seed |
| **🎲** | Generate new random seed |

### Seed Control
- Enter a specific seed number to reproduce exact scenarios
- Same seed = same initial positions and random events
- Useful for comparing different checkpoints on identical scenarios

### Speed Slider
- Scale from 1 (slowest) to 10 (fastest)
- Lower speeds help observe individual animal behaviors
- Higher speeds for quickly testing population dynamics

### Simulation Field

The main canvas shows:

| Element | Appearance | Description |
|---------|------------|-------------|
| **Prey** | Blue circles | Herbivores that eat grass and flee predators |
| **Predators** | Red/orange triangles | Carnivores that hunt prey |
| **Hungry Predator** | Orange triangle | Predator that hasn't eaten recently |
| **Grass** | Dark green cells | Food source for prey |
| **Danger Pheromone** | Red haze | Warning signals from prey |
| **Mating Pheromone** | Yellow haze | Attraction signals |

### Field of View Cones
- Each animal displays its vision cone
- Prey have wider FOV (240°) but shorter range
- Predators have narrower FOV (180°) but longer range
- Animals can only see others within their cone

### Statistics Panel
Real-time counters on the right side:
- **Step**: Current simulation step
- **Prey**: Current prey count
- **Predators**: Current predator count
- **Births**: Total births this episode
- **Deaths**: Total deaths this episode
- **Meals**: Successful predator hunts
- **Grass Eaten**: Prey foraging events

### Death Effects
When animals die, colored splash effects indicate cause:
- **Red splash**: Prey eaten by predator
- **Brown splash**: Starvation
- **Blue splash**: Exhaustion (energy depleted)
- **Gray splash**: Old age

---

## Charts Tab

Live graphs showing population dynamics over time.

![Charts Tab](screenshots/demo_charts.png)
*Screenshot: Charts tab with population graphs*

### Population Chart
- **Blue line**: Prey population
- **Red line**: Predator population
- X-axis: Simulation step
- Y-axis: Population count

### Events Chart
- Tracks cumulative births, deaths, meals, grass eaten
- Helps identify trends and ecosystem balance

### Chart Features
- Auto-scales to fit data
- Updates in real-time during simulation
- Resets when simulation restarts

---

## Evaluation Tab

Displays performance metrics for the current simulation run.

![Evaluation Tab](screenshots/demo_evaluation.png)
*Screenshot: Evaluation metrics display*

### Metrics Displayed

| Metric | Description |
|--------|-------------|
| **Current Step** | Simulation progress |
| **Current Prey/Predators** | Live population counts |
| **Total Births/Deaths** | Cumulative events |
| **Detection Events** | Times predator spotted prey |
| **Capture %** | Successful hunts / detections |
| **Escape %** | Prey escapes / detections |
| **Avg Capture Time** | Steps from detection to capture |
| **Meals/Pred** | Hunting efficiency per predator |

### Using Evaluation Data
- Compare different checkpoints on same scenario (use fixed seed)
- Track learning progress across training phases
- Identify if predators or prey need more training

---

## Configuration Tab

View and modify simulation parameters.

![Configuration Tab](screenshots/demo_config.png)
*Screenshot: Configuration editor*

### Parameter Categories
- **Grid Settings**: World size, boundaries
- **Population Settings**: Initial counts, max populations
- **Animal Behavior**: Vision ranges, FOV angles, hunger thresholds
- **Movement Speeds**: How fast animals move
- **Energy System**: Energy costs and gains
- **Pheromone System**: Decay rates, sensing range

### Editing Parameters
1. Modify values in the entry fields
2. Click **💾 Save to config.py** to persist changes
3. Click **🔄 Reload** to discard changes and reload from file

**Note**: Changes affect the next simulation restart, not the current run.

---

## Tips & Tricks

### Debugging Behavior Issues
1. Pause the simulation
2. Use **➡ Step** to advance frame-by-frame
3. Observe individual animal decisions
4. Check if vision cones align with targets

### Comparing Checkpoints
1. Note a specific seed that produces interesting scenarios
2. Select checkpoint A, enter the seed, run simulation
3. Record final stats (prey count, meals, etc.)
4. Select checkpoint B, enter same seed, run again
5. Compare results

### Finding Good Seeds
- Seeds that produce early predator-prey encounters test hunting/evasion
- Seeds with clustered initial positions test crowding behavior
- Try several random seeds to find interesting scenarios

### Performance Tips
- Lower speed for detailed observation
- Higher speed to quickly evaluate population outcomes
- Close Charts tab if experiencing lag (matplotlib rendering)

### Recommended Workflow
1. Start with speed 5 to observe general behavior
2. Increase to speed 8-10 to see population trends
3. If interesting event occurs, pause and step through slowly
4. Use fixed seeds to reproduce and analyze specific scenarios

---

## Troubleshooting

### "No checkpoints found"
- Ensure training has completed at least one episode
- Check `outputs/checkpoints/` contains `.pth` files
- Click **🔄 Refresh** to rescan

### Simulation runs very slowly
- Reduce population in Configuration tab
- Close other applications
- Use smaller grid size for testing

### Animals not moving intelligently
- Ensure correct checkpoint is loaded
- Early training checkpoints may show random behavior
- Use later phase checkpoints (phase3, phase4) for learned behaviors

### Window appears blurry
- Windows DPI scaling should be handled automatically
- If still blurry, try running at 100% display scaling

---

*For training monitoring, see the [Dashboard Guide](Dashboard_Guide.md)*
