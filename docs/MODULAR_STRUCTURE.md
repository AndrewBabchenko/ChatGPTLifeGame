# Life Game Demo - Modular Structure

## Overview
The Life Game Demo has been refactored into multiple smaller, more maintainable files.

## File Structure

### Core Files

1. **`config.py`** (33 lines)
   - Contains `SimulationConfig` class
   - All simulation parameters in one place
   - Grid settings, population settings, behavior parameters

2. **`animal.py`** (220 lines)
   - Contains `Animal` class
   - Movement logic with neural network
   - Eating, mating, and communication methods
   - Position and visibility calculations

3. **`neural_network.py`** (37 lines)
   - Contains `SimpleNN` class (PyTorch model)
   - GRU-based neural network architecture
   - Forward pass for action prediction
   - 4 input → 16 → GRU → 32 → 16 → 8 output

4. **`visualizer.py`** (280 lines)
   - Contains `GameFieldVisualizer` class
   - **Game field display** (not a chart!)
   - Dark green game field with grid pattern
   - Animals displayed as colored circles/sprites
   - No X/Y axis labels - pure game aesthetic
   - Population graph, live statistics, legend
   - Pause and Restart buttons
   - Final statistics window

5. **`simulation.py`** (189 lines)
   - Contains simulation engine functions
   - `run_simulation()` - main loop
   - `build_spatial_grid()` - optimization
   - `create_population()` - initialization
   - Movement, eating, mating phases

6. **`Life_Game_Demo_New.py`** (110 lines)
   - Main entry point
   - Model loading
   - Game loop with restart capability
   - Console output with emoji
   - Error handling

## Key Improvements

### 1. Modular Design
- Each file has a single responsibility
- Easy to find and modify specific functionality
- Clear separation of concerns

### 2. Game Field Visualization
- **Replaced chart with game field**
- Dark green (#2d5016) game field background
- Grid pattern for retro game aesthetic
- No X/Y axis labels or ticks
- Animals drawn as colored circles with glow effect:
  - 🐰 Prey: Bright green (#00ff00)
  - 🦊 Predator (Normal): Bright red (#ff4444)
  - 🐺 Predator (Hungry): Dark red (#880000)
- HUD-style title bar with game stats

### 3. Better Organization
- Configuration centralized
- Neural network isolated
- Visualization separate from logic
- Easy to test individual components

### 4. Enhanced UX
- Emoji indicators throughout
- Clear console messages
- Interactive buttons (Pause/Resume, Restart)
- Final statistics window

## How to Use

### Running the Demo
```bash
python Life_Game_Demo_New.py
```

### Modifying Parameters
Edit `config.py` to change:
- Grid size
- Population counts
- Movement speeds
- Mating probabilities
- Hunger thresholds

### Extending the Code
- Add new animal behaviors in `animal.py`
- Modify visualization in `visualizer.py`
- Change network architecture in `neural_network.py`
- Add new simulation phases in `simulation.py`

## Dependencies
- PyTorch (neural networks)
- Matplotlib (visualization)
- NumPy (statistics)

## Visual Changes

### Before
- Scatter plot with X/Y axes
- Chart-like appearance
- Basic circles

### After
- **Game field** with no axes
- Retro game aesthetic
- Dark green field background
- Grid pattern overlay
- Colored sprites with glow effects
- HUD-style information display

## File Sizes (Approximate)
- config.py: ~0.8 KB
- animal.py: ~7 KB
- neural_network.py: ~1 KB
- visualizer.py: ~11 KB
- simulation.py: ~6 KB
- Life_Game_Demo_New.py: ~3.5 KB

**Total: ~29 KB** (previously 838 lines / ~35 KB in one file)

## Benefits
1. ✅ **Readability**: Each file is focused and under 300 lines
2. ✅ **Maintainability**: Easy to locate and fix issues
3. ✅ **Testability**: Can test components independently
4. ✅ **Extensibility**: Easy to add new features
5. ✅ **Game-like**: Visual field instead of chart
6. ✅ **Professional**: Better code organization
