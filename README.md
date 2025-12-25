# Life Game - Predator-Prey Ecosystem

An optimized predator-prey simulation using neural networks with reinforcement learning.

## Project Structure

```
ChatGPTLifeGame/
├── main.py                  # Main entry point (OPTIMIZED)
├── src/                     # Source code
│   ├── config.py           # Configuration settings
│   ├── animal.py           # Animal class and behavior
│   ├── neural_network.py   # PyTorch neural network
│   ├── visualizer.py       # Optimized game field display
│   └── simulation.py       # Simulation engine
├── models/                  # Trained model files
│   ├── model_A_fixed.pth   # Prey model
│   └── model_B_fixed.pth   # Predator model
├── docs/                    # Documentation
│   ├── CODE_REVIEW.md
│   ├── IMPROVEMENTS.md
│   └── MODULAR_STRUCTURE.md
├── Life_Game_Fixed.py       # Training script
└── Life_Game.py             # Original version
```

## Quick Start

### Running the Demo (Recommended)
```bash
python main.py
```

### Training Models
```bash
python Life_Game_Fixed.py
```

## Features

### Optimized Performance
- ✅ Fast rendering with batch scatter plots
- ✅ Graph updates every 5 steps (configurable)
- ✅ Reduced pause time (0.0001s)
- ✅ Efficient spatial grid for mating detection

### Fixed Layout
- ✅ Proper spacing between elements
- ✅ No overlapping panels
- ✅ Tight grid layout (16:9 aspect ratio)
- ✅ Consistent margins and padding

### Clean UI
- ✅ No emoji (fixed strange box symbols)
- ✅ Clear text labels
- ✅ Professional game field aesthetic
- ✅ Dark green field with subtle grid

### Interactive Controls
- ✅ Pause/Resume button
- ✅ Restart button
- ✅ Real-time statistics
- ✅ Population graph
- ✅ Final statistics window

## Game Field

- Dark green background (#2d5016)
- No X/Y axis labels (pure game aesthetic)
- Animals displayed as colored circles:
  - **Prey**: Bright green (#00ff00)
  - **Predator (Normal)**: Red (#ff4444)
  - **Predator (Hungry)**: Dark red (#880000)

## Performance Improvements

### Before
- Full redraw every frame
- Graph updated every step
- Individual circle patches
- plt.pause(0.001)

### After
- Batch scatter plots (10x faster)
- Graph updates every 5 steps
- Optimized animal rendering
- plt.pause(0.0001) - minimal delay

## Configuration

Edit `src/config.py` to customize:
- Grid size
- Population counts
- Movement speeds
- Hunger thresholds
- Mating probabilities

## Requirements

```bash
pip install torch matplotlib numpy
```

## Controls

- **Pause Button**: Pause/Resume simulation
- **Restart Button**: Start new simulation
- **Close Window**: Exit application

## Statistics Tracked

- Current population (prey/predators)
- Birth and death counts
- Predator meals
- Average survival times
- Population peaks and minimums
- Prey:Predator ratio

## Notes

- Models are loaded from `models/` directory
- If no trained models found, uses random behavior
- Train models first with `Life_Game_Fixed.py`
- Simulation runs for max 1000 steps (configurable)

## Performance Tips

1. Reduce `MAX_ANIMALS` in config for better speed
2. Increase graph update interval in visualizer
3. Lower initial population counts
4. Adjust `plt.pause()` value in visualizer

## Troubleshooting

**Slow performance?**
- Check `MAX_ANIMALS` setting
- Reduce graph update frequency
- Close other applications

**Layout issues?**
- Check figure size in visualizer
- Adjust `hspace` and `wspace` values
- Verify matplotlib backend

**Models not found?**
- Run `Life_Game_Fixed.py` first
- Check `models/` directory exists
- Verify .pth files present

## License

Educational project for learning reinforcement learning and ecosystem simulation.
