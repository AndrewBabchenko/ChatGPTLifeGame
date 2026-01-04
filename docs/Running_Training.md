# Running Training

This guide explains how to run training safely without interruptions.

## Quick Start

### Option 1: Direct Python (Recommended)
```powershell
python scripts/train.py
```

**Benefits**:
- ✅ Quick and simple
- ✅ See progress in real-time
- ✅ Checkpoints saved every episode
- ✅ Ctrl+C saves interrupt checkpoint

### Option 2: With Logging
```powershell
python scripts/train.py 2>&1 | Tee-Object -FilePath outputs/logs/training.log
```

**Benefits**:
- ✅ Output shown on screen AND saved to file
- ✅ Can review later

### Option 3: Using Dashboard
```powershell
python scripts/run_dashboard.py
```

**Benefits**:
- ✅ GUI interface
- ✅ Real-time monitoring
- ✅ Start/stop training from GUI
- ✅ View metrics and charts

### Option 4: Phase-Based Training
```powershell
python scripts/run_phase.py --phase 1
python scripts/run_phase.py --phase 2
python scripts/run_phase.py --phase 3
python scripts/run_phase.py --phase 4
```

**Benefits**:
- ✅ Curriculum learning (easy → hard)
- ✅ Each phase builds on previous
- ✅ Better final performance

## Progress Features

The training shows detailed progress:

### Within Each Episode:
```
Episode 5/150
  Step 50/300: Prey=38, Predators=18
  Step 100/300: Prey=35, Predators=16
  Step 150/300: Prey=32, Predators=14
```

### After Each Episode:
```
  Final: Prey=28, Predators=11
  Births=5, Deaths=17, Meals=8
  Exhaustion=3, Old Age=1
  Rewards: Prey=1234.5, Predator=678.9
  Losses: Policy(P=0.12/Pr=0.23), Value(P=0.34/Pr=0.45)
  Time: 8.3s | Avg: 7.5s | ETA: 18min
```

### Checkpoints:
Models saved every episode to `outputs/checkpoints/`:
```
phase1_ep10_model_A.pth  (prey)
phase1_ep10_model_B.pth  (predator)
```

## Running Outside VS Code (Optional)

### Why Run Outside VS Code?
- ✅ More stable (no editor overhead)
- ✅ Lower memory usage
- ✅ Won't stop if VS Code crashes

### How to Run in PowerShell:

1. **Open PowerShell** (not VS Code terminal)
   - Press `Win + X`, select "Windows PowerShell"

2. **Navigate to project**:
   ```powershell
   cd C:\Users\Andrey\.github\ChatGPTLifeGame
   ```

3. **Activate virtual environment** (if using one):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. **Run training**:
   ```powershell
   python scripts/train.py
   ```

## Safety Tips

### ✅ DO:
1. **Run in dedicated PowerShell window** (not VS Code)
2. **Keep computer awake** during training
3. **Check logs/** folder for outputs
4. **Let it save checkpoints** (every 10 episodes)
5. **Wait for "Training Complete" message**

### ❌ DON'T:
1. **Don't close PowerShell window** during training
2. **Don't hibernate/sleep computer**
3. **Don't run in VS Code terminal** (less stable)
4. **Don't manually stop without Ctrl+C** (may lose progress)
5. **Don't delete outputs/checkpoints/** folder during training

## If Training Stops

### Recovery Steps:

1. **Check for saved models**:
   ```powershell
   dir outputs/checkpoints/
   ```
   You should see:
   - `model_A_ppo.pth` (best prey model)
   - `model_B_ppo.pth` (best predator model)
   - `model_A_ppo_ep10.pth`, `model_A_ppo_ep20.pth`, etc. (checkpoints)

2. **Check the log file**:
   ```powershell
   dir outputs/logs/
   ```
   Open most recent log to see where it stopped

3. **Resume training**:
   - Training always starts fresh (by design)
   - Previous best models are kept
   - Just run `.\run_training_safe.ps1` again

## Expected Training Time

### With GPU (DirectML/CUDA):
- **Per episode**: 5-15 seconds
- **150 episodes**: 15-40 minutes
- **Full 4-phase training**: 1-2 hours

### Progress Indicators:
```
Episode 1/150   (Just started)
Episode 15/150  (10% done)
Episode 75/150  (50% done)
Episode 150/150 (Complete!)
```

## Monitoring Training

### While Training is Running:

**Watch for**:
- Episode number increasing (1, 2, 3...)
- Step progress within episodes (50/200, 100/200...)
- Population counts (Prey=X, Predators=Y)
- "New best!" messages (model improving)
- Time estimates (ETA: Xmin)

**Good signs**:
- ✅ Episode times consistent (1-3s each)
- ✅ Populations surviving (not all dying)
- ✅ Births happening regularly
- ✅ Models saving occasionally

**Bad signs** (need attention):
- ⚠️ Episodes taking 10+ seconds (something wrong)
- ⚠️ All animals dying immediately
- ⚠️ No output for 30+ seconds
- ⚠️ Errors in red text

## Output Files

After training completes:

### outputs/checkpoints/
```
phase1_ep50_model_A.pth   - Prey model at episode 50, phase 1
phase1_ep50_model_B.pth   - Predator model at episode 50, phase 1
model_A_ppo.pth           - Current best prey model
model_B_ppo.pth           - Current best predator model
model_A_interrupt.pth     - Saved on Ctrl+C interrupt
```

### outputs/logs/
```
training_20251225_143022.log - Training output (if using Tee-Object)
```

## Troubleshooting

### Training is very slow (5+ seconds per episode):
```powershell
# Check CPU usage is normal
# Make sure no other heavy programs running
# Close VS Code if open
```

### Can't see progress:
```powershell
# Make sure PowerShell window is large enough
# Progress updates every 50 steps
# Should see output every 1-2 seconds
```

### Training stopped unexpectedly:
```powershell
# Check outputs/logs/ for error messages
# Check outputs/checkpoints/ for saved checkpoints
# Computer didn't sleep/hibernate?
# Restart training - it will continue from best model
```

## Best Practice Workflow

1. **Before starting**:
   ```powershell
   # Open PowerShell (not VS Code)
   cd C:\Users\Andrey\.github\ChatGPTLifeGame
   dir outputs/checkpoints/  # Check existing models
   ```

2. **Start training**:
   ```powershell
   .\scripts\run_training_safe.ps1
   # Press Y to confirm
   ```

3. **During training**:
   - Keep PowerShell window open
   - Can minimize but don't close
   - Can use computer for other tasks
   - Watch for "Episode X/100" progress

4. **After completion**:
   ```powershell
   # Check results
   dir outputs/checkpoints/
   
   # Run demo with trained models
   python scripts/run_demo.py
   ```

## Quick Reference

### Start Training:
```powershell
python scripts/train.py
```

### Stop Training Safely:
- Press `Ctrl + C` in terminal
- Interrupt checkpoint auto-saved
- Check outputs/checkpoints/ folder

### Check Progress:
- Watch terminal output
- Look for "Episode X/150"
- Check ETA times

### After Training:
```powershell
python scripts/run_demo.py  # See trained agents in action
```

## Summary

**Recommended Setup**:
1. Run `python scripts/train.py` in terminal
2. Keep window open
3. Wait for completion (~20-40 minutes per phase)
4. Check outputs/checkpoints/ folder

**For curriculum training**:
1. Run phase 1: `python scripts/run_phase.py --phase 1`
2. Run phase 2: `python scripts/run_phase.py --phase 2`
3. Run phase 3: `python scripts/run_phase.py --phase 3`
4. Run phase 4: `python scripts/run_phase.py --phase 4`

**Or use the dashboard**:
```powershell
python scripts/run_dashboard.py
```
