# Running Training Safely

This guide explains how to run training safely without interruptions.

## Quick Start (Recommended)

### Option 1: Safe Runner Script (Best)
```powershell
.\scripts\run_training_safe.ps1
```

Non-interactive (auto-confirm):
```powershell
.\scripts\run_training_safe.ps1 -Force
```

**Benefits**:
- ✅ Automatic log file creation (outputs/logs/training_YYYYMMDD_HHMMSS.log)
- ✅ Output shown on screen AND saved to file
- ✅ Proper error handling
- ✅ Safe to run in background
- ✅ Resume from checkpoints if interrupted

### Option 2: Direct Python (Simple)
```powershell
python scripts/train_advanced.py
```

**Benefits**:
- ✅ Quick and simple
- ✅ See progress in real-time
- ⚠️ Output lost if window closed

### Option 3: Background with Logging
```powershell
python scripts/train_advanced.py > outputs/logs/training.log 2>&1
```

**Benefits**:
- ✅ Runs in background
- ✅ All output saved
- ⚠️ Can't see progress live

## New Progress Features

The training now shows detailed progress:

### Within Each Episode:
```
Episode 5/100
  Step 50/200: Prey=95, Predators=18
  Step 100/200: Prey=87, Predators=15
  Step 150/200: Prey=82, Predators=12
```

### After Each Episode:
```
  Final: Prey=79, Predators=11
  Births=23, Deaths=44, Meals=12
  Exhaustion=8, Old Age=2
  Rewards: Prey=1234.5, Predator=678.9
  Losses: Policy(P=0.123/Pr=0.234), Value(P=0.345/Pr=0.456)
  Time: 2.3s | Avg: 2.1s | ETA: 45min
  ✓ New best! Saved models (prey survival: 79)
```

### Checkpoints:
```
Episode 10/100
  ✓ Checkpoint saved (episode 10)
```

## Running Outside VS Code (Recommended)

### Why Run Outside VS Code?
- ✅ More stable (no editor overhead)
- ✅ Lower memory usage
- ✅ Better performance
- ✅ Won't stop if VS Code crashes
- ✅ Can close VS Code while training

### How to Run in PowerShell:

1. **Open PowerShell** (not VS Code terminal)
   - Press `Win + X`, select "Windows PowerShell"
   - Or search "PowerShell" in Start menu

2. **Navigate to project**:
   ```powershell
   cd C:\Users\Andrey\.github\ChatGPTLifeGame
   ```

3. **Run training**:
   ```powershell
   .\scripts\run_training_safe.ps1
   ```

4. **Keep window open**
   - Don't minimize or close
   - Can work in other windows
   - Training continues in background

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

### With CPU (Current Setup):
- **Per episode**: 1-3 seconds
- **100 episodes**: 3-6 minutes
- **Total**: Under 10 minutes

### Progress Indicators:
```
Episode 1/100   (Just started)
Episode 10/100  (10% done, checkpoint saved)
Episode 25/100  (25% done)
Episode 50/100  (50% done, checkpoint saved)
Episode 75/100  (75% done)
Episode 100/100 (Complete!)
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
model_A_ppo.pth        - Best prey model
model_B_ppo.pth        - Best predator model
model_A_ppo_ep10.pth   - Checkpoint at episode 10
model_B_ppo_ep10.pth   - Checkpoint at episode 10
model_A_ppo_ep20.pth   - Checkpoint at episode 20
... (more checkpoints)
```

### outputs/logs/
```
training_20251225_143022.log - Full training output with timestamp
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
   dir outputs/logs/
   
   # Run demo with trained models
   python scripts/demo.py
   ```

## Quick Reference

### Start Training:
```powershell
.\scripts\run_training_safe.ps1
```

### Stop Training Safely:
- Press `Ctrl + C` in PowerShell
- Models auto-save before exit
- Check outputs/checkpoints/ folder

### Check Progress:
- Watch PowerShell output
- Look for "Episode X/100"
- Check ETA times

### After Training:
```powershell
python scripts/demo.py  # See trained agents in action
```

## Summary

**Recommended Setup**:
1. Use `scripts/run_training_safe.ps1` in PowerShell
2. Run outside VS Code
3. Keep window open
4. Wait for completion (~5-10 minutes)
5. Check outputs/checkpoints/ and outputs/logs/ folders

**Expected Experience**:
- See live progress updates
- Episode every 1-3 seconds
- Checkpoints every 10 episodes
- Complete training in under 10 minutes
- Models auto-saved throughout
