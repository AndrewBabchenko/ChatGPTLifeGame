# GPU Setup Guide for Life Game Training

## Current Status
✓ Training script is GPU-ready  
✗ PyTorch CPU-only version detected  

## Performance Impact
- **CPU Training**: ~2-5 minutes per episode (100 episodes = 3-8 hours)
- **GPU Training**: ~5-15 seconds per episode (100 episodes = 10-25 minutes)
- **Speedup**: ~10-20x faster with GPU

## To Enable GPU Training

### Step 1: Identify Your GPU

**For NVIDIA GPUs:**
```powershell
nvidia-smi
```

**For AMD GPUs:**
```powershell
# Check Windows Device Manager or run:
wmic path win32_VideoController get name
```

---

## 🟢 For AMD GPUs (Radeon)

### ⚠️ Python 3.13 Limitation

**Important**: As of December 2025, torch-directml does not support Python 3.13 yet.

**Workaround Options:**

1. **Use Python 3.11** (Recommended if you need GPU)
   - Install Python 3.11 from python.org
   - Create a virtual environment with Python 3.11
   - Then install torch-directml

2. **Wait for torch-directml Python 3.13 support** (Coming soon)
   - Check: https://github.com/microsoft/DirectML

3. **Use CPU with optimizations** (Current solution)
   - Stay on Python 3.13
   - Use optimized CPU training (see below)

### Option 1: DirectML (For Python 3.10 - 3.11 only)

DirectML provides GPU acceleration for AMD (and Intel) GPUs on Windows.

**Install PyTorch with DirectML:**
```powershell
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Install torch-directml
pip install torch-directml
pip install torchvision torchaudio
```

**Verify:**
```powershell
python -c "import torch_directml; dml = torch_directml.device(); print('DirectML Device:', dml)"
```

**Update train_advanced.py to use DirectML:**

Find line 323 (in main function):
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Replace with:
```python
# Try DirectML first (AMD/Intel), then CUDA (NVIDIA), then CPU
try:
    import torch_directml
    device = torch_directml.device()
    print("Using DirectML (AMD/Intel GPU)")
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Performance:**
- 3-8x faster than CPU
- Not as fast as NVIDIA CUDA (~50-70% of CUDA performance)
- Still much better than CPU!

### Option 2: ROCm (Linux Only - Advanced)

ROCm provides full PyTorch support for AMD GPUs but requires Linux:
- Ubuntu 20.04/22.04
- Install ROCm drivers
- Install PyTorch ROCm build

**Not recommended for Windows** - DirectML is easier and works well.

---

## 🔵 For NVIDIA GPUs (GeForce/RTX)

### Step 1: Install CUDA-Enabled PyTorch

**Uninstall current PyTorch:**
```powershell
pip uninstall torch torchvision torchaudio
```

**Install CUDA version (choose based on your CUDA version):**

For CUDA 11.8:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Verify GPU is Available
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Should output:
```
CUDA Available: True
GPU: <Your GPU Name>
```

### Step 4: Run Training
```powershell
python train_advanced.py
```

**For DirectML (AMD/Intel), you should see:**
```
Using DirectML (AMD/Intel GPU)
Device: privateuseone:0
Model Size: 93,641 parameters
```

**For CUDA (NVIDIA), you should see:**
```
Device: cuda
GPU: <Your GPU Name>
CUDA Version: 11.8
GPU Memory: X.X GB
```

## Training on CPU (Current Setup)

If you don't have an NVIDIA GPU or prefer to use CPU:

**Pros:**
- No setup required (already working)
- Works on any machine
- Stable and compatible

**Cons:**
- Much slower training (3-8 hours for 100 episodes)
- CPU may get very hot during training

**Tips for CPU Training:**
1. **Reduce episodes**: Change `num_episodes = 100` to `num_episodes = 20` in train_advanced.py
2. **Monitor progress**: Check output every 10-15 minutes
3. **Run overnight**: Let it train while you're not using the computer
4. **Close other programs**: Free up CPU resources

## Expected Training Times

| Hardware | Time per Episode | 100 Episodes |
|----------|------------------|--------------|
| CPU (i5/i7) | 2-3 min | 3-5 hours |
| CPU (Ryzen) | 1.5-2.5 min | 2.5-4 hours |
| AMD GPU (DirectML) | 15-30 sec | 25-50 min |
| AMD GPU (RX 6000) | 12-20 sec | 20-35 min |
| NVIDIA GTX 1060 | 10-15 sec | 15-25 min |
| NVIDIA RTX 3060 | 5-10 sec | 8-15 min |
| NVIDIA RTX 4080 | 3-5 sec | 5-8 min |

## Current Training Status

Your training is currently running on CPU. You can:

1. **Let it complete**: Will take several hours but will work
2. **Stop and enable GPU**: Press Ctrl+C, follow GPU setup, restart
3. **Reduce episodes**: Edit train_advanced.py, change line 365 from `num_episodes = 100` to `num_episodes = 20`

## Monitoring Training

Check progress:
```powershell
# In another terminal window
Get-Content -Path "training.log" -Wait -Tail 20
```

Or just watch the terminal output - it updates after each episode with:
- Population sizes (prey/predators)
- Births, deaths, meals
- Episode rewards
- Loss values
- Model saves (when performance improves)

## After Training

Best models are saved to:
- `models/model_A_ppo.pth` (Prey)
- `models/model_B_ppo.pth` (Predators)

Checkpoints every 10 episodes:
- `models/model_A_ppo_ep10.pth`
- `models/model_A_ppo_ep20.pth`
- etc.

To test the trained models:
```powershell
python main.py
```

This will load the best models and show the ecosystem in action!

## Troubleshooting

### "CUDA out of memory"
- Your GPU VRAM is full
- Solution: Reduce batch size in config.py: `PPO_BATCH_SIZE = 32`

### "RuntimeError: Expected all tensors to be on the same device"
- Mixed CPU/GPU tensors
- This shouldn't happen - the code handles it automatically
- Report this if you see it!

### Training seems stuck
- It's just slow on CPU
- Episode 1 takes longer (initialization)
- Subsequent episodes will be faster
- Check terminal - should show progress every 2-3 minutes

### GPU not detected after installing CUDA PyTorch
- Check CUDA drivers: `nvidia-smi`
- Update GPU drivers from NVIDIA website
- Verify CUDA toolkit is installed
- Try restarting your computer

## Advanced: Monitor GPU Usage

If you have a GPU and want to monitor it during training:

**Windows:**
```powershell
# Open another terminal
nvidia-smi -l 1  # Updates every second
```

You should see:
- GPU utilization: 80-100%
- Memory usage: 1-2 GB
- Temperature: 60-80°C

---

**Happy Training! 🚀**
