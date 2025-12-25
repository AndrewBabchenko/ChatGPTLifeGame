# WSL2 ROCm Setup Guide

## Why WSL2?

**Windows ROCm Limitation**: AMD officially lists "No ML training support" for ROCm on Windows. Forward passes work, but backward passes hang with large models (confirmed with our 2.9M parameter model).

**Solution**: Train on Linux via WSL2 where ROCm training is fully supported, while keeping Windows for development.

---

## Setup Steps

### 1. Install WSL2 with Ubuntu

```powershell
# Open PowerShell as Administrator
wsl --install -d Ubuntu-22.04

# Restart your computer if prompted
```

After restart, Ubuntu will launch and ask for a username/password.

### 2. Install ROCm in WSL2

```bash
# Inside WSL2 Ubuntu terminal

# Update system
sudo apt update && sudo apt upgrade -y

# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/6.2.4/ubuntu/jammy/amdgpu-install_6.2.60204-1_all.deb
sudo apt install ./amdgpu-install_6.2.60204-1_all.deb

# Install ROCm
sudo amdgpu-install --usecase=rocm --no-dkms

# Add user to video/render groups
sudo usermod -a -G render,video $LOGNAME

# Restart WSL (from PowerShell)
# wsl --shutdown
# Then reopen Ubuntu
```

### 3. Verify GPU Access

```bash
# Check if GPU is visible
rocm-smi

# Should show your AMD Radeon RX 9070 XT
```

### 4. Install Python Environment

```bash
# Install Python and pip
sudo apt install python3.10 python3.10-venv python3-pip -y

# Navigate to project (WSL can access Windows files)
cd /mnt/c/Users/Andrey/.github/ChatGPTLifeGame

# Create virtual environment
python3.10 -m venv .venv_wsl_rocm

# Activate
source .venv_wsl_rocm/bin/activate

# Install PyTorch with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

### 5. Verify PyTorch ROCm

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output:
```
PyTorch: 2.x.x+rocm6.2
CUDA: True
GPU: AMD Radeon RX 9070 XT
```

### 6. Run Training

```bash
# Set environment
export PYTHONPATH=/mnt/c/Users/Andrey/.github/ChatGPTLifeGame
export HSA_ENABLE_SDMA=0
export HIP_LAUNCH_BLOCKING=1

# Run training
python scripts/train_advanced.py
```

---

## Quick Reference

### Access Windows Files from WSL2
```bash
cd /mnt/c/Users/Andrey/.github/ChatGPTLifeGame
```

### Access WSL2 Files from Windows
In File Explorer: `\\wsl$\Ubuntu-22.04\home\<username>\`

### Restart WSL2 (from PowerShell)
```powershell
wsl --shutdown
```

### Check WSL2 Status
```powershell
wsl --list --verbose
```

---

## Training Workflow

**Development** (Windows):
- Edit code in VS Code on Windows
- Files are in `C:\Users\Andrey\.github\ChatGPTLifeGame`

**Training** (WSL2):
- Open WSL2 terminal
- Navigate to `/mnt/c/Users/Andrey/.github/ChatGPTLifeGame`
- Run training with GPU support

**Best of Both Worlds**:
- ✅ Full GPU training support (Linux ROCm)
- ✅ Windows development environment
- ✅ Same codebase, no duplication

---

## Troubleshooting

### GPU Not Visible in WSL2
```bash
# Check WSL2 version (must be 2, not 1)
wsl --list --verbose

# If version 1, convert to version 2
wsl --set-version Ubuntu-22.04 2
```

### ROCm Not Installing
```bash
# Check Ubuntu version (needs 22.04)
lsb_release -a

# Reinstall if wrong version
```

### Training Still Hangs
```bash
# Try different environment flags
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export GPU_MAX_HW_QUEUES=2
```

---

## Performance Notes

- **WSL2 GPU Performance**: Nearly native Linux performance (~95%)
- **Training Time**: 2.9M model with 100 steps/episode should complete in reasonable time
- **Memory**: Full access to 17.1GB GPU memory
- **No Windows ROCm Limitations**: Backward passes work reliably

---

## Alternative: CPU Training on Windows

If you don't want to set up WSL2:

```powershell
# Run on CPU (slower but stable)
python scripts/train_advanced.py --cpu
```

**Note**: CPU training is ~10-20x slower but requires no setup.
