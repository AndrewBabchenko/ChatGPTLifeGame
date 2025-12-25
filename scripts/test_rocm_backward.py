"""
Minimal test to reproduce ROCm backward hang
Tests the actor-critic network with typical PPO batch shapes
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SimulationConfig
from src.models.actor_critic_network import ActorCriticNetwork
from datetime import datetime

def test_backward(use_sdpa_math=False):
    """
    Test forward + backward pass with typical PPO shapes
    
    Args:
        use_sdpa_math: If True, force SDPA to use math backend (not flash attention)
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] ==========================================")
    print(f"[{timestamp}] ROCm Backward Pass Test")
    print(f"[{timestamp}] SDPA Math Backend: {use_sdpa_math}")
    print(f"[{timestamp}] ==========================================\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Device: {device}")
    
    if device.type == "cuda":
        print(f"[{timestamp}] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[{timestamp}] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create network
    config = SimulationConfig()
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Creating ActorCriticNetwork...")
    net = ActorCriticNetwork(config).to(device)
    
    total_params = sum(p.numel() for p in net.parameters())
    print(f"[{timestamp}] Model: {total_params:,} parameters")
    
    # Test shapes matching PPO update
    batch_size = 256  # PPO_BATCH_SIZE
    max_visible = config.MAX_VISIBLE_ANIMALS
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Test batch: {batch_size} samples, {max_visible} visible animals")
    
    # Create test data
    self_state = torch.randn(batch_size, 21, device=device)
    visible_animals = torch.randn(batch_size, max_visible, 8, device=device)
    
    # Test forward pass
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Running forward pass...")
    
    try:
        if use_sdpa_math:
            # Force SDPA to use math backend
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with sdpa_kernel(SDPBackend.MATH):
                    action_probs, state_value = net(self_state, visible_animals)
            except ImportError:
                # Fallback for older PyTorch
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                action_probs, state_value = net(self_state, visible_animals)
        else:
            action_probs, state_value = net(self_state, visible_animals)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] ✓ Forward pass completed")
        print(f"[{timestamp}]   Action probs shape: {action_probs.shape}")
        print(f"[{timestamp}]   State value shape: {state_value.shape}")
        
        # Test backward pass
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Running backward pass...")
        
        loss = (action_probs.mean() + state_value.mean())
        loss.backward()
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] ✓ Backward pass completed!")
        print(f"[{timestamp}] ✓ Test PASSED - No hang detected")
        
        return True
        
    except Exception as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] ✗ Test FAILED with exception:")
        print(f"[{timestamp}]   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*70)
    print("  ROCm Backward Pass Test")
    print("  Tests if attention backward hangs on Windows ROCm")
    print("="*70)
    
    # Test without SDPA fix
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] Test 1: Default SDPA (may use flash attention)")
    result1 = test_backward(use_sdpa_math=False)
    
    # Test with SDPA math backend
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] Test 2: Force SDPA math backend (ROCm fix)")
    result2 = test_backward(use_sdpa_math=True)
    
    # Summary
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] ==========================================")
    print(f"[{timestamp}] Test Summary:")
    print(f"[{timestamp}] ==========================================")
    print(f"[{timestamp}] Default SDPA: {'PASSED' if result1 else 'FAILED'}")
    print(f"[{timestamp}] Math SDPA:    {'PASSED' if result2 else 'FAILED'}")
    print(f"[{timestamp}] ==========================================")
    
    if result2 and not result1:
        print(f"[{timestamp}] → SDPA math backend fixes the issue!")
        print(f"[{timestamp}] → Training should work now")
    elif result1 and result2:
        print(f"[{timestamp}] → Both tests passed - no hang detected")
    else:
        print(f"[{timestamp}] → Issue persists - may need WSL2/Linux")
