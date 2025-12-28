"""
GPU smoke test: forward/backward/optimizer step works on detected device
"""
import torch
from tests._device import pick_device, is_gpu_device

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimulationConfig
from src.models.actor_critic_network import ActorCriticNetwork


def test_gpu_smoke_forward_backward_step():
    """Validates GPU can do forward/backward/step without errors or NaNs"""
    device = pick_device(prefer_gpu=True)
    
    # Uncomment to make this test GPU-only:
    # assert is_gpu_device(device), f"GPU not detected. device={device}"
    
    config = SimulationConfig()
    model = ActorCriticNetwork(config).to(device)
    model.train()
    
    B = 8
    obs = torch.randn(B, 34, device=device)
    vis = torch.zeros(B, config.MAX_VISIBLE_ANIMALS, 8, device=device)
    vis[:, :, 7] = 0.0  # All padding
    vis[:, 0, 7] = 1.0  # First slot present
    vis[:, 0, 4] = 1.0  # is_prey
    vis[:, 0, 0] = 0.5  # dx
    vis[:, 0, 1] = 0.0  # dy
    vis[:, 0, 2] = 0.5  # dist
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Forward pass
    turn_probs, move_probs, values = model.forward(obs, vis)
    loss = (turn_probs.mean() + move_probs.mean() + values.mean())
    
    # Backward pass
    opt.zero_grad(set_to_none=True)
    loss.backward()
    
    # Check gradients exist and are finite
    finite_grads = True
    any_grad = False
    for p in model.parameters():
        if p.grad is not None:
            any_grad = True
            if not torch.isfinite(p.grad).all():
                finite_grads = False
                break
    
    assert any_grad, "No gradients computed"
    assert finite_grads, "Non-finite gradients detected"
    
    # Optimizer step
    opt.step()
    
    print(f"✓ GPU smoke test passed on {device}")


if __name__ == "__main__":
    test_gpu_smoke_forward_backward_step()
