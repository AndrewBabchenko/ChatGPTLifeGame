"""
Edge case tests: directional losses must be finite even with no targets
"""
import torch
from tests._device import pick_device

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import compute_directional_loss, compute_supervised_directional_loss


def test_directional_losses_are_finite_when_no_targets():
    """
    Edge case: when all visible slots are padding (is_present=0),
    losses should be finite (typically 0), not NaN/Inf
    """
    device = pick_device(prefer_gpu=True)
    
    B = 16
    N = 12
    actions = torch.zeros(B, dtype=torch.long, device=device)
    
    # All padding: is_present=0
    vis = torch.zeros(B, N, 8, device=device)
    vis[:, :, 7] = 0.0  # All padding
    
    # directional loss should be finite (and typically 0)
    dl = compute_directional_loss(actions, vis, is_predator=True, device=device)
    assert torch.isfinite(dl).all(), f"directional_loss not finite: {dl}"
    print(f"✓ directional_loss with no targets: {dl.item():.6f} (finite)")
    
    # supervised loss also should be finite
    log_probs = torch.log(torch.full((B, 8), 1/8, device=device).clamp_min(1e-8))
    sl = compute_supervised_directional_loss(log_probs, vis, is_predator=True, device=device)
    assert torch.isfinite(sl).all(), f"supervised_loss not finite: {sl}"
    print(f"✓ supervised_loss with no targets: {sl.item():.6f} (finite)")


def test_directional_losses_finite_with_mixed_targets():
    """
    Test with some batches having targets, some not
    """
    device = pick_device(prefer_gpu=True)
    
    B = 16
    N = 12
    actions = torch.zeros(B, dtype=torch.long, device=device)
    
    vis = torch.zeros(B, N, 8, device=device)
    # Half the batch has targets
    vis[:B//2, 0, 7] = 1.0  # present
    vis[:B//2, 0, 4] = 1.0  # is_prey
    vis[:B//2, 0, 0] = 0.5  # dx
    vis[:B//2, 0, 1] = 0.0  # dy
    vis[:B//2, 0, 2] = 0.3  # dist
    
    dl = compute_directional_loss(actions, vis, is_predator=True, device=device)
    assert torch.isfinite(dl).all(), f"directional_loss not finite: {dl}"
    print(f"✓ directional_loss with mixed targets: {dl.item():.6f} (finite)")
    
    log_probs = torch.log(torch.full((B, 8), 1/8, device=device).clamp_min(1e-8))
    sl = compute_supervised_directional_loss(log_probs, vis, is_predator=True, device=device)
    assert torch.isfinite(sl).all(), f"supervised_loss not finite: {sl}"
    print(f"✓ supervised_loss with mixed targets: {sl.item():.6f} (finite)")


if __name__ == "__main__":
    test_directional_losses_are_finite_when_no_targets()
    test_directional_losses_finite_with_mixed_targets()
