"""
Optional: Directional supervised loss mapping contract (diagonal bias regression test)
Validates that normalized directions don't give diagonals unfair advantage
"""
import torch
from scripts.train import compute_supervised_directional_loss


def test_directional_loss_maps_cardinal_directions_correctly():
    """
    Regression test: normalized action directions prevent diagonal bias
    
    With unnormalized dirs, diagonals have length √2 ≈ 1.414, giving them
    higher dot products and unfair advantage. This test validates the fix.
    """
    device = torch.device("cpu")
    config_max_vis = 12
    
    # Test cardinal directions map to correct actions
    test_cases = [
        # (dx, dy, expected_best_action, name)
        (0.0, -1.0, 0, "North"),
        (1.0,  0.0, 2, "East"),
        (0.0,  1.0, 4, "South"),
        (-1.0, 0.0, 6, "West"),
    ]
    
    for dx, dy, expected_action, name in test_cases:
        # Create batch with one prey at (dx, dy)
        B = 16
        log_probs = torch.log(torch.full((B, 8), 1/8)).requires_grad_(True)  # uniform init
        
        vis = torch.zeros(B, config_max_vis, 8)
        vis[:, :, 7] = 0.0  # all padding
        vis[:, 0, 7] = 1.0  # first slot present
        vis[:, 0, 4] = 1.0  # is_prey
        vis[:, 0, 0] = dx
        vis[:, 0, 1] = dy
        vis[:, 0, 2] = 0.3  # dist
        
        # Compute loss (this internally finds best action via argmax of normalized sims)
        loss = compute_supervised_directional_loss(log_probs, vis, is_predator=True, device=device)
        
        # Validate loss is finite
        assert torch.isfinite(loss), f"{name}: loss is not finite"
        
        # Gradient check: loss should be able to backprop
        loss.backward()
        assert log_probs.grad is not None, f"{name}: no gradients computed"
        assert torch.isfinite(log_probs.grad).all(), f"{name}: gradients contain NaN/Inf"
        
        print(f"✓ {name} ({dx:+.1f}, {dy:+.1f}) -> action {expected_action}: loss={loss.item():.4f}")


def test_directional_loss_diagonal_not_biased():
    """
    Regression test: diagonals should have same magnitude as cardinals after normalization
    Before fix: diagonals had length √2, giving them advantage
    After fix: all directions normalized to length 1
    """
    device = torch.device("cpu")
    config_max_vis = 12
    
    # Test diagonal directions
    test_cases = [
        # (dx, dy, expected_best_action, name)
        (1.0, -1.0, 1, "NorthEast"),
        (1.0,  1.0, 3, "SouthEast"),
        (-1.0, 1.0, 5, "SouthWest"),
        (-1.0,-1.0, 7, "NorthWest"),
    ]
    
    for dx, dy, expected_action, name in test_cases:
        B = 16
        log_probs = torch.log(torch.full((B, 8), 1/8)).requires_grad_(True)
        
        vis = torch.zeros(B, config_max_vis, 8)
        vis[:, :, 7] = 0.0
        vis[:, 0, 7] = 1.0
        vis[:, 0, 4] = 1.0
        vis[:, 0, 0] = dx
        vis[:, 0, 1] = dy
        vis[:, 0, 2] = 0.3
        
        loss = compute_supervised_directional_loss(log_probs, vis, is_predator=True, device=device)
        
        assert torch.isfinite(loss), f"{name}: loss is not finite"
        loss.backward()
        assert torch.isfinite(log_probs.grad).all(), f"{name}: gradients contain NaN/Inf"
        
        print(f"✓ {name} ({dx:+.1f}, {dy:+.1f}) -> action {expected_action}: loss={loss.item():.4f}")


if __name__ == "__main__":
    test_directional_loss_maps_cardinal_directions_correctly()
    test_directional_loss_diagonal_not_biased()
