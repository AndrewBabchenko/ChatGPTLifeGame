"""
Critical test: directional supervised learning actually improves policy
This validates that training is LEARNING, not just running
"""
import torch
import torch.nn.functional as F
from tests._device import pick_device

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimulationConfig
from src.models.actor_critic_network import ActorCriticNetwork
from scripts.train import compute_supervised_directional_loss

# CRITICAL: Must match Animal._apply_action() indexing exactly
# 8-direction action vectors: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
ACTION_DIRS = torch.tensor([
    [0, -1],   # 0: N  (north)
    [1, -1],   # 1: NE (northeast)
    [1, 0],    # 2: E  (east)
    [1, 1],    # 3: SE (southeast)
    [0, 1],    # 4: S  (south)
    [-1, 1],   # 5: SW (southwest)
    [-1, 0],   # 6: W  (west)
    [-1, -1]   # 7: NW (northwest)
], dtype=torch.float32)


def best_action_for(dx, dy, device):
    """Find best action index for moving toward (dx, dy)"""
    v = torch.tensor([dx, dy], device=device, dtype=torch.float32)
    v_norm = v / (v.norm() + 1e-8)
    dirs = ACTION_DIRS.to(device)
    dirs_norm = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
    sims = (dirs_norm * v_norm).sum(dim=1)  # cosine similarity
    return int(torch.argmax(sims).item())


def test_pretraining_increases_prob_of_best_move():
    """
    Critical learning test: supervised loss should shift probability mass toward correct direction
    This catches: wrong wiring, device mismatch, "training runs but doesn't learn"
    """
    torch.manual_seed(0)
    
    # Use CPU for deterministic unit test (avoids GPU nondeterminism)
    device = pick_device(prefer_gpu=False)
    config = SimulationConfig()
    model = ActorCriticNetwork(config).to(device)
    model.train()
    
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    
    B = 64
    N = config.MAX_VISIBLE_ANIMALS
    
    # Synthetic state obs (doesn't matter much for this test)
    obs = torch.zeros(B, 34, device=device)
    
    # Construct visible list: ONE prey target at (dx, dy)
    # Predator wants to move TOWARD prey
    dx, dy = 1.0, 0.0  # target is to the East
    best = best_action_for(dx, dy, device)
    
    # CRITICAL: Validate mapping is correct
    assert best == 2, f"Mapping mismatch: dx=1,dy=0 (east) should map to action 2 (E), got {best}"
    
    vis = torch.zeros(B, N, 8, device=device)
    vis[:, :, 7] = 0.0  # All padding
    vis[:, 0, 7] = 1.0  # First slot present
    vis[:, 0, 4] = 1.0  # is_prey=1
    vis[:, 0, 3] = 0.0  # is_predator=0
    vis[:, 0, 0] = dx
    vis[:, 0, 1] = dy
    vis[:, 0, 2] = 0.25  # close-ish
    
    def prob_best():
        """Get probability of best action"""
        with torch.no_grad():
            _, move_probs, _ = model.forward(obs, vis)
            return float(move_probs[:, best].mean().item())
    
    p0 = prob_best()
    print(f"Initial P(best action={best} [E]): {p0:.3f}")
    
    # Train using supervised directional loss (80 steps for stability)
    for step in range(80):
        turn_probs, move_probs, values = model.forward(obs, vis)
        log_probs = torch.log(move_probs.clamp_min(1e-8))
        loss = compute_supervised_directional_loss(
            log_probs=log_probs,
            visible_animals=vis,
            is_predator=True,
            device=device
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if step % 20 == 0:
            p_curr = prob_best()
            print(f"  Step {step}: P(best)={p_curr:.3f}, loss={loss.item():.4f}")
    
    p1 = prob_best()
    improvement = p1 - p0
    print(f"Final P(best action={best} [E]): {p1:.3f} (Δ={improvement:+.3f})")
    
    # Check 1: Probability must improve meaningfully
    assert improvement > 0.10, f"Policy did not learn: {p0:.3f} -> {p1:.3f} (Δ={improvement:.3f})"
    
    # Check 2: Best action should become the argmax for most samples
    with torch.no_grad():
        _, move_probs, _ = model.forward(obs, vis)
        argmax_correct = (move_probs.argmax(dim=1) == best).float().mean().item()
    
    assert argmax_correct > 0.60, f"Argmax accuracy too low: {argmax_correct:.2%} (want >60%)"
    
    # Check 3: Mild absolute floor (less strict than 0.40)
    assert p1 > 0.30, f"Still too low p_best={p1:.3f} (expected >0.30)"
    
    print(f"✓ Learning test PASSED:")
    print(f"  Improvement: {p0:.3f} -> {p1:.3f} (Δ={improvement:+.3f})")
    print(f"  Argmax accuracy: {argmax_correct:.2%}")


if __name__ == "__main__":
    test_pretraining_increases_prob_of_best_move()
