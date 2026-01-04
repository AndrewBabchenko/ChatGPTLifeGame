"""
Test: PPO pretraining mode actually updates the model (integration test)
Catches: optimizer never steps, gradients don't flow, wrong tensors/head, device mismatch
"""
import copy
import torch

from src.config import SimulationConfig
from src.models.actor_critic_network import ActorCriticNetwork

# Import ppo_update from your training script
from scripts.train import ppo_update


class DummyHierMemory:
    """
    Minimal object to satisfy ppo_update(pretraining_mode=True) for hierarchical batches.
    """
    def __init__(self, obs_move, vis_move):
        # These lengths matter for pretraining_mode bookkeeping
        self.rewards = [0.0] * obs_move.shape[0]
        self.dones = [False] * obs_move.shape[0]
        self.next_values = [torch.zeros(1, 1) for _ in range(obs_move.shape[0])]

        self._obs_move = [obs_move[i:i+1].cpu() for i in range(obs_move.shape[0])]
        self._vis_move = [vis_move[i:i+1].cpu() for i in range(obs_move.shape[0])]

        # present for safety in pretraining_mode path
        self.returns = None
        self.advantages = None

    def compute_returns_and_advantages(self, gamma):
        raise AssertionError("compute_returns_and_advantages should NOT be called in pretraining_mode=True")

    def get_batches(self):
        # match what ppo_update expects when it sees 'obs_turn' in batch
        batch = {
            "obs_turn": [],  # only used to detect hierarchical mode
            "obs_move": self._obs_move,
            "vis_move": self._vis_move,
            "move_actions": torch.zeros(len(self._obs_move), dtype=torch.long),
            "returns": torch.zeros(len(self._obs_move)),
            "advantages": torch.zeros(len(self._obs_move)),
        }
        yield batch


def test_pretraining_mode_updates_model_weights_cpu():
    """
    Validates that pretraining mode actually:
    - Computes gradients
    - Takes optimizer steps
    - Changes model weights
    
    This catches silent failures where training "runs" but doesn't learn
    """
    torch.manual_seed(0)
    device = torch.device("cpu")

    config = SimulationConfig()
    config.PPO_EPOCHS = 4
    config.PPO_BATCH_SIZE = 64

    model = ActorCriticNetwork(config).to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    B = 64
    N = config.MAX_VISIBLE_ANIMALS

    # obs: set is_predator=1 at index 4 so model recognizes this is a predator batch
    obs_move = torch.zeros(B, config.SELF_FEATURE_DIM, dtype=torch.float32)
    obs_move[:, 4] = 1.0  # is_predator flag (predators look for prey)

    # visible: one prey at dx=+1, dy=0 (east)
    vis_move = torch.zeros(B, N, 9, dtype=torch.float32)
    vis_move[:, :, 8] = 0.0
    vis_move[:, 0, 8] = 1.0
    vis_move[:, 0, 4] = 1.0  # is_prey
    vis_move[:, 0, 0] = 1.0
    vis_move[:, 0, 1] = 0.0
    vis_move[:, 0, 2] = 0.25

    mem = DummyHierMemory(obs_move, vis_move)

    # Snapshot weights
    before = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})

    # One pretraining PPO update pass
    sup_loss, _, entropy = ppo_update(
        model, opt, mem, config, device,
        use_amp=False,
        accumulation_steps=1,
        pretraining_mode=True,
        species="predator"  # Explicit species flag (Fix 3)
    )

    after = model.state_dict()

    # DEBUG: Check gradient computation
    has_grad = any(p.grad is not None for p in model.parameters())
    if not has_grad:
        print("WARNING: No gradients computed!")
    else:
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        print(f"Gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}")

    # Verify at least one tensor changed
    changed = any(not torch.equal(before[k], after[k].cpu()) for k in before.keys())
    if not changed:
        print("DEBUG: Model weights did NOT change")
        print(f"  sup_loss={sup_loss:.6f}")
        print(f"  entropy={entropy:.6f}")
        print(f"  has_grad={has_grad}")
    assert changed, "Model weights did not change after pretraining update"

    # Sanity: supervised loss is finite
    assert float(sup_loss) == float(sup_loss), "supervised loss is NaN"
    
    print(f"✓ Pretraining update changed model weights (supervised_loss={sup_loss:.4f}, entropy={entropy:.3f})")


if __name__ == "__main__":
    test_pretraining_mode_updates_model_weights_cpu()
