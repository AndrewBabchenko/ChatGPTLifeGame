"""
Advanced Training Script with PPO Algorithm
Integrates Actor-Critic network, pheromones, energy, and age systems
"""

import sys
import os
from pathlib import Path

# Add project root to path (required when running from subprocess)
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
from datetime import datetime

# ROCm compatibility: Force SDPA to use math backend (not flash attention)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    SDPA_AVAILABLE = True
except ImportError:
    # Fallback for older PyTorch versions
    SDPA_AVAILABLE = False
    print("[WARNING] torch.nn.attention not available, using legacy SDPA settings")

# Force unbuffered output for logging
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

from src.config import SimulationConfig
from src.core.animal import Prey, Predator
from src.models.actor_critic_network import ActorCriticNetwork
from src.core.pheromone_system import PheromoneMap
from src.models.replay_buffer import PPOMemory


# Global cached action directions tensor (avoid rebuilding in every function)
# Must match Animal._apply_action() exactly: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
_ACTION_DIRS_CPU = torch.tensor([
    [0.0, -1.0],   # 0: N  (north)
    [1.0, -1.0],   # 1: NE (northeast)
    [1.0,  0.0],   # 2: E  (east)
    [1.0,  1.0],   # 3: SE (southeast)
    [0.0,  1.0],   # 4: S  (south)
    [-1.0, 1.0],   # 5: SW (southwest)
    [-1.0, 0.0],   # 6: W  (west)
    [-1.0,-1.0]    # 7: NW (northwest)
], dtype=torch.float32)
# Normalize to avoid diagonal bias (diagonals have length √2)
_ACTION_DIRS_CPU = _ACTION_DIRS_CPU / (_ACTION_DIRS_CPU.norm(dim=1, keepdim=True) + 1e-8)

# Device-specific cache (will be populated on first use)
_ACTION_DIRS_CACHE = {}

def _ts() -> str:
    """Timestamp with milliseconds for logs."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def compute_supervised_directional_loss(log_probs: torch.Tensor, visible_animals: torch.Tensor, 
                                       is_predator: bool, device, obs: torch.Tensor = None, config=None) -> torch.Tensor:
    """
    Supervised loss: choose the action whose direction best matches the nearest target.
    Predators -> toward prey
    Prey -> STEP 3: Context-aware behavior:
        - If predator CLOSE: flee (away from predator)
        - Else if high energy & mate visible: approach mate
        - Else: no supervision
    Uses hard labels (argmax) for strongest learning signal.
    
    Args:
        log_probs: (B, 8) log probabilities for each action from policy
        visible_animals: (B, N, 8) visible animals features
        is_predator: species type
        device: torch device
        obs: (B, OBS_DIM) observations for energy/species check (STEP 3)
        config: config module for thresholds
    
    Returns:
        supervised_loss: NLL loss with best-matching action as target
    """
    B = log_probs.size(0)
    
    # Use cached action directions (cache per device)
    device_key = f"{device.type}:{getattr(device, 'index', None)}"
    if device_key not in _ACTION_DIRS_CACHE:
        _ACTION_DIRS_CACHE[device_key] = _ACTION_DIRS_CPU.to(device)
    action_dirs = _ACTION_DIRS_CACHE[device_key]
    
    # Find nearest target by distance (idx 2), ignoring padding and wrong type
    dist = visible_animals[:, :, 2]  # (B, N)
    pad = visible_animals[:, :, 7] < 0.5  # is_present==0
    
    if is_predator:
        wrong_type = visible_animals[:, :, 4] < 0.5  # not prey
        # Predators always approach prey
        mask = pad | wrong_type
        dist_filtered = dist.masked_fill(mask, float("inf"))
        min_dist, nearest_idx = dist_filtered.min(dim=1)
        has_target = torch.isfinite(min_dist)
        
        if has_target.sum() == 0:
            return torch.zeros((), device=device, requires_grad=True)
        
        b_idx = torch.arange(B, device=device)
        target_dx = visible_animals[b_idx, nearest_idx, 0]
        target_dy = visible_animals[b_idx, nearest_idx, 1]
        target_dir = torch.stack([target_dx, target_dy], dim=-1)  # Toward prey
        
    else:
        # --- Prey context-aware supervision ---
        dist = visible_animals[:, :, 2]  # (B, N)
        pad = visible_animals[:, :, 7] < 0.5

        # Nearest predator (visible)
        predator_mask = pad | (visible_animals[:, :, 3] < 0.5)
        dist_pred = dist.masked_fill(predator_mask, float("inf"))
        min_pred_dist, nearest_pred_idx = dist_pred.min(dim=1)
        has_predator = torch.isfinite(min_pred_dist)

        # Nearest mate = prey + same_species + same_type
        mate_mask = (
            pad
            | (visible_animals[:, :, 4] < 0.5)   # not prey
            | (visible_animals[:, :, 5] < 0.5)   # not same species
            | (visible_animals[:, :, 6] < 0.5)   # not same type
        )
        dist_mate = dist.masked_fill(mate_mask, float("inf"))
        min_mate_dist, nearest_mate_idx = dist_mate.min(dim=1)
        has_mate = torch.isfinite(min_mate_dist)

        # Use correct indices from OBS contract
        # obs[16] = energy, obs[6] = mating cooldown normalized, obs[15] = age normalized
        if obs is not None:
            energy_norm = obs[:, 16]
            cooldown_norm = obs[:, 6]
            age_norm = obs[:, 15]
            nearest_pred_obs = obs[:, 7]  # normalized nearest predator dist in obs

            if config is not None:
                energy_thr = float(config.PREY_MATING_ENERGY_THRESHOLD / config.MAX_ENERGY)
                maturity_thr = float(config.MATURITY_AGE / config.MAX_AGE)
                # Flee supervision only if predator is actually "close"
                flee_thr = float(getattr(config, "PREY_FLEE_SUPERVISION_DIST", 0.35))
            else:
                energy_thr = 0.6
                maturity_thr = 0.2
                flee_thr = 0.35

            high_energy = energy_norm > energy_thr
            mature = age_norm > maturity_thr
            ready = (cooldown_norm < 1e-3) & high_energy & mature

            predator_close = has_predator & (nearest_pred_obs < flee_thr)
        else:
            # fallback if obs not passed
            ready = torch.ones(B, dtype=torch.bool, device=device)
            predator_close = has_predator

        # Priority:
        # 1) Flee only if predator is CLOSE (otherwise mating never happens)
        # 2) If safe + ready + mate exists, approach mate
        flee_mode = predator_close
        mate_mode = (~predator_close) & has_mate & ready
        has_target = flee_mode | mate_mode

        if has_target.sum() == 0:
            return torch.zeros((), device=device, requires_grad=True)

        b_idx = torch.arange(B, device=device)
        target_dir = torch.zeros(B, 2, device=device)

        # Flee: away from predator
        flee_idx = flee_mode.nonzero(as_tuple=False).squeeze(-1)
        if flee_idx.numel() > 0:
            dx = visible_animals[flee_idx, nearest_pred_idx[flee_idx], 0]
            dy = visible_animals[flee_idx, nearest_pred_idx[flee_idx], 1]
            target_dir[flee_idx] = -torch.stack([dx, dy], dim=-1)

        # Mate: toward mate
        mate_idx = mate_mode.nonzero(as_tuple=False).squeeze(-1)
        if mate_idx.numel() > 0:
            dx = visible_animals[mate_idx, nearest_mate_idx[mate_idx], 0]
            dy = visible_animals[mate_idx, nearest_mate_idx[mate_idx], 1]
            target_dir[mate_idx] = torch.stack([dx, dy], dim=-1)
    
    # Normalize target direction
    target_dir = target_dir / (target_dir.norm(dim=-1, keepdim=True) + 1e-8)  # (B, 2)
    
    # Compute cosine similarities
    sims = target_dir @ action_dirs.T  # (B, 8)
    
    # Hard label = best matching action (strongest learning signal)
    labels = torch.argmax(sims, dim=1)  # (B,)
    
    # CRITICAL: Only compute loss on rows with valid targets (avoids NaN propagation)
    valid_idx = has_target.nonzero(as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        # No valid targets - return zero loss
        return torch.zeros((), device=device, requires_grad=True)
    
    # Compute NLL loss only on filtered rows
    loss = F.nll_loss(log_probs[valid_idx], labels[valid_idx], reduction='mean')
    
    return loss


def compute_directional_loss(actions: torch.Tensor, visible_animals: torch.Tensor, 
                             is_predator: bool, device) -> tuple[torch.Tensor, float, float, int]:
    """
    Compute auxiliary loss that penalizes actions misaligned with target direction.
    This provides direct supervision for directional learning.
    
    Args:
        actions: (B,) action indices (0-7 for 8 directions)
        visible_animals: (B, N, 8) visible animals features
        is_predator: True for predators (move toward), False for prey (move away)
        device: torch device
    
    Returns:
        directional_loss: scalar loss penalizing wrong direction choices
    """
    batch_size = actions.size(0)
    
    # Use cached action directions (cache per device)
    device_key = f"{device.type}:{getattr(device, 'index', None)}"
    if device_key not in _ACTION_DIRS_CACHE:
        _ACTION_DIRS_CACHE[device_key] = _ACTION_DIRS_CPU.to(device)
    action_dirs = _ACTION_DIRS_CACHE[device_key]
    
    # Find nearest target (distance at index 2)
    distances = visible_animals[:, :, 2]  # (B, N)
    
    # Mask padding using is_present flag (index 7)
    padding_mask = (visible_animals[:, :, 7] < 0.5)  # (B, N)
    
    # Filter by target type: predators target prey (is_prey=1 at index 4), prey target predators (is_predator=1 at index 3)
    if is_predator:
        # Predator: only target prey
        wrong_type_mask = (visible_animals[:, :, 4] < 0.5)  # Not prey
    else:
        # Prey: only target predators
        wrong_type_mask = (visible_animals[:, :, 3] < 0.5)  # Not predator
    
    # Combine masks
    combined_mask = padding_mask | wrong_type_mask
    distances = distances.masked_fill(combined_mask, float('inf'))
    
    # Get nearest target index and check validity FIRST (avoid argmin on all-inf)
    min_dist, nearest_idx = distances.min(dim=1)  # (B,)
    has_target = torch.isfinite(min_dist)  # (B,)
    
    # Track supervision metrics (helps tune weight)
    has_target_pct = has_target.float().mean().item()
    n_has_target = int(has_target.sum().item())
    # Use MEAN distance (not min) - more representative than single closest target
    mean_target_dist = min_dist[has_target].mean().item() if has_target.any() else 0.0
    
    if has_target.sum() == 0:
        # No targets in entire batch - return zero loss with gradient
        return torch.zeros((), device=device, requires_grad=True), has_target_pct, mean_target_dist, n_has_target
    
    # Extract direction to nearest target (dx at 0, dy at 1)
    batch_indices = torch.arange(batch_size, device=device)
    target_dx = visible_animals[batch_indices, nearest_idx, 0]  # (B,)
    target_dy = visible_animals[batch_indices, nearest_idx, 1]  # (B,)
    target_dir = torch.stack([target_dx, target_dy], dim=-1)  # (B, 2)
    
    # Get chosen action directions
    chosen_dir = action_dirs[actions]  # (B, 2)
    
    # For prey, flip target direction (move AWAY from predator)
    if not is_predator:
        target_dir = -target_dir
    
    # Normalize directions for fair comparison (avoid diagonal bias)
    target_norm = torch.norm(target_dir, dim=-1, keepdim=True)
    chosen_norm = torch.norm(chosen_dir, dim=-1, keepdim=True)
    similarity = torch.sum(target_dir * chosen_dir, dim=-1) / ((target_norm * chosen_norm).squeeze() + 1e-8)
    
    # Mask invalid rows BEFORE applying loss (similarity undefined for no-target rows)
    similarity = similarity.masked_fill(~has_target, 1.0)  # 1.0 = perfect alignment -> penalty 0
    
    # Loss: penalize negative similarity (moving away when should move toward)
    direction_penalty = F.relu(-similarity)  # Max penalty when similarity = -1 (opposite)
    direction_loss = (direction_penalty * has_target.float()).sum() / has_target.float().sum()
    
    return direction_loss, has_target_pct, mean_target_dist, n_has_target

def create_population(config: SimulationConfig) -> list:
    """Create initial population of animals"""
    animals = []
    
    # Create prey (species A)
    for _ in range(config.INITIAL_PREY_COUNT):
        x = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        y = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        animal = Prey(x, y, "A", "#00ff00")
        animal.energy = config.INITIAL_ENERGY
        animals.append(animal)
    
    # Create predators (species B)
    for _ in range(config.INITIAL_PREDATOR_COUNT):
        x = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        y = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        animal = Predator(x, y, "B", "#ff0000")
        animal.energy = config.INITIAL_ENERGY
        animals.append(animal)
    
    return animals


def compute_directional_correctness(actions: torch.Tensor, visible_animals: torch.Tensor, 
                                   is_predator: bool, device) -> torch.Tensor:
    """
    Compute directional correctness score for experience prioritization.
    Returns score from 0 (wrong direction) to 1 (perfect direction).
    
    Args:
        actions: (B,) action indices
        visible_animals: (B, N, 8) visible animals
        is_predator: species type
        device: torch device
    
    Returns:
        scores: (B,) correctness scores for prioritization
    """
    if actions.numel() == 0 or visible_animals.size(1) == 0:
        return torch.ones(actions.size(0), device=device)
    
    # Use cached action directions (cache per device)
    device_key = f"{device.type}:{getattr(device, 'index', None)}"
    if device_key not in _ACTION_DIRS_CACHE:
        _ACTION_DIRS_CACHE[device_key] = _ACTION_DIRS_CPU.to(device)
    action_dirs = _ACTION_DIRS_CACHE[device_key]
    
    # Find nearest target (distance at index 2)
    distances = visible_animals[:, :, 2]
    padding_mask = (visible_animals[:, :, 7] < 0.5)  # Use is_present flag
    
    # Filter by target type
    if is_predator:
        wrong_type_mask = (visible_animals[:, :, 4] < 0.5)  # Not prey
    else:
        wrong_type_mask = (visible_animals[:, :, 3] < 0.5)  # Not predator
    
    combined_mask = padding_mask | wrong_type_mask
    distances = distances.masked_fill(combined_mask, float('inf'))
    
    has_target = (distances < float('inf')).any(dim=1)
    if not has_target.any():
        return torch.ones(actions.size(0), device=device)
    
    # Find nearest target (already safe since we checked has_target)
    nearest_idx = torch.argmin(distances, dim=1)
    batch_indices = torch.arange(actions.size(0), device=device)
    
    # Target direction (dx at 0, dy at 1)
    target_dx = visible_animals[batch_indices, nearest_idx, 0]
    target_dy = visible_animals[batch_indices, nearest_idx, 1]
    target_dir = torch.stack([target_dx, target_dy], dim=-1)
    
    # Prey: move away (flip direction)
    if not is_predator:
        target_dir = -target_dir
    
    # Compute cosine similarity
    chosen_dir = action_dirs[actions]
    target_norm = torch.norm(target_dir, dim=-1, keepdim=True)
    chosen_norm = torch.norm(chosen_dir, dim=-1, keepdim=True)
    norms = target_norm * chosen_norm
    
    similarity = torch.sum(target_dir * chosen_dir, dim=-1) / (norms.squeeze() + 1e-8)
    
    # CRITICAL: Mask similarity to avoid NaN propagation
    # Set invalid rows to 1.0 (perfect alignment -> score 1.0)
    similarity = similarity.masked_fill(~has_target, 1.0)
    
    # Convert to 0-1 score (similarity ranges -1 to 1)
    # -1 (opposite) -> 0, 0 (perpendicular) -> 0.5, 1 (aligned) -> 1
    scores = (similarity + 1.0) / 2.0
    scores = scores * has_target.float() + (~has_target).float()
    
    return scores


def ppo_update(model, optimizer, memory, config, device, use_amp=False, accumulation_steps=4, pretraining_mode=False, species="prey"):
    """
    Perform PPO update on the model with gradient accumulation and mixed precision
    
    Args:
        model: Actor-Critic network
        optimizer: PyTorch optimizer
        memory: PPOMemory with stored experiences
        config: SimulationConfig
        device: torch.device for GPU/CPU
        use_amp: Use automatic mixed precision (faster on GPU)
        accumulation_steps: Number of mini-batches to accumulate before optimizer step
        pretraining_mode: If True, train ONLY directional loss (no policy/value optimization)
        species: "prey" or "predator" - explicitly specify for directional loss
    """
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    # Compute returns and advantages only when PPO is active
    if not pretraining_mode:
        returns, advantages = memory.compute_returns_and_advantages(gamma=config.GAMMA)
        # Keep as tensors for efficient indexing (don't convert to Python lists)
        memory.returns = returns.detach().flatten()
        memory.advantages = advantages.detach().flatten()
    
    # S3: Removed advantage weighting by correctness - PPO learns better from failures
    # (previously weighted correct experiences higher, but this reduces learning from mistakes)
    
    # CRITICAL: Disable dropout during PPO update (common RL practice)
    # Avoids inconsistent policy from multiple forward passes
    was_training = model.training
    model.eval()
    
    # PPO epochs (use fewer for pretraining to avoid overfitting)
    epochs = 4 if pretraining_mode else config.PPO_EPOCHS
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_supervised = 0.0
    n_updates = 0
    
    # PPO diagnostics (accumulate across minibatches)
    total_approx_kl = 0.0
    total_clipped_samples = 0  # For clip fraction calculation
    # Supervision metrics (proper weighting)
    total_targets = 0      # Total samples with targets
    total_samples = 0      # Total samples processed
    total_dist_sum = 0.0   # Sum of (mean_dist * n_has_target) for weighted average
    
    update_start_time = time.time()
    last_heartbeat = update_start_time
    heartbeat_interval_s = 30
    
    # Explicit species flag for directional loss (avoid brittle obs[0,4] inference)
    is_predator_species = (species == "predator")
    
    # KL divergence early stopping thresholds (prevents policy collapse)
    KL_TARGET_MINIBATCH = 0.05  # Stop immediately if minibatch KL exceeds this
    KL_TARGET_EPOCH = 0.03      # Stop epoch if average KL exceeds this
    epoch_kl_values = []
    
    for epoch in range(epochs):
        # Per-epoch KL tracking (fixes bug #3: epoch KL using cumulative instead of per-epoch)
        epoch_kl_sum = 0.0
        epoch_updates = 0
        break_out = False  # Flag for breaking out of nested loops
        # Get all batches for this epoch
        # Ensure returns/advantages exist for get_batches() even in pretraining mode
        if pretraining_mode:
            if not hasattr(memory, 'returns') or memory.returns is None or len(memory.returns) != len(memory.rewards):
                memory.returns = torch.zeros(len(memory.rewards), dtype=torch.float32)  # CPU (same as values)
            if not hasattr(memory, 'advantages') or memory.advantages is None or len(memory.advantages) != len(memory.rewards):
                memory.advantages = torch.zeros(len(memory.rewards), dtype=torch.float32)  # CPU
        
        # Stream batches instead of materializing (saves memory)
        for batch_idx, batch in enumerate(memory.get_batches()):
            # Extract batch data (hierarchical or simple depending on memory mode)
            is_hierarchical = 'obs_turn' in batch
            
            if is_hierarchical and pretraining_mode:
                # Hierarchical pretraining: use move observations for directional learning
                obs_move = torch.cat(batch['obs_move'], dim=0).to(device)
                vis_move = torch.cat(batch['vis_move'], dim=0).to(device)
                move_actions = batch['move_actions'].to(device)
                # Build simple-mode-like variables for pretraining path
                animal_inputs = obs_move
                visible_inputs = vis_move
                # Keep full tensor for stable slicing in minibatch loop
                actions_all = move_actions
            elif is_hierarchical and not pretraining_mode:
                # Hierarchical mode: separate turn and move observations
                obs_turn = torch.cat(batch['obs_turn'], dim=0).to(device)
                vis_turn = torch.cat(batch['vis_turn'], dim=0).to(device)
                turn_actions = batch['turn_actions'].to(device)
                turn_logp_old = batch['turn_log_probs_old'].to(device)
                
                obs_move = torch.cat(batch['obs_move'], dim=0).to(device)
                vis_move = torch.cat(batch['vis_move'], dim=0).to(device)
                move_actions = batch['move_actions'].to(device)
                move_logp_old = batch['move_log_probs_old'].to(device)
                
                returns_batch = batch['returns'].view(-1).to(device)
                advantages_batch = batch['advantages'].view(-1).to(device)
            else:
                # Simple mode: single observation (shouldn't happen but keep for compatibility)
                states = batch['states']
                actions = batch['actions'].to(device)
                old_log_probs = batch['old_log_probs'].to(device)
                returns_batch = batch['returns'].view(-1).to(device)
                advantages_batch = batch['advantages'].view(-1).to(device)
            
            # Split batch into smaller mini-batches for gradient accumulation
            if is_hierarchical and pretraining_mode:
                batch_size = obs_move.size(0)
            elif is_hierarchical:
                batch_size = obs_turn.size(0)
            else:
                batch_size = len(states)
            
            # Decide acc_steps first, then compute minibatch_size from it
            acc_steps = min(accumulation_steps, batch_size)
            minibatch_size = max(1, batch_size // acc_steps)
            
            # Zero gradients at start of each outer batch
            optimizer.zero_grad(set_to_none=True)
            
            # Process mini-batches with gradient accumulation
            for mini_idx in range(acc_steps):
                start_idx = mini_idx * minibatch_size
                end_idx = start_idx + minibatch_size if mini_idx < acc_steps - 1 else batch_size
                
                if start_idx >= batch_size:
                    break
                
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval_s:
                    elapsed = now - update_start_time
                    print(
                        f"[{_ts()}] PPO update: "
                        f"epoch {epoch + 1}/{epochs}, "
                        f"batch {batch_idx + 1}, "
                        f"minibatch {mini_idx + 1}/{acc_steps}, "
                        f"elapsed {elapsed:.1f}s",
                        flush=True
                    )
                    last_heartbeat = now
                
                # Get mini-batch slices (hierarchical or simple)
                if is_hierarchical and pretraining_mode:
                    # Hierarchical pretraining: slice from stable full tensor
                    animal_inputs_mini = animal_inputs[start_idx:end_idx]
                    visible_inputs_mini = visible_inputs[start_idx:end_idx]
                elif is_hierarchical and not pretraining_mode:
                    mini_obs_turn = obs_turn[start_idx:end_idx]
                    mini_vis_turn = vis_turn[start_idx:end_idx]
                    mini_turn_actions = turn_actions[start_idx:end_idx]
                    mini_turn_logp_old = turn_logp_old[start_idx:end_idx]
                    
                    mini_obs_move = obs_move[start_idx:end_idx]
                    mini_vis_move = vis_move[start_idx:end_idx]
                    mini_move_actions = move_actions[start_idx:end_idx]
                    mini_move_logp_old = move_logp_old[start_idx:end_idx]
                    
                    mini_returns = returns_batch[start_idx:end_idx]
                    mini_advantages = advantages_batch[start_idx:end_idx]
                    
                    # Per-minibatch advantage normalization (critical for PPO stability)
                    # DML-friendly std (no aten::std.correction - avoids CPU fallback)
                    adv = mini_advantages
                    adv_mean = adv.mean()
                    adv_var = (adv - adv_mean).pow(2).mean()  # population variance
                    adv_std = torch.sqrt(adv_var + 1e-8)
                    
                    if adv_std > 1e-6 and adv.numel() > 2:
                        mini_advantages = (adv - adv_mean) / adv_std
                    else:
                        # Skip normalization for degenerate cases (just center)
                        mini_advantages = adv - adv_mean
                else:
                    mini_states = states[start_idx:end_idx]
                    mini_actions = actions[start_idx:end_idx]
                    mini_old_log_probs = old_log_probs[start_idx:end_idx]
                    mini_returns = returns_batch[start_idx:end_idx]
                    mini_advantages = advantages_batch[start_idx:end_idx]
                    
                    # Per-minibatch advantage normalization (critical for PPO stability)
                    # DML-friendly std (no aten::std.correction - avoids CPU fallback)
                    adv = mini_advantages
                    adv_mean = adv.mean()
                    adv_var = (adv - adv_mean).pow(2).mean()  # population variance
                    adv_std = torch.sqrt(adv_var + 1e-8)
                    
                    if adv_std > 1e-6 and adv.numel() > 2:
                        mini_advantages = (adv - adv_mean) / adv_std
                    else:
                        # Skip normalization for degenerate cases (just center)
                        mini_advantages = adv - adv_mean
                
                # Evaluate actions with current policy (hierarchical or simple)
                if is_hierarchical and pretraining_mode:
                    # Hierarchical pretraining: use move observations
                    mini_visible = visible_inputs_mini
                elif is_hierarchical and not pretraining_mode:
                    # Hierarchical mode: recompute log probs for both heads
                    # Ensure actions are 1D
                    mini_turn_actions = mini_turn_actions.view(-1)
                    mini_move_actions = mini_move_actions.view(-1)
                    mini_turn_logp_old = mini_turn_logp_old.view(-1)
                    mini_move_logp_old = mini_move_logp_old.view(-1)
                    
                    # Use mixed precision for forward pass
                    # CRITICAL: Use correct observations for each head
                    # Turn head evaluated on turn observation, move head on move observation
                    
                    # --- Turn head + VALUE on TURN observation (matches rollout value_old) ---
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            turn_probs, _, values = model.forward(mini_obs_turn, mini_vis_turn)
                    else:
                        turn_probs, _, values = model.forward(mini_obs_turn, mini_vis_turn)
                    
                    turn_log_probs = torch.log(turn_probs + 1e-8)  # (B, 3)
                    turn_logp_new = turn_log_probs.gather(1, mini_turn_actions.unsqueeze(1)).squeeze(1)
                    turn_entropy = -(turn_probs * turn_log_probs).sum(dim=-1)
                    
                    # --- Move head on MOVE observation (value ignored here) ---
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            _, move_probs, _ = model.forward(mini_obs_move, mini_vis_move)
                    else:
                        _, move_probs, _ = model.forward(mini_obs_move, mini_vis_move)
                    
                    move_log_probs = torch.log(move_probs + 1e-8)  # (B, 8)
                    move_logp_new = move_log_probs.gather(1, mini_move_actions.unsqueeze(1)).squeeze(1)
                    move_entropy = -(move_probs * move_log_probs).sum(dim=-1)
                    
                    # CRITICAL: One-time check that forward() returns probabilities (not logits)
                    if epoch == 0 and batch_idx == 0 and mini_idx == 0:
                        with torch.no_grad():
                            # Test on small sample
                            test_obs = mini_obs_move[:4] if mini_obs_move.size(0) >= 4 else mini_obs_move
                            test_vis = mini_vis_move[:4] if mini_vis_move.size(0) >= 4 else mini_vis_move
                            _, m_test, _ = model.forward(test_obs, test_vis)
                            prob_sums = m_test.sum(dim=-1)
                            if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-3):
                                print(
                                    f"[WARN] model.forward move head does not sum to 1 (mean sum={prob_sums.mean().item():.4f}). "
                                    f"forward() likely returns logits. Ensure log_prob_* and any log() use log_softmax.",
                                    flush=True
                                )
                    
                    # Combined log probs for PPO
                    log_probs = turn_logp_new + move_logp_new
                    old_log_probs_combined = mini_turn_logp_old + mini_move_logp_old
                    entropies = turn_entropy + move_entropy
                    
                    # Flatten
                    log_probs = log_probs.view(-1)
                    old_log_probs_combined = old_log_probs_combined.view(-1)
                    values = values.view(-1)
                    entropies = entropies.view(-1)
                    
                    # Use move visibility for directional loss
                    mini_visible = mini_vis_move
                    mini_actions_for_loss = mini_move_actions
                else:
                    # Simple mode: single observation
                    animal_inputs = torch.cat([s[0] for s in mini_states], dim=0).to(device)
                    visible_inputs = torch.cat([s[1] for s in mini_states], dim=0).to(device)
                    mini_actions = mini_actions.view(-1)
                    
                    # Extract visible animals for directional loss
                    mini_visible = torch.cat([s[1] for s in mini_states], dim=0).to(device)

                # In pretraining mode, get full action distribution for supervised loss
                if pretraining_mode:
                    # Get full action log probabilities for move head (8 actions)
                    # forward() returns (turn_probs, move_probs, values) for dual-head model
                    if is_hierarchical:
                        inputs = animal_inputs_mini
                        vis = visible_inputs_mini
                    else:
                        inputs = torch.cat([s[0] for s in mini_states], dim=0).to(device)
                        vis = torch.cat([s[1] for s in mini_states], dim=0).to(device)
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            turn_probs, move_probs, values = model.forward(inputs, vis)
                    else:
                        turn_probs, move_probs, values = model.forward(inputs, vis)
                    
                    # CRITICAL: Verify model returns probabilities (fail-fast if network changes)
                    if mini_idx == 0 and epoch == 0:  # Check once per update
                        prob_sums = move_probs.sum(dim=-1)
                        if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-3):
                            raise RuntimeError(
                                f"Model must return probabilities (sum=1), got sum={prob_sums.mean():.4f}. "
                                f"Network changed to return logits? Ensure forward() uses F.softmax()."
                            )
                    
                    # Use move head for directional learning (8 directions)
                    action_probs = move_probs
                    full_log_probs = torch.log(action_probs.clamp_min(1e-8))  # (B, 8)
                    
                    # Compute entropy for monitoring (no sampling needed in supervised mode)
                    dist = torch.distributions.Categorical(action_probs)
                    entropies = dist.entropy()
                    
                    values = values.view(-1)
                    entropies = entropies.view(-1)
                    
                    # Compute supervised loss (use explicit species flag) - STEP 3: pass obs for context-aware supervision
                    supervised_loss = compute_supervised_directional_loss(full_log_probs, mini_visible, is_predator_species, device, obs=inputs, config=config)
                    loss = supervised_loss / acc_steps
                    policy_loss = torch.zeros(1, device=device)
                    value_loss = torch.zeros(1, device=device)
                    entropy_loss = torch.zeros(1, device=device)
                    
                    # Track for logging
                    total_supervised += supervised_loss.item()
                    total_entropy += entropies.mean().item()
                elif is_hierarchical:
                    # Hierarchical PPO: use combined log probs
                    ratio = torch.exp(log_probs - old_log_probs_combined)
                    surr1 = ratio * mini_advantages
                    surr2 = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPSILON, 
                                       1.0 + config.PPO_CLIP_EPSILON) * mini_advantages
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss with proper shapes
                    value_loss = nn.MSELoss()(values, mini_returns)
                    
                    # Entropy (track positive metric separately from loss term)
                    entropy = entropies.mean()
                    entropy_loss = -entropy
                    
                    # DIFFERENTIABLE auxiliary directional loss (uses current policy distribution) - STEP 3: pass obs + config
                    directional_loss = compute_supervised_directional_loss(move_log_probs, mini_vis_move, is_predator_species, device, obs=mini_obs_move, config=config)
                    
                    # Track metrics (non-differentiable, for logging only) - single call
                    with torch.no_grad():
                        _, _, mean_dist, n_has_target = compute_directional_loss(mini_actions_for_loss, mini_visible, is_predator_species, device)
                        mb_size = mini_actions_for_loss.numel()
                        total_samples += mb_size
                        total_targets += n_has_target
                        if n_has_target > 0:
                            total_dist_sum += mean_dist * n_has_target
                    
                    # Compute diagnostics
                    with torch.no_grad():
                        # Approximate KL divergence (standard formula: stays >=0)
                        log_ratio = log_probs - old_log_probs_combined
                        approx_kl = (torch.exp(log_ratio) - 1.0 - log_ratio).mean().item()
                        # Clip fraction (sample-weighted)
                        clipped = (ratio - 1.0).abs() > config.PPO_CLIP_EPSILON
                        total_clipped_samples += clipped.sum().item()
                        total_samples += clipped.numel()
                        
                        # Sanity checks (once per epoch for performance)
                        if mini_idx == 0:
                            mean_ratio = ratio.mean().item()
                            extreme_ratio_pct = ((ratio > 10.0) | (ratio < 0.1)).float().mean().item()
                            # Always log extreme ratio percentage for dashboard tracking
                            if extreme_ratio_pct > 0.3:
                                print(f"  [WARN] Extreme ratios: {extreme_ratio_pct*100:.1f}% (mean={mean_ratio:.3f})", flush=True)
                            else:
                                print(f"  [INFO] Extreme ratios: {extreme_ratio_pct*100:.1f}%", flush=True)
                        
                        # Accumulate for logging
                        total_approx_kl += approx_kl  # Overall diagnostic

                        epoch_kl_sum += approx_kl  # Per-epoch tracking
                        epoch_updates += 1
                        
                        # CRITICAL: Per-minibatch KL early stopping (prevents policy collapse)
                        if approx_kl > KL_TARGET_MINIBATCH:
                            print(f"  [STOP] KL spike {approx_kl:.4f} > {KL_TARGET_MINIBATCH}, skipping remaining minibatches", flush=True)
                            break_out = True
                    
                    if break_out:
                        break  # Exit minibatch loop
                    
                    loss = (policy_loss + 
                           config.VALUE_LOSS_COEF * value_loss + 
                           config.ENTROPY_COEF * entropy_loss +
                           0.5 * directional_loss) / acc_steps  # Differentiable supervised loss (reduced from 5.0)
                else:
                    # Simple PPO training
                    # Compute everything from single forward pass (avoid dropout inconsistency)
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            _, move_probs, values = model.forward(animal_inputs, visible_inputs)
                    else:
                        _, move_probs, values = model.forward(animal_inputs, visible_inputs)
                    
                    # Compute log probs and entropy manually
                    move_log_probs = torch.log(move_probs + 1e-8)  # (B, 8)
                    log_probs = move_log_probs.gather(1, mini_actions.unsqueeze(1)).squeeze(1)
                    entropies = -(move_probs * move_log_probs).sum(dim=-1)
                    
                    # Ensure all tensors are 1D for loss calculation
                    log_probs = log_probs.view(-1)
                    values = values.view(-1)
                    entropies = entropies.view(-1)
                    mini_old_log_probs = mini_old_log_probs.view(-1)
                    
                    # Compute PPO loss
                    ratio = torch.exp(log_probs - mini_old_log_probs)
                    surr1 = ratio * mini_advantages
                    surr2 = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPSILON, 
                                       1.0 + config.PPO_CLIP_EPSILON) * mini_advantages
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss with proper shapes
                    value_loss = nn.MSELoss()(values, mini_returns)
                    
                    # Entropy (track positive metric separately from loss term)
                    entropy = entropies.mean()
                    entropy_loss = -entropy
                    
                    # DIFFERENTIABLE auxiliary directional loss (uses current policy distribution) - STEP 3: pass obs + config
                    directional_loss = compute_supervised_directional_loss(move_log_probs, mini_visible, is_predator_species, device, obs=animal_inputs, config=config)
                    
                    # Track metrics (non-differentiable, for logging only) - single call
                    with torch.no_grad():
                        _, _, mean_dist, n_has_target = compute_directional_loss(mini_actions, mini_visible, is_predator_species, device)
                        mb_size = mini_actions.numel()
                        total_samples += mb_size
                        total_targets += n_has_target
                        if n_has_target > 0:
                            total_dist_sum += mean_dist * n_has_target
                    
                    # Compute diagnostics
                    with torch.no_grad():
                        # Approximate KL divergence (standard formula: stays >=0)
                        log_ratio = log_probs - mini_old_log_probs
                        approx_kl = (torch.exp(log_ratio) - 1.0 - log_ratio).mean().item()
                        # Clip fraction (sample-weighted)
                        clipped = (ratio - 1.0).abs() > config.PPO_CLIP_EPSILON
                        total_clipped_samples += clipped.sum().item()
                        total_samples += clipped.numel()
                        
                        # Sanity checks (once per epoch for performance)
                        if mini_idx == 0:
                            mean_ratio = ratio.mean().item()
                            extreme_ratio_pct = ((ratio > 10.0) | (ratio < 0.1)).float().mean().item()
                            # Always log extreme ratio percentage for dashboard tracking
                            if extreme_ratio_pct > 0.3:
                                print(f"  [WARN] Extreme ratios: {extreme_ratio_pct*100:.1f}% (mean={mean_ratio:.3f})", flush=True)
                            else:
                                print(f"  [INFO] Extreme ratios: {extreme_ratio_pct*100:.1f}%", flush=True)
                        
                        # Accumulate for logging
                        total_approx_kl += approx_kl  # Overall diagnostic

                        epoch_kl_sum += approx_kl  # Per-epoch tracking
                        epoch_updates += 1
                        
                        # CRITICAL: Per-minibatch KL early stopping (prevents policy collapse)
                        if approx_kl > KL_TARGET_MINIBATCH:
                            print(f"  [STOP] KL spike {approx_kl:.4f} > {KL_TARGET_MINIBATCH}, skipping remaining minibatches", flush=True)
                            break_out = True
                    
                    if break_out:
                        break  # Exit minibatch loop
                    
                    loss = (policy_loss + 
                           config.VALUE_LOSS_COEF * value_loss + 
                           config.ENTROPY_COEF * entropy_loss +
                           0.5 * directional_loss) / acc_steps  # Differentiable supervised loss (reduced from 5.0)
                
                # Backward pass (accumulate gradients)
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Accumulate losses for logging
                if not pretraining_mode:
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                n_updates += 1
            
            # Optimizer step after accumulating gradients (skip if KL spike detected)
            if not break_out:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                    optimizer.step()
            else:
                # Clear gradients without stepping when KL spike detected
                optimizer.zero_grad(set_to_none=True)
        
        # Break out of batch loop if KL spike detected
        if break_out:
            break
        
        # KL divergence early stopping check (per epoch) - use per-epoch average
        if not pretraining_mode and epoch_updates > 0:
            epoch_avg_kl = epoch_kl_sum / epoch_updates
            epoch_kl_values.append(epoch_avg_kl)
            if epoch_avg_kl > KL_TARGET_EPOCH:
                print(f"  [Early Stop] KL divergence {epoch_avg_kl:.6f} > {KL_TARGET_EPOCH}, stopping at epoch {epoch+1}/{epochs}", flush=True)
                break
    
    # Restore model training state
    if was_training:
        model.train()
    
    if n_updates > 0:
        if pretraining_mode:
            # Return supervised loss and entropy for pretraining
            avg_supervised = total_supervised / n_updates
            avg_entropy = total_entropy / n_updates
            return avg_supervised, 0.0, avg_entropy
        else:
            # Return policy/value/entropy for normal training
            avg_policy_loss = total_policy_loss / n_updates
            avg_value_loss = total_value_loss / n_updates
            avg_entropy = total_entropy / n_updates
            avg_approx_kl = total_approx_kl / n_updates
            avg_clip_frac = total_clipped_samples / max(1, total_samples)
            
            # Log diagnostics every update (helps detect instability early)
            print(f"  [PPO Diagnostics] KL: {avg_approx_kl:.6f}, ClipFrac: {avg_clip_frac:.3f}", flush=True)
            
            # Log supervision metrics (helps tune directional loss weight)
            if total_samples > 0 and total_targets > 0:
                avg_has_target = total_targets / total_samples
                avg_mean_dist = total_dist_sum / total_targets
                print(f"  [Supervision] Target visible: {avg_has_target*100:.1f}%, Mean target dist: {avg_mean_dist:.1f}", flush=True)
            
            return avg_policy_loss, avg_value_loss, avg_entropy
    
    return 0, 0, 0


def process_animal_hierarchical(animal, model, animals, config, pheromone_map, device, 
                                 action_counts, previous_distances, prey_list, predator_list):
    """
    Process single animal with hierarchical turn→move policy.
    Returns (transitions, reward, exhausted, moved)
    
    Args:
        prey_list: Pre-computed list of all prey (avoids repeated isinstance scans)
        predator_list: Pre-computed list of all predators
    """
    # S4: Track actual position before move_training
    pos_before = (animal.x, animal.y)
    
    # Use move_training which returns hierarchical transitions
    # Wrap in no_grad to avoid keeping computation graph during rollout
    with torch.no_grad():
        transitions = animal.move_training(model, animals, config, pheromone_map)
    
    # S4: Check if animal actually moved (position changed)
    moved = (animal.x, animal.y) != pos_before
    
    # Update energy (use actual moved flag)
    animal.update_energy(config, moved)
    
    # Track actions from last transition for stats
    if len(transitions) > 0:
        last_trans = transitions[-1]
        if isinstance(last_trans['move_action'], torch.Tensor):
            action_counts[last_trans['move_action'].item()] += 1
        else:
            action_counts[last_trans['move_action']] += 1
    
    # Check exhaustion
    if animal.is_exhausted():
        reward = config.DEATH_PENALTY + config.EXHAUSTION_PENALTY
        return transitions, reward, True, moved
    
    # Base reward
    reward = config.SURVIVAL_REWARD
    if not moved:
        reward += 0.1
    
    # Distance-based reward shaping
    animal_id = animal.id  # Use stable animal.id not Python id()
    is_prey = isinstance(animal, Prey)
    
    if is_prey:
        # Prey: reward for escaping predators (use pre-computed predator_list)
        closest_predator = None
        min_dist = float('inf')
        for other in predator_list:
            dx = abs(other.x - animal.x)
            dy = abs(other.y - animal.y)
            dx = min(dx, config.GRID_SIZE - dx)
            dy = min(dy, config.GRID_SIZE - dy)
            dist = (dx**2 + dy**2)**0.5
            if dist < min_dist:
                min_dist = dist
                closest_predator = other
        
        if closest_predator and moved:
            predator_id = closest_predator.id  # Use stable animal.id
            current_dist = min_dist
            
            if animal_id in previous_distances and predator_id in previous_distances[animal_id]:
                prev_dist = previous_distances[animal_id][predator_id]
                distance_change = current_dist - prev_dist
                
                if distance_change > 0 and current_dist < 15:
                    reward += 3.0 * min(distance_change / 5.0, 1.0)
                elif distance_change < 0 and current_dist < 10:
                    reward -= 1.5 * min(abs(distance_change) / 5.0, 0.5)
            
            if animal_id not in previous_distances:
                previous_distances[animal_id] = {}
            previous_distances[animal_id][predator_id] = current_dist
        
        # STEP 2: Mate-approach reward (dense shaping for prey mating behavior)
        # Only when no predator visible nearby AND ready to reproduce AND moved
        if moved and animal.can_reproduce(config):
            # Check if predators are far enough away (safe to mate)
            predator_safe = True
            if closest_predator:
                if min_dist < config.PREY_SAFE_MATING_DISTANCE:
                    predator_safe = False
            
            if predator_safe:
                # Find nearest potential mate (same species prey, also ready to mate)
                closest_mate = None
                mate_min_dist = float('inf')
                for other in prey_list:
                    if other.id == animal.id:  # Skip self
                        continue
                    if other.name != animal.name:  # Must be same species
                        continue
                    if not other.can_reproduce(config):  # Mate must also be ready
                        continue
                    dx = abs(other.x - animal.x)
                    dy = abs(other.y - animal.y)
                    dx = min(dx, config.GRID_SIZE - dx)
                    dy = min(dy, config.GRID_SIZE - dy)
                    dist = (dx**2 + dy**2)**0.5
                    if dist < mate_min_dist:
                        mate_min_dist = dist
                        closest_mate = other
                
                if closest_mate:
                    mate_id = closest_mate.id
                    current_mate_dist = mate_min_dist
                    
                    if animal_id in previous_distances and mate_id in previous_distances[animal_id]:
                        prev_mate_dist = previous_distances[animal_id][mate_id]
                        distance_change = prev_mate_dist - current_mate_dist  # Positive = got closer
                        
                        if distance_change > 0:  # Reward approaching mate when safe
                            reward += config.PREY_MATE_APPROACH_REWARD * min(distance_change / 3.0, 1.0)
                    
                    if animal_id not in previous_distances:
                        previous_distances[animal_id] = {}
                    previous_distances[animal_id][mate_id] = current_mate_dist
        
        # Overcrowding penalty (use cached count, not scan)
        same_species_count = config._prey_count if hasattr(config, '_prey_count') else sum(1 for a in animals if isinstance(a, Prey))
        if same_species_count > config.MAX_PREY:
            overcrowd_ratio = (same_species_count - config.MAX_PREY) / config.MAX_PREY
            reward += config.OVERPOPULATION_PENALTY * overcrowd_ratio
    else:
        # Predator: reward for chasing prey (use pre-computed prey_list)
        closest_prey = None
        min_dist = float('inf')
        for other in prey_list:
            dx = abs(other.x - animal.x)
            dy = abs(other.y - animal.y)
            dx = min(dx, config.GRID_SIZE - dx)
            dy = min(dy, config.GRID_SIZE - dy)
            dist = (dx**2 + dy**2)**0.5
            if dist < min_dist:
                min_dist = dist
                closest_prey = other
        
        if closest_prey and moved:
            prey_id = closest_prey.id  # Use stable animal.id
            current_dist = min_dist
            
            if animal_id in previous_distances and prey_id in previous_distances[animal_id]:
                prev_dist = previous_distances[animal_id][prey_id]
                distance_change = prev_dist - current_dist
                
                if distance_change > 0:
                    reward += 5.0 * min(distance_change / 5.0, 1.0)
                elif distance_change < 0 and animal.steps_since_last_meal > config.HUNGER_THRESHOLD:
                    reward -= 2.5 * min(abs(distance_change) / 5.0, 0.5)
            
            if animal_id not in previous_distances:
                previous_distances[animal_id] = {}
            previous_distances[animal_id][prey_id] = current_dist
        
        # Hunger penalties
        if animal.steps_since_last_meal > config.HUNGER_THRESHOLD and closest_prey is not None:
            if min_dist < 0.8:
                reward += 1.0 * (1.0 - min_dist)
        
        if animal.steps_since_last_meal > config.STARVATION_THRESHOLD * 0.8:
            reward -= 2.0
        
        # Overcrowding penalty (use cached count, not scan)
        same_species_count = config._pred_count if hasattr(config, '_pred_count') else sum(1 for a in animals if isinstance(a, Predator))
        if same_species_count > config.MAX_PREDATORS:
            overcrowd_ratio = (same_species_count - config.MAX_PREDATORS) / config.MAX_PREDATORS
            reward += config.OVERPOPULATION_PENALTY * overcrowd_ratio
    
    return transitions, reward, False, moved


def run_episode(animals, model_prey, model_predator, pheromone_map, config, steps, device):
    """
    Run a single training episode with hierarchical turn→move policy
    
    Returns:
        Episode statistics and memories
    """
    # S1: Set models to eval mode during rollout (disables Dropout)
    model_prey.eval()
    model_predator.eval()
    
    memory_prey = PPOMemory(config.PPO_BATCH_SIZE, hierarchical=True)
    memory_predator = PPOMemory(config.PPO_BATCH_SIZE, hierarchical=True)
    
    episode_reward_prey = 0
    episode_reward_predator = 0
    episode_stats = {
        'births': 0,
        'deaths': 0,
        'meals': 0,
        'exhaustion_deaths': 0,
        'old_age_deaths': 0,
        'starvation_deaths': 0
    }
    
    # Track action distributions
    action_counts_prey = [0] * 8  # 8 directions
    action_counts_predator = [0] * 8
    
    # R4/R5: Track last memory index for each animal across entire episode (for penalties)
    last_idx_prey = {}  # {animal_id: memory_index}
    last_idx_predator = {}  # {animal_id: memory_index}
    
    # Track previous distances for distance-based reward shaping
    previous_distances = {}  # {animal_id: {target_id: distance}}
    
    for step in range(steps):
        step_reward_prey = 0
        step_reward_predator = 0
        animals_to_remove = []
        
        # === STEP START: Pre-compute species lists once (avoid repeated scans) ===
        prey_list = [a for a in animals if isinstance(a, Prey)]
        predator_list = [a for a in animals if isinstance(a, Predator)]
        prey_count = len(prey_list)
        predator_count = len(predator_list)
        
        # Cache counts on config for get_enhanced_input() optimization (avoids N scans)
        config._prey_count = prey_count
        config._pred_count = predator_count
        
        # Log progress every 50 steps (reduced overhead)
        if (step + 1) % 50 == 0:
            timestamp = _ts()
            print(f"[{timestamp}] Step {step + 1}/{steps}: {len(animals)} animals (Prey={prey_count}, Pred={predator_count})", flush=True)
        
        # Age and energy updates
        for animal in animals:
            animal.update_age()
            
            # Check for old age (apply small penalty - natural death)
            if animal.is_old(config):
                animals_to_remove.append(animal)
                episode_stats['old_age_deaths'] += 1
                
                # R4: Apply old age death penalty to THIS animal's last experience
                animal_id = animal.id  # Use stable animal.id not Python id()
                if isinstance(animal, Predator):
                    if animal_id in last_idx_predator:
                        memory_idx = last_idx_predator[animal_id]
                        if memory_idx < len(memory_predator.rewards):
                            memory_predator.rewards[memory_idx] += config.DEATH_PENALTY + config.OLD_AGE_PENALTY
                            memory_predator.next_values[memory_idx] = torch.zeros_like(memory_predator.next_values[memory_idx])
                            memory_predator.dones[memory_idx] = True
                else:
                    if animal_id in last_idx_prey:
                        memory_idx = last_idx_prey[animal_id]
                        if memory_idx < len(memory_prey.rewards):
                            memory_prey.rewards[memory_idx] += config.DEATH_PENALTY + config.OLD_AGE_PENALTY
                            memory_prey.next_values[memory_idx] = torch.zeros_like(memory_prey.next_values[memory_idx])
                            memory_prey.dones[memory_idx] = True
                continue
        
        # Movement phase - HIERARCHICAL (turn→reobserve→move per animal)
        # Filter out animals marked for removal
        active_animals = [a for a in animals if a not in animals_to_remove]
        
        # Recompute species lists AFTER old-age filtering (exclude dying animals from distance shaping)
        prey_list = [a for a in active_animals if isinstance(a, Prey)]
        predator_list = [a for a in active_animals if isinstance(a, Predator)]
        prey_count = len(prey_list)
        predator_count = len(predator_list)
        
        # Update cached counts for get_enhanced_input() optimization
        config._prey_count = prey_count
        config._pred_count = predator_count
        
        # Track memory indices for reward adjustments
        prey_memory_indices = {}
        predator_memory_indices = {}
        
        if len(active_animals) > 0:
            # Process each animal with hierarchical policy
            for animal in active_animals:
                is_prey = isinstance(animal, Prey)
                model = model_prey if is_prey else model_predator
                memory = memory_prey if is_prey else memory_predator
                action_counts = action_counts_prey if is_prey else action_counts_predator
                
                # Get hierarchical transitions and reward
                transitions, reward, exhausted, moved = process_animal_hierarchical(
                    animal, model, animals, config, pheromone_map, device,
                    action_counts, previous_distances, prey_list, predator_list
                )
                
                # Store all transitions with the computed reward
                memory_start_idx = len(memory.rewards)
                for trans_idx, transition in enumerate(transitions):
                    # Apply reward only to last transition (final outcome of all micro-steps)
                    trans_reward = reward if trans_idx == len(transitions) - 1 else 0.0
                    memory.add(transition=transition, reward=trans_reward, done=False)
                
                # Track memory index for this animal (use last transition index)
                if len(transitions) > 0:
                    animal_id = animal.id  # Use stable animal.id not Python id()
                    memory_idx = memory_start_idx + len(transitions) - 1
                    
                    # CRITICAL FIX: Link micro-step transitions within same trajectory
                    # Each transition's next_value should be the next transition's value_old
                    for i in range(len(transitions) - 1):
                        trans_idx = memory_start_idx + i
                        next_trans_idx = trans_idx + 1
                        if trans_idx < len(memory.next_values) and next_trans_idx < len(memory.values):
                            # Use view_as() to ensure shape consistency
                            memory.next_values[trans_idx] = memory.values[next_trans_idx].view_as(memory.next_values[trans_idx])
                    # Last transition in this step: will be linked to next step (or terminal)
                    
                    # TD(0) bootstrapping: patch previous step's last transition's next_value with current step's first value
                    # Use first transition's value (state when animal acts this step)
                    cur_value_old = transitions[0]['value_old'].detach().view(1, 1)
                    prev_idx = last_idx_prey.get(animal_id) if is_prey else last_idx_predator.get(animal_id)
                    # Don't overwrite next_value for already-terminal transitions
                    if (
                        prev_idx is not None
                        and prev_idx < len(memory.next_values)
                        and prev_idx < len(memory.dones)
                        and not memory.dones[prev_idx]
                    ):
                        # Use view_as() to ensure shape consistency
                        memory.next_values[prev_idx] = cur_value_old.view_as(memory.next_values[prev_idx])
                    
                    if is_prey:
                        prey_memory_indices[animal_id] = memory_idx
                        last_idx_prey[animal_id] = memory_idx  # R4: Persistent tracking
                        step_reward_prey += reward
                    else:
                        predator_memory_indices[animal_id] = memory_idx
                        last_idx_predator[animal_id] = memory_idx  # R4: Persistent tracking
                        step_reward_predator += reward
                
                # Mark for removal if exhausted
                if exhausted:
                    animals_to_remove.append(animal)
                    episode_stats['exhaustion_deaths'] += 1
                    # Terminal state: zero next_value and mark done
                    if len(transitions) > 0 and memory_idx < len(memory.rewards):
                        memory.next_values[memory_idx] = torch.zeros_like(memory.next_values[memory_idx])
                        memory.dones[memory_idx] = True
        
        # Remove dead animals
        for animal in animals_to_remove:
            if animal in animals:
                animals.remove(animal)
        animals_to_remove.clear()
        
        # Eating phase - use predator snapshot to avoid double-eating
        predators_snapshot = [a for a in animals if isinstance(a, Predator)]
        eaten_prey_animals = []  # Track which prey objects were eaten (for penalty application)
        
        for predator in predators_snapshot:
            has_eaten, eat_reward, eaten_prey = predator.perform_eat(animals, config)
            
            if has_eaten and eaten_prey is not None:
                # Remove prey immediately to prevent double-eating
                if eaten_prey in animals:
                    animals.remove(eaten_prey)
                    eaten_prey_animals.append(eaten_prey)
                    
                    episode_stats['meals'] += 1
                    episode_stats['deaths'] += 1
                    step_reward_predator += eat_reward
                    
                    # CRITICAL: Add hunting reward to THIS predator's stored experience
                    # Use the tracked memory index to ensure we modify the correct predator's reward
                    predator_id = predator.id  # Use stable animal.id
                    if predator_id in predator_memory_indices:
                        memory_idx = predator_memory_indices[predator_id]
                        if memory_idx < len(memory_predator.rewards):
                            memory_predator.rewards[memory_idx] += eat_reward
            else:
                # No eat -> hunger progresses
                predator.steps_since_last_meal += 1
                
                # Check if starvation death is enabled (curriculum learning may disable it)
                if config.STARVATION_ENABLED and predator.steps_since_last_meal >= config.STARVATION_THRESHOLD:
                    animals_to_remove.append(predator)
                    episode_stats['starvation_deaths'] += 1
                # If starvation disabled, apply reduced penalty for being very hungry
                elif not config.STARVATION_ENABLED and predator.steps_since_last_meal >= config.STARVATION_THRESHOLD:
                    # R5: Apply reduced penalty using persistent index
                    predator_id = predator.id  # Use stable animal.id
                    if predator_id in last_idx_predator:
                        memory_idx = last_idx_predator[predator_id]
                        if memory_idx < len(memory_predator.rewards):
                            memory_predator.rewards[memory_idx] += config.STARVATION_PENALTY * 0.5  # Half penalty
        
        # Remove starved predators and apply starvation penalty
        for animal in animals_to_remove:
            if animal in animals:
                # R5: Apply heavy starvation penalty using persistent index
                animal_id = animal.id  # Use stable animal.id
                if animal_id in last_idx_predator:
                    memory_idx = last_idx_predator[animal_id]
                    if memory_idx < len(memory_predator.rewards):
                        memory_predator.rewards[memory_idx] += config.DEATH_PENALTY + config.STARVATION_PENALTY
                        memory_predator.next_values[memory_idx] = torch.zeros_like(memory_predator.next_values[memory_idx])
                        memory_predator.dones[memory_idx] = True
                animals.remove(animal)
        
        # Apply eaten penalty to the specific prey that were eaten this step
        if len(eaten_prey_animals) > 0:
            # Apply penalty to each eaten prey's stored experience
            for eaten_prey in eaten_prey_animals:
                prey_id = eaten_prey.id  # Use stable animal.id
                # Fallback to last_idx_prey if not in this step's indices
                idx = prey_memory_indices.get(prey_id, last_idx_prey.get(prey_id))
                if idx is not None and idx < len(memory_prey.rewards):
                    memory_prey.rewards[idx] += config.DEATH_PENALTY + config.EATEN_PENALTY
                    memory_prey.next_values[idx] = torch.zeros_like(memory_prey.next_values[idx])
                    memory_prey.dones[idx] = True
        
        # Mating phase - OPTIMIZED with spatial hashing (was O(N²), now O(N))
        new_animals = []
        mated_animals = set()
        
        # Build position map for local mating checks
        from collections import defaultdict
        pos_map = defaultdict(list)
        for a in animals:
            pos_map[(a.x, a.y)].append(a)
        
        for animal1 in animals:
            if animal1.id in mated_animals or not animal1.can_reproduce(config):
                continue
            
            # Use flag to properly break out of nested loops
            mated = False
            # Only check neighboring cells (9 cells: 3x3 grid centered on animal1)
            for dx in (-1, 0, 1):
                if mated:
                    break
                for dy in (-1, 0, 1):
                    nx = (animal1.x + dx) % config.GRID_SIZE
                    ny = (animal1.y + dy) % config.GRID_SIZE
                    
                    for animal2 in pos_map[(nx, ny)]:
                        if animal2.id <= animal1.id:  # Avoid duplicate pairs and self
                            continue
                        if animal2.id in mated_animals or not animal2.can_reproduce(config):
                            continue
                        
                        if animal1.can_mate(animal2, config):
                            mating_prob = (config.MATING_PROBABILITY_PREY 
                                         if animal1.name == "A" 
                                         else config.MATING_PROBABILITY_PREDATOR)
                            
                            if random.random() < mating_prob:
                                # Create offspring (same type as parents)
                                child_x = (animal1.x + animal2.x) // 2
                                child_y = (animal1.y + animal2.y) // 2
                                if isinstance(animal1, Prey):
                                    child = Prey(child_x, child_y, animal1.name, animal1.color,
                                               {animal1.id, animal2.id})
                                else:  # Predator
                                    child = Predator(child_x, child_y, animal1.name, animal1.color,
                                                   {animal1.id, animal2.id})
                                child.energy = config.INITIAL_ENERGY
                            # Store parent IDs with child for reward tracking
                            child._parent_ids = (animal1.id, animal2.id)
                            new_animals.append(child)
                            
                            # Update parents
                            animal1.energy -= config.MATING_ENERGY_COST
                            animal2.energy -= config.MATING_ENERGY_COST
                            animal1.move_away(config)
                            animal2.move_away(config)
                            animal1.mating_cooldown = config.MATING_COOLDOWN
                            animal2.mating_cooldown = config.MATING_COOLDOWN
                            animal1.num_children += 1
                            animal2.num_children += 1
                            
                            mated_animals.add(animal1.id)
                            mated_animals.add(animal2.id)
                            
                            episode_stats['births'] += 1
        predator_count = sum(1 for a in animals if isinstance(a, Predator))
        
        # Separate offspring by species
        new_prey = [a for a in new_animals if isinstance(a, Prey)]
        new_predators = [a for a in new_animals if isinstance(a, Predator)]
        
        # Add prey up to their capacity
    added_prey = []
    if prey_count + len(new_prey) <= config.MAX_PREY:
        added_prey = new_prey
        animals.extend(new_prey)
    else:
        available_prey_slots = max(0, config.MAX_PREY - prey_count)
        if available_prey_slots > 0:
            added_prey = new_prey[:available_prey_slots]
            animals.extend(added_prey)
    
    # Add predators up to their capacity
    added_predators = []
    if predator_count + len(new_predators) <= config.MAX_PREDATORS:
        added_predators = new_predators
        animals.extend(new_predators)
    else:
        available_predator_slots = max(0, config.MAX_PREDATORS - predator_count)
        if available_predator_slots > 0:
            added_predators = new_predators[:available_predator_slots]
            animals.extend(added_predators)
    
    # CRITICAL: Apply reproduction reward ONLY for offspring that were actually added
    # Track unique parent pairs to avoid double-rewarding
    rewarded_parent_pairs = set()
    
    for child in added_prey:
        if hasattr(child, '_parent_ids'):
            parent_ids = tuple(sorted(child._parent_ids))
            if parent_ids not in rewarded_parent_pairs:
                rewarded_parent_pairs.add(parent_ids)
                for parent_id in parent_ids:
                    idx = last_idx_prey.get(parent_id)
                    if idx is not None and idx < len(memory_prey.rewards) and not memory_prey.dones[idx]:
                        memory_prey.rewards[idx] += config.REPRODUCTION_REWARD
                step_reward_prey += config.REPRODUCTION_REWARD
    
    for child in added_predators:
        if hasattr(child, '_parent_ids'):
            parent_ids = tuple(sorted(child._parent_ids))
            if parent_ids not in rewarded_parent_pairs:
                rewarded_parent_pairs.add(parent_ids)
                for parent_id in parent_ids:
                    idx = last_idx_predator.get(parent_id)
                    if idx is not None and idx < len(memory_predator.rewards) and not memory_predator.dones[idx]:
                        memory_predator.rewards[idx] += config.REPRODUCTION_REWARD
                step_reward_predator += config.REPRODUCTION_REWARD
        # Update cooldowns
        for animal in animals:
            if animal.mating_cooldown > 0:
                animal.mating_cooldown -= 1
            animal.survival_time += 1
            
            # Deposit pheromones
            animal.deposit_pheromones(animals, pheromone_map, config)
        
        # Update pheromone map
        pheromone_map.update()
        
        # Check extinction
        if len(animals) == 0:
            timestamp = _ts()
            print(f"[{timestamp}] Episode ended at step {step + 1}: All animals extinct")
            break
        
        # Store step rewards
        episode_reward_prey += step_reward_prey
        episode_reward_predator += step_reward_predator
    
    episode_stats['final_prey'] = sum(1 for a in animals if isinstance(a, Prey))
    episode_stats['final_predators'] = sum(1 for a in animals if isinstance(a, Predator))
    episode_stats['total_reward_prey'] = episode_reward_prey
    episode_stats['total_reward_predator'] = episode_reward_predator
    episode_stats['action_dist_prey'] = action_counts_prey
    episode_stats['action_dist_predator'] = action_counts_predator
    
    # REMOVED: Experience filtering that was creating survivorship bias
    # Network now learns from ALL experiences (successful and failed attempts)
    # This allows it to learn: "I moved E when prey was S = bad" (failed attempts)
    # Not just: "I moved randomly and got lucky = good" (successful attempts)
    episode_stats['predator_experiences'] = len(memory_predator.rewards)
    print(f"DEBUG: Training on all {len(memory_predator.rewards)} predator experiences (no filtering)", flush=True)
    
    # End-of-episode: mark last transitions terminal for all remaining animals
    # This prevents value leakage across episode boundaries
    for mem, last_idx in ((memory_prey, last_idx_prey), (memory_predator, last_idx_predator)):
        for _, idx in last_idx.items():
            if idx is not None and idx < len(mem.dones) and not mem.dones[idx]:
                mem.next_values[idx] = torch.zeros_like(mem.next_values[idx])
                mem.dones[idx] = True
    
    return memory_prey, memory_predator, episode_stats


def main():
    print("\n" + "="*70, flush=True)
    print("  ADVANCED LIFE GAME TRAINING (PPO + Pheromones + Energy)", flush=True)
    print("="*70, flush=True)
    
    # Fix 6: Set consistent seed for reproducibility
    SEED = int(os.environ.get("LIFEGAME_SEED", "0"))
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"Seed: {SEED}", flush=True)
    
    # Device detection helpers
    def is_cuda_device(dev) -> bool:
        return hasattr(dev, "type") and dev.type == "cuda"
    
    def is_directml_device(dev) -> bool:
        # torch_directml devices show as "privateuseone"
        return hasattr(dev, "type") and dev.type == "privateuseone"
    
    # Check for --cpu flag
    force_cpu = '--cpu' in sys.argv or os.environ.get('FORCE_CPU', '0') == '1'
    
    # Setup device (GPU REQUIRED)
    # Priority: DirectML (AMD/Intel) > CUDA (NVIDIA) > Error
    device = None
    device_name = "cpu"
    
    try:
        import torch_directml
        device = torch_directml.device()
        device_name = "DirectML (AMD/Intel GPU)"
        print(f"Device: {device}")
        print("Using GPU backend: DirectML")
        print("Note: DirectML detected - Good performance on AMD/Intel GPUs!")
        print("Expected: 3-8x faster than CPU")
    except ImportError:
        if force_cpu:
            device = torch.device('cpu')
            device_name = "CPU (forced via --cpu flag)"
            print(f"Device: {device}", flush=True)
            print("Note: CPU mode forced. Remove --cpu flag to use GPU.", flush=True)
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            device_name_check = torch.cuda.get_device_name(0)
            device_name = f"CUDA - {device_name_check}"
            print(f"Device: {device}", flush=True)
            print("Using GPU backend: CUDA/ROCm")
            print(f"GPU: {device_name_check}", flush=True)
            if 'ROCm' in torch.__version__:
                rocm_version = torch.__version__.split('+')[1] if '+' in torch.__version__ else 'Unknown'
                print(f"ROCm Version: {rocm_version}", flush=True)
                
                # ROCm-specific optimizations
                # Try different BLAS backends (experimental, can help with kernel hangs)
                # Options: "hipblaslt" (default), "hipblas", "ck" (composable kernels)
                try:
                    blas_lib = os.environ.get('PYTORCH_HIP_BLAS_LIBRARY', 'hipblaslt')
                    torch.backends.cuda.preferred_blas_library = blas_lib
                    print(f"ROCm BLAS: {blas_lib}", flush=True)
                except Exception as e:
                    print(f"ROCm BLAS: default (setting failed: {e})", flush=True)
            else:
                print(f"CUDA Version: {torch.version.cuda}", flush=True)
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
        else:
            # GPU REQUIRED - DO NOT FALL BACK TO CPU
            print("\n" + "="*70, flush=True)
            print("ERROR: GPU NOT DETECTED - TRAINING ABORTED", flush=True)
            print("="*70, flush=True)
            print("\nThis training script requires GPU acceleration.", flush=True)
            print("\nYou are likely using the wrong Python environment:", flush=True)
            print("  Current: CPU-only PyTorch", flush=True)
            print("  Required: .venv_rocm with ROCm PyTorch", flush=True)
            print("\nRun training using:", flush=True)
            print("  .\\scripts\\run_training_safe.ps1", flush=True)
            print("\nOr manually activate the correct environment:", flush=True)
            print("  .venv_rocm\\Scripts\\python.exe scripts\\train.py", flush=True)
            print("\n" + "="*70 + "\n", flush=True)
            sys.exit(1)
    
    # Configuration
    config = SimulationConfig()
    CFG = type(config)  # Get class for class-level modifications
    
    # Fix 7: Save base config snapshot to restore at Stage 4 (prevent curriculum leakage)
    BASE_CONFIG = {k: getattr(CFG, k) for k in dir(CFG) if k.isupper() and not k.startswith('_')}
    
    # Aggressive CPU Optimizations
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    
    # Create models and move to device
    model_prey = ActorCriticNetwork(config).to(device)
    model_predator = ActorCriticNetwork(config).to(device)
    
    # Display model size
    total_params = sum(p.numel() for p in model_prey.parameters())
    trainable_params = sum(p.numel() for p in model_prey.parameters() if p.requires_grad)
    print(f"\nModel Size: {total_params:,} parameters ({trainable_params:,} trainable)", flush=True)
    
    # Fix 5: Add Ctrl+C handler for safe checkpoint
    def sigint_handler(signum, frame):
        print("\n[SIGINT] Saving interrupt checkpoint...", flush=True)
        torch.save(model_prey.state_dict(), "models/model_A_interrupt.pth")
        torch.save(model_predator.state_dict(), "models/model_B_interrupt.pth")
        print("[SIGINT] Checkpoint saved. Exiting.", flush=True)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, sigint_handler)
    
    # Optimizers
    optimizer_prey = optim.Adam(model_prey.parameters(), lr=config.LEARNING_RATE_PREY)
    optimizer_predator = optim.Adam(model_predator.parameters(), lr=config.LEARNING_RATE_PREDATOR)
    
    # Initialize pheromone map
    pheromone_map = PheromoneMap(config.GRID_SIZE, 
                                 decay_rate=config.PHEROMONE_DECAY,
                                 diffusion_rate=config.PHEROMONE_DIFFUSION)
    
    # Training parameters (reduced for 2.9M parameter model)
    num_episodes = config.NUM_EPISODES  # Read from centralized config
    steps_per_episode = config.STEPS_PER_EPISODE  # Read from centralized config
    
    print(f"\nTraining for {num_episodes} episodes", flush=True)
    print(f"Steps per episode: {steps_per_episode}", flush=True)
    print(f"Using Actor-Critic with PPO algorithm", flush=True)
    print(f"Advanced features: Energy, Age, Pheromones, Multi-Head Attention\n", flush=True)
    
    # Create models directory
    os.makedirs("outputs/checkpoints", exist_ok=True)
    
    best_prey_survival = 0
    current_curriculum_stage = 0
    
    # NEW: Directional pre-training phase (Episodes -4 to 0)
    # Train with ONLY directional loss, no value/policy loss
    # This forces the network to learn correct directions first
    pretraining_episodes = 0  # DISABLED - PPO overwrites it anyway, saves ~1 hour
    
    if pretraining_episodes > 0:
        print(f"\n{'='*70}", flush=True)
        print("DIRECTIONAL PRE-TRAINING PHASE", flush=True)
        print(f"Episodes: {pretraining_episodes}", flush=True)
        print("Training with DIRECTIONAL LOSS ONLY (no value/policy optimization)", flush=True)
        print("This forces correct spatial behavior before survival optimization", flush=True)
        print(f"{'='*70}\n", flush=True)
    
    for episode in range(1, num_episodes + 1 + pretraining_episodes):
        # Determine if in pre-training phase
        is_pretraining = (episode <= pretraining_episodes)
        actual_episode = episode if not is_pretraining else episode  # Keep sequential numbering
        
        timestamp = _ts()
        if is_pretraining:
            print(f"\n[{timestamp}] PRE-TRAINING Episode {episode}/{pretraining_episodes}", flush=True)
        else:
            adjusted_ep = episode - pretraining_episodes
            print(f"\n[{timestamp}] Episode {adjusted_ep}/{num_episodes}", flush=True)
        sys.stdout.flush()  # Force immediate write
        
        # Apply curriculum learning stage for this episode (skip during pre-training)
        curriculum_stage = None if is_pretraining else config.apply_curriculum_stage(episode - pretraining_episodes)
        
        # Restore base config at Stage 2 to prevent curriculum leakage
        if not is_pretraining and curriculum_stage:
            new_stage_dict, new_stage_idx = config.get_curriculum_stage(episode - pretraining_episodes)
            if new_stage_idx == 1:  # Stage 2 (index 1) - restore to baseline
                # Restore original base values to CLASS, not instance
                for k, v in BASE_CONFIG.items():
                    setattr(CFG, k, v)
                print(f"[CURRICULUM STAGE 2] Restored base config values (no overrides)", flush=True)
        
        # NEW: Curriculum-aware learning rate adjustment
        # Reset/increase LR when curriculum changes to help adapt to new conditions
        if not is_pretraining and curriculum_stage:
            new_stage_dict, new_stage_idx = config.get_curriculum_stage(episode - pretraining_episodes)
            if new_stage_idx != current_curriculum_stage:
                current_curriculum_stage = new_stage_idx
                # Boost learning rate for 3 episodes after curriculum change
                new_lr_prey = config.LEARNING_RATE_PREY * 2.0
                new_lr_pred = config.LEARNING_RATE_PREDATOR * 2.0
                for param_group in optimizer_prey.param_groups:
                    param_group['lr'] = new_lr_prey
                for param_group in optimizer_predator.param_groups:
                    param_group['lr'] = new_lr_pred
                print(f"[CURRICULUM CHANGE] Stage {new_stage_idx}: {new_stage_dict['name']} - LR boosted to {new_lr_prey:.6f}", flush=True)
            # Decay back to normal over 3 episodes after curriculum change
            elif current_curriculum_stage >= 0:
                # Calculate episodes since last curriculum change
                stage_start_episode = config.CURRICULUM_STAGES[current_curriculum_stage]['episodes'][0]
                episodes_in_stage = (episode - pretraining_episodes) - stage_start_episode
                if 0 < episodes_in_stage <= 3:
                    decay_factor = 0.8
                    for param_group in optimizer_prey.param_groups:
                        param_group['lr'] = max(param_group['lr'] * decay_factor, config.LEARNING_RATE_PREY)
                    for param_group in optimizer_predator.param_groups:
                        param_group['lr'] = max(param_group['lr'] * decay_factor, config.LEARNING_RATE_PREDATOR)
        
        if curriculum_stage:
            curriculum_info = config.get_curriculum_info(episode - pretraining_episodes)
            print(curriculum_info, flush=True)
            sys.stdout.flush()
        
        # Create fresh population
        animals = create_population(config)
        pheromone_map.reset()
        
        # Run episode with timing
        env_start = time.time()
        memory_prey, memory_predator, stats = run_episode(
            animals, model_prey, model_predator, pheromone_map, config, steps_per_episode, device
        )
        env_time = time.time() - env_start
        
        # Log before starting GPU update
        timestamp = _ts()
        print(f"[{timestamp}] Starting PPO update (Prey experiences={len(memory_prey.rewards)}, Predator={len(memory_predator.rewards)})...", flush=True)
        sys.stdout.flush()
        
        # S1: Set models to train mode for optimization (enables Dropout)
        model_prey.train()
        model_predator.train()
        
        # PPO updates with timing
        use_amp = False  # Disable mixed precision for ROCm compatibility
        gpu_start = time.time()
        
        # Use gradient accumulation to reduce peak memory usage
        accumulation_steps = 1  # Single large mini-batch for maximum GPU work
        
        # ROCm Fix: Force SDPA to use math backend (not flash attention)
        # This avoids attention kernel hangs on Windows ROCm
        use_sdpa_math = SDPA_AVAILABLE and is_cuda_device(device)
        
        if use_sdpa_math:
            with sdpa_kernel(SDPBackend.MATH):
                policy_loss_prey, value_loss_prey, entropy_prey = ppo_update(
                    model_prey, optimizer_prey, memory_prey, config, device, use_amp, accumulation_steps, is_pretraining, species="prey"
                )
                policy_loss_pred, value_loss_pred, entropy_pred = ppo_update(
                    model_predator, optimizer_predator, memory_predator, config, device, use_amp, accumulation_steps, is_pretraining, species="predator"
                )
        else:
            # DirectML or CPU: no CUDA backend toggles
            policy_loss_prey, value_loss_prey, entropy_prey = ppo_update(
                model_prey, optimizer_prey, memory_prey, config, device, use_amp, accumulation_steps, is_pretraining, species="prey"
            )
            policy_loss_pred, value_loss_pred, entropy_pred = ppo_update(
                model_predator, optimizer_predator, memory_predator, config, device, use_amp, accumulation_steps, is_pretraining, species="predator"
            )
        
        if is_cuda_device(device):
            torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - gpu_start
        
        timestamp = _ts()
        print(f"[{timestamp}] PPO update completed!", flush=True)
        sys.stdout.flush()
        
        # Show timing breakdown
        total_time = env_time + gpu_time
        env_pct = (env_time / total_time) * 100
        gpu_pct = (gpu_time / total_time) * 100
        timestamp = _ts()
        print(f"[{timestamp}] Timing: Env={env_time:.1f}s ({env_pct:.0f}%), GPU={gpu_time:.1f}s ({gpu_pct:.0f}%), Total={total_time:.1f}s", flush=True)
        
        if is_cuda_device(device):
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1e9
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1e9
            timestamp = _ts()
            print(f"[{timestamp}] GPU Memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved", flush=True)
        
        # Print stats
        timestamp = _ts()
        print(f"[{timestamp}] Final: Prey={stats['final_prey']}, Predators={stats['final_predators']}", flush=True)
        print(f"[{timestamp}] Births={stats['births']}, Deaths={stats['deaths']}, Meals={stats['meals']}", flush=True)
        if 'filtered_predator_experiences' in stats:
            print(f"[{timestamp}] Filtered Experiences: {stats['filtered_predator_experiences']}", flush=True)
        print(f"[{timestamp}] Exhaustion={stats['exhaustion_deaths']}, Old Age={stats['old_age_deaths']}, Starvation={stats['starvation_deaths']}", flush=True)
        print(f"[{timestamp}] Rewards: Prey={stats['total_reward_prey']:.1f}, Predator={stats['total_reward_predator']:.1f}", flush=True)
        if is_pretraining:
            print(f"[{timestamp}] Pretrain: SupLoss(P={policy_loss_prey:.3f}/Pr={policy_loss_pred:.3f}), "
                  f"Entropy(P={entropy_prey:.3f}/Pr={entropy_pred:.3f})", flush=True)
        else:
            print(f"[{timestamp}] Losses: Policy(P={policy_loss_prey:.3f}/Pr={policy_loss_pred:.3f}), "
                  f"Value(P={value_loss_prey:.3f}/Pr={value_loss_pred:.3f}), "
                  f"Entropy(P={entropy_prey:.3f}/Pr={entropy_pred:.3f})", flush=True)
        
        # Action distribution analysis (detect bias)
        action_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]  # 0=N, 1=NE, 2=E, ...
        total_prey_actions = sum(stats['action_dist_prey'])
        total_pred_actions = sum(stats['action_dist_predator'])
        
        if total_prey_actions > 0:
            prey_dist = [f"{name}:{count*100/total_prey_actions:.0f}%" 
                        for name, count in zip(action_names, stats['action_dist_prey']) if count > 0]
            print(f"[{timestamp}] Prey Actions: {', '.join(prey_dist[:5])}", flush=True)  # Top 5
            
            # Warn if any action > 30% (bias detected)
            max_prey_pct = max(stats['action_dist_prey']) * 100 / total_prey_actions if total_prey_actions > 0 else 0
            if max_prey_pct > 30:
                max_idx = stats['action_dist_prey'].index(max(stats['action_dist_prey']))
                print(f"[{timestamp}] WARNING: Prey bias detected - {action_names[max_idx]} = {max_prey_pct:.0f}%", flush=True)
        
        if total_pred_actions > 0:
            pred_dist = [f"{name}:{count*100/total_pred_actions:.0f}%" 
                        for name, count in zip(action_names, stats['action_dist_predator']) if count > 0]
            print(f"[{timestamp}] Predator Actions: {', '.join(pred_dist[:5])}", flush=True)  # Top 5
            
            # Warn if any action > 30% (bias detected)
            max_pred_pct = max(stats['action_dist_predator']) * 100 / total_pred_actions if total_pred_actions > 0 else 0
            if max_pred_pct > 30:
                max_idx = stats['action_dist_predator'].index(max(stats['action_dist_predator']))
                print(f"[{timestamp}] WARNING: Predator bias detected - {action_names[max_idx]} = {max_pred_pct:.0f}%", flush=True)
        
        # Save best model (skip during pre-training)
        if not is_pretraining and stats['final_prey'] > best_prey_survival:
            best_prey_survival = stats['final_prey']
            torch.save(model_prey.state_dict(), "outputs/checkpoints/model_A_ppo.pth")
            torch.save(model_predator.state_dict(), "outputs/checkpoints/model_B_ppo.pth")
            timestamp = _ts()
            print(f"[{timestamp}] Saved models", flush=True)
        
        # Save checkpoint every episode
        checkpoint_episode = episode if is_pretraining else (episode - pretraining_episodes)
        checkpoint_name = f"pretrain{episode}" if is_pretraining else f"ep{checkpoint_episode}"
        torch.save(model_prey.state_dict(), f"outputs/checkpoints/model_A_ppo_{checkpoint_name}.pth")
        torch.save(model_predator.state_dict(), f"outputs/checkpoints/model_B_ppo_{checkpoint_name}.pth")
        timestamp = _ts()
        print(f"[{timestamp}] Checkpoint saved (episode {episode})")
    
    timestamp = _ts()
    print(f"\n[{timestamp}] " + "="*70)
    print(f"[{timestamp}]   TRAINING COMPLETE!")
    print(f"[{timestamp}] " + "="*70)
    print(f"[{timestamp}] Best prey survival: {best_prey_survival}")
    print(f"[{timestamp}] Models saved to: outputs/checkpoints/model_A_ppo.pth, outputs/checkpoints/model_B_ppo.pth")


if __name__ == "__main__":
    main()
