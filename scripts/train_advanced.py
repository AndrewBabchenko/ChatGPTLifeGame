"""
Advanced Training Script with PPO Algorithm
Integrates Actor-Critic network, pheromones, energy, and age systems
"""

import torch
import torch.optim as optim
import torch.nn as nn
import random
import os
import sys
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
from src.core.animal import Animal
from src.models.actor_critic_network import ActorCriticNetwork
from src.core.pheromone_system import PheromoneMap
from src.models.replay_buffer import PPOMemory

def _ts() -> str:
    """Timestamp with milliseconds for logs."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def create_population(config: SimulationConfig) -> list:
    """Create initial population of animals"""
    animals = []
    
    # Create prey (species A)
    for _ in range(config.INITIAL_PREY_COUNT):
        x = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        y = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        animal = Animal(x, y, "A", "#00ff00", predator=False)
        animal.energy = config.INITIAL_ENERGY
        animals.append(animal)
    
    # Create predators (species B)
    for _ in range(config.INITIAL_PREDATOR_COUNT):
        x = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        y = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        animal = Animal(x, y, "B", "#ff0000", predator=True)
        animal.energy = config.INITIAL_ENERGY
        animals.append(animal)
    
    return animals


def ppo_update(model, optimizer, memory, config, device, use_amp=False, accumulation_steps=4):
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
    """
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    # Compute returns and advantages
    returns, advantages = memory.compute_returns_and_advantages(
        torch.tensor([0.0]), config.GAMMA, config.GAE_LAMBDA
    )
    # Flatten and convert tensors to lists of scalars for proper batching
    memory.returns = returns.flatten().tolist()
    memory.advantages = advantages.flatten().tolist()
    
    # PPO epochs
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    n_updates = 0
    update_start_time = time.time()
    last_heartbeat = update_start_time
    heartbeat_interval_s = 30
    
    for epoch in range(config.PPO_EPOCHS):
        # Get all batches for this epoch
        batches = list(memory.get_batches())
        total_batches = len(batches)
        
        for batch_idx, batch in enumerate(batches):
            # Extract batch data
            states = batch['states']
            actions = batch['actions'].to(device)
            old_log_probs = batch['old_log_probs'].to(device)
            returns_batch = batch['returns'].view(-1).to(device)  # Flatten to 1D
            advantages_batch = batch['advantages'].view(-1).to(device)  # Flatten to 1D
            
            # Split batch into smaller mini-batches for gradient accumulation
            minibatch_size = len(states) // accumulation_steps
            if minibatch_size == 0:
                minibatch_size = 1
                accumulation_steps = len(states)
            
            # Zero gradients once per batch
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()
            
            # Process mini-batches with gradient accumulation
            for mini_idx in range(accumulation_steps):
                start_idx = mini_idx * minibatch_size
                end_idx = start_idx + minibatch_size if mini_idx < accumulation_steps - 1 else len(states)
                
                if start_idx >= len(states):
                    break
                
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval_s:
                    elapsed = now - update_start_time
                    print(
                        f"[{_ts()}] PPO update in progress: "
                        f"epoch {epoch + 1}/{config.PPO_EPOCHS}, "
                        f"batch {batch_idx + 1}/{total_batches}, "
                        f"minibatch {mini_idx + 1}/{accumulation_steps}, "
                        f"elapsed {elapsed:.1f}s",
                        flush=True
                    )
                    last_heartbeat = now
                
                # Get mini-batch slices
                mini_states = states[start_idx:end_idx]
                mini_actions = actions[start_idx:end_idx]
                mini_old_log_probs = old_log_probs[start_idx:end_idx]
                mini_returns = returns_batch[start_idx:end_idx]
                mini_advantages = advantages_batch[start_idx:end_idx]
                
                # Evaluate actions with current policy (batched for speed)
                animal_inputs = torch.cat([s[0] for s in mini_states], dim=0).to(device)
                visible_inputs = torch.cat([s[1] for s in mini_states], dim=0).to(device)
                mini_actions = mini_actions.view(-1)

                # Use mixed precision for forward pass
                if use_amp:
                    with torch.cuda.amp.autocast():
                        log_probs, values, entropies = model.evaluate_actions(
                            animal_inputs, visible_inputs, mini_actions
                        )
                else:
                    log_probs, values, entropies = model.evaluate_actions(
                        animal_inputs, visible_inputs, mini_actions
                    )
                
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
                entropy_loss = -entropies.mean()
                
                # Total loss (divide by accumulation steps to normalize)
                loss = (policy_loss + 
                       config.VALUE_LOSS_COEF * value_loss + 
                       config.ENTROPY_COEF * entropy_loss) / accumulation_steps
                
                # Backward pass (accumulate gradients)
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Accumulate losses for logging
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                n_updates += 1
            
            # Optimizer step after accumulating gradients
            if scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
    
    if n_updates > 0:
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy = total_entropy / n_updates
        return avg_policy_loss, avg_value_loss, avg_entropy
    
    return 0, 0, 0


def run_episode(animals, model_prey, model_predator, pheromone_map, config, steps, device):
    """
    Run a single training episode with advanced features
    
    Returns:
        Episode statistics and memories
    """
    memory_prey = PPOMemory(config.PPO_BATCH_SIZE)
    memory_predator = PPOMemory(config.PPO_BATCH_SIZE)
    
    episode_reward_prey = 0
    episode_reward_predator = 0
    episode_stats = {
        'births': 0,
        'deaths': 0,
        'meals': 0,
        'exhaustion_deaths': 0,
        'old_age_deaths': 0
    }
    
    for step in range(steps):
        step_reward_prey = 0
        step_reward_predator = 0
        animals_to_remove = []
        
        # Log progress every 10 steps
        if (step + 1) % 10 == 0:
            prey_count = sum(1 for a in animals if not a.predator)
            predator_count = sum(1 for a in animals if a.predator)
            timestamp = _ts()
            print(f"[{timestamp}] Step {step + 1}/{steps}: Animals={len(animals)} (Prey={prey_count}, Pred={predator_count})", flush=True)
        
        # Age and energy updates
        for animal in animals:
            animal.update_age()
            
            # Check for old age
            if animal.is_old(config):
                animals_to_remove.append(animal)
                episode_stats['old_age_deaths'] += 1
                continue
        
        # Movement phase with energy costs - BATCHED
        # Filter out animals marked for removal
        active_animals = [a for a in animals if a not in animals_to_remove]
        
        if len(active_animals) > 0:
            # Separate prey and predators for batched processing
            prey_animals = [a for a in active_animals if not a.predator]
            pred_animals = [a for a in active_animals if a.predator]
            
            # Process prey in batch
            if len(prey_animals) > 0:
                # Collect all inputs
                prey_inputs = []
                prey_visible = []
                for animal in prey_animals:
                    animal_input = animal.get_enhanced_input(animals, config, pheromone_map).squeeze(0)  # Remove batch dim
                    visible_animals = animal.communicate(animals, config)
                    visible_animals_input = torch.tensor(visible_animals, dtype=torch.float32)
                    prey_inputs.append(animal_input)
                    prey_visible.append(visible_animals_input)
                
                # Stack into batches
                prey_inputs_batch = torch.stack(prey_inputs).to(device)  # [B, 20]
                prey_visible_batch = torch.stack(prey_visible).to(device)  # [B, N, 8]
                
                # Single batched forward pass
                with torch.no_grad():
                    actions, log_probs, values = model_prey.get_action(prey_inputs_batch, prey_visible_batch)
                
                # Apply actions and collect rewards
                for idx, animal in enumerate(prey_animals):
                    state_tuple = (prey_inputs_batch[idx].unsqueeze(0).detach(), prey_visible_batch[idx].unsqueeze(0).detach())
                    action = actions[idx]
                    log_prob = log_probs[idx]
                    value = values[idx]
                    
                    # Execute movement
                    old_pos = (animal.x, animal.y)
                    animal._apply_action(action.item(), config)
                    moved = (animal.x, animal.y) != old_pos
                    
                    # Update energy
                    animal.update_energy(config, moved)
                    
                    # Check exhaustion
                    if animal.is_exhausted():
                        animals_to_remove.append(animal)
                        episode_stats['exhaustion_deaths'] += 1
                        reward = -10.0
                    else:
                        reward = config.SURVIVAL_REWARD
                        if not moved:
                            reward += 0.1
                        
                        # Reward for avoiding predators (species-aware behavior)
                        threat_info = animal._get_threat_info(animals, config, pheromone_map)
                        nearest_pred_dist = threat_info['nearest_predator_dist']
                        if nearest_pred_dist < 1.0:  # Predator is very close
                            # Calculate distance change (did we move away?)
                            old_dist_to_pred = nearest_pred_dist
                            # Check if we moved away from threat
                            if moved and nearest_pred_dist > 0.1:  # Successfully evading
                                reward += config.PREY_EVASION_REWARD * (1.0 - nearest_pred_dist)
                        
                        # Penalty for overcrowding (respect OTHER_SPECIES_CAPACITY)
                        same_species_count = sum(1 for a in animals if not a.predator)
                        usable_capacity = config.MAX_ANIMALS - config.OTHER_SPECIES_CAPACITY
                        if same_species_count > usable_capacity:
                            overcrowd_ratio = (same_species_count - usable_capacity) / usable_capacity
                            reward += config.OVERPOPULATION_PENALTY * overcrowd_ratio
                    
                    # Store experience
                    memory_prey.add(state_tuple, action, log_prob, value, reward, False)
                    step_reward_prey += reward
            
            # Process predators in batch
            if len(pred_animals) > 0:
                # Collect all inputs
                pred_inputs = []
                pred_visible = []
                for animal in pred_animals:
                    animal_input = animal.get_enhanced_input(animals, config, pheromone_map).squeeze(0)  # Remove batch dim
                    visible_animals = animal.communicate(animals, config)
                    visible_animals_input = torch.tensor(visible_animals, dtype=torch.float32)
                    pred_inputs.append(animal_input)
                    pred_visible.append(visible_animals_input)
                
                # Stack into batches
                pred_inputs_batch = torch.stack(pred_inputs).to(device)  # [B, 20]
                pred_visible_batch = torch.stack(pred_visible).to(device)  # [B, N, 8]
                
                # Single batched forward pass
                with torch.no_grad():
                    actions, log_probs, values = model_predator.get_action(pred_inputs_batch, pred_visible_batch)
                
                # Apply actions and collect rewards
                for idx, animal in enumerate(pred_animals):
                    state_tuple = (pred_inputs_batch[idx].unsqueeze(0).detach(), pred_visible_batch[idx].unsqueeze(0).detach())
                    action = actions[idx]
                    log_prob = log_probs[idx]
                    value = values[idx]
                    
                    # Execute movement
                    old_pos = (animal.x, animal.y)
                    animal._apply_action(action.item(), config)
                    moved = (animal.x, animal.y) != old_pos
                    
                    # Update energy
                    animal.update_energy(config, moved)
                    
                    # Check exhaustion
                    if animal.is_exhausted():
                        animals_to_remove.append(animal)
                        episode_stats['exhaustion_deaths'] += 1
                        reward = -10.0
                    else:
                        reward = config.SURVIVAL_REWARD
                        if not moved:
                            reward += 0.1
                        
                        # Reward for approaching prey (species-aware behavior)
                        threat_info = animal._get_threat_info(animals, config, pheromone_map)
                        nearest_prey_dist = threat_info['nearest_prey_dist']
                        if nearest_prey_dist < 1.0:  # Prey is nearby
                            # Reward getting closer to prey
                            if moved and nearest_prey_dist < 0.5:  # Getting close
                                reward += config.PREDATOR_APPROACH_REWARD * (1.0 - nearest_prey_dist)
                        
                        # Extra reward for hungry predators moving toward prey
                        if animal.steps_since_last_meal > config.HUNGER_THRESHOLD:
                            if nearest_prey_dist < 0.8:
                                reward += 1.0 * (1.0 - nearest_prey_dist)
                        
                        # Penalty for overcrowding (respect OTHER_SPECIES_CAPACITY)
                        same_species_count = sum(1 for a in animals if a.predator)
                        usable_capacity = config.MAX_ANIMALS - config.OTHER_SPECIES_CAPACITY
                        if same_species_count > usable_capacity:
                            overcrowd_ratio = (same_species_count - usable_capacity) / usable_capacity
                            reward += config.OVERPOPULATION_PENALTY * overcrowd_ratio
                    
                    # Store experience
                    memory_predator.add(state_tuple, action, log_prob, value, reward, False)
                    step_reward_predator += reward
        
        # Remove dead animals
        for animal in animals_to_remove:
            if animal in animals:
                animals.remove(animal)
        animals_to_remove.clear()
        
        # Eating phase
        for animal in animals:
            if animal.predator:
                has_eaten, eat_reward = animal.eat_training(animals, config)
                if has_eaten:
                    episode_stats['meals'] += 1
                    episode_stats['deaths'] += 1
                    step_reward_predator += eat_reward
                if not has_eaten:
                    animal.steps_since_last_meal += 1
                    if animal.steps_since_last_meal >= config.STARVATION_THRESHOLD:
                        animals_to_remove.append(animal)
        
        # Remove starved predators
        for animal in animals_to_remove:
            if animal in animals:
                animals.remove(animal)
        
        # Mating phase
        new_animals = []
        mated_animals = set()
        
        for i, animal1 in enumerate(animals):
            if animal1.id in mated_animals or not animal1.can_reproduce(config):
                continue
            
            for animal2 in animals[i+1:]:
                if animal2.id in mated_animals or not animal2.can_reproduce(config):
                    continue
                
                if animal1.can_mate(animal2):
                    mating_prob = (config.MATING_PROBABILITY_PREY 
                                 if animal1.name == "A" 
                                 else config.MATING_PROBABILITY_PREDATOR)
                    
                    if random.random() < mating_prob:
                        # Create offspring
                        child_x = (animal1.x + animal2.x) // 2
                        child_y = (animal1.y + animal2.y) // 2
                        child = Animal(child_x, child_y, animal1.name, animal1.color,
                                     {animal1.id, animal2.id}, animal1.predator)
                        child.energy = config.INITIAL_ENERGY
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
                        
                        # Reproduction reward
                        if animal1.name == "A":
                            step_reward_prey += config.REPRODUCTION_REWARD
                        else:
                            step_reward_predator += config.REPRODUCTION_REWARD
                        
                        break
        
        # Add new animals (reserve capacity for future species)
        max_animals_current = max(0, config.MAX_ANIMALS - config.OTHER_SPECIES_CAPACITY)
        if len(animals) + len(new_animals) <= max_animals_current:
            animals.extend(new_animals)
        else:
            available_slots = max_animals_current - len(animals)
            if available_slots > 0:
                animals.extend(new_animals[:available_slots])
        
        # Update cooldowns
        for animal in animals:
            if animal.mating_cooldown > 0:
                animal.mating_cooldown -= 1
            animal.survival_time += 1
            
            # Deposit pheromones
            animal.deposit_pheromones(pheromone_map, config)
        
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
    
    episode_stats['final_prey'] = sum(1 for a in animals if not a.predator)
    episode_stats['final_predators'] = sum(1 for a in animals if a.predator)
    episode_stats['total_reward_prey'] = episode_reward_prey
    episode_stats['total_reward_predator'] = episode_reward_predator
    
    return memory_prey, memory_predator, episode_stats


def main():
    print("\n" + "="*70, flush=True)
    print("  ADVANCED LIFE GAME TRAINING (PPO + Pheromones + Energy)", flush=True)
    print("="*70, flush=True)
    
    # Check for --cpu flag
    force_cpu = '--cpu' in sys.argv or os.environ.get('FORCE_CPU', '0') == '1'
    
    # Setup device (GPU if available)
    # Priority: DirectML (AMD/Intel) > CUDA (NVIDIA) > CPU
    device = None
    device_name = "cpu"
    
    try:
        import torch_directml
        device = torch_directml.device()
        device_name = "DirectML (AMD/Intel GPU)"
        print(f"Device: {device}")
        print(f"Using: {device_name}")
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
            device = torch.device('cpu')
            device_name = "CPU"
            print(f"Device: {device}", flush=True)
            print("Note: GPU not available, training on CPU", flush=True)
            print("For AMD GPU: pip install torch-directml")
            print("For NVIDIA GPU: Install CUDA-enabled PyTorch from pytorch.org")
    
    # Configuration
    config = SimulationConfig()
    
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
    
    # Optimizers
    optimizer_prey = optim.Adam(model_prey.parameters(), lr=config.LEARNING_RATE_PREY)
    optimizer_predator = optim.Adam(model_predator.parameters(), lr=config.LEARNING_RATE_PREDATOR)
    
    # Initialize pheromone map
    pheromone_map = PheromoneMap(config.GRID_SIZE, 
                                 decay_rate=config.PHEROMONE_DECAY,
                                 diffusion_rate=config.PHEROMONE_DIFFUSION)
    
    # Training parameters (reduced for 2.9M parameter model)
    num_episodes = 50  # Full training run
    steps_per_episode = 200
    
    print(f"\nTraining for {num_episodes} episodes", flush=True)
    print(f"Steps per episode: {steps_per_episode}", flush=True)
    print(f"Using Actor-Critic with PPO algorithm", flush=True)
    print(f"Advanced features: Energy, Age, Pheromones, Multi-Head Attention\n", flush=True)
    
    # Create models directory
    os.makedirs("outputs/checkpoints", exist_ok=True)
    
    best_prey_survival = 0
    
    for episode in range(1, num_episodes + 1):
        timestamp = _ts()
        print(f"\n[{timestamp}] Episode {episode}/{num_episodes}", flush=True)
        sys.stdout.flush()  # Force immediate write
        
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
        print(f"[{timestamp}] Starting PPO update (Prey experiences={len(memory_prey.states)}, Predator={len(memory_predator.states)})...", flush=True)
        sys.stdout.flush()
        
        # PPO updates with timing
        use_amp = False  # Disable mixed precision for ROCm compatibility
        gpu_start = time.time()
        
        # Use gradient accumulation to reduce peak memory usage
        accumulation_steps = 1  # Single large mini-batch for maximum GPU work
        
        # ROCm Fix: Force SDPA to use math backend (not flash attention)
        # This avoids attention kernel hangs on Windows ROCm
        if SDPA_AVAILABLE:
            with sdpa_kernel(SDPBackend.MATH):
                policy_loss_prey, value_loss_prey, entropy_prey = ppo_update(
                    model_prey, optimizer_prey, memory_prey, config, device, use_amp, accumulation_steps
                )
                policy_loss_pred, value_loss_pred, entropy_pred = ppo_update(
                    model_predator, optimizer_predator, memory_predator, config, device, use_amp, accumulation_steps
                )
        else:
            # Fallback for older PyTorch: disable flash/mem_efficient SDPA
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            policy_loss_prey, value_loss_prey, entropy_prey = ppo_update(
                model_prey, optimizer_prey, memory_prey, config, device, use_amp, accumulation_steps
            )
            policy_loss_pred, value_loss_pred, entropy_pred = ppo_update(
                model_predator, optimizer_predator, memory_predator, config, device, use_amp, accumulation_steps
            )
        
        if device.type == 'cuda':
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
        
        if torch.cuda.is_available() and device.type == 'cuda':
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1e9
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1e9
            timestamp = _ts()
            print(f"[{timestamp}] GPU Memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved", flush=True)
        
        # Print stats
        timestamp = _ts()
        print(f"[{timestamp}] Final: Prey={stats['final_prey']}, Predators={stats['final_predators']}", flush=True)
        print(f"[{timestamp}] Births={stats['births']}, Deaths={stats['deaths']}, Meals={stats['meals']}", flush=True)
        print(f"[{timestamp}] Exhaustion={stats['exhaustion_deaths']}, Old Age={stats['old_age_deaths']}", flush=True)
        print(f"[{timestamp}] Rewards: Prey={stats['total_reward_prey']:.1f}, Predator={stats['total_reward_predator']:.1f}", flush=True)
        print(f"[{timestamp}] Losses: Policy(P={policy_loss_prey:.3f}/Pr={policy_loss_pred:.3f}), "
              f"Value(P={value_loss_prey:.3f}/Pr={value_loss_pred:.3f})", flush=True)
        
        # Save best model
        if stats['final_prey'] > best_prey_survival:
            best_prey_survival = stats['final_prey']
            torch.save(model_prey.state_dict(), "outputs/checkpoints/model_A_ppo.pth")
            torch.save(model_predator.state_dict(), "outputs/checkpoints/model_B_ppo.pth")
        timestamp = _ts()
        print(f"[{timestamp}] * New best! Saved models")
        
        # Save checkpoint every 5 episodes
        if episode % 5 == 0:
            torch.save(model_prey.state_dict(), f"outputs/checkpoints/model_A_ppo_ep{episode}.pth")
            torch.save(model_predator.state_dict(), f"outputs/checkpoints/model_B_ppo_ep{episode}.pth")
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
