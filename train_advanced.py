"""
Advanced Training Script with PPO Algorithm
Integrates Actor-Critic network, pheromones, energy, and age systems
"""

import torch
import torch.optim as optim
import torch.nn as nn
import random
import os

from src.config import SimulationConfig
from src.animal import Animal
from src.actor_critic_network import ActorCriticNetwork
from src.pheromone_system import PheromoneMap
from src.replay_buffer import PPOMemory

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


def ppo_update(model, optimizer, memory, config, device):
    """
    Perform PPO update on the model
    
    Args:
        model: Actor-Critic network
        optimizer: PyTorch optimizer
        memory: PPOMemory with stored experiences
        config: SimulationConfig
        device: torch.device for GPU/CPU
    """
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
    
    for epoch in range(config.PPO_EPOCHS):
        for batch in memory.get_batches():
            # Extract batch data
            states = batch['states']
            actions = batch['actions']
            old_log_probs = batch['old_log_probs']
            returns_batch = batch['returns'].view(-1)  # Flatten to 1D
            advantages_batch = batch['advantages'].view(-1)  # Flatten to 1D
            
            # Evaluate actions with current policy
            all_log_probs = []
            all_values = []
            all_entropies = []
            
            for idx, state_tuple in enumerate(states):
                animal_input, visible_animals_input = state_tuple
                action = actions[idx] if actions.dim() > 0 else actions
                log_prob, value, entropy = model.evaluate_actions(
                    animal_input, visible_animals_input, action
                )
                all_log_probs.append(log_prob)
                all_values.append(value.squeeze())  # Remove extra dimensions
                all_entropies.append(entropy)
            
            log_probs = torch.stack(all_log_probs)
            values = torch.stack(all_values)  # Stack instead of cat to maintain shape
            entropies = torch.stack(all_entropies)
            
            # Ensure all tensors are 1D for loss calculation
            log_probs = log_probs.view(-1)
            values = values.view(-1)
            old_log_probs = old_log_probs.view(-1)
            
            # Compute PPO loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPSILON, 
                               1.0 + config.PPO_CLIP_EPSILON) * advantages_batch
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with proper shapes
            value_loss = nn.MSELoss()(values, returns_batch)
            entropy_loss = -entropies.mean()
            
            # Total loss
            loss = (policy_loss + 
                   config.VALUE_LOSS_COEF * value_loss + 
                   config.ENTROPY_COEF * entropy_loss)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropies.mean().item()
            n_updates += 1
    
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
        
        # Age and energy updates
        for animal in animals:
            animal.update_age()
            
            # Check for old age
            if animal.is_old(config):
                animals_to_remove.append(animal)
                episode_stats['old_age_deaths'] += 1
                continue
        
        # Movement phase with energy costs
        for animal in animals:
            if animal in animals_to_remove:
                continue
                
            # Get state before action
            animal_input = animal.get_enhanced_input(animals, config, pheromone_map).to(device)
            visible_animals = animal.communicate(animals, config)
            visible_animals_input = torch.tensor(visible_animals, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Select model
            model = model_prey if animal.name == "A" else model_predator
            memory = memory_prey if animal.name == "A" else memory_predator
            
            # Get action and value
            with torch.no_grad():
                action, log_prob, value = model.get_action(animal_input, visible_animals_input)
            
            # Store state
            state_tuple = (animal_input.detach(), visible_animals_input.detach())
            
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
                reward = -10.0  # Penalty for exhaustion
            else:
                reward = config.SURVIVAL_REWARD
                # Energy efficiency bonus
                if not moved:
                    reward += 0.1
            
            # Store experience
            memory.add(state_tuple, action, log_prob, value, reward, False)
            
            if animal.name == "A":
                step_reward_prey += reward
            else:
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
        
        # Add new animals
        if len(animals) + len(new_animals) <= config.MAX_ANIMALS:
            animals.extend(new_animals)
        else:
            animals.extend(new_animals[:config.MAX_ANIMALS - len(animals)])
        
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
            print(f"  Episode ended at step {step + 1}: All animals extinct")
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
    print("\n" + "="*70)
    print("  ADVANCED LIFE GAME TRAINING (PPO + Pheromones + Energy)")
    print("="*70)
    
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
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = f"CUDA - {torch.cuda.get_device_name(0)}"
            print(f"Device: {device}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device('cpu')
            device_name = "CPU"
            print(f"Device: {device}")
            print("Note: GPU not available, training on CPU")
            print("For AMD GPU: pip install torch-directml")
            print("For NVIDIA GPU: Install CUDA-enabled PyTorch from pytorch.org")
    
    # Configuration
    config = SimulationConfig()
    
    # Aggressive CPU Optimizations
    import os
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
    print(f"\nModel Size: {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Optimizers
    optimizer_prey = optim.Adam(model_prey.parameters(), lr=config.LEARNING_RATE_PREY)
    optimizer_predator = optim.Adam(model_predator.parameters(), lr=config.LEARNING_RATE_PREDATOR)
    
    # Initialize pheromone map
    pheromone_map = PheromoneMap(config.GRID_SIZE, 
                                 decay_rate=config.PHEROMONE_DECAY,
                                 diffusion_rate=config.PHEROMONE_DIFFUSION)
    
    # Training parameters
    num_episodes = 100
    steps_per_episode = 200
    
    print(f"\nTraining for {num_episodes} episodes")
    print(f"Steps per episode: {steps_per_episode}")
    print(f"Using Actor-Critic with PPO algorithm")
    print(f"Advanced features: Energy, Age, Pheromones, Multi-Head Attention\n")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    best_prey_survival = 0
    
    for episode in range(1, num_episodes + 1):
        print(f"\nEpisode {episode}/{num_episodes}")
        
        # Create fresh population
        animals = create_population(config)
        pheromone_map.reset()
        
        # Run episode
        memory_prey, memory_predator, stats = run_episode(
            animals, model_prey, model_predator, pheromone_map, config, steps_per_episode, device
        )
        
        # PPO updates
        policy_loss_prey, value_loss_prey, entropy_prey = ppo_update(
            model_prey, optimizer_prey, memory_prey, config, device
        )
        policy_loss_pred, value_loss_pred, entropy_pred = ppo_update(
            model_predator, optimizer_predator, memory_predator, config, device
        )
        
        # Print stats
        print(f"  Final: Prey={stats['final_prey']}, Predators={stats['final_predators']}")
        print(f"  Births={stats['births']}, Deaths={stats['deaths']}, Meals={stats['meals']}")
        print(f"  Exhaustion={stats['exhaustion_deaths']}, Old Age={stats['old_age_deaths']}")
        print(f"  Rewards: Prey={stats['total_reward_prey']:.1f}, Predator={stats['total_reward_predator']:.1f}")
        print(f"  Losses: Policy(P={policy_loss_prey:.3f}/Pr={policy_loss_pred:.3f}), "
              f"Value(P={value_loss_prey:.3f}/Pr={value_loss_pred:.3f})")
        
        # Save best model
        if stats['final_prey'] > best_prey_survival:
            best_prey_survival = stats['final_prey']
            torch.save(model_prey.state_dict(), "models/model_A_ppo.pth")
            torch.save(model_predator.state_dict(), "models/model_B_ppo.pth")
            print(f"  ✓ New best! Saved models")
        
        # Save checkpoint every 10 episodes
        if episode % 10 == 0:
            torch.save(model_prey.state_dict(), f"models/model_A_ppo_ep{episode}.pth")
            torch.save(model_predator.state_dict(), f"models/model_B_ppo_ep{episode}.pth")
            print(f"  Checkpoint saved (episode {episode})")
    
    print("\n" + "="*70)
    print("  TRAINING COMPLETE!")
    print("="*70)
    print(f"Best prey survival: {best_prey_survival}")
    print(f"Models saved to: models/model_A_ppo.pth, models/model_B_ppo.pth")


if __name__ == "__main__":
    main()
