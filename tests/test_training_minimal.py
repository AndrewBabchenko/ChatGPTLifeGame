"""Minimal test to reproduce training issue"""
import torch
import torch.optim as optim
import sys
sys.path.insert(0, '.')

from src.config import SimulationConfig
from src.models.actor_critic_network import ActorCriticNetwork
from src.models.replay_buffer import PPOMemory
from src.core.animal import Animal
from src.core.pheromone_system import PheromoneMap

print("Testing minimal training loop...")

# Setup
config = SimulationConfig()
device = torch.device('cpu')
print(f"Device: {device}")

# Create model
model = ActorCriticNetwork(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_PREY)
print(f"Model created: {sum(p.numel() for p in model.parameters())} params")

# Create minimal memory
memory = PPOMemory(config.PPO_BATCH_SIZE)
pheromone_map = PheromoneMap(config.GRID_SIZE, 
                             decay_rate=config.PHEROMONE_DECAY,
                             diffusion_rate=config.PHEROMONE_DIFFUSION)

# Create one animal
animal = Animal(50, 50, "A", "#00ff00", predator=False)
animal.energy = config.INITIAL_ENERGY
animals = [animal]

print("\nGenerating experiences...")
# Generate a few experiences
for step in range(5):
    animal_input = animal.get_enhanced_input(animals, config, pheromone_map).to(device)
    visible_animals = animal.communicate(animals, config)
    visible_animals_input = torch.tensor(visible_animals, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        action, log_prob, value = model.get_action(animal_input, visible_animals_input)
    
    state_tuple = (animal_input.detach(), visible_animals_input.detach())
    reward = 1.0
    done = False
    
    memory.add(state_tuple, action, log_prob, value, reward, done)
    print(f"  Step {step+1}: action={action.item()}, log_prob={log_prob.item():.3f}, value={value.item():.3f}")

print(f"\nMemory size: {len(memory.states)} experiences")

# Compute returns and advantages
print("\nComputing returns and advantages...")
returns, advantages = memory.compute_returns_and_advantages(
    torch.tensor([0.0]), config.GAMMA, config.GAE_LAMBDA
)
print(f"  Returns shape: {returns.shape}, Advantages shape: {advantages.shape}")

# Flatten and store
memory.returns = returns.flatten().tolist()
memory.advantages = advantages.flatten().tolist()

print("\nPerforming PPO update...")
# Get batch
for batch in memory.get_batches():
    print(f"  Batch size: {len(batch['states'])}")
    
    states = batch['states']
    actions = batch['actions'].to(device)
    old_log_probs = batch['old_log_probs'].to(device)
    returns_batch = batch['returns'].view(-1).to(device)
    advantages_batch = batch['advantages'].view(-1).to(device)
    
    print(f"  Actions: {actions}")
    print(f"  Old log probs shape: {old_log_probs.shape}")
    print(f"  Returns shape: {returns_batch.shape}")
    print(f"  Advantages shape: {advantages_batch.shape}")
    
    # Evaluate actions
    all_log_probs = []
    all_values = []
    all_entropies = []
    
    print("\n  Evaluating actions...")
    for idx, state_tuple in enumerate(states):
        animal_input, visible_animals_input = state_tuple
        action = actions[idx] if actions.dim() > 0 else actions
        
        print(f"    State {idx+1}: animal_input shape={animal_input.shape}, visible_animals shape={visible_animals_input.shape}, action={action}")
        
        log_prob, value, entropy = model.evaluate_actions(
            animal_input, visible_animals_input, action
        )
        
        print(f"      -> log_prob={log_prob.item():.3f}, value={value.item():.3f}, entropy={entropy.item():.3f}")
        
        all_log_probs.append(log_prob)
        all_values.append(value.squeeze())
        all_entropies.append(entropy)
    
    log_probs = torch.stack(all_log_probs).view(-1)
    values = torch.stack(all_values).view(-1)
    entropies = torch.stack(all_entropies)
    
    print(f"\n  Stacked log_probs shape: {log_probs.shape}")
    print(f"  Stacked values shape: {values.shape}")
    print(f"  Stacked entropies shape: {entropies.shape}")
    
    # Compute loss
    print("\n  Computing loss...")
    ratio = torch.exp(log_probs - old_log_probs)
    print(f"    Ratio: {ratio}")
    
    surr1 = ratio * advantages_batch
    surr2 = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPSILON, 
                       1.0 + config.PPO_CLIP_EPSILON) * advantages_batch
    
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = torch.nn.MSELoss()(values, returns_batch)
    entropy_loss = -entropies.mean()
    
    loss = (policy_loss + 
           config.VALUE_LOSS_COEF * value_loss + 
           config.ENTROPY_COEF * entropy_loss)
    
    print(f"    Policy loss: {policy_loss.item():.4f}")
    print(f"    Value loss: {value_loss.item():.4f}")
    print(f"    Entropy loss: {entropy_loss.item():.4f}")
    print(f"    Total loss: {loss.item():.4f}")
    
    # Backward
    print("\n  Performing backward pass...")
    optimizer.zero_grad()
    loss.backward()
    print("  Backward pass completed!")
    
    optimizer.step()
    print("  Optimizer step completed!")
    
    break  # Only test first batch

print("\n✓ Training loop test completed successfully!")
