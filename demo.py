"""
Life Game Demo - Visual Simulation
Shows trained Actor-Critic agents in action with pheromones, energy, and age systems
"""

import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from src.config import SimulationConfig
from src.animal import Animal
from src.actor_critic_network import ActorCriticNetwork
from src.pheromone_system import PheromoneMap


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


def run_demo():
    """Run visual demo of trained models"""
    
    print("\n" + "=" * 70)
    print("  LIFE GAME - VISUAL DEMO")
    print("  Actor-Critic PPO Models with Advanced Features")
    print("=" * 70)
    
    # Initialize
    config = SimulationConfig()
    device = torch.device('cpu')
    
    # Load models
    print("\nLoading models...")
    model_prey = ActorCriticNetwork(config).to(device)
    model_predator = ActorCriticNetwork(config).to(device)
    
    try:
        model_prey.load_state_dict(torch.load('models/model_A_ppo.pth', map_location=device))
        model_predator.load_state_dict(torch.load('models/model_B_ppo.pth', map_location=device))
        print("✓ Loaded trained PPO models")
    except FileNotFoundError:
        print("! No trained models found, using random initialization")
    
    model_prey.eval()
    model_predator.eval()
    
    # Create population and pheromone map
    animals = create_population(config)
    pheromone_map = PheromoneMap(config)
    
    # Setup visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(config.FIELD_MIN, config.FIELD_MAX)
    ax.set_ylim(config.FIELD_MIN, config.FIELD_MAX)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    step_count = [0]
    
    def update(frame):
        """Animation update function"""
        ax.clear()
        ax.set_xlim(config.FIELD_MIN, config.FIELD_MAX)
        ax.set_ylim(config.FIELD_MIN, config.FIELD_MAX)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Simulation step
        for animal in animals[:]:
            if not animal.is_alive():
                animals.remove(animal)
                continue
            
            # Age and energy updates
            animal.age += 1
            animal.energy -= config.ENERGY_DECAY_RATE
            
            # Check death conditions
            if animal.energy <= 0 or animal.age >= config.MAX_AGE:
                animals.remove(animal)
                continue
            
            # Get visible animals
            visible = [a for a in animals if a != animal and 
                      abs(a.x - animal.x) <= config.VISION_RANGE and 
                      abs(a.y - animal.y) <= config.VISION_RANGE]
            
            # Prepare inputs
            animal_input = animal.get_state_vector(config)
            visible_animals_input = [a.get_state_vector(config) for a in visible[:config.MAX_VISIBLE_ANIMALS]]
            
            # Pad if needed
            while len(visible_animals_input) < config.MAX_VISIBLE_ANIMALS:
                visible_animals_input.append([0] * config.VISION_INPUT_SIZE)
            
            # Get action from model
            model = model_prey if not animal.predator else model_predator
            with torch.no_grad():
                action_probs, _ = model(animal_input, visible_animals_input)
                action = torch.argmax(action_probs).item()
            
            # Execute action (8-direction movement)
            dx_map = [-1, -1, -1, 0, 0, 1, 1, 1]
            dy_map = [-1, 0, 1, -1, 1, -1, 0, 1]
            
            if action < 8:
                animal.x = max(config.FIELD_MIN, min(config.FIELD_MAX, animal.x + dx_map[action]))
                animal.y = max(config.FIELD_MIN, min(config.FIELD_MAX, animal.y + dy_map[action]))
            
            # Leave pheromone
            if not animal.predator:
                pheromone_map.add_pheromone(animal.x, animal.y, 'prey', 1.0)
            else:
                pheromone_map.add_pheromone(animal.x, animal.y, 'predator', 1.0)
        
        # Decay pheromones
        pheromone_map.decay()
        
        # Handle interactions
        for animal in animals[:]:
            if not animal in animals:
                continue
            
            if animal.predator:
                # Predator hunts prey
                for prey in animals[:]:
                    if not prey.predator and abs(prey.x - animal.x) <= 1 and abs(prey.y - animal.y) <= 1:
                        animals.remove(prey)
                        animal.energy = min(config.MAX_ENERGY, animal.energy + config.ENERGY_FROM_FOOD)
                        break
            else:
                # Prey reproduction
                if animal.energy >= config.REPRODUCTION_ENERGY_THRESHOLD:
                    neighbors = [a for a in animals if not a.predator and 
                               abs(a.x - animal.x) <= config.REPRODUCTION_RANGE and 
                               abs(a.y - animal.y) <= config.REPRODUCTION_RANGE]
                    if len(neighbors) >= config.REPRODUCTION_MIN_NEIGHBORS:
                        child = Animal(animal.x + random.randint(-1, 1), 
                                     animal.y + random.randint(-1, 1), 
                                     "A", "#00ff00", predator=False)
                        child.energy = config.INITIAL_ENERGY
                        animals.append(child)
                        animal.energy -= config.REPRODUCTION_ENERGY_COST
        
        # Draw animals
        for animal in animals:
            color = '#00ff00' if not animal.predator else '#ff0000'
            size = 100 if not animal.predator else 150
            ax.scatter(animal.x, animal.y, c=color, s=size, alpha=0.7)
        
        # Statistics
        prey_count = sum(1 for a in animals if not a.predator)
        predator_count = sum(1 for a in animals if a.predator)
        step_count[0] += 1
        
        ax.set_title(f'Step {step_count[0]} | Prey: {prey_count} | Predators: {predator_count}', 
                    fontsize=14, fontweight='bold')
        
        return ax.patches
    
    print("\nStarting visualization...")
    print("Close the window to exit")
    
    anim = FuncAnimation(fig, update, frames=1000, interval=100, blit=False, repeat=True)
    plt.show()


if __name__ == "__main__":
    run_demo()
