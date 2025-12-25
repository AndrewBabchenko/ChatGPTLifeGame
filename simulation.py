"""
Simulation engine for the Life Game
Handles the main simulation loop and game logic
"""

import random
import time
from typing import List, Dict, Tuple
from collections import defaultdict
import torch
import torch.nn as nn
from animal import Animal
from visualizer import GameFieldVisualizer


def build_spatial_grid(animals: List[Animal], config) -> Dict[Tuple[int, int], List[Animal]]:
    """Build spatial grid for efficient neighbor queries"""
    grid = defaultdict(list)
    cell_size = 5
    
    for animal in animals:
        grid_key = (animal.x // cell_size, animal.y // cell_size)
        grid[grid_key].append(animal)
    
    return grid


def run_simulation(animals: List[Animal], steps: int, 
                   model_prey: nn.Module, model_predator: nn.Module,
                   config) -> Dict:
    """Run simulation in inference mode (no training)
    
    Returns:
        Dictionary containing detailed statistics about the simulation run
    """
    start_time = time.time()
    
    # Initialize statistics tracking
    stats = {
        'total_births': 0,
        'total_deaths': 0,
        'total_meals': 0,
        'peak_prey': len([a for a in animals if not a.predator]),
        'peak_predators': len([a for a in animals if a.predator]),
        'min_prey': len([a for a in animals if not a.predator]),
    }
    
    # Create game field visualizer
    viz = GameFieldVisualizer(config)
    
    for step in range(steps):
        # Check for restart request
        if viz.restart_requested:
            print("\n⟳ Restart requested by user")
            stats['restart_requested'] = True
            break
        
        # Handle pause
        while viz.paused:
            import matplotlib.pyplot as plt
            plt.pause(0.1)
            if viz.restart_requested:
                print("\n⟳ Restart requested by user")
                stats['restart_requested'] = True
                break
        
        if stats.get('restart_requested'):
            break
        
        # Track animals to remove (starvation)
        animals_to_remove = []
        
        # === MOVEMENT PHASE ===
        for animal in animals:
            if animal.name == "A":
                animal.move(model_prey, animals, config)
            else:
                animal.move(model_predator, animals, config)

            # === EATING PHASE ===
            has_eaten = animal.eat(animals, stats)
            
            if animal.predator:
                if has_eaten:
                    stats['total_meals'] += 1
                else:
                    animal.steps_since_last_meal += 1
                    
                    # Mark starved predators for removal
                    if animal.steps_since_last_meal >= config.STARVATION_THRESHOLD:
                        animals_to_remove.append(animal)

        # Remove starved predators and track deaths
        for animal in animals_to_remove:
            if animal in animals:
                animals.remove(animal)
                stats['total_deaths'] += 1

        # === MATING PHASE ===
        spatial_grid = build_spatial_grid(animals, config)
        new_animals = []
        mated_animals = set()
        
        for cell_animals in spatial_grid.values():
            for i, animal1 in enumerate(cell_animals):
                if animal1.id in mated_animals:
                    continue
                    
                for animal2 in cell_animals[i+1:]:
                    if animal2.id in mated_animals:
                        continue
                        
                    if animal1.can_mate(animal2):
                        mating_prob = (config.MATING_PROBABILITY_PREY 
                                     if animal1.name == "A" 
                                     else config.MATING_PROBABILITY_PREDATOR)
                        
                        if random.random() < mating_prob:
                            # Create offspring
                            child_x = (animal1.x + animal2.x) // 2
                            child_y = (animal1.y + animal2.y) // 2
                            child_parent_ids = {animal1.id, animal2.id}
                            
                            new_animal = Animal(child_x, child_y, animal1.name, 
                                              animal1.color, child_parent_ids, 
                                              animal1.predator)
                            new_animals.append(new_animal)
                            stats['total_births'] += 1
                            
                            # Update parents
                            animal1.move_away(config)
                            animal2.move_away(config)
                            animal1.mating_cooldown = config.MATING_COOLDOWN
                            animal2.mating_cooldown = config.MATING_COOLDOWN
                            animal1.num_children += 1
                            animal2.num_children += 1
                            
                            mated_animals.add(animal1.id)
                            mated_animals.add(animal2.id)
                            
                            break

        # Add new animals (up to max capacity)
        animals_added = min(len(new_animals), config.MAX_ANIMALS - len(animals))
        if animals_added > 0:
            animals.extend(new_animals[:animals_added])

        # === UPDATE PHASE ===
        # Update cooldowns and survival time
        for animal in animals:
            if animal.mating_cooldown > 0:
                animal.mating_cooldown -= 1
            animal.survival_time += 1
        
        # Track population peaks and minimums
        prey_count = sum(1 for a in animals if not a.predator)
        predator_count = sum(1 for a in animals if a.predator)
        stats['peak_prey'] = max(stats['peak_prey'], prey_count)
        stats['peak_predators'] = max(stats['peak_predators'], predator_count)
        stats['min_prey'] = min(stats['min_prey'], prey_count)

        # === VISUALIZATION ===
        viz.update(animals, step, stats)
        
        # Console output every 20 steps
        if step % 20 == 0:
            print(f"Step {step:4d}: Prey={prey_count:3d}, Predators={predator_count:3d}, "
                  f"Births={stats['total_births']:3d}, Deaths={stats['total_deaths']:3d}")
    
    # Calculate final statistics
    end_time = time.time()
    stats['total_steps'] = steps
    stats['duration'] = end_time - start_time
    stats['final_prey'] = sum(1 for a in animals if not a.predator)
    stats['final_predators'] = sum(1 for a in animals if a.predator)
    stats['viz'] = viz  # Store visualizer for final stats display
    
    return stats


def create_population(config) -> List[Animal]:
    """Create initial population of animals"""
    animals = []
    
    # Create prey - spread across entire map
    for _ in range(config.INITIAL_PREY_COUNT):
        x = random.randint(5, 95)
        y = random.randint(5, 95)
        animals.append(Animal(x, y, 'A', 'green', predator=False))
    
    # Create predators - spread across map
    for _ in range(config.INITIAL_PREDATOR_COUNT):
        x = random.randint(5, 95)
        y = random.randint(5, 95)
        animals.append(Animal(x, y, 'B', 'red', predator=True))
    
    return animals
