"""
Animal class for the Life Game simulation
"""

import random
import torch
import torch.nn as nn
from typing import List, Set, Optional, Tuple, Dict


class Animal:
    """Represents an individual animal (prey or predator) in the simulation"""
    _next_id = 1

    def __init__(self, x: int, y: int, name: str, color: str, 
                 parent_ids: Optional[Set[int]] = None, 
                 predator: bool = False) -> None:
        self.id = Animal._next_id
        Animal._next_id += 1
        self.x = x
        self.y = y
        self.name = name
        self.color = color
        self.parent_ids = parent_ids if parent_ids and len(parent_ids) <= 2 else set()
        self.predator = predator
        self.steps_since_last_meal = 0
        self.mating_cooldown = 0
        self.survival_time = 0
        self.num_children = 0

    def move(self, model: nn.Module, animals: List['Animal'], config) -> None:
        """Move the animal based on neural network decision (inference only)"""
        if self.predator:
            moves = (config.PREDATOR_HUNGRY_MOVES 
                    if self.steps_since_last_meal >= config.HUNGER_THRESHOLD 
                    else config.PREDATOR_NORMAL_MOVES)
        else:
            moves = config.PREY_MOVES
        
        # Get visible animals
        visible_animals = self.communicate(animals, config)
        
        with torch.no_grad():  # No gradient computation
            animal_input = torch.tensor(
                [self.x / config.GRID_SIZE, self.y / config.GRID_SIZE, 
                 int(self.name == 'A'), int(self.name == 'B')], 
                dtype=torch.float32
            ).unsqueeze(0)
            visible_animals_input = torch.tensor(
                visible_animals, dtype=torch.float32
            ).unsqueeze(0)
            
            # Get action probabilities from model
            action_prob = model(animal_input, visible_animals_input)
        
        for _ in range(moves):
            # Sample action (no log prob needed for inference)
            action_idx = torch.multinomial(action_prob, 1).item()
            
            # Convert action to movement
            new_x, new_y = self.x, self.y
            
            # Find nearest prey for predators
            if self.predator:
                nearest_prey = self._find_nearest_prey(animals)
                if nearest_prey:
                    dx = nearest_prey.x - self.x
                    dy = nearest_prey.y - self.y
                    if abs(dx) > abs(dy):
                        new_x = (self.x + (1 if dx > 0 else -1)) % config.GRID_SIZE
                    else:
                        new_y = (self.y + (1 if dy > 0 else -1)) % config.GRID_SIZE
                else:
                    new_x, new_y = self._apply_action(action_idx, config)
            else:
                new_x, new_y = self._apply_action(action_idx, config)

            if not self._position_occupied(animals, new_x, new_y):
                self.x, self.y = new_x, new_y

    def _apply_action(self, action: int, config) -> Tuple[int, int]:
        """Apply action to get new position with wrapping"""
        new_x, new_y = self.x, self.y
        
        if action == 0:
            new_y = (self.y + 1) % config.GRID_SIZE
        elif action == 1:
            new_y = (self.y - 1) % config.GRID_SIZE
        elif action == 2:
            new_x = (self.x - 1) % config.GRID_SIZE
        elif action == 3:
            new_x = (self.x + 1) % config.GRID_SIZE
        elif action == 4:
            new_x = (self.x + 1) % config.GRID_SIZE
            new_y = (self.y + 1) % config.GRID_SIZE
        elif action == 5:
            new_x = (self.x + 1) % config.GRID_SIZE
            new_y = (self.y - 1) % config.GRID_SIZE
        elif action == 6:
            new_x = (self.x - 1) % config.GRID_SIZE
            new_y = (self.y + 1) % config.GRID_SIZE
        elif action == 7:
            new_x = (self.x - 1) % config.GRID_SIZE
            new_y = (self.y - 1) % config.GRID_SIZE
            
        return new_x, new_y

    def _find_nearest_prey(self, animals: List['Animal']) -> Optional['Animal']:
        """Find the nearest prey animal"""
        nearest = None
        min_distance = float('inf')
        
        for animal in animals:
            if not animal.predator:
                dx = abs(animal.x - self.x)
                dy = abs(animal.y - self.y)
                distance = dx + dy
                if distance < min_distance:
                    min_distance = distance
                    nearest = animal
        
        return nearest

    def move_away(self, config) -> None:
        """Move to a random adjacent position"""
        move_directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        random.shuffle(move_directions)
        for dx, dy in move_directions:
            new_x = (self.x + dx) % config.GRID_SIZE
            new_y = (self.y + dy) % config.GRID_SIZE
            self.x = new_x
            self.y = new_y
            break

    def can_mate(self, other: 'Animal') -> bool:
        """Check if this animal can mate with another"""
        return (
            self.name == other.name
            and abs(self.x - other.x) <= 1
            and abs(self.y - other.y) <= 1
            and self.id not in other.parent_ids
            and other.id not in self.parent_ids
            and self.mating_cooldown == 0
            and other.mating_cooldown == 0
        )

    def communicate(self, animals: List['Animal'], config) -> List[List[float]]:
        """Get information about nearby visible animals with padding"""
        visible_animals = []

        for animal in animals:
            if animal != self and len(visible_animals) < config.MAX_VISIBLE_ANIMALS:
                dx = abs(animal.x - self.x)
                dy = abs(animal.y - self.y)

                if dx <= config.VISION_RANGE and dy <= config.VISION_RANGE:
                    visible_animals.append([
                        animal.x / config.GRID_SIZE, 
                        animal.y / config.GRID_SIZE, 
                        float(animal.name == 'A'), 
                        float(animal.name == 'B')
                    ])

        while len(visible_animals) < config.MAX_VISIBLE_ANIMALS:
            visible_animals.append([0.0, 0.0, 0.0, 0.0])

        return visible_animals[:config.MAX_VISIBLE_ANIMALS]

    def eat(self, animals: List['Animal'], stats: Dict = None) -> bool:
        """Predator attempts to eat nearby prey"""
        if not self.predator:
            return False
        
        prey_to_eat = None
        for prey in animals:
            if not prey.predator and abs(self.x - prey.x) <= 1 and abs(self.y - prey.y) <= 1:
                prey_to_eat = prey
                break
        
        if prey_to_eat:
            animals.remove(prey_to_eat)
            self.steps_since_last_meal = 0
            if stats is not None:
                stats['total_deaths'] += 1  # Track prey death
            return True
        
        return False

    def display_color(self, config) -> str:
        """Get color for visualization"""
        if self.predator and self.steps_since_last_meal >= config.HUNGER_THRESHOLD:
            return 'darkred'
        return self.color

    def _position_occupied(self, animals: List['Animal'], 
                          new_x: int, new_y: int) -> bool:
        """Check if a position is occupied"""
        for animal in animals:
            if animal != self and animal.x == new_x and animal.y == new_y:
                return True
        return False
