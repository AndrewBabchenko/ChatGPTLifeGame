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
        
        # Advanced features
        self.age = 0  # Age in simulation steps
        self.energy = 100.0  # Energy/stamina (0-100)
        self.max_energy = 100.0
        self.experience = 0.0  # Accumulated experience (affects learning)
        self.successful_hunts = 0  # For predators
        self.successful_evasions = 0  # For prey
        self.last_danger_time = -999  # Last time saw predator (for pheromone trails)

    def get_enhanced_input(self, animals: List['Animal'], config, pheromone_map=None) -> torch.Tensor:
        """
        Build enhanced input vector with all 20 features
        Includes position, state, threat info, age, energy, and pheromones
        """
        threat_info = self._get_threat_info(animals, config, pheromone_map)
        
        # 20 features for actor-critic network
        features = [
            self.x / config.GRID_SIZE,  # 0: x position
            self.y / config.GRID_SIZE,  # 1: y position
            int(self.name == 'A'),  # 2: species A
            int(self.name == 'B'),  # 3: species B
            int(self.predator),  # 4: is predator
            self.steps_since_last_meal / config.STARVATION_THRESHOLD if self.predator else 0.0,  # 5: hunger
            self.mating_cooldown / config.MATING_COOLDOWN,  # 6: mating readiness
            threat_info['nearest_predator_dist'],  # 7: nearest threat distance
            threat_info['nearest_predator_dx'],  # 8: threat direction x
            threat_info['nearest_predator_dy'],  # 9: threat direction y
            threat_info['nearest_prey_dist'],  # 10: nearest prey distance
            threat_info['nearest_prey_dx'],  # 11: prey direction x
            threat_info['nearest_prey_dy'],  # 12: prey direction y
            threat_info['predator_count'] / config.MAX_VISIBLE_ANIMALS,  # 13: visible predators
            threat_info['prey_count'] / config.MAX_VISIBLE_ANIMALS,  # 14: visible prey
            self.age / config.MAX_AGE,  # 15: age (normalized)
            self.energy / config.MAX_ENERGY,  # 16: energy level
            threat_info.get('pheromone_danger', 0.0),  # 17: danger pheromone
            threat_info.get('pheromone_mating', 0.0),  # 18: mating pheromone
            threat_info.get('pheromone_food', 0.0),  # 19: food pheromone
        ]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def move(self, model: nn.Module, animals: List['Animal'], config, pheromone_map=None) -> None:
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
            # Get enhanced 20-feature input
            animal_input = self.get_enhanced_input(animals, config, pheromone_map)
            visible_animals_input = torch.tensor(
                visible_animals, dtype=torch.float32
            ).unsqueeze(0)
            
            # Get action probabilities from model (handle both old SimpleNN and new ActorCritic)
            model_output = model(animal_input, visible_animals_input)
            if isinstance(model_output, tuple):  # Actor-Critic returns (probs, value)
                action_prob, _ = model_output
            else:  # Old SimpleNN returns just probs
                action_prob = model_output
        
        for _ in range(moves):
            # Sample action (no log prob needed for inference)
            action_idx = torch.multinomial(action_prob, 1).item()
            
            # Convert action to movement
            new_x, new_y = self.x, self.y
            
            # Find nearest prey for predators
            if self.predator:
                nearest_prey = self._find_nearest_prey(animals, config)
                if nearest_prey:
                    # Calculate direction with toroidal wrapping
                    dx = nearest_prey.x - self.x
                    dy = nearest_prey.y - self.y
                    
                    # Check shorter path through wrapping
                    if abs(dx) > config.GRID_SIZE / 2:
                        dx = -(config.GRID_SIZE - abs(dx)) * (1 if dx > 0 else -1)
                    if abs(dy) > config.GRID_SIZE / 2:
                        dy = -(config.GRID_SIZE - abs(dy)) * (1 if dy > 0 else -1)
                    
                    # Move toward prey (8 directions)
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

    def move_training(self, model: nn.Module, animals: List['Animal'], 
                     log_probs_list: List, config, pheromone_map=None, values_list: List = None) -> None:
        """Move the animal during training (collects log probabilities and values for gradient)"""
        if self.predator:
            moves = (config.PREDATOR_HUNGRY_MOVES 
                    if self.steps_since_last_meal >= config.HUNGER_THRESHOLD 
                    else config.PREDATOR_NORMAL_MOVES)
        else:
            moves = config.PREY_MOVES
        
        # Get visible animals
        visible_animals = self.communicate(animals, config)
        
        # Get enhanced 20-feature input
        animal_input = self.get_enhanced_input(animals, config, pheromone_map)
        visible_animals_input = torch.tensor(
            visible_animals, dtype=torch.float32
        ).unsqueeze(0)
        
        # Get action probabilities from model
        model_output = model(animal_input, visible_animals_input)
        if isinstance(model_output, tuple):  # Actor-Critic
            action_prob, state_value = model_output
        else:  # Old SimpleNN
            action_prob = model_output
            state_value = None
        
        for _ in range(moves):
            # Sample action and collect log probability for policy gradient
            dist = torch.distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs_list.append(log_prob)
            
            # Store value if using Actor-Critic
            if state_value is not None and values_list is not None:
                values_list.append(state_value)
            
            action_idx = action.item()
            new_x, new_y = self.x, self.y
            
            # Find nearest prey for predators
            if self.predator:
                nearest_prey = self._find_nearest_prey(animals, config)
                if nearest_prey:
                    # Calculate direction with toroidal wrapping
                    dx = nearest_prey.x - self.x
                    dy = nearest_prey.y - self.y
                    
                    # Check shorter path through wrapping
                    if abs(dx) > config.GRID_SIZE / 2:
                        dx = -(config.GRID_SIZE - abs(dx)) * (1 if dx > 0 else -1)
                    if abs(dy) > config.GRID_SIZE / 2:
                        dy = -(config.GRID_SIZE - abs(dy)) * (1 if dy > 0 else -1)
                    
                    # Move toward prey (8 directions)
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

    def _get_threat_info(self, animals: List['Animal'], config, pheromone_map=None) -> dict:
        """Calculate threat and opportunity information for decision making"""
        nearest_predator_dist = 1.0  # Normalized to 1.0 (no threat)
        nearest_predator_dx = 0.0
        nearest_predator_dy = 0.0
        nearest_prey_dist = 1.0
        nearest_prey_dx = 0.0
        nearest_prey_dy = 0.0
        predator_count = 0
        prey_count = 0
        
        # Get pheromone information if available
        pheromone_danger = 0.0
        pheromone_mating = 0.0
        pheromone_food = 0.0
        
        if pheromone_map is not None:
            local_pheromones = pheromone_map.get_local_pheromones(self.x, self.y, config.PHEROMONE_SENSING_RANGE)
            pheromone_danger = local_pheromones['danger_max']
            pheromone_mating = local_pheromones['mating_max']
            pheromone_food = local_pheromones['food_max']
        
        for animal in animals:
            if animal == self:
                continue
                
            # Calculate toroidal distance
            dx_raw = animal.x - self.x
            dy_raw = animal.y - self.y
            
            if abs(dx_raw) > config.GRID_SIZE / 2:
                dx_raw = -(config.GRID_SIZE - abs(dx_raw)) * (1 if dx_raw > 0 else -1)
            if abs(dy_raw) > config.GRID_SIZE / 2:
                dy_raw = -(config.GRID_SIZE - abs(dy_raw)) * (1 if dy_raw > 0 else -1)
            
            dx = abs(dx_raw)
            dy = abs(dy_raw)
            
            if dx <= config.VISION_RANGE and dy <= config.VISION_RANGE:
                distance = (dx**2 + dy**2)**0.5 / (config.VISION_RANGE * 1.414)
                
                if animal.predator:
                    predator_count += 1
                    if not self.predator and distance < nearest_predator_dist:
                        nearest_predator_dist = distance
                        nearest_predator_dx = dx_raw / config.VISION_RANGE
                        nearest_predator_dy = dy_raw / config.VISION_RANGE
                else:
                    prey_count += 1
                    if self.predator and distance < nearest_prey_dist:
                        nearest_prey_dist = distance
                        nearest_prey_dx = dx_raw / config.VISION_RANGE
                        nearest_prey_dy = dy_raw / config.VISION_RANGE
        
        return {
            'nearest_predator_dist': nearest_predator_dist,
            'nearest_predator_dx': nearest_predator_dx,
            'nearest_predator_dy': nearest_predator_dy,
            'nearest_prey_dist': nearest_prey_dist,
            'nearest_prey_dx': nearest_prey_dx,
            'nearest_prey_dy': nearest_prey_dy,
            'predator_count': predator_count,
            'prey_count': prey_count,
            'pheromone_danger': pheromone_danger,
            'pheromone_mating': pheromone_mating,
            'pheromone_food': pheromone_food
        }

    def _find_nearest_prey(self, animals: List['Animal'], config) -> Optional['Animal']:
        """Find the nearest prey animal using toroidal distance"""
        nearest = None
        min_distance = float('inf')
        
        for animal in animals:
            if not animal.predator:
                # Calculate toroidal distance
                dx = abs(animal.x - self.x)
                dy = abs(animal.y - self.y)
                dx = min(dx, config.GRID_SIZE - dx)
                dy = min(dy, config.GRID_SIZE - dy)
                distance = dx + dy
                if distance < min_distance:
                    min_distance = distance
                    nearest = animal
        
        return nearest

    def move_away(self, config) -> None:
        """Move to a random adjacent position (8 directions)"""
        move_directions = [(0, 1), (0, -1), (-1, 0), (1, 0),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]
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
        """Get enhanced information about nearby visible animals (toroidal vision)"""
        visible_animals = []

        for animal in animals:
            if animal != self and len(visible_animals) < config.MAX_VISIBLE_ANIMALS:
                # Calculate toroidal distance (wrapping around borders)
                dx_raw = animal.x - self.x
                dy_raw = animal.y - self.y
                
                # Adjust for wrapping to get shortest path
                if abs(dx_raw) > config.GRID_SIZE / 2:
                    dx_raw = -(config.GRID_SIZE - abs(dx_raw)) * (1 if dx_raw > 0 else -1)
                if abs(dy_raw) > config.GRID_SIZE / 2:
                    dy_raw = -(config.GRID_SIZE - abs(dy_raw)) * (1 if dy_raw > 0 else -1)
                
                dx = abs(dx_raw)
                dy = abs(dy_raw)
                distance = (dx**2 + dy**2)**0.5

                if dx <= config.VISION_RANGE and dy <= config.VISION_RANGE:
                    visible_animals.append([
                        animal.x / config.GRID_SIZE,  # Absolute position x
                        animal.y / config.GRID_SIZE,  # Absolute position y
                        dx_raw / config.VISION_RANGE,  # Relative direction x (signed)
                        dy_raw / config.VISION_RANGE,  # Relative direction y (signed)
                        distance / (config.VISION_RANGE * 1.414),  # Normalized distance
                        float(animal.predator),  # Is predator?
                        float(not animal.predator),  # Is prey?
                        float(animal.name == self.name)  # Same species?
                    ])

        # Pad with zeros if fewer animals visible
        while len(visible_animals) < config.MAX_VISIBLE_ANIMALS:
            visible_animals.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return visible_animals[:config.MAX_VISIBLE_ANIMALS]

    def eat(self, animals: List['Animal'], stats: Dict = None) -> bool:
        """Predator attempts to eat nearby prey (inference mode)"""
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

    def eat_training(self, animals: List['Animal'], config) -> tuple:
        """Predator attempts to eat nearby prey (training mode - returns reward)"""
        if not self.predator:
            return False, 0.0
        
        prey_to_eat = None
        for prey in animals:
            if not prey.predator and abs(self.x - prey.x) <= 1 and abs(self.y - prey.y) <= 1:
                prey_to_eat = prey
                break
        
        if prey_to_eat:
            animals.remove(prey_to_eat)
            self.steps_since_last_meal = 0
            self.energy = min(self.max_energy, self.energy + config.EATING_ENERGY_GAIN)
            self.successful_hunts += 1
            self.experience += config.EXPERIENCE_GAIN_RATE
            return True, config.PREDATOR_EAT_REWARD
        
        return False, 0.0
    
    def update_energy(self, config, moved: bool = True):
        """Update energy levels based on activity"""
        # Base energy decay
        self.energy -= config.ENERGY_DECAY_RATE
        
        # Additional cost for movement
        if moved:
            self.energy -= config.MOVE_ENERGY_COST
        else:
            # Gain energy when resting
            self.energy += config.REST_ENERGY_GAIN
        
        # Clamp energy
        self.energy = max(0.0, min(self.max_energy, self.energy))
    
    def update_age(self):
        """Increment age"""
        self.age += 1
    
    def is_exhausted(self) -> bool:
        """Check if animal is out of energy"""
        return self.energy <= 0.0
    
    def is_old(self, config) -> bool:
        """Check if animal has reached maximum age"""
        return self.age >= config.MAX_AGE
    
    def can_reproduce(self, config) -> bool:
        """Check if animal is old enough and has enough energy to reproduce"""
        return (self.age >= config.MATURITY_AGE and 
                self.energy >= config.MATING_ENERGY_COST and
                self.mating_cooldown == 0)
    
    def deposit_pheromones(self, pheromone_map, config):
        """Deposit pheromones based on current state"""
        if pheromone_map is None:
            return
        
        # Prey deposits danger pheromone when seeing predator
        if not self.predator:
            threat_info = self._get_threat_info([], config, pheromone_map)
            if threat_info['predator_count'] > 0:
                pheromone_map.deposit_pheromone(
                    self.x, self.y, 'danger', config.DANGER_PHEROMONE_STRENGTH
                )
                self.last_danger_time = self.age
        
        # Animals deposit mating pheromone when ready to mate
        if self.can_reproduce(config):
            pheromone_map.deposit_pheromone(
                self.x, self.y, 'mating', config.MATING_PHEROMONE_STRENGTH
            )
        
        # Predators deposit food pheromone after successful hunt
        if self.predator and self.steps_since_last_meal == 0:
            pheromone_map.deposit_pheromone(
                self.x, self.y, 'food', 0.9
            )

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
