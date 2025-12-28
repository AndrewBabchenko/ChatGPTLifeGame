"""
Animal class hierarchy for the Life Game simulation
Base Animal class with Prey and Predator subclasses for type-specific behavior
"""

import random
import math
import torch
import torch.nn as nn
from typing import List, Set, Optional, Tuple, Dict
from abc import ABC, abstractmethod


class Animal(ABC):
    """Abstract base class for all animals in the simulation"""
    _next_id = 1
    
    # Observation contract version (increment when feature structure changes)
    OBS_VERSION = 2  # Updated: 31→34 features (added pheromone gradient magnitudes)
    
    # Turn action constants (limited turn rate: ±1 per step)
    TURN_LEFT = 0
    TURN_STRAIGHT = 1
    TURN_RIGHT = 2
    
    # 8-direction heading system (0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW)
    DIRECTIONS = [
        (0, -1),   # 0: North (up)
        (1, -1),   # 1: Northeast
        (1, 0),    # 2: East (right)
        (1, 1),    # 3: Southeast
        (0, 1),    # 4: South (down)
        (-1, 1),   # 5: Southwest
        (-1, 0),   # 6: West (left)
        (-1, -1),  # 7: Northwest
    ]

    def __init__(self, x: int, y: int, name: str, color: str, 
                 parent_ids: Optional[Set[int]] = None) -> None:
        self.id = Animal._next_id
        Animal._next_id += 1
        self.x = x
        self.y = y
        self.name = name
        self.color = color
        self.parent_ids = parent_ids if parent_ids and len(parent_ids) <= 2 else set()
        self.mating_cooldown = 0
        self.survival_time = 0
        self.num_children = 0
        self._last_visible_step = -1  # Track cache freshness
        
        # Advanced features
        self.age = 0  # Age in simulation steps
        self.energy = 100.0  # Energy/stamina (0-100)
        self.max_energy = 100.0
        self.experience = 0.0  # Accumulated experience (affects learning)
        
        # Heading/facing direction (0-7 index into DIRECTIONS)
        self.heading_idx = random.randint(0, 7)  # Random initial heading
        self.heading_dx, self.heading_dy = self.DIRECTIONS[self.heading_idx]
        self.last_heading_idx = self.heading_idx  # Track previous heading
        self.previous_turn_action = 1  # 1=TURN_STRAIGHT default (no turn)

    @abstractmethod
    def is_predator(self) -> bool:
        """Return whether this animal is a predator"""
        pass

    @abstractmethod
    def get_vision_range(self, config) -> int:
        """Get the vision range for this animal type"""
        pass

    @abstractmethod
    def get_move_count(self, config) -> int:
        """Get number of moves per step for this animal type"""
        pass

    @abstractmethod
    def can_eat(self, other: 'Animal') -> bool:
        """Check if this animal can eat another animal"""
        pass

    @abstractmethod
    def perform_eat(self, animals: List['Animal'], config, stats: Dict = None) -> Tuple[bool, float, Optional['Animal']]:
        """Attempt to eat nearby animals - returns (success, reward, eaten_animal)"""
        pass

    @abstractmethod
    def update_post_action(self, config):
        """Update animal state after taking action (type-specific logic)"""
        pass

    def get_enhanced_input(self, animals: List['Animal'], config, pheromone_map=None, 
                          visible_animals: Optional[List[List[float]]] = None) -> torch.Tensor:
        """
        Build enhanced input vector with all 34 features (OBS_VERSION=2)
        Includes position, heading, state, threat info, age, energy, and pheromones with gradients + magnitudes
        
        Args:
            visible_animals: Optional pre-computed visible list. If None, will compute it.
                            Pass this to avoid redundant world scan when you already called communicate().
        
        STABLE FEATURE ORDER (critical for NN - bump OBS_VERSION if changed):
        [0-1]: Position (x, y) - normalized [0,1]
        [2-3]: Species (A, B) - binary
        [4]: Predator flag - binary
        [5-6]: State (hunger, mating_cooldown) - normalized [0,1], CLAMPED
        [7-12]: Nearest threat/prey (dist, dx, dy for each) - normalized, CLAMPED
        [13-14]: Visible counts (predators, prey) - normalized [0,1], CLAMPED
        [15-16]: Age and energy - normalized [0,1], CLAMPED
        [17-19]: Pheromone intensities (danger, mating, food max) - [0,1]
        [20-21]: Current heading direction (heading_dx, heading_dy) - [-1,1]
        [22-27]: Pheromone gradients (danger_x/y, mating_x/y, food_x/y) - [-1,1]
        [28-30]: Pheromone gradient magnitudes (danger, mating, food) - [0,1]
        [31]: Danger memory (high=recent threat) - [0,1]
        [32]: Population ratio - [0,1]
        [33]: Previous turn action (0=left, 1=straight, 2=right) - normalized [0,1]
        """
        # Compute threat info from visible list (avoids redundant world scan)
        if visible_animals is None:
            visible_animals = self.communicate(animals, config)
        
        vis_info = self.summarize_visible(visible_animals)
        
        # Get pheromone information (single call - get_sensory_input includes local pheromones)
        threat_info = {'pheromone_danger': 0.0, 'pheromone_mating': 0.0, 'pheromone_food': 0.0}
        pheromone_grads = {'danger_grad_x': 0.0, 'danger_grad_y': 0.0, 'danger_grad_mag': 0.0,
                          'mating_grad_x': 0.0, 'mating_grad_y': 0.0, 'mating_grad_mag': 0.0,
                          'food_grad_x': 0.0, 'food_grad_y': 0.0, 'food_grad_mag': 0.0}
        
        if pheromone_map is not None:
            sensory = pheromone_map.get_sensory_input(self.x, self.y, config.PHEROMONE_SENSING_RANGE)
            
            # Max intensities (no extra neighborhood scan - already in sensory)
            threat_info['pheromone_danger'] = sensory['danger_max']
            threat_info['pheromone_mating'] = sensory['mating_max']
            threat_info['pheromone_food'] = sensory['food_max']
            
            # Gradients + magnitudes
            pheromone_grads = {
                'danger_grad_x': sensory['danger_grad_x'],
                'danger_grad_y': sensory['danger_grad_y'],
                'danger_grad_mag': sensory.get('danger_grad_mag', 0.0),
                'mating_grad_x': sensory['mating_grad_x'],
                'mating_grad_y': sensory['mating_grad_y'],
                'mating_grad_mag': sensory.get('mating_grad_mag', 0.0),
                'food_grad_x': sensory['food_grad_x'],
                'food_grad_y': sensory['food_grad_y'],
                'food_grad_mag': sensory.get('food_grad_mag', 0.0),
            }
            
            # Hide predator-hunt pheromones from prey (food = hunting signal)
            if not isinstance(self, Predator):
                threat_info['pheromone_food'] = 0.0
                pheromone_grads['food_grad_x'] = 0.0
                pheromone_grads['food_grad_y'] = 0.0
                pheromone_grads['food_grad_mag'] = 0.0
        
        # Calculate max capacity based on species type
        max_capacity = config.MAX_PREDATORS if self.is_predator() else config.MAX_PREY
        # Use cached counts if available (avoids N scans per step)
        if hasattr(config, "_prey_count") and hasattr(config, "_pred_count"):
            same_species_count = config._pred_count if self.is_predator() else config._prey_count
        else:
            same_species_count = sum(1 for a in animals if type(a) == type(self))
        current_population_ratio = min(1.0, same_species_count / max_capacity)
        
        # Time since last danger (for prey memory)
        # HIGH value = recent threat (better for learning)
        danger_memory = 0.0
        if not self.is_predator() and hasattr(self, 'last_danger_time'):
            time_since_danger = self.age - self.last_danger_time
            # Exponential decay: 1.0 = just saw threat, decays to 0.0 over time
            danger_memory = max(0.0, 1.0 - time_since_danger / 50.0)  # Decays over ~50 steps
        
        features = [
            self.x / config.GRID_SIZE,  # 0: x position
            self.y / config.GRID_SIZE,  # 1: y position
            int(self.name == 'A'),  # 2: species A
            int(self.name == 'B'),  # 3: species B
            int(self.is_predator()),  # 4: is predator
            min(1.0, self._get_hunger_level(config)),  # 5: hunger level (CLAMPED)
            min(1.0, self.mating_cooldown / config.MATING_COOLDOWN),  # 6: mating readiness (CLAMPED)
            min(1.0, vis_info['nearest_predator_dist']),  # 7: nearest threat distance (CLAMPED)
            vis_info['nearest_predator_dx'],  # 8: threat direction x
            vis_info['nearest_predator_dy'],  # 9: threat direction y
            min(1.0, vis_info['nearest_prey_dist']),  # 10: nearest prey distance (CLAMPED)
            vis_info['nearest_prey_dx'],  # 11: prey direction x
            vis_info['nearest_prey_dy'],  # 12: prey direction y
            min(1.0, vis_info['predator_count'] / config.MAX_VISIBLE_ANIMALS),  # 13: visible predators (CLAMPED)
            min(1.0, vis_info['prey_count'] / config.MAX_VISIBLE_ANIMALS),  # 14: visible prey (CLAMPED)
            min(1.0, self.age / config.MAX_AGE),  # 15: age (CLAMPED)
            min(1.0, self.energy / config.MAX_ENERGY),  # 16: energy level (CLAMPED)
            threat_info.get('pheromone_danger', 0.0),  # 17: danger pheromone max
            threat_info.get('pheromone_mating', 0.0),  # 18: mating pheromone max
            threat_info.get('pheromone_food', 0.0),  # 19: food pheromone max
            self.heading_dx,  # 20: current heading direction x (NEW)
            self.heading_dy,  # 21: current heading direction y (NEW)
            pheromone_grads['danger_grad_x'],  # 22: danger gradient x (NEW)
            pheromone_grads['danger_grad_y'],  # 23: danger gradient y (NEW)
            pheromone_grads['mating_grad_x'],  # 24: mating gradient x (NEW)
            pheromone_grads['mating_grad_y'],  # 25: mating gradient y (NEW)
            pheromone_grads['food_grad_x'],  # 26: food gradient x (NEW)
            pheromone_grads['food_grad_y'],  # 27: food gradient y (NEW)
            pheromone_grads['danger_grad_mag'],  # 28: danger gradient magnitude (NEW)
            pheromone_grads['mating_grad_mag'],  # 29: mating gradient magnitude (NEW)
            pheromone_grads['food_grad_mag'],  # 30: food gradient magnitude (NEW)
            danger_memory,  # 31: time since last danger (high=recent)
            current_population_ratio,  # 32: current population vs usable capacity
            self.previous_turn_action / 2.0,  # 33: previous turn action, normalized to [0,1]
        ]
        
        # Contract check (SHOULD DO for safety)
        assert len(features) == 34, f"OBS_VERSION={Animal.OBS_VERSION} expects 34 features, got {len(features)}"
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    @abstractmethod
    def _get_hunger_level(self, config) -> float:
        """Get normalized hunger level (type-specific)"""
        pass
    
    def _toroidal_delta(self, target_x: int, target_y: int, config) -> Tuple[float, float, float]:
        """
        Calculate shortest toroidal distance and direction to target
        Returns: (dx, dy, distance) where dx/dy are signed deltas
        """
        dx = target_x - self.x
        dy = target_y - self.y
        
        # Wrap around if shorter path exists
        if abs(dx) > config.GRID_SIZE / 2:
            dx = -(config.GRID_SIZE - abs(dx)) * (1 if dx > 0 else -1)
        if abs(dy) > config.GRID_SIZE / 2:
            dy = -(config.GRID_SIZE - abs(dy)) * (1 if dy > 0 else -1)
        
        distance = (dx**2 + dy**2)**0.5
        return dx, dy, distance
    
    def is_in_vision(self, target_x: int, target_y: int, config) -> bool:
        """
        Check if target position is within this animal's vision
        Uses circular range boundary and cone-shaped FOV based on heading
        Optimized with dot-product FOV test (avoids atan2 per candidate)
        """
        dx, dy, distance = self._toroidal_delta(target_x, target_y, config)
        
        # Check circular range boundary
        vision_range = self.get_vision_range(config)
        if distance > vision_range:
            return False
        
        # Target at same position
        if distance < 1e-6:
            return True
        
        # Unit vector to target
        inv = 1.0 / (distance + 1e-8)
        ux = dx * inv
        uy = dy * inv
        
        # Normalize heading (diagonals have length sqrt(2))
        hx = float(self.heading_dx)
        hy = float(self.heading_dy)
        hnorm = (hx*hx + hy*hy) ** 0.5 + 1e-8
        hx /= hnorm
        hy /= hnorm
        
        # Cache cos(fov/2) per heading to avoid repeated trig
        fov_deg = self.get_fov_deg(config)
        key = (self.heading_idx, fov_deg)
        cos_half = getattr(self, "_cos_half_fov_cache", {}).get(key, None)
        if cos_half is None:
            cos_half = math.cos(math.radians(fov_deg * 0.5))
            cache = getattr(self, "_cos_half_fov_cache", None)
            if cache is None:
                cache = {}
                setattr(self, "_cos_half_fov_cache", cache)
            cache[key] = cos_half
        
        # Dot-product FOV check
        dot = hx * ux + hy * uy
        return dot >= cos_half
    
    @abstractmethod
    def get_fov_deg(self, config) -> float:
        """Get the field of view angle in degrees for this animal type"""
        pass

    def move(self, model: nn.Module, animals: List['Animal'], config, pheromone_map=None) -> None:
        """Move the animal based on neural network decision (inference only)
        
        Two-phase action per micro-step:
        1. Sample turn action from current observation
        2. Apply turn (changes heading/FOV)
        3. Recompute observation (CRITICAL: FOV changed)
        4. Sample move action from new observation
        5. Apply movement
        """
        moves = self.get_move_count(config)
        
        # Set model to eval mode and disable gradients for inference
        model.eval()
        with torch.no_grad():
            for _ in range(moves):
                # PHASE 1: TURN
                # Get current observation (pre-turn)
                animal_input = self.get_enhanced_input(animals, config, pheromone_map)
                visible_animals = self.communicate(animals, config)
                visible_animals_input = torch.tensor(
                    visible_animals, dtype=torch.float32
                ).unsqueeze(0)
                
                # Sample turn action from pre-turn observation (deterministic=True for argmax)
                if hasattr(model, 'get_action'):
                    # New dual-head network - use deterministic argmax for evaluation
                    turn_action, _, _, _, _ = model.get_action(animal_input, visible_animals_input, deterministic=True)
                    turn_action = int(turn_action.item())
                else:
                    # Fallback: old single-head model (no turn control)
                    turn_action = 1  # TURN_STRAIGHT (no turn for old models)
                
                # Apply turn (changes heading → changes FOV)
                self.apply_turn_action(turn_action)
                
                # PHASE 2: MOVE
                # Recompute observation after turn (CRITICAL: FOV changed!)
                animal_input = self.get_enhanced_input(animals, config, pheromone_map)
                visible_animals = self.communicate(animals, config)
                visible_animals_input = torch.tensor(
                    visible_animals, dtype=torch.float32
                ).unsqueeze(0)
                
                # Sample move action from post-turn observation (deterministic=True for argmax)
                if hasattr(model, 'get_action'):
                    # Use deterministic argmax for evaluation
                    _, move_action, _, _, _ = model.get_action(animal_input, visible_animals_input, deterministic=True)
                    move_action = int(move_action.item())
                else:
                    # Fallback: old single-head model
                    model_output = model(animal_input, visible_animals_input)
                    if isinstance(model_output, tuple):
                        action_prob, _ = model_output
                    else:
                        action_prob = model_output
                    # multinomial handles unnormalized probs directly, no need to clamp
                    move_action = torch.multinomial(action_prob, 1).item()
                
                # Apply movement
                new_x, new_y = self._apply_action_logic(move_action, animals, config, is_training=False)
                
                if not self._position_occupied(animals, new_x, new_y):
                    self.x, self.y = new_x, new_y
        
        # Cache last visible_animals for pheromone deposit in inference mode
        self._last_visible_animals = visible_animals
        self._last_visible_step = getattr(config, 'CURRENT_STEP', -1)

    @abstractmethod
    def _apply_action_logic(self, action_idx: int, animals: List['Animal'], config, is_training: bool = False) -> Tuple[int, int]:
        """Apply action and return new position - type-specific logic"""
        pass

    def move_training(self, model: nn.Module, animals: List['Animal'], config, 
                     pheromone_map=None) -> List[Dict]:
        """Move the animal during training (returns hierarchical transitions for PPO)
        
        GPU-optimized version:
        - Infers device from model parameters
        - Creates observations on CPU, moves to device for forward pass
        - Uses multinomial sampling (DirectML-friendly)
        - Stores transitions on CPU to avoid VRAM explosion
        
        Hierarchical policy (turn→move) per micro-step:
        1. Sample turn action from pre-turn observation
        2. Apply turn (changes heading/FOV)
        3. Recompute observation (CRITICAL: FOV changed)
        4. Sample move action from post-turn observation
        5. Return structured transition with both observations and actions
        
        Returns:
            List of transition dicts, one per micro-step:
            {
                'obs_turn': pre-turn animal_input (CPU),
                'vis_turn': pre-turn visible_animals (CPU),
                'turn_action': turn action taken,
                'turn_logp_old': log prob of turn action (CPU),
                
                'obs_move': post-turn animal_input (CPU),
                'vis_move': post-turn visible_animals (CPU),
                'move_action': move action taken,
                'move_logp_old': log prob of move action (CPU),
                
                'value_old': pre-turn state value (CPU),
                'value_next': placeholder for TD(0) bootstrapping (CPU)
            }
        """
        moves = self.get_move_count(config)
        transitions = []
        
        # Infer device from model (DirectML / CUDA / CPU)
        dev = next(model.parameters()).device
        
        for _ in range(moves):
            # -------- TURN PHASE --------
            # Build pre-turn observation on CPU, then move to dev for forward
            visible_animals_turn = self.communicate(animals, config)                                 # Python list
            animal_input_turn_cpu = self.get_enhanced_input(animals, config, pheromone_map, visible_animals=visible_animals_turn)  # CPU tensor
            visible_input_turn_cpu = torch.as_tensor(visible_animals_turn, dtype=torch.float32).unsqueeze(0)
            
            animal_input_turn = animal_input_turn_cpu.to(dev)
            visible_input_turn = visible_input_turn_cpu.to(dev)
            
            # Forward for turn head
            turn_probs, _, state_value_turn = model.forward(animal_input_turn, visible_input_turn)
            # Sample (DirectML friendly)
            turn_action_t = torch.multinomial(turn_probs, 1).view(-1)  # (1,)
            turn_action = int(turn_action_t.item())
            turn_log_prob = torch.log(turn_probs.gather(1, turn_action_t.view(1,1)).clamp_min(1e-8)).view(-1)
            
            # Apply turn
            self.apply_turn_action(turn_action)
            
            # -------- MOVE PHASE --------
            visible_animals_move = self.communicate(animals, config)
            animal_input_move_cpu = self.get_enhanced_input(animals, config, pheromone_map, visible_animals=visible_animals_move)
            visible_input_move_cpu = torch.as_tensor(visible_animals_move, dtype=torch.float32).unsqueeze(0)
            
            animal_input_move = animal_input_move_cpu.to(dev)
            visible_input_move = visible_input_move_cpu.to(dev)
            
            _, move_probs, _ = model.forward(animal_input_move, visible_input_move)
            move_action_t = torch.multinomial(move_probs, 1).view(-1)  # (1,)
            move_action_item = int(move_action_t.item())
            move_log_prob = torch.log(move_probs.gather(1, move_action_t.view(1,1)).clamp_min(1e-8)).view(-1)
            
            # Apply movement
            new_x, new_y = self._apply_action_logic(move_action_item, animals, config, is_training=True)
            if not self._position_occupied(animals, new_x, new_y):
                self.x, self.y = new_x, new_y
            
            # Store transition (store CPU copies to keep VRAM low)
            transition = {
                'traj_id': self.id,  # Use stable id (better than id(self))
                'obs_turn': animal_input_turn_cpu.detach(),
                'vis_turn': visible_input_turn_cpu.detach(),
                'turn_action': turn_action,
                'turn_logp_old': turn_log_prob.detach().cpu(),  # ensure CPU scalar tensor
                
                'obs_move': animal_input_move_cpu.detach(),
                'vis_move': visible_input_move_cpu.detach(),
                'move_action': move_action_t.detach().cpu(),     # tensor is OK for your trainer
                'move_logp_old': move_log_prob.detach().cpu(),
                
                'value_old': state_value_turn.detach().cpu(),
                'value_next': torch.zeros_like(state_value_turn.detach().cpu()),
            }
            transitions.append(transition)
        
        # Cache last visible_animals for pheromone deposit (avoids extra communicate() call)
        self._last_visible_animals = visible_animals_move
        self._last_visible_step = getattr(config, 'CURRENT_STEP', -1)
        
        return transitions

    def summarize_visible(self, visible_animals: List[List[float]]) -> dict:
        """
        Compute nearest predator/prey + counts from the fixed visible list.
        Avoids redundant world scan - use this instead of _get_threat_info() when visible_animals is already computed.
        
        visible_animals entries are 8 floats:
          [0]=dx_norm, [1]=dy_norm, [2]=dist_norm,
          [3]=is_predator, [4]=is_prey,
          [5]=same_species, [6]=same_type, [7]=is_present
        """
        nearest_pred_dist = 1.0
        nearest_pred_dx = 0.0
        nearest_pred_dy = 0.0
        
        nearest_prey_dist = 1.0
        nearest_prey_dx = 0.0
        nearest_prey_dy = 0.0
        
        predator_count = 0
        prey_count = 0
        
        for v in visible_animals:
            is_present = (v[7] >= 0.5)
            if not is_present:
                continue
            
            dist = float(v[2])
            is_pred = (v[3] >= 0.5)
            is_prey = (v[4] >= 0.5)
            
            if is_pred:
                predator_count += 1
                if dist < nearest_pred_dist:
                    nearest_pred_dist = dist
                    nearest_pred_dx = float(v[0])
                    nearest_pred_dy = float(v[1])
            
            if is_prey:
                prey_count += 1
                if dist < nearest_prey_dist:
                    nearest_prey_dist = dist
                    nearest_prey_dx = float(v[0])
                    nearest_prey_dy = float(v[1])
        
        return {
            "nearest_predator_dist": nearest_pred_dist,
            "nearest_predator_dx": nearest_pred_dx,
            "nearest_predator_dy": nearest_pred_dy,
            "nearest_prey_dist": nearest_prey_dist,
            "nearest_prey_dx": nearest_prey_dx,
            "nearest_prey_dy": nearest_prey_dy,
            "predator_count": predator_count,
            "prey_count": prey_count,
        }

    def _apply_action(self, action: int, config) -> Tuple[int, int]:
        """
        Convert action index to new position with toroidal wrapping
        Actions match DIRECTIONS: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
        """
        new_x, new_y = self.x, self.y
        
        if action == 0:  # North (up): y - 1
            new_y = (self.y - 1) % config.GRID_SIZE
        elif action == 1:  # Northeast: x + 1, y - 1
            new_x = (self.x + 1) % config.GRID_SIZE
            new_y = (self.y - 1) % config.GRID_SIZE
        elif action == 2:  # East (right): x + 1
            new_x = (self.x + 1) % config.GRID_SIZE
        elif action == 3:  # Southeast: x + 1, y + 1
            new_x = (self.x + 1) % config.GRID_SIZE
            new_y = (self.y + 1) % config.GRID_SIZE
        elif action == 4:  # South (down): y + 1
            new_y = (self.y + 1) % config.GRID_SIZE
        elif action == 5:  # Southwest: x - 1, y + 1
            new_x = (self.x - 1) % config.GRID_SIZE
            new_y = (self.y + 1) % config.GRID_SIZE
        elif action == 6:  # West (left): x - 1
            new_x = (self.x - 1) % config.GRID_SIZE
        elif action == 7:  # Northwest: x - 1, y - 1
            new_x = (self.x - 1) % config.GRID_SIZE
            new_y = (self.y - 1) % config.GRID_SIZE
            
        return new_x, new_y
    
    def apply_turn_action(self, turn_action: int):
        """
        Apply NN turn action (limited turn rate: -1, 0, +1)
        This is the ONLY way heading should change (NN-controlled)
        
        Args:
            turn_action: 0=TURN_LEFT, 1=TURN_STRAIGHT, 2=TURN_RIGHT
        """
        self.previous_turn_action = turn_action
        
        # Convert action to turn delta
        turn_delta = turn_action - 1  # 0→-1, 1→0, 2→+1
        
        # Update heading with limited turn rate
        self.last_heading_idx = self.heading_idx
        self.heading_idx = (self.heading_idx + turn_delta) % 8
        self.heading_dx, self.heading_dy = self.DIRECTIONS[self.heading_idx]
    
    # DISABLED: Automatic heading update breaks NN heading control
    # def _update_heading_from_movement(self, new_x: int, new_y: int, config):
    #     """
    #     [DISABLED] This method is no longer used. Heading is controlled by NN turn actions.
    #     Keeping for reference only.
    #     """
    #     pass

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
        
        # Use appropriate vision range based on animal type
        my_vision_range = self.get_vision_range(config)
        
        for animal in animals:
            if animal == self:
                continue
            
            # Use new vision system (circular range + cone FOV)
            if not self.is_in_vision(animal.x, animal.y, config):
                continue
            
            # Calculate toroidal distance and direction
            dx, dy, distance = self._toroidal_delta(animal.x, animal.y, config)
            
            # Normalize distance (circular vision boundary)
            normalized_distance = distance / my_vision_range
            
            if animal.is_predator():
                predator_count += 1
                if not self.is_predator() and normalized_distance < nearest_predator_dist:
                    nearest_predator_dist = normalized_distance
                    nearest_predator_dx = dx / my_vision_range
                    nearest_predator_dy = dy / my_vision_range
            else:
                prey_count += 1
                if self.is_predator() and normalized_distance < nearest_prey_dist:
                    nearest_prey_dist = normalized_distance
                    nearest_prey_dx = dx / my_vision_range
                    nearest_prey_dy = dy / my_vision_range
        
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

    def can_mate(self, other: 'Animal', config) -> bool:
        """Check if this animal can mate with another (toroidal adjacency)"""
        # Calculate toroidal distance
        dx = abs(other.x - self.x)
        dy = abs(other.y - self.y)
        
        # Wrap around if needed
        dx = min(dx, config.GRID_SIZE - dx)
        dy = min(dy, config.GRID_SIZE - dy)
        
        return (
            type(self) == type(other)  # Same class (Prey or Predator)
            and self.name == other.name
            and dx <= 1  # Toroidal adjacency
            and dy <= 1  # Toroidal adjacency
            and self.id not in other.parent_ids
            and other.id not in self.parent_ids
            and self.mating_cooldown == 0
            and other.mating_cooldown == 0
        )

    def communicate(self, animals: List['Animal'], config) -> List[List[float]]:
        """
        Get enhanced information about nearby visible animals with 50/50 same/opposite split
        
        Returns fixed-size list with reserved slots:
        - k_same = floor(MAX_VISIBLE_ANIMALS / 2) for same-type animals
        - k_opp = remaining slots for opposite-type animals
        - Backfills from other group if one is undersubscribed
        - Distance-sorted for deterministic selection
        
        Optimized with heap-based top-K selection (avoids sorting all visible)
        """
        import heapq
        
        # Use appropriate vision range based on animal type
        my_vision_range = self.get_vision_range(config)
        
        # Calculate reserved slots
        k_same = config.MAX_VISIBLE_ANIMALS // 2
        k_opp = config.MAX_VISIBLE_ANIMALS - k_same
        
        # Keep 2x quota for backfill (still tiny compared to N)
        keep_same = k_same * 2
        keep_opp = k_opp * 2
        
        # Max-heaps (negative distance) to keep smallest distances
        heap_same = []
        heap_opp = []
        
        for animal in animals:
            if animal is self:
                continue
            
            # Use consistent vision system (circular range + cone FOV)
            if not self.is_in_vision(animal.x, animal.y, config):
                continue
            
            # Calculate toroidal distance and direction
            dx, dy, distance = self._toroidal_delta(animal.x, animal.y, config)
            
            # Build feature vector for this animal (8 features)
            feats = [
                dx / my_vision_range,  # 0: Relative direction x (signed, normalized)
                dy / my_vision_range,  # 1: Relative direction y (signed, normalized)
                distance / my_vision_range,  # 2: Normalized distance (circular, [0,1])
                float(animal.is_predator()),  # 3: Is predator? (binary)
                float(not animal.is_predator()),  # 4: Is prey? (binary)
                float(animal.name == self.name),  # 5: Same species name? (binary)
                float(animal.is_predator() == self.is_predator()),  # 6: Same type? (binary)
                1.0,  # 7: is_present flag (1.0 = real animal, padding will be 0.0)
            ]
            
            # Push to appropriate heap (keep only top-K by distance)
            item = (-distance, feats)  # negative for max-heap behavior
            if animal.is_predator() == self.is_predator():
                if len(heap_same) < keep_same:
                    heapq.heappush(heap_same, item)
                elif distance < -heap_same[0][0]:
                    heapq.heapreplace(heap_same, item)
            else:
                if len(heap_opp) < keep_opp:
                    heapq.heappush(heap_opp, item)
                elif distance < -heap_opp[0][0]:
                    heapq.heapreplace(heap_opp, item)
        
        # Convert heaps to sorted lists by distance ascending
        same_type = [f for _, f in sorted(heap_same, key=lambda t: -t[0])]
        opposite_type = [f for _, f in sorted(heap_opp, key=lambda t: -t[0])]
        
        # Select from each group up to quota
        selected_same = same_type[:k_same]
        selected_opp = opposite_type[:k_opp]
        
        # Exhaustive backfill: keep taking from either group until slots are full
        remaining_same = same_type[k_same:]
        remaining_opp = opposite_type[k_opp:]
        
        while len(selected_same) + len(selected_opp) < config.MAX_VISIBLE_ANIMALS:
            # Try to take from remaining groups
            if remaining_same:
                selected_same.append(remaining_same.pop(0))
            elif remaining_opp:
                selected_opp.append(remaining_opp.pop(0))
            else:
                # No more animals to add
                break
        
        # Combine lists (same-type first, then opposite-type)
        visible_animals = selected_same + selected_opp
        
        # Pad to MAX_VISIBLE_ANIMALS with zeros (is_present=0.0 at index 7 distinguishes padding)
        padding_row = [0.0] * 8
        while len(visible_animals) < config.MAX_VISIBLE_ANIMALS:
            visible_animals.append(padding_row)
        
        return visible_animals

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

    @abstractmethod
    def can_reproduce(self, config) -> bool:
        """Check if animal can reproduce - type-specific requirements"""
        pass

    @abstractmethod
    def deposit_pheromones(self, animals: List['Animal'], pheromone_map, config):
        """Deposit pheromones based on current state - type-specific"""
        pass

    def display_color(self, config) -> str:
        """Get color for visualization - can be overridden"""
        return self.color

    def _position_occupied(self, animals: List['Animal'], 
                          new_x: int, new_y: int) -> bool:
        """Check if a position is occupied"""
        for animal in animals:
            if animal != self and animal.x == new_x and animal.y == new_y:
                return True
        return False


class Prey(Animal):
    """Prey animal - herbivore that avoids predators"""
    
    def __init__(self, x: int, y: int, name: str, color: str, 
                 parent_ids: Optional[Set[int]] = None) -> None:
        super().__init__(x, y, name, color, parent_ids)
        self.successful_evasions = 0
        self.last_danger_time = -999  # Last time saw predator

    def is_predator(self) -> bool:
        return False

    def get_vision_range(self, config) -> int:
        return config.PREY_VISION_RANGE
    
    def get_fov_deg(self, config) -> float:
        """Prey have wide peripheral vision"""
        return config.PREY_FOV_DEG

    def get_move_count(self, config) -> int:
        return config.PREY_MOVES

    def can_eat(self, other: 'Animal') -> bool:
        """Prey cannot eat other animals"""
        return False

    def perform_eat(self, animals: List['Animal'], config, stats: Dict = None) -> Tuple[bool, float, Optional['Animal']]:
        """Prey don't eat other animals"""
        return False, 0.0, None

    def update_post_action(self, config):
        """No special post-action updates for prey"""
        pass

    def _get_hunger_level(self, config) -> float:
        """Prey don't have hunger (always 0)"""
        return 0.0

    def _apply_action_logic(self, action_idx: int, animals: List['Animal'], config, is_training: bool = False) -> Tuple[int, int]:
        """Prey use action directly without special movement logic"""
        return self._apply_action(action_idx, config)

    def can_reproduce(self, config) -> bool:
        """Check if prey is old enough and has enough energy to reproduce"""
        return (self.age >= config.MATURITY_AGE and 
                self.energy >= config.MATING_ENERGY_COST and
                self.mating_cooldown == 0)

    def deposit_pheromones(self, animals: List['Animal'], pheromone_map, config):
        """Deposit danger pheromone when seeing predators, mating pheromone when ready"""
        if pheromone_map is None:
            return
        
        # Use cached visibility only if from current step (avoid stale data)
        current_step = getattr(config, 'CURRENT_STEP', -1)
        visible_animals = getattr(self, "_last_visible_animals", None)
        if visible_animals is None or getattr(self, "_last_visible_step", -1) != current_step:
            visible_animals = self.communicate(animals, config)
        vis_info = self.summarize_visible(visible_animals)
        
        # Deposit danger pheromone when seeing predator
        if vis_info["predator_count"] > 0:
            pheromone_map.deposit_pheromone(
                self.x, self.y, 'danger', config.DANGER_PHEROMONE_STRENGTH
            )
            self.last_danger_time = self.age
        
        # Deposit mating pheromone when ready to mate
        if self.can_reproduce(config):
            pheromone_map.deposit_pheromone(
                self.x, self.y, 'mating', config.MATING_PHEROMONE_STRENGTH
            )


class Predator(Animal):
    """
    Predator animal - carnivore that hunts prey
    
    TRAINING POLICY: During move_training(), NN always controls movement.
    Chase override is ONLY used during inference (move() without training).
    This ensures NN learns hunting behavior rather than relying on hardcoded chase.
    """
    
    def __init__(self, x: int, y: int, name: str, color: str, 
                 parent_ids: Optional[Set[int]] = None) -> None:
        super().__init__(x, y, name, color, parent_ids)
        self.steps_since_last_meal = 0
        self.successful_hunts = 0

    def is_predator(self) -> bool:
        return True

    def get_vision_range(self, config) -> int:
        return config.PREDATOR_VISION_RANGE
    
    def get_fov_deg(self, config) -> float:
        """Predators have forward-focused vision"""
        return config.PREDATOR_FOV_DEG

    def get_move_count(self, config) -> int:
        """Predators move more when hungry"""
        if self.steps_since_last_meal >= config.HUNGER_THRESHOLD:
            return config.PREDATOR_HUNGRY_MOVES
        else:
            return config.PREDATOR_NORMAL_MOVES

    def can_eat(self, other: 'Animal') -> bool:
        """Check if this predator can eat another animal"""
        return not other.is_predator()

    def perform_eat(self, animals: List['Animal'], config, stats: Dict = None) -> Tuple[bool, float, Optional['Animal']]:
        """Attempt to eat nearby prey - returns (success, reward, eaten_prey)
        
        NOTE: Does NOT remove prey from animals list (caller must do that)
        This avoids modifying the list during iteration.
        """
        prey_to_eat = None
        for prey in animals:
            if not prey.is_predator():
                # Check toroidal adjacency
                dx = abs(prey.x - self.x)
                dy = abs(prey.y - self.y)
                
                # Wrap around if needed
                dx = min(dx, config.GRID_SIZE - dx)
                dy = min(dy, config.GRID_SIZE - dy)
                
                if dx <= 1 and dy <= 1:
                    prey_to_eat = prey
                    break
        
        if prey_to_eat:
            # Update predator state (but don't modify animals list)
            self.steps_since_last_meal = 0
            self.energy = min(self.max_energy, self.energy + config.EATING_ENERGY_GAIN)
            self.successful_hunts += 1
            self.experience += config.EXPERIENCE_GAIN_RATE
            
            if stats is not None:
                stats['total_deaths'] = stats.get('total_deaths', 0) + 1
            
            return True, config.PREDATOR_EAT_REWARD, prey_to_eat
        
        return False, 0.0, None

    def update_post_action(self, config):
        """Increment hunger counter"""
        self.steps_since_last_meal += 1

    def _get_hunger_level(self, config) -> float:
        """Get normalized hunger level for predators"""
        return self.steps_since_last_meal / config.STARVATION_THRESHOLD

    def _apply_action_logic(self, action_idx: int, animals: List['Animal'], config, is_training: bool = False) -> Tuple[int, int]:
        """
        Predators can optionally move toward nearest prey if visible (inference only).
        During training (is_training=True), NN always controls movement.
        Chase override controlled by config.CHASE_OVERRIDE_IN_INFERENCE.
        Set to False when benchmarking learned hunting behavior.
        """
        if is_training:
            # During training, NN always controls movement
            return self._apply_action(action_idx, config)
        
        # During inference, check if chase override is enabled
        if not getattr(config, 'CHASE_OVERRIDE_IN_INFERENCE', False):
            # Chase override disabled - NN controls hunting
            return self._apply_action(action_idx, config)
        
        # Chase override enabled - use hardcoded hunting
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
            
            # Move toward prey (8 directions including diagonals)
            move_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
            move_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
            
            new_x = (self.x + move_x) % config.GRID_SIZE
            new_y = (self.y + move_y) % config.GRID_SIZE
            
            return new_x, new_y
        else:
            # No prey visible, use neural network action
            return self._apply_action(action_idx, config)

    def _find_nearest_prey(self, animals: List['Animal'], config) -> Optional['Animal']:
        """Find the nearest VISIBLE prey animal using toroidal distance"""
        nearest = None
        min_distance = float('inf')
        
        for animal in animals:
            if not animal.is_predator():
                # Only chase prey that is actually visible (circular range + cone FOV)
                if not self.is_in_vision(animal.x, animal.y, config):
                    continue
                
                # Calculate toroidal distance
                _, _, distance = self._toroidal_delta(animal.x, animal.y, config)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest = animal
        
        return nearest

    def can_reproduce(self, config) -> bool:
        """Check if predator can reproduce - must be well-fed"""
        base_requirements = (self.age >= config.MATURITY_AGE and 
                            self.energy >= config.MATING_ENERGY_COST and
                            self.mating_cooldown == 0)
        
        # Fitness-based reproduction: predators must have hunted successfully
        if base_requirements:
            # Only allow reproduction if predator has caught prey recently
            return self.steps_since_last_meal < config.STARVATION_THRESHOLD * 0.5
        
        return False

    def deposit_pheromones(self, animals: List['Animal'], pheromone_map, config):
        """Deposit food pheromone when seeing prey (hunting ground marker), mating pheromone when ready"""
        if pheromone_map is None:
            return
        
        # Use cached visibility only if from current step (avoid stale data)
        current_step = getattr(config, 'CURRENT_STEP', -1)
        visible_animals = getattr(self, "_last_visible_animals", None)
        if visible_animals is None or getattr(self, "_last_visible_step", -1) != current_step:
            visible_animals = self.communicate(animals, config)
        vis_info = self.summarize_visible(visible_animals)
        
        # Deposit food pheromone when seeing prey (marks hunting grounds for cooperation)
        if vis_info["prey_count"] > 0:
            # Deposit toward prey location (not just at predator position)
            # Use normalized direction and scale by 1-2 cells
            dx, dy = vis_info["nearest_prey_dx"], vis_info["nearest_prey_dy"]
            target_x = int(self.x + dx * 2) % config.GRID_SIZE
            target_y = int(self.y + dy * 2) % config.GRID_SIZE
            pheromone_map.deposit_pheromone(
                target_x, target_y, 'food', 0.7  # Lower strength than successful hunt
            )
        
        # Stronger pheromone after successful hunt (confirms good hunting ground)
        if self.steps_since_last_meal == 0:
            pheromone_map.deposit_pheromone(
                self.x, self.y, 'food', 0.9
            )
        
        # Deposit mating pheromone when ready to mate
        if self.can_reproduce(config):
            pheromone_map.deposit_pheromone(
                self.x, self.y, 'mating', config.MATING_PHEROMONE_STRENGTH
            )

    def display_color(self, config) -> str:
        """Show darker color when hungry"""
        if self.steps_since_last_meal >= config.HUNGER_THRESHOLD:
            return 'darkred'
        return self.color
