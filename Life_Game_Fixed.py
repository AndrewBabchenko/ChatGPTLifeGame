# =============================================================================
# Life Game - Predator-Prey Simulation with Reinforcement Learning (TRAINING)
# =============================================================================
# This version trains neural networks using policy gradient reinforcement learning
# to learn optimal survival behaviors for both prey and predators
#
# Key Features:
# - Proper REINFORCE policy gradient implementation
# - Fixed GRU input handling for variable-length sequences
# - Safe list operations (no modification during iteration)
# - Comprehensive configuration management
# - Performance optimizations (spatial grid for mating)
# - Balanced reward structure to maintain ecosystem equilibrium
# =============================================================================

import random
import time
from typing import List, Set, Optional, Tuple, Dict
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class SimulationConfig:
    """
    Central configuration class for all simulation parameters
    
    This class contains all tunable hyperparameters for the simulation,
    making it easy to experiment with different settings without modifying code.
    """
    
    # === GRID SETTINGS ===
    GRID_SIZE = 100          # Size of the simulation grid (100x100)
    FIELD_MIN = 20           # Minimum spawn coordinate
    FIELD_MAX = 80           # Maximum spawn coordinate
    
    # === POPULATION SETTINGS ===
    INITIAL_PREY_COUNT = 120         # Starting number of prey animals
    INITIAL_PREDATOR_COUNT = 10      # Starting number of predators
    MAX_ANIMALS = 400                # Maximum total population (prevents overflow)
    
    # === ANIMAL BEHAVIOR ===
    VISION_RANGE = 8                 # How far animals can see (grid units)
    MAX_VISIBLE_ANIMALS = 15         # Max animals processed by neural network
    HUNGER_THRESHOLD = 30            # Steps before predator becomes "hungry"
    STARVATION_THRESHOLD = 60        # Steps before predator dies from starvation
    MATING_COOLDOWN = 15             # Steps between mating attempts
    
    # === MOVEMENT SPEEDS ===
    PREDATOR_HUNGRY_MOVES = 2        # Moves per step when hungry
    PREDATOR_NORMAL_MOVES = 1        # Moves per step when fed
    PREY_MOVES = 1                   # Moves per step for prey
    
    # === REPRODUCTION PROBABILITIES ===
    MATING_PROBABILITY_PREY = 0.9    # 90% chance prey mate when adjacent
    MATING_PROBABILITY_PREDATOR = 0.15  # 15% chance predators mate
    
    # === REINFORCEMENT LEARNING SETTINGS ===
    LEARNING_RATE_PREY = 0.001       # Adam optimizer learning rate for prey
    LEARNING_RATE_PREDATOR = 0.001   # Adam optimizer learning rate for predators
    GAMMA = 0.99                     # Discount factor for future rewards
    
    # === REWARD STRUCTURE ===
    SURVIVAL_REWARD = 0.2            # Reward per step survived
    REPRODUCTION_REWARD = 10.0       # Reward for successful reproduction
    EXTINCTION_PENALTY = -1000.0     # Penalty if species goes extinct
    PREDATOR_EAT_REWARD = 15.0       # Reward for catching prey
    OVERPOPULATION_PENALTY = -10.0   # Penalty for predator overpopulation


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
        # Only keep immediate parents to avoid memory leak
        self.parent_ids = parent_ids if parent_ids and len(parent_ids) <= 2 else set()
        self.predator = predator
        self.steps_since_last_meal = 0
        self.mating_cooldown = 0
        self.survival_time = 0
        self.num_children = 0

    def move(self, model: nn.Module, animals: List['Animal'], 
             log_probs_list: List[torch.Tensor], 
             config: SimulationConfig) -> None:
        """
        Move the animal based on neural network decision
        
        Args:
            model: Neural network that outputs action probabilities
            animals: List of all animals in simulation (for vision/collision)
            log_probs_list: List to store log probabilities for policy gradient
            config: Simulation configuration
        """
        # Determine number of moves based on species and hunger state
        if self.predator:
            moves = (config.PREDATOR_HUNGRY_MOVES 
                    if self.steps_since_last_meal >= config.HUNGER_THRESHOLD 
                    else config.PREDATOR_NORMAL_MOVES)
        else:
            moves = config.PREY_MOVES
        
        # Get visual input: nearby animals within vision range
        visible_animals = self.communicate(animals, config)
        
        # Prepare neural network inputs
        animal_input = torch.tensor(
            [self.x / config.GRID_SIZE, self.y / config.GRID_SIZE, 
             int(self.name == 'A'), int(self.name == 'B')], 
            dtype=torch.float32
        ).unsqueeze(0)
        visible_animals_input = torch.tensor(
            visible_animals, dtype=torch.float32
        ).unsqueeze(0)
        
        # Get action probabilities from neural network
        action_prob = model(animal_input, visible_animals_input)
        
        # Execute moves
        for _ in range(moves):
            # Sample action from probability distribution (stochastic policy)
            dist = torch.distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs_list.append(log_prob)  # Store for gradient calculation
            
            # Convert action to movement
            new_x, new_y = self.x, self.y
            action_idx = action.item()
            
            # PREDATOR BEHAVIOR: Chase nearest prey if visible
            if self.predator:
                nearest_prey = self._find_nearest_prey(animals)
                if nearest_prey:
                    # Hardcoded chase behavior (move toward prey)
                    dx = nearest_prey.x - self.x
                    dy = nearest_prey.y - self.y
                    if abs(dx) > abs(dy):
                        new_x = (self.x + (1 if dx > 0 else -1)) % config.GRID_SIZE
                    else:
                        new_y = (self.y + (1 if dy > 0 else -1)) % config.GRID_SIZE
                else:
                    # No prey visible: use neural network decision
                    new_x, new_y = self._apply_action(action_idx, config)
            else:
                # PREY BEHAVIOR: Always use neural network (learns evasion)
                new_x, new_y = self._apply_action(action_idx, config)

            # Only move if position is not occupied (collision avoidance)
            if not self._position_occupied(animals, new_x, new_y):
                self.x, self.y = new_x, new_y

    def _apply_action(self, action: int, config: SimulationConfig) -> Tuple[int, int]:
        """Apply action to get new position with wrapping"""
        new_x, new_y = self.x, self.y
        
        if action == 0:  # Up
            new_y = (self.y + 1) % config.GRID_SIZE
        elif action == 1:  # Down
            new_y = (self.y - 1) % config.GRID_SIZE
        elif action == 2:  # Left
            new_x = (self.x - 1) % config.GRID_SIZE
        elif action == 3:  # Right
            new_x = (self.x + 1) % config.GRID_SIZE
        elif action == 4:  # Up-Right
            new_x = (self.x + 1) % config.GRID_SIZE
            new_y = (self.y + 1) % config.GRID_SIZE
        elif action == 5:  # Down-Right
            new_x = (self.x + 1) % config.GRID_SIZE
            new_y = (self.y - 1) % config.GRID_SIZE
        elif action == 6:  # Up-Left
            new_x = (self.x - 1) % config.GRID_SIZE
            new_y = (self.y + 1) % config.GRID_SIZE
        elif action == 7:  # Down-Left
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

    def move_away(self, config: SimulationConfig) -> None:
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

    def communicate(self, animals: List['Animal'], 
                   config: SimulationConfig) -> List[List[float]]:
        """
        Get information about nearby visible animals
        
        Returns a padded list of visible animals' features for neural network input.
        Padding ensures consistent input size for the GRU layer.
        
        Args:
            animals: All animals in simulation
            config: Simulation configuration
            
        Returns:
            List of animal features [x, y, is_prey, is_predator] with padding
        """
        visible_animals = []

        for animal in animals:
            if animal != self and len(visible_animals) < config.MAX_VISIBLE_ANIMALS:
                dx = abs(animal.x - self.x)
                dy = abs(animal.y - self.y)

                # Check if within vision range
                if dx <= config.VISION_RANGE and dy <= config.VISION_RANGE:
                    visible_animals.append([
                        animal.x / config.GRID_SIZE,  # Normalized x position
                        animal.y / config.GRID_SIZE,  # Normalized y position
                        float(animal.name == 'A'),    # Is prey? (1.0 or 0.0)
                        float(animal.name == 'B')     # Is predator? (1.0 or 0.0)
                    ])

        # Pad to fixed length to prevent GRU input size errors
        while len(visible_animals) < config.MAX_VISIBLE_ANIMALS:
            visible_animals.append([0.0, 0.0, 0.0, 0.0])  # Zero padding

        return visible_animals[:config.MAX_VISIBLE_ANIMALS]

    def eat(self, animals: List['Animal'], config: SimulationConfig) -> Tuple[bool, float]:
        """
        Predator attempts to eat nearby prey
        Returns (success, reward)
        Fixed: Doesn't modify list during iteration
        """
        if not self.predator:
            return False, 0.0
        
        # Find prey to eat (don't modify list yet)
        prey_to_eat = None
        for prey in animals:
            if not prey.predator and abs(self.x - prey.x) <= 1 and abs(self.y - prey.y) <= 1:
                prey_to_eat = prey
                break
        
        # Now safely remove the prey
        if prey_to_eat:
            animals.remove(prey_to_eat)
            self.steps_since_last_meal = 0
            return True, config.PREDATOR_EAT_REWARD
        
        return False, 0.0

    def display_color(self, config: SimulationConfig) -> str:
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


class SimpleNN(nn.Module):
    """
    Neural Network for Animal Decision Making
    
    Architecture:
    - Input Layer 1: Animal's own state (position, species) → 16 neurons
    - Input Layer 2: Visible neighbors (sequence) → GRU(16 hidden) 
    - Combined: Concatenate both → 32 neurons
    - Hidden: 32 → 16 neurons (ReLU)
    - Output: 16 → 8 actions (softmax probabilities)
    
    The GRU processes sequences of variable length (visible animals),
    allowing the network to understand spatial relationships and threats.
    """
    def __init__(self, config: SimulationConfig):
        super(SimpleNN, self).__init__()
        # Process animal's own state (x, y, is_prey, is_predator)
        self.fc1 = nn.Linear(4, 16)
        
        # Process sequence of visible animals using GRU for temporal/spatial patterns
        self.rnn = nn.GRU(4, 16, batch_first=True)
        
        # Combine both information streams
        self.fc2 = nn.Linear(32, 16)  # 16 (own) + 16 (neighbors) = 32
        
        # Output layer: 8 possible actions
        self.fc3 = nn.Linear(16, 8)

    def forward(self, animal_input: torch.Tensor, 
                visible_animals_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            animal_input: Tensor [batch, 4] - own state
            visible_animals_input: Tensor [batch, seq_len, 4] - neighbors
            
        Returns:
            Action probabilities [batch, 8]
        """
        # Process own state
        animal_output = F.relu(self.fc1(animal_input))
        
        # Process visible animals with GRU
        # FIXED: Handle empty sequences properly (prevents crashes)
        if visible_animals_input.size(1) > 0:
            _, rnn_output = self.rnn(visible_animals_input)
            rnn_output = rnn_output.squeeze(0)
        else:
            # No visible animals: use zeros
            rnn_output = torch.zeros(animal_input.size(0), 16)
        
        # Combine both information sources
        combined_output = torch.cat((animal_output, rnn_output), dim=-1)
        
        # Hidden layer with ReLU activation
        hidden = F.relu(self.fc2(combined_output))
        
        # Output layer: 8 movement directions
        output = self.fc3(hidden)
        
        # Softmax to get probability distribution
        return F.softmax(output, dim=-1)


def plot_animals(animals: List[Animal], step: int, config: SimulationConfig) -> None:
    """
    Visualize the current state of the simulation (Training Mode)
    
    Creates a simple real-time visualization showing animal positions.
    For a more advanced dashboard, see Life_Game_Demo.py
    """
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(0, config.GRID_SIZE)
    ax.set_ylim(0, config.GRID_SIZE)
    ax.set_facecolor('#f5f5f5')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    prey_count = sum(1 for a in animals if not a.predator)
    predator_count = sum(1 for a in animals if a.predator)
    
    ax.set_title(f\"Training Step: {step} | Prey: {prey_count} | Predators: {predator_count}\",
                fontsize=12, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    # Draw animals as circles
    for animal in animals:
        circle = plt.Circle((animal.x, animal.y), 0.8, 
                          color=animal.display_color(config),
                          alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.add_artist(circle)

    plt.pause(0.0001)


def build_spatial_grid(animals: List[Animal], 
                      config: SimulationConfig) -> Dict[Tuple[int, int], List[Animal]]:
    """
    Build spatial grid for efficient neighbor queries
    Performance optimization for mating detection
    """
    grid = defaultdict(list)
    cell_size = 5
    
    for animal in animals:
        grid_key = (animal.x // cell_size, animal.y // cell_size)
        grid[grid_key].append(animal)
    
    return grid


def compute_returns(rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
    """
    Compute discounted returns for policy gradient (REINFORCE algorithm)
    
    This is the correct way to implement reinforcement learning.
    Returns are computed backwards from future rewards, discounted by gamma.
    
    Formula: R_t = r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + ...
    
    Args:
        rewards: List of rewards received at each step
        gamma: Discount factor (0.99 = value future rewards at 99% of current)
        
    Returns:
        Normalized tensor of discounted returns
    """
    returns = []
    R = 0
    
    # Calculate discounted returns backwards
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    if len(returns) == 0:
        return torch.tensor([])
    
    returns = torch.tensor(returns, dtype=torch.float32)
    
    # Normalize returns for stable learning (reduces variance)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns


def simulate_episode(animals: List[Animal], steps: int, 
                    model_prey: nn.Module, model_predator: nn.Module,
                    config: SimulationConfig, plot: bool = False) -> Tuple[List, List, List, List]:
    """
    Run one episode of simulation
    Returns log probabilities and rewards for both species
    """
    log_probs_prey = []
    log_probs_predator = []
    rewards_prey = []
    rewards_predator = []

    for step in range(steps):
        step_reward_prey = 0
        step_reward_predator = 0
        
        # Track animals to remove (starvation)
        animals_to_remove = []
        
        # Movement phase
        for animal in animals:
            if animal.name == "A":
                animal.move(model_prey, animals, log_probs_prey, config)
                step_reward_prey += config.SURVIVAL_REWARD
            else:
                animal.move(model_predator, animals, log_probs_predator, config)
                step_reward_predator += config.SURVIVAL_REWARD

            # Eating phase
            has_eaten, eat_reward = animal.eat(animals, config)
            
            if animal.predator:
                step_reward_predator += eat_reward
                if not has_eaten:
                    animal.steps_since_last_meal += 1
                    
                    # Mark for removal if starved
                    if animal.steps_since_last_meal >= config.STARVATION_THRESHOLD:
                        animals_to_remove.append(animal)

        # Remove starved predators
        for animal in animals_to_remove:
            if animal in animals:
                animals.remove(animal)

        # Mating phase with spatial optimization
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
                            # Create child
                            child_x = (animal1.x + animal2.x) // 2
                            child_y = (animal1.y + animal2.y) // 2
                            child_parent_ids = {animal1.id, animal2.id}
                            
                            new_animal = Animal(child_x, child_y, animal1.name, 
                                              animal1.color, child_parent_ids, 
                                              animal1.predator)
                            new_animals.append(new_animal)
                            
                            # Update parents
                            animal1.move_away(config)
                            animal2.move_away(config)
                            animal1.mating_cooldown = config.MATING_COOLDOWN
                            animal2.mating_cooldown = config.MATING_COOLDOWN
                            animal1.num_children += 1
                            animal2.num_children += 1
                            
                            mated_animals.add(animal1.id)
                            mated_animals.add(animal2.id)
                            
                            # Reward for reproduction
                            if animal1.name == "A":
                                step_reward_prey += config.REPRODUCTION_REWARD
                            else:
                                step_reward_predator += config.REPRODUCTION_REWARD
                            
                            break

        # Add new animals up to max population
        if len(animals) + len(new_animals) <= config.MAX_ANIMALS:
            animals.extend(new_animals)
        else:
            animals.extend(new_animals[:config.MAX_ANIMALS - len(animals)])

        # Update cooldowns and survival time
        for animal in animals:
            if animal.mating_cooldown > 0:
                animal.mating_cooldown -= 1
            animal.survival_time += 1

        # Check for extinction and overpopulation
        prey_count = sum(1 for a in animals if not a.predator)
        predator_count = sum(1 for a in animals if a.predator)
        
        if prey_count == 0:
            step_reward_prey += config.EXTINCTION_PENALTY
        if predator_count == 0:
            step_reward_predator += config.EXTINCTION_PENALTY
        
        # Heavily penalize predator overpopulation and reward balance
        ratio_penalty = 0
        if prey_count > 0:
            ratio = predator_count / prey_count
            if ratio > 0.3:  # More than 30% predators
                ratio_penalty = config.OVERPOPULATION_PENALTY * predator_count * (ratio - 0.3)
                step_reward_predator += ratio_penalty

        # Store rewards
        rewards_prey.append(step_reward_prey)
        rewards_predator.append(step_reward_predator)

        # Visualization
        if plot:
            plot_animals(animals, step, config)

    return log_probs_prey, log_probs_predator, rewards_prey, rewards_predator


def train(initial_animals: List[Animal], episodes: int, steps: int, 
         model_prey: nn.Module, model_predator: nn.Module,
         optimizer_prey: optim.Optimizer, optimizer_predator: optim.Optimizer,
         config: SimulationConfig, plot: bool = False) -> None:
    """
    Train the models using proper policy gradient
    Fixed: Actual reinforcement learning implementation
    """
    balanced_episodes = 0
    required_balanced = 5
    
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        # Create fresh copy of animals for this episode
        episode_animals = [
            Animal(a.x, a.y, a.name, a.color, 
                  parent_ids=a.parent_ids.copy() if a.parent_ids else None,
                  predator=a.predator) 
            for a in initial_animals
        ]
        
        # Run episode
        log_probs_prey, log_probs_predator, rewards_prey, rewards_predator = \
            simulate_episode(episode_animals, steps, model_prey, model_predator, 
                           config, plot=plot)
        
        # Compute returns
        returns_prey = compute_returns(rewards_prey, config.GAMMA)
        returns_predator = compute_returns(rewards_predator, config.GAMMA)
        
        # Update prey model
        if len(log_probs_prey) > 0 and len(returns_prey) > 0:
            optimizer_prey.zero_grad()
            
            # Ensure same length
            min_len_prey = min(len(log_probs_prey), len(returns_prey))
            log_probs_prey = log_probs_prey[:min_len_prey]
            returns_prey = returns_prey[:min_len_prey]
            
            # Policy gradient loss
            policy_loss_prey = []
            for log_prob, R in zip(log_probs_prey, returns_prey):
                policy_loss_prey.append(-log_prob * R)
            
            loss_prey = torch.stack(policy_loss_prey).sum()
            loss_prey.backward()
            optimizer_prey.step()
            
            print(f"Prey - Loss: {loss_prey.item():.4f}, "
                  f"Total Reward: {sum(rewards_prey):.2f}")
        
        # Update predator model
        if len(log_probs_predator) > 0 and len(returns_predator) > 0:
            optimizer_predator.zero_grad()
            
            # Ensure same length
            min_len_predator = min(len(log_probs_predator), len(returns_predator))
            log_probs_predator = log_probs_predator[:min_len_predator]
            returns_predator = returns_predator[:min_len_predator]
            
            # Policy gradient loss
            policy_loss_predator = []
            for log_prob, R in zip(log_probs_predator, returns_predator):
                policy_loss_predator.append(-log_prob * R)
            
            loss_predator = torch.stack(policy_loss_predator).sum()
            loss_predator.backward()
            optimizer_predator.step()
            
            print(f"Predator - Loss: {loss_predator.item():.4f}, "
                  f"Total Reward: {sum(rewards_predator):.2f}")
        
        # Stats and balance check
        prey_count = sum(1 for a in episode_animals if not a.predator)
        predator_count = sum(1 for a in episode_animals if a.predator)
        print(f"Final population - Prey: {prey_count}, Predators: {predator_count}")
        
        # Check if balanced (prey survive well)
        if prey_count > 100:
            balanced_episodes += 1
            print(f"✓ BALANCED ({balanced_episodes}/{required_balanced})")
        else:
            balanced_episodes = 0
            
        # Early stopping if balanced
        if balanced_episodes >= required_balanced:
            print(f"\n{'='*50}")
            print(f"🎉 MODEL BALANCED! Prey survived >100 for {required_balanced} consecutive episodes")
            print(f"{'='*50}")
            break


def main():
    """Main entry point"""
    print("Starting Fixed Life Game Simulation")
    print("=" * 50)
    
    # Initialize configuration
    config = SimulationConfig()
    
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Create models
    model_prey = SimpleNN(config)
    model_predator = SimpleNN(config)
    
    # Don't load old models - start fresh for balanced training
    # try:
    #     model_prey.load_state_dict(torch.load("model_A_fixed.pth"))
    #     model_predator.load_state_dict(torch.load("model_B_fixed.pth"))
    #     print("✓ Loaded existing models")
    # except FileNotFoundError:
    #     print("✓ Starting with fresh models")
    print("✓ Starting with fresh models (old models disabled for rebalancing)")
    
    # Set to training mode
    model_prey.train()
    model_predator.train()
    
    # Create initial population
    animals = []
    
    # Create prey - spread across entire map for maximum dispersal
    for _ in range(config.INITIAL_PREY_COUNT):
        x = random.randint(5, 95)
        y = random.randint(5, 95)
        animals.append(Animal(x, y, 'A', 'green', predator=False))
    
    # Create predators - very concentrated in small center area
    for _ in range(config.INITIAL_PREDATOR_COUNT):
        x = random.randint(45, 55)
        y = random.randint(45, 55)
        animals.append(Animal(x, y, 'B', 'red', predator=True))
    
    print(f"✓ Created {len(animals)} initial animals")
    print(f"  - Prey: {config.INITIAL_PREY_COUNT}")
    print(f"  - Predators: {config.INITIAL_PREDATOR_COUNT}")
    
    # Create optimizers
    optimizer_prey = optim.Adam(model_prey.parameters(), 
                               lr=config.LEARNING_RATE_PREY)
    optimizer_predator = optim.Adam(model_predator.parameters(), 
                                   lr=config.LEARNING_RATE_PREDATOR)
    
    # Train continuously until balanced
    print("\nStarting continuous training until balanced...")
    print("Balance criteria: Prey survive >100 at end for 5 consecutive episodes")
    episodes = 100
    steps = 200
    
    train(animals, episodes, steps, model_prey, model_predator,
          optimizer_prey, optimizer_predator, config, plot=True)
    
    # Save models
    model_prey.eval()
    model_predator.eval()
    
    torch.save(model_prey.state_dict(), "model_A_fixed.pth")
    torch.save(model_predator.state_dict(), "model_B_fixed.pth")
    print("\n✓ Models saved successfully")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
