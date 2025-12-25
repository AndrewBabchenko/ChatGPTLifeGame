"""
Life Game - Demo Mode (Inference Only)

This script runs the simulation without training.
It loads pre-trained models and displays the learned behaviors.
"""

# =============================================================================
# Life Game - Demo Mode (Inference Only)
# A predator-prey simulation using trained neural networks
# =============================================================================

import random
from typing import List, Set, Optional, Tuple, Dict
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimulationConfig:
    """Configuration class for all simulation parameters"""
    # Grid settings
    GRID_SIZE = 100
    FIELD_MIN = 20
    FIELD_MAX = 80
    
    # Population settings
    INITIAL_PREY_COUNT = 120
    INITIAL_PREDATOR_COUNT = 10
    MAX_ANIMALS = 400
    
    # Animal behavior
    VISION_RANGE = 8
    MAX_VISIBLE_ANIMALS = 15
    HUNGER_THRESHOLD = 30
    STARVATION_THRESHOLD = 60
    MATING_COOLDOWN = 15
    
    # Movement
    PREDATOR_HUNGRY_MOVES = 2
    PREDATOR_NORMAL_MOVES = 1
    PREY_MOVES = 1
    
    # Mating probabilities
    MATING_PROBABILITY_PREY = 0.9
    MATING_PROBABILITY_PREDATOR = 0.15


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

    def move(self, model: nn.Module, animals: List['Animal'], config: SimulationConfig) -> None:
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

    def _apply_action(self, action: int, config: SimulationConfig) -> Tuple[int, int]:
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
    """Neural network for animal decision making"""
    def __init__(self, config: SimulationConfig):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.rnn = nn.GRU(4, 16, batch_first=True)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)

    def forward(self, animal_input: torch.Tensor, 
                visible_animals_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        animal_output = F.relu(self.fc1(animal_input))
        
        if visible_animals_input.size(1) > 0:
            _, rnn_output = self.rnn(visible_animals_input)
            rnn_output = rnn_output.squeeze(0)
        else:
            rnn_output = torch.zeros(animal_input.size(0), 16)
        
        combined_output = torch.cat((animal_output, rnn_output), dim=-1)
        hidden = F.relu(self.fc2(combined_output))
        output = self.fc3(hidden)
        
        return F.softmax(output, dim=-1)


class SimulationVisualizer:
    """Modern dashboard for visualization with statistics and controls"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle('Life Game - Predator-Prey Ecosystem Simulation', 
                         fontsize=16, fontweight='bold')
        
        # Create grid layout with 5 rows for better spacing
        gs = self.fig.add_gridspec(5, 3, hspace=0.35, wspace=0.4, 
                                  left=0.05, right=0.95, top=0.93, bottom=0.08)
        
        # Main simulation area (left side, larger) - spans 4 rows
        self.ax_main = self.fig.add_subplot(gs[:4, :2])
        self.ax_main.set_xlim(0, config.GRID_SIZE)
        self.ax_main.set_ylim(0, config.GRID_SIZE)
        self.ax_main.set_facecolor('#f0f0f0')
        self.ax_main.grid(True, alpha=0.3, linestyle='--')
        self.ax_main.set_xlabel('X Position', fontsize=10)
        self.ax_main.set_ylabel('Y Position', fontsize=10)
        
        # Population graph (top right) - RESTORED
        self.ax_pop = self.fig.add_subplot(gs[0, 2])
        self.ax_pop.set_title('Population Over Time', fontsize=10, fontweight='bold')
        self.ax_pop.set_xlabel('Step', fontsize=8)
        self.ax_pop.set_ylabel('Count', fontsize=8)
        self.ax_pop.grid(True, alpha=0.3)
        
        # Live statistics panel (middle right)
        self.ax_stats = self.fig.add_subplot(gs[1:3, 2])
        self.ax_stats.axis('off')
        self.ax_stats.set_title('Live Statistics', fontsize=10, fontweight='bold')
        
        # Legend panel (bottom right) - now with more spacing
        self.ax_legend = self.fig.add_subplot(gs[3, 2])
        self.ax_legend.axis('off')
        
        # Control buttons panel (bottom row)
        self.ax_controls = self.fig.add_subplot(gs[4, :])
        self.ax_controls.axis('off')
        
        # Data for population tracking
        self.prey_history = []
        self.predator_history = []
        self.step_history = []
        
        # Simulation control state
        self.paused = False
        self.restart_requested = False
        
        # Create legend and buttons
        self._create_legend()
        self._create_buttons()
        
        plt.ion()  # Interactive mode
        plt.show()
    
    def _create_legend(self):
        """Create visual legend for animal types"""
        legend_elements = [
            mpatches.Patch(facecolor='green', edgecolor='black', label='Prey (Well-fed)'),
            mpatches.Patch(facecolor='red', edgecolor='black', label='Predator (Normal)'),
            mpatches.Patch(facecolor='darkred', edgecolor='black', label='Predator (Hungry)'),
        ]
        self.ax_legend.legend(handles=legend_elements, loc='center', 
                             frameon=True, fontsize=9, ncol=1)
        self.ax_legend.set_title('Legend', fontsize=10, fontweight='bold', loc='center', pad=5)
    
    def _create_buttons(self):
        """Create interactive control buttons"""
        # Pause/Resume button
        ax_pause = plt.axes([0.35, 0.02, 0.12, 0.04])
        self.btn_pause = Button(ax_pause, 'Pause', color='lightblue', hovercolor='skyblue')
        self.btn_pause.on_clicked(self._toggle_pause)
        
        # Restart button
        ax_restart = plt.axes([0.53, 0.02, 0.12, 0.04])
        self.btn_restart = Button(ax_restart, 'Restart', color='lightcoral', hovercolor='salmon')
        self.btn_restart.on_clicked(self._request_restart)
    
    def _toggle_pause(self, event):
        """Toggle simulation pause state"""
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.label.set_text('Resume')
            self.btn_pause.color = 'lightgreen'
        else:
            self.btn_pause.label.set_text('Pause')
            self.btn_pause.color = 'lightblue'
        self.fig.canvas.draw_idle()
    
    def _request_restart(self, event):
        """Request simulation restart"""
        self.restart_requested = True
    
    def update(self, animals: List[Animal], step: int, stats: Dict = None):
        """Update all visualization components"""
        # Count populations
        prey_count = sum(1 for a in animals if not a.predator)
        predator_count = sum(1 for a in animals if a.predator)
        
        # Update history
        self.prey_history.append(prey_count)
        self.predator_history.append(predator_count)
        self.step_history.append(step)
        
        # Clear main plot
        self.ax_main.clear()
        self.ax_main.set_xlim(0, self.config.GRID_SIZE)
        self.ax_main.set_ylim(0, self.config.GRID_SIZE)
        self.ax_main.set_facecolor('#f0f0f0')
        self.ax_main.grid(True, alpha=0.3, linestyle='--')
        self.ax_main.set_xlabel('X Position', fontsize=10)
        self.ax_main.set_ylabel('Y Position', fontsize=10)
        self.ax_main.set_title(f'Step: {step} | Total Animals: {len(animals)}', 
                              fontsize=12, fontweight='bold')
        
        # Draw animals as scatter plot for better performance
        prey_x = [a.x for a in animals if not a.predator]
        prey_y = [a.y for a in animals if not a.predator]
        pred_x = [a.x for a in animals if a.predator and a.steps_since_last_meal < self.config.HUNGER_THRESHOLD]
        pred_y = [a.y for a in animals if a.predator and a.steps_since_last_meal < self.config.HUNGER_THRESHOLD]
        hungry_x = [a.x for a in animals if a.predator and a.steps_since_last_meal >= self.config.HUNGER_THRESHOLD]
        hungry_y = [a.y for a in animals if a.predator and a.steps_since_last_meal >= self.config.HUNGER_THRESHOLD]
        
        if prey_x:
            self.ax_main.scatter(prey_x, prey_y, c='green', s=50, alpha=0.6, 
                               edgecolors='darkgreen', linewidth=1, label='Prey')
        if pred_x:
            self.ax_main.scatter(pred_x, pred_y, c='red', s=70, alpha=0.7, 
                               edgecolors='darkred', linewidth=1, marker='^', label='Predator')
        if hungry_x:
            self.ax_main.scatter(hungry_x, hungry_y, c='darkred', s=80, alpha=0.8, 
                               edgecolors='black', linewidth=1.5, marker='^', label='Hungry')
        
        # Update population graph - RESTORED
        self.ax_pop.clear()
        self.ax_pop.set_title('Population Over Time', fontsize=10, fontweight='bold')
        self.ax_pop.set_xlabel('Step', fontsize=8)
        self.ax_pop.set_ylabel('Count', fontsize=8)
        self.ax_pop.grid(True, alpha=0.3)
        
        if len(self.step_history) > 1:
            self.ax_pop.plot(self.step_history, self.prey_history, 'g-', 
                           linewidth=2, label='Prey')
            self.ax_pop.plot(self.step_history, self.predator_history, 'r-', 
                           linewidth=2, label='Predators')
            self.ax_pop.legend(loc='upper right', fontsize=8)
            self.ax_pop.fill_between(self.step_history, self.prey_history, alpha=0.3, color='green')
            self.ax_pop.fill_between(self.step_history, self.predator_history, alpha=0.3, color='red')
        
        # Update statistics panel
        self._update_stats(animals, step, prey_count, predator_count, stats)
        
        # Refresh display
        plt.pause(0.001)
    
    def _update_stats(self, animals: List[Animal], step: int, 
                     prey_count: int, predator_count: int, stats: Dict):
        """Update the statistics text panel"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Calculate additional statistics
        if animals:
            avg_prey_survival = np.mean([a.survival_time for a in animals if not a.predator]) if prey_count > 0 else 0
            avg_pred_survival = np.mean([a.survival_time for a in animals if a.predator]) if predator_count > 0 else 0
            avg_hunger = np.mean([a.steps_since_last_meal for a in animals if a.predator]) if predator_count > 0 else 0
        else:
            avg_prey_survival = avg_pred_survival = avg_hunger = 0
        
        # Create stats text
        stats_text = f"""CURRENT STATUS
━━━━━━━━━━━━━━━━━━━
Step: {step}

POPULATION
  Prey: {prey_count}
  Predators: {predator_count}
  Ratio: {prey_count/max(predator_count,1):.1f}:1

AVERAGES
  Prey Survival: {avg_prey_survival:.1f}
  Pred Survival: {avg_pred_survival:.1f}
  Pred Hunger: {avg_hunger:.1f}
"""
        
        if stats:
            stats_text += f"\nSUMMARY\n  Total Births: {stats.get('total_births', 0)}\n  Total Deaths: {stats.get('total_deaths', 0)}"
        
        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=9, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def show_final_stats(self, stats: Dict):
        """Display final statistics in a dedicated window"""
        fig_stats = plt.figure(figsize=(10, 8))
        fig_stats.suptitle('Simulation Summary', fontsize=16, fontweight='bold')
        
        # Create subplots for different statistics
        gs = fig_stats.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        # Population history
        ax1 = fig_stats.add_subplot(gs[0, :])
        ax1.plot(self.step_history, self.prey_history, 'g-', linewidth=2, label='Prey')
        ax1.plot(self.step_history, self.predator_history, 'r-', linewidth=2, label='Predators')
        ax1.set_title('Population Dynamics', fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Population')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(self.step_history, self.prey_history, alpha=0.2, color='green')
        ax1.fill_between(self.step_history, self.predator_history, alpha=0.2, color='red')
        
        # Statistics text (left middle)
        ax2 = fig_stats.add_subplot(gs[1, 0])
        ax2.axis('off')
        summary_text = f"""FINAL STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Steps: {stats['total_steps']}
Duration: {stats['duration']:.2f}s

POPULATION
  Final Prey: {stats['final_prey']}
  Final Predators: {stats['final_predators']}
  Peak Prey: {stats['peak_prey']}
  Peak Predators: {stats['peak_predators']}
  Min Prey: {stats['min_prey']}

EVENTS
  Total Births: {stats['total_births']}
  Total Deaths: {stats['total_deaths']}
  Predator Meals: {stats['total_meals']}
"""
        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Ratio over time (right middle)
        ax3 = fig_stats.add_subplot(gs[1, 1])
        ratios = [p/max(pr, 1) for p, pr in zip(self.prey_history, self.predator_history)]
        ax3.plot(self.step_history, ratios, 'b-', linewidth=2)
        ax3.axhline(y=10, color='g', linestyle='--', alpha=0.5, label='Healthy (10:1)')
        ax3.set_title('Prey:Predator Ratio', fontweight='bold')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Population distribution (bottom)
        ax4 = fig_stats.add_subplot(gs[2, :])
        categories = ['Prey\n(Start)', 'Predators\n(Start)', 'Prey\n(Final)', 'Predators\n(Final)']
        values = [self.prey_history[0], self.predator_history[0], 
                 stats['final_prey'], stats['final_predators']]
        colors = ['lightgreen', 'lightcoral', 'green', 'red']
        bars = ax4.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
        ax4.set_title('Population Comparison', fontweight='bold')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        plt.show()


def build_spatial_grid(animals: List[Animal], 
                      config: SimulationConfig) -> Dict[Tuple[int, int], List[Animal]]:
    """Build spatial grid for efficient neighbor queries"""
    grid = defaultdict(list)
    cell_size = 5
    
    for animal in animals:
        grid_key = (animal.x // cell_size, animal.y // cell_size)
        grid[grid_key].append(animal)
    
    return grid


def run_simulation(animals: List[Animal], steps: int, 
                   model_prey: nn.Module, model_predator: nn.Module,
                   config: SimulationConfig) -> Dict:
    """Run simulation in inference mode (no training)
    
    Returns:
        Dictionary containing detailed statistics about the simulation run
    """
    import time
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
    
    # Create modern visualizer
    viz = SimulationVisualizer(config)
    
    for step in range(steps):
        # Check for restart request
        if viz.restart_requested:
            print("\n⟳ Restart requested by user")
            stats['restart_requested'] = True
            break
        
        # Handle pause
        while viz.paused:
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


def main():
    """Main entry point for demo mode with restart capability"""
    
    # Header
    print("\n" + "=" * 70)
    print("  LIFE GAME - DEMO MODE (Inference Only)")
    print("  Predator-Prey Ecosystem with Neural Networks")
    print("=" * 70)
    
    # Initialize configuration
    config = SimulationConfig()
    
    # Create and load models
    model_prey = SimpleNN(config)
    model_predator = SimpleNN(config)
    
    try:
        model_prey.load_state_dict(torch.load("model_A_fixed.pth"))
        model_predator.load_state_dict(torch.load("model_B_fixed.pth"))
        print("\n✓ Loaded trained models successfully")
        models_loaded = True
    except FileNotFoundError:
        print("\n⚠  WARNING: No trained models found!")
        print("   Using untrained models (random behavior)")
        print("   Run Life_Game_Fixed.py first to train models\n")
        models_loaded = False
    
    # Set to evaluation mode
    model_prey.eval()
    model_predator.eval()
    
    def create_population():
        """Create initial population of animals"""
        animals = []
        
        # Create prey - spread across entire map
        for _ in range(config.INITIAL_PREY_COUNT):
            x = random.randint(5, 95)
            y = random.randint(5, 95)
            animals.append(Animal(x, y, 'A', 'green', predator=False))
        
        # Create predators - concentrated in center
        for _ in range(config.INITIAL_PREDATOR_COUNT):
            x = random.randint(45, 55)
            y = random.randint(45, 55)
            animals.append(Animal(x, y, 'B', 'red', predator=True))
        
        return animals
    
    # Main simulation loop with restart capability
    while True:
        # Reset animal ID counter
        Animal._next_id = 1
        
        # Create population
        animals = create_population()
        print(f"\n✓ Created {len(animals)} initial animals")
        print(f"  • Prey: {config.INITIAL_PREY_COUNT}")
        print(f"  • Predators: {config.INITIAL_PREDATOR_COUNT}")
        
        # Run simulation
        print("\n" + "-" * 70)
        print("SIMULATION RUNNING")
        print("-" * 70)
        print("Close the plot window to exit | Press Ctrl+C to restart\n")
        
        steps = 1000  # Run for 1000 steps
        
        try:
            stats = run_simulation(animals, steps, model_prey, model_predator, config)
            
            # Check if restart was requested during simulation
            if stats.get('restart_requested'):
                print("\n⟳ Restarting simulation...")
                plt.close('all')
                continue
            
            # Display final statistics
            print("\n" + "=" * 70)
            print("SIMULATION COMPLETE")
            print("=" * 70)
            print(f"\nFinal Population:")
            print(f"  • Prey: {stats['final_prey']}")
            print(f"  • Predators: {stats['final_predators']}")
            print(f"\nTotal Events:")
            print(f"  • Births: {stats['total_births']}")
            print(f"  • Deaths: {stats['total_deaths']}")
            print(f"  • Predator Meals: {stats['total_meals']}")
            print(f"\nPopulation Extremes:")
            print(f"  • Peak Prey: {stats['peak_prey']}")
            print(f"  • Peak Predators: {stats['peak_predators']}")
            print(f"  • Minimum Prey: {stats['min_prey']}")
            print(f"\nDuration: {stats['duration']:.2f} seconds")
            
            # Show detailed statistics window
            print("\n📊 Opening detailed statistics window...")
            viz = stats.get('viz')
            if viz:
                viz.show_final_stats(stats)
            
            # Wait for user to close stats window
            plt.show(block=False)
            plt.pause(3)  # Show for 3 seconds
            
            # Ask for restart
            print("\n" + "=" * 70)
            response = input("\nRun again? (y/n): ").strip().lower()
            plt.close('all')  # Close all windows before restart
            if response != 'y':
                break
                
        except KeyboardInterrupt:
            print("\n\n⚠  Simulation interrupted by user")
            response = input("Restart simulation? (y/n): ").strip().lower()
            if response != 'y':
                break
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            break
    
    print("\n" + "=" * 70)
    print("  Thank you for using Life Game Demo!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
