"""
Simulation Tab - Interactive simulation visualization
"""

import random
from collections import defaultdict
import math
import torch
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

from src.core.animal import Prey, Predator
from src.models.actor_critic_network import ActorCriticNetwork
from src.core.pheromone_system import PheromoneMap
from src.core.grass_field import GrassField


class SimulationTab:
    """Tab for running and visualizing the simulation"""
    
    def __init__(self, parent, app):
        self.app = app
        self.frame = ttk.Frame(parent)
        
        # Simulation state
        self.animals = []
        self.pheromone_map = None
        self.grass_field = None
        self.model_prey = None
        self.model_predator = None
        self.paused = True
        self.step_count = 0
        self.births = 0
        self.deaths = 0
        self.meals = 0
        self.grass_eaten = 0
        self.speed = 101
        self.update_id = None
        self.current_seed = None
        
        # Death splash effects: [(x, y, color, lifetime)]
        self.splashes = []
        
        # Setup UI
        self.setup_ui()
        
        # Load initial models
        self.load_models()
        
        # Initialize simulation
        self.frame.after(100, self.reset_simulation)
    
    def setup_ui(self):
        """Build simulation tab UI"""
        # Top controls
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.play_btn = ttk.Button(control_frame, text="▶ Play",
                                   command=self.toggle_pause)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="➡ Step",
                  command=self.step_once).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Seed:").pack(side=tk.LEFT, padx=(15, 5))
        self.seed_var = tk.StringVar()
        seed_entry = ttk.Entry(control_frame, textvariable=self.seed_var, width=12)
        seed_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="🎲", width=3,
                  command=self.randomize_seed).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="🔄 Restart",
                  command=self.reset_simulation).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Speed:").pack(side=tk.LEFT, padx=(15, 5))
        
        speed_container = ttk.Frame(control_frame)
        speed_container.pack(side=tk.LEFT, padx=5)
        
        self.speed_var = tk.IntVar(value=5)
        speed_scale = ttk.Scale(speed_container, from_=1, to=10, variable=self.speed_var,
                               orient=tk.HORIZONTAL, length=200, command=self.update_speed)
        speed_scale.grid(row=0, column=0, columnspan=10)
        
        for i in range(1, 11):
            label = ttk.Label(speed_container, text=str(i), font=('Arial', 7))
            label.grid(row=1, column=i-1, sticky='w')
        
        for i in range(10):
            speed_container.grid_columnconfigure(i, weight=1, uniform="speed")
        
        # Content area
        content_frame = ttk.Frame(self.frame)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10)
        
        # Left: Canvas
        canvas_frame = ttk.LabelFrame(content_frame, text="Simulation Field", padding="10")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_container, width=700, height=700, bg='#1a1a2e')
        self.canvas.pack(padx=20, pady=20)
        
        self.draw_grid()
        
        # Right: Stats panel
        stats_frame = ttk.LabelFrame(content_frame, text="Statistics", padding="10")
        stats_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        
        self.stats_labels = {}
        stats = ['Step', 'Prey', 'Predators', 'Births', 'Deaths', 'Meals', 'Grass Eaten']
        for stat in stats:
            frame = ttk.Frame(stats_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{stat}:", font=('Arial', 11)).pack(side=tk.LEFT)
            label = ttk.Label(frame, text="0", font=('Arial', 11, 'bold'))
            label.pack(side=tk.RIGHT)
            self.stats_labels[stat] = label
        
        # Legend
        ttk.Separator(stats_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(stats_frame, text="Legend:", font=('Arial', 11, 'bold')).pack(anchor=tk.W)
        
        self.create_legend(stats_frame)
    
    def create_legend(self, parent):
        """Create legend showing animal types and pheromones"""
        legend_frame = ttk.Frame(parent)
        legend_frame.pack(fill=tk.X, pady=5)
        
        prey_canvas = tk.Canvas(legend_frame, width=20, height=20, bg='white', highlightthickness=0)
        prey_canvas.pack(side=tk.LEFT, padx=5)
        prey_canvas.create_oval(5, 5, 15, 15, fill='#5ac8fa', outline='#0a84ff')
        ttk.Label(legend_frame, text="Prey").pack(side=tk.LEFT)
        
        pred_canvas = tk.Canvas(legend_frame, width=20, height=20, bg='white', highlightthickness=0)
        pred_canvas.pack(side=tk.LEFT, padx=(15, 5))
        pred_canvas.create_polygon(10, 5, 5, 15, 15, 15, fill='#ff6464', outline='#ff3b30')
        ttk.Label(legend_frame, text="Predator").pack(side=tk.LEFT)
        
        legend_frame2 = ttk.Frame(parent)
        legend_frame2.pack(fill=tk.X, pady=2)
        
        hungry_canvas = tk.Canvas(legend_frame2, width=20, height=20, bg='white', highlightthickness=0)
        hungry_canvas.pack(side=tk.LEFT, padx=5)
        hungry_canvas.create_polygon(10, 5, 5, 15, 15, 15, fill='#ff9500', outline='#ff3b30')
        ttk.Label(legend_frame2, text="Hungry", font=('Arial', 9)).pack(side=tk.LEFT)
        
        ttk.Label(parent, text="Pheromones:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        pheromone_frame = ttk.Frame(parent)
        pheromone_frame.pack(fill=tk.X, pady=2)
        
        danger_canvas = tk.Canvas(pheromone_frame, width=20, height=20, bg='white', highlightthickness=0)
        danger_canvas.pack(side=tk.LEFT, padx=5)
        danger_canvas.create_rectangle(2, 2, 18, 18, fill='#ff0000', outline='')
        ttk.Label(pheromone_frame, text="Danger", font=('Arial', 9)).pack(side=tk.LEFT)
        
        mating_canvas = tk.Canvas(pheromone_frame, width=20, height=20, bg='white', highlightthickness=0)
        mating_canvas.pack(side=tk.LEFT, padx=(15, 5))
        mating_canvas.create_rectangle(2, 2, 18, 18, fill='#ffff00', outline='')
        ttk.Label(pheromone_frame, text="Mating", font=('Arial', 9)).pack(side=tk.LEFT)
        
        # Grass legend
        grass_frame = ttk.Frame(parent)
        grass_frame.pack(fill=tk.X, pady=2)
        
        grass_canvas = tk.Canvas(grass_frame, width=20, height=20, bg='white', highlightthickness=0)
        grass_canvas.pack(side=tk.LEFT, padx=5)
        grass_canvas.create_rectangle(2, 2, 18, 18, fill='#2d5016', outline='')
        ttk.Label(grass_frame, text="Grass", font=('Arial', 9)).pack(side=tk.LEFT)
    
    def load_models(self, model_a_path=None):
        """Load trained models"""
        print("Loading models...")
        self.model_prey = ActorCriticNetwork(self.app.config).to(self.app.device)
        self.model_predator = ActorCriticNetwork(self.app.config).to(self.app.device)
        
        if model_a_path is None:
            model_a_path = PROJECT_ROOT / "outputs" / "checkpoints" / "model_A_ppo.pth"
            model_b_path = PROJECT_ROOT / "outputs" / "checkpoints" / "model_B_ppo.pth"
        else:
            model_a_path = Path(model_a_path)
            model_b_path = model_a_path.parent / model_a_path.name.replace("model_A", "model_B")
        
        try:
            if model_a_path.exists():
                self.model_prey.load_state_dict(
                    torch.load(str(model_a_path), map_location=self.app.device)
                )
            if model_b_path.exists():
                self.model_predator.load_state_dict(
                    torch.load(str(model_b_path), map_location=self.app.device)
                )
            print(f"Loaded models from {model_a_path.name}")
        except Exception as e:
            print(f"Error loading models: {e}")
        
        self.model_prey.eval()
        self.model_predator.eval()
        
        # Reset simulation with new models
        if hasattr(self, 'canvas'):
            self.reset_simulation()
    
    def draw_grid(self):
        """Draw grid lines"""
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1:
            w = h = 700
        
        for i in range(1, 10):
            x = i * w // 10
            y = i * h // 10
            self.canvas.create_line(x, 0, x, h, fill='#2d3561', tags='grid')
            self.canvas.create_line(0, y, w, y, fill='#2d3561', tags='grid')
    
    def reset_simulation(self):
        """Reset simulation"""
        seed_input = self.seed_var.get().strip()
        if seed_input:
            try:
                self.current_seed = int(seed_input)
            except ValueError:
                self.current_seed = hash(seed_input) % (2**31)
        else:
            self.current_seed = random.randint(0, 2**31 - 1)
        
        self.seed_var.set(str(self.current_seed))
        
        random.seed(self.current_seed)
        torch.manual_seed(self.current_seed)
        
        self.animals = self.create_population()
        self.pheromone_map = PheromoneMap(
            self.app.config.GRID_SIZE,
            decay_rate=self.app.config.PHEROMONE_DECAY,
            diffusion_rate=self.app.config.PHEROMONE_DIFFUSION
        )
        self.grass_field = GrassField(
            self.app.config.GRID_SIZE,
            getattr(self.app.config, 'GRASS_REGROW_INTERVAL', 5)
        )
        self.app.config.GRASS_FIELD = self.grass_field
        self.step_count = 0
        self.births = 0
        self.deaths = 0
        self.meals = 0
        self.grass_eaten = 0
        self.splashes = []  # Clear death splashes
        self.paused = True
        self.play_btn.config(text="▶ Play")
        
        # Reset evaluation metrics
        if hasattr(self.app, 'evaluation_tab'):
            self.app.evaluation_tab.reset_metrics()
        
        # Notify chart tab to reset
        if hasattr(self.app, 'chart_tab'):
            self.app.chart_tab.reset()
        
        self.update_stats()
        self.render()
    
    def randomize_seed(self):
        """Generate new random seed"""
        self.seed_var.set("")
        self.reset_simulation()
    
    def create_population(self):
        """Create initial population"""
        animals = []
        
        for _ in range(self.app.config.INITIAL_PREY_COUNT):
            x = random.randint(self.app.config.FIELD_MIN, self.app.config.FIELD_MAX)
            y = random.randint(self.app.config.FIELD_MIN, self.app.config.FIELD_MAX)
            animal = Prey(x, y, "A", "#00ff00")
            animal.energy = self.app.config.INITIAL_ENERGY
            animals.append(animal)
        
        for _ in range(self.app.config.INITIAL_PREDATOR_COUNT):
            x = random.randint(self.app.config.FIELD_MIN, self.app.config.FIELD_MAX)
            y = random.randint(self.app.config.FIELD_MIN, self.app.config.FIELD_MAX)
            animal = Predator(x, y, "B", "#ff0000")
            animal.energy = self.app.config.INITIAL_ENERGY
            animals.append(animal)
        
        return animals
    
    def toggle_pause(self):
        """Toggle pause"""
        self.paused = not self.paused
        self.play_btn.config(text="⏸ Pause" if not self.paused else "▶ Play")
        if not self.paused:
            self.run_step()
    
    def step_once(self):
        """Execute one step"""
        if not self.paused:
            return
        self.simulate_one_step()
        self.update_stats()
        self.render()
        
        # Update chart
        if hasattr(self.app, 'chart_tab'):
            self.app.chart_tab.update_chart(self.step_count, 
                                           sum(1 for a in self.animals if isinstance(a, Prey)),
                                           sum(1 for a in self.animals if isinstance(a, Predator)),
                                           self.births, self.deaths, self.meals, self.grass_eaten)
    
    def update_speed(self, value):
        """Update simulation speed"""
        speed_level = int(float(value))
        if speed_level >= 10:
            self.speed = 1
        else:
            self.speed = max(1, 201 - (speed_level * 22))
    
    def run_step(self):
        """Main simulation loop"""
        if self.update_id:
            self.frame.after_cancel(self.update_id)
            self.update_id = None
        
        if not self.paused:
            self.simulate_one_step()
            self.update_stats()
            self.render()
            
            # Update chart
            if hasattr(self.app, 'chart_tab'):
                self.app.chart_tab.update_chart(self.step_count,
                                               sum(1 for a in self.animals if isinstance(a, Prey)),
                                               sum(1 for a in self.animals if isinstance(a, Predator)),
                                               self.births, self.deaths, self.meals, self.grass_eaten)
            
            if not self.animals:
                self.paused = True
                self.play_btn.config(text="▶ Play")
                messagebox.showinfo("Simulation Ended", "All animals are dead!")
                return
            
            self.update_id = self.frame.after(self.speed, self.run_step)
    
    def simulate_one_step(self):
        """Advance simulation by one step"""
        animals_to_remove = []
        
        prey_list = [a for a in self.animals if isinstance(a, Prey)]
        predator_list = [a for a in self.animals if isinstance(a, Predator)]
        self.app.config._prey_count = len(prey_list)
        self.app.config._pred_count = len(predator_list)
        
        # Age updates
        for animal in self.animals:
            animal.update_age()
            if animal.is_old(self.app.config):
                animals_to_remove.append(animal)
                self.deaths += 1
                # Add gray splash for old age death
                self.splashes.append((animal.x, animal.y, '#888888', 20))
        
        # Movement
        active_animals = [a for a in self.animals if a not in animals_to_remove]
        
        # Track escapes: check active chases before movement
        if hasattr(self.app, 'evaluation_tab'):
            for chase_key in list(self.app.evaluation_tab.chase_step_counts.keys()):
                pred_id, prey_id = chase_key
                # Find the predator and prey
                predator = next((a for a in self.animals if a.id == pred_id), None)
                prey = next((a for a in self.animals if a.id == prey_id), None)
                
                # If prey no longer exists (died but not captured) or predator lost sight
                if not prey or not predator:
                    self.app.evaluation_tab.track_escape(None, None, chase_key)
                elif not predator.is_in_vision(prey.x, prey.y, self.app.config):
                    # Predator lost sight - prey escaped
                    self.app.evaluation_tab.track_escape(predator, prey, chase_key)
        
        for animal in active_animals:
            model = self.model_prey if isinstance(animal, Prey) else self.model_predator
            pos_before = (animal.x, animal.y)
            
            # Track predator-prey detections for evaluation
            if isinstance(animal, Predator) and hasattr(self.app, 'evaluation_tab'):
                # Check if predator can see any prey
                for other in self.animals:
                    if isinstance(other, Prey) and animal.is_in_vision(other.x, other.y, self.app.config):
                        self.app.evaluation_tab.track_detection(animal, other)
            
            with torch.no_grad():
                animal.move_training(model, self.animals, self.app.config, self.pheromone_map)
            moved = (animal.x, animal.y) != pos_before
            
            animal.update_energy(self.app.config, moved)
            if animal.is_exhausted():
                animals_to_remove.append(animal)
                self.deaths += 1
                # Add dark blue splash for exhaustion
                self.splashes.append((animal.x, animal.y, '#4a5d7a', 20))
        
        for animal in animals_to_remove:
            if animal in self.animals:
                self.animals.remove(animal)
        animals_to_remove.clear()
        
        # Eating
        for predator in [a for a in self.animals if isinstance(a, Predator)]:
            ate, _, eaten = predator.perform_eat(self.animals, self.app.config)
            if ate:
                self.meals += 1
                if eaten and eaten in self.animals:
                    # Track capture for evaluation
                    if hasattr(self.app, 'evaluation_tab'):
                        self.app.evaluation_tab.track_capture(predator, eaten)
                    # Add red splash for eaten prey
                    self.splashes.append((eaten.x, eaten.y, '#ff0000', 25))
                    self.animals.remove(eaten)
                    self.deaths += 1
            else:
                predator.steps_since_last_meal += 1
                if (self.app.config.STARVATION_ENABLED and
                        predator.steps_since_last_meal >= self.app.config.STARVATION_THRESHOLD):
                    animals_to_remove.append(predator)
                    self.deaths += 1
                    # Add brown splash for starvation
                    self.splashes.append((predator.x, predator.y, '#8b4513', 20))
        
        for animal in animals_to_remove:
            if animal in self.animals:
                self.animals.remove(animal)
        animals_to_remove.clear()
        
        # Mating
        new_animals = []
        mated_animals = set()
        pos_map = defaultdict(list)
        for a in self.animals:
            pos_map[(a.x, a.y)].append(a)
        
        for animal1 in self.animals:
            if animal1.id in mated_animals or not animal1.can_reproduce(self.app.config):
                continue
            
            mated = False
            for dx in (-1, 0, 1):
                if mated:
                    break
                for dy in (-1, 0, 1):
                    nx = (animal1.x + dx) % self.app.config.GRID_SIZE
                    ny = (animal1.y + dy) % self.app.config.GRID_SIZE
                    for animal2 in pos_map[(nx, ny)]:
                        if animal2.id <= animal1.id:
                            continue
                        if animal2.id in mated_animals or not animal2.can_reproduce(self.app.config):
                            continue
                        if animal1.can_mate(animal2, self.app.config):
                            mating_prob = (self.app.config.MATING_PROBABILITY_PREY
                                           if animal1.name == "A"
                                           else self.app.config.MATING_PROBABILITY_PREDATOR)
                            if random.random() < mating_prob:
                                child_x = (animal1.x + animal2.x) // 2
                                child_y = (animal1.y + animal2.y) // 2
                                if isinstance(animal1, Prey):
                                    child = Prey(child_x, child_y, animal1.name, animal1.color,
                                                 {animal1.id, animal2.id})
                                else:
                                    child = Predator(child_x, child_y, animal1.name, animal1.color,
                                                     {animal1.id, animal2.id})
                                child.energy = self.app.config.INITIAL_ENERGY
                                new_animals.append(child)
                                self.births += 1
                                
                                animal1.energy -= self.app.config.MATING_ENERGY_COST
                                animal2.energy -= self.app.config.MATING_ENERGY_COST
                                animal1.move_away(self.app.config)
                                animal2.move_away(self.app.config)
                                animal1.mating_cooldown = self.app.config.MATING_COOLDOWN
                                animal2.mating_cooldown = self.app.config.MATING_COOLDOWN
                                animal1.num_children += 1
                                animal2.num_children += 1
                                
                                mated_animals.add(animal1.id)
                                mated_animals.add(animal2.id)
                                mated = True
                                break
        
        # Add new animals
        prey_count = sum(1 for a in self.animals if isinstance(a, Prey))
        predator_count = sum(1 for a in self.animals if isinstance(a, Predator))
        new_prey = [a for a in new_animals if isinstance(a, Prey)]
        new_predators = [a for a in new_animals if isinstance(a, Predator)]
        
        if prey_count + len(new_prey) <= self.app.config.MAX_PREY:
            self.animals.extend(new_prey)
        else:
            available_prey_slots = max(0, self.app.config.MAX_PREY - prey_count)
            if available_prey_slots > 0:
                self.animals.extend(new_prey[:available_prey_slots])
        
        if predator_count + len(new_predators) <= self.app.config.MAX_PREDATORS:
            self.animals.extend(new_predators)
        else:
            available_predator_slots = max(0, self.app.config.MAX_PREDATORS - predator_count)
            if available_predator_slots > 0:
                self.animals.extend(new_predators[:available_predator_slots])
        
        # Cooldowns and pheromones
        for animal in self.animals:
            if animal.mating_cooldown > 0:
                animal.mating_cooldown -= 1
            animal.survival_time += 1
            animal.deposit_pheromones(self.animals, self.pheromone_map, self.app.config)
        
        # Prey eating grass
        for prey in [a for a in self.animals if isinstance(a, Prey)]:
            if prey.energy < getattr(self.app.config, 'PREY_HUNGER_THRESHOLD', 60.0):
                if self.grass_field.consume(prey.x, prey.y):
                    grass_energy = getattr(self.app.config, 'GRASS_ENERGY', 20.0)
                    prey.energy = min(prey.max_energy, prey.energy + grass_energy)
                    self.grass_eaten += 1
        
        self.pheromone_map.update()
        self.grass_field.step_regrow(self.step_count)
        
        # Update evaluation chase tracking
        if hasattr(self.app, 'evaluation_tab'):
            self.app.evaluation_tab.update_chase_steps()
        
        # Update splash effects (decay lifetime)
        self.splashes = [(x, y, color, life - 1) for x, y, color, life in self.splashes if life > 1]
        
        self.step_count += 1
    
    def update_stats(self):
        """Update statistics display"""
        prey_count = sum(1 for a in self.animals if isinstance(a, Prey))
        predator_count = sum(1 for a in self.animals if isinstance(a, Predator))
        
        self.stats_labels['Step'].config(text=str(self.step_count))
        self.stats_labels['Prey'].config(text=str(prey_count))
        self.stats_labels['Predators'].config(text=str(predator_count))
        self.stats_labels['Births'].config(text=str(self.births))
        self.stats_labels['Deaths'].config(text=str(self.deaths))
        self.stats_labels['Meals'].config(text=str(self.meals))
        self.stats_labels['Grass Eaten'].config(text=str(self.grass_eaten))
    
    def render(self):
        """Render animals on canvas"""
        self.canvas.delete('animal')
        self.canvas.delete('pheromone')
        self.canvas.delete('grid')
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1:
            w = h = 700
        
        field_range = self.app.config.GRID_SIZE
        scale_x = w / field_range
        scale_y = h / field_range
        
        def to_canvas(x, y):
            return x * scale_x, y * scale_y
        
        # Draw grass field (dark green cells)
        grass_cell_size = scale_x
        for x in range(self.app.config.GRID_SIZE):
            for y in range(self.app.config.GRID_SIZE):
                if self.grass_field.has_grass(x, y):
                    cx, cy = to_canvas(x, y)
                    # Light green for grass
                    self.canvas.create_rectangle(
                        cx, cy, cx + grass_cell_size, cy + grass_cell_size,
                        fill='#2d5016', outline='', tags='pheromone'
                    )
        
        # Draw pheromones
        sample_rate = 2
        cell_size = scale_x * sample_rate
        
        for x in range(0, self.app.config.GRID_SIZE, sample_rate):
            for y in range(0, self.app.config.GRID_SIZE, sample_rate):
                danger = self.pheromone_map.get_pheromone(x, y, 'danger')
                mating = self.pheromone_map.get_pheromone(x, y, 'mating')
                food = self.pheromone_map.get_pheromone(x, y, 'food')
                
                if danger > 0.05:
                    intensity = min(255, int(danger * 255))
                    red = max(80, intensity)
                    color = f'#{red:02x}0000'
                    cx, cy = to_canvas(x, y)
                    self.canvas.create_rectangle(
                        cx, cy, cx + cell_size, cy + cell_size,
                        fill=color, outline='', tags='pheromone'
                    )
                
                if mating > 0.05:
                    intensity = min(255, int(mating * 255))
                    yellow_val = max(80, intensity)
                    color = f'#{yellow_val:02x}{yellow_val:02x}00'
                    cx, cy = to_canvas(x, y)
                    self.canvas.create_rectangle(
                        cx, cy, cx + cell_size, cy + cell_size,
                        fill=color, outline='', tags='pheromone'
                    )
                
                if food > 0.05:
                    intensity = min(255, int(food * 255))
                    blue = max(80, intensity)
                    color = f'#0000{blue:02x}'
                    cx, cy = to_canvas(x, y)
                    self.canvas.create_rectangle(
                        cx, cy, cx + cell_size, cy + cell_size,
                        fill=color, outline='', tags='pheromone'
                    )
        
        # Draw grid on top of grass and pheromones
        for i in range(1, 10):
            x = i * w // 10
            y = i * h // 10
            self.canvas.create_line(x, 0, x, h, fill='#2d3561', tags='grid')
            self.canvas.create_line(0, y, w, y, fill='#2d3561', tags='grid')
        
        # Draw death splashes with fadeout effect
        for sx, sy, color, lifetime in self.splashes:
            cx, cy = to_canvas(sx, sy)
            # Size and opacity based on lifetime
            max_lifetime = 25  # Maximum lifetime
            alpha = int((lifetime / max_lifetime) * 255)
            size = scale_x * 0.8 * (lifetime / max_lifetime)
            
            # Draw splash as a star/splatter pattern
            splash_color = color
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                dx = math.cos(rad) * size
                dy = math.sin(rad) * size
                self.canvas.create_line(
                    cx, cy, cx + dx, cy + dy,
                    fill=splash_color, width=2, tags='pheromone'
                )
            # Center circle
            self.canvas.create_oval(
                cx - size/2, cy - size/2,
                cx + size/2, cy + size/2,
                fill=splash_color, outline='', tags='pheromone'
            )
        
        # Draw FOV helper function
        def draw_fov_cone(cx, cy, animal, color):
            """Draw cone-shaped field of view"""
            vision_range = animal.get_vision_range(self.app.config)
            vision_radius = vision_range * scale_x
            fov_deg = animal.get_fov_deg(self.app.config)
            
            heading_angle = math.atan2(animal.heading_dy, animal.heading_dx)
            heading_deg = math.degrees(heading_angle)
            
            center_angle = -heading_deg
            half_fov = fov_deg / 2
            start_angle = center_angle - half_fov
            extent_angle = fov_deg
            
            self.canvas.create_arc(
                cx - vision_radius, cy - vision_radius,
                cx + vision_radius, cy + vision_radius,
                start=start_angle, extent=extent_angle,
                outline=color, width=1, style=tk.PIESLICE,
                fill='', tags='animal'
            )
        
        # Draw FOV cones for prey
        for animal in self.animals:
            if isinstance(animal, Predator):
                continue
            cx, cy = to_canvas(animal.x, animal.y)
            draw_fov_cone(cx, cy, animal, '#5ac8fa')
            
            # Direction arrow
            arrow_len = 15
            dx, dy = animal.heading_dx, animal.heading_dy
            self.canvas.create_line(cx, cy, cx + dx*arrow_len, cy + dy*arrow_len,
                                   arrow=tk.LAST, fill='#0a84ff', width=2, tags='animal')
        
        # Draw FOV cones for predators
        for animal in self.animals:
            if isinstance(animal, Prey):
                continue
            cx, cy = to_canvas(animal.x, animal.y)
            color = '#ff9500' if animal.steps_since_last_meal >= self.app.config.HUNGER_THRESHOLD else '#ff6464'
            draw_fov_cone(cx, cy, animal, color)
            
            # Direction arrow
            arrow_len = 18
            dx, dy = animal.heading_dx, animal.heading_dy
            self.canvas.create_line(cx, cy, cx + dx*arrow_len, cy + dy*arrow_len,
                                   arrow=tk.LAST, fill='#ff3b30', width=2, tags='animal')
        
        # Draw prey on top (circles)
        for animal in self.animals:
            if isinstance(animal, Predator):
                continue
            cx, cy = to_canvas(animal.x, animal.y)
            self.canvas.create_oval(cx-7, cy-7, cx+7, cy+7,
                                   fill='#5ac8fa', outline='#0a84ff', width=2, tags='animal')
        
        # Draw predators on top (triangles)
        for animal in self.animals:
            if isinstance(animal, Prey):
                continue
            cx, cy = to_canvas(animal.x, animal.y)
            color = '#ff9500' if animal.steps_since_last_meal >= self.app.config.HUNGER_THRESHOLD else '#ff6464'
            self.canvas.create_polygon(cx, cy-8, cx-8, cy+8, cx+8, cy+8,
                                      fill=color, outline='#ff3b30', width=2, tags='animal')
    
    def cleanup(self):
        """Cleanup"""
        if self.update_id:
            self.frame.after_cancel(self.update_id)
