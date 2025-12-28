"""
Life Game Demo - Tkinter Visualization
Shows trained Actor-Critic agents in action with pheromones, energy, and age systems
"""

import os
import sys
import random
from collections import defaultdict
import torch
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import math
import ctypes
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Fix blurry text on Windows with DPI scaling
if sys.platform == 'win32':
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except:
            pass

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SimulationConfig
from src.core.animal import Prey, Predator
from src.models.actor_critic_network import ActorCriticNetwork
from src.core.pheromone_system import PheromoneMap


class LifeGameDemo:
    """Tkinter-based visualization of trained predator-prey models"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Life Game - Predator/Prey Simulation")
        self.root.geometry("2200x900")
        
        # Simulation state
        self.config = SimulationConfig()
        self.device = torch.device("cpu")
        self.animals = []
        self.pheromone_map = None
        self.model_prey = None
        self.model_predator = None
        self.paused = True
        self.step_count = 0
        self.births = 0
        self.deaths = 0
        self.meals = 0
        self.speed = 101  # Default delay ms (corresponds to speed_var=5)
        self.update_id = None
        self.current_seed = None
        
        # History tracking for charts
        self.history = {
            'step': [],
            'prey': [],
            'predators': [],
            'births': [],
            'deaths': [],
            'meals': []
        }
        
        # Load models
        self.load_models()
        
        # Build UI
        self.setup_ui()
        
        # Force layout to complete before initializing simulation
        self.root.update_idletasks()
        
        # Initialize simulation (deferred chart update)
        self.reset_simulation()
        
        # Schedule chart update after window is fully shown
        self.root.after(100, self.update_chart)
    
    def load_models(self):
        """Load trained PPO models"""
        print("Loading models...")
        self.model_prey = ActorCriticNetwork(self.config).to(self.device)
        self.model_predator = ActorCriticNetwork(self.config).to(self.device)
        
        try:
            self.model_prey.load_state_dict(
                torch.load("outputs/checkpoints/model_A_ppo.pth", map_location=self.device)
            )
            self.model_predator.load_state_dict(
                torch.load("outputs/checkpoints/model_B_ppo.pth", map_location=self.device)
            )
            print("Loaded trained PPO models")
        except FileNotFoundError:
            print("No trained models found, using random initialization")
        
        self.model_prey.eval()
        self.model_predator.eval()
    
    def setup_ui(self):
        """Build the UI"""
        # Top control frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(control_frame, text="Life Game Demo", 
                 font=('Arial', 16, 'bold')).pack(side=tk.LEFT, padx=5)
        
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
        ttk.Button(control_frame, text="🔄 Restart with seed",
                  command=self.reset_simulation).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Speed:").pack(side=tk.LEFT, padx=(15, 5))
        
        speed_container = ttk.Frame(control_frame)
        speed_container.pack(side=tk.LEFT, padx=5)
        
        self.speed_var = tk.IntVar(value=5)
        speed_scale = ttk.Scale(speed_container, from_=1, to=10, variable=self.speed_var,
                               orient=tk.HORIZONTAL, length=200, command=self.update_speed)
        speed_scale.grid(row=0, column=0, columnspan=10)
        
        # Speed labels (1 through 10) - use grid for precise alignment
        for i in range(1, 11):
            label = ttk.Label(speed_container, text=str(i), font=('Arial', 7))
            label.grid(row=1, column=i-1, sticky='w')
        
        # Configure columns to be equal width
        for i in range(10):
            speed_container.grid_columnconfigure(i, weight=1, uniform="speed")
        
        # Main content frame with tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Simulation
        simulation_tab = ttk.Frame(notebook)
        notebook.add(simulation_tab, text="Simulation")
        
        # Tab 2: Configuration
        config_tab = ttk.Frame(notebook)
        notebook.add(config_tab, text="Configuration")
        
        # Build simulation tab content
        self.setup_simulation_tab(simulation_tab)
        
        # Build configuration tab content
        self.setup_config_tab(config_tab)
    
    def setup_simulation_tab(self, parent):
        """Build the simulation visualization tab"""
        content_frame = parent
        
        # Left: Canvas for simulation (square)
        canvas_frame = ttk.LabelFrame(content_frame, text="Simulation Field", padding="10")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        # Create a container to center the square canvas
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_container, width=700, height=700, bg='#1a1a2e')
        self.canvas.pack(padx=20, pady=20)
        
        # Draw grid
        self.draw_grid()
        
        # Middle: Stats panel (pack first, between canvas and chart)
        stats_frame = ttk.LabelFrame(content_frame, text="Statistics", padding="10")
        stats_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        
        # Stats labels
        self.stats_labels = {}
        stats = ['Step', 'Prey', 'Predators', 'Births', 'Deaths', 'Meals']
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
        
        legend_frame = ttk.Frame(stats_frame)
        legend_frame.pack(fill=tk.X, pady=5)
        
        prey_canvas = tk.Canvas(legend_frame, width=20, height=20, bg='white', highlightthickness=0)
        prey_canvas.pack(side=tk.LEFT, padx=5)
        prey_canvas.create_oval(5, 5, 15, 15, fill='#5ac8fa', outline='#0a84ff')
        ttk.Label(legend_frame, text="Prey").pack(side=tk.LEFT)
        
        pred_canvas = tk.Canvas(legend_frame, width=20, height=20, bg='white', highlightthickness=0)
        pred_canvas.pack(side=tk.LEFT, padx=(15, 5))
        pred_canvas.create_polygon(10, 5, 5, 15, 15, 15, fill='#ff6464', outline='#ff3b30')
        ttk.Label(legend_frame, text="Predator").pack(side=tk.LEFT)
        
        # Second row for hungry predator
        legend_frame2 = ttk.Frame(stats_frame)
        legend_frame2.pack(fill=tk.X, pady=2)
        
        hungry_canvas = tk.Canvas(legend_frame2, width=20, height=20, bg='white', highlightthickness=0)
        hungry_canvas.pack(side=tk.LEFT, padx=5)
        hungry_canvas.create_polygon(10, 5, 5, 15, 15, 15, fill='#ff9500', outline='#ff3b30')
        ttk.Label(legend_frame2, text="Hungry", font=('Arial', 9)).pack(side=tk.LEFT)
        
        # Field of view indicators
        ttk.Label(stats_frame, text="Field of View:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        fov_frame = ttk.Frame(stats_frame)
        fov_frame.pack(fill=tk.X, pady=2)
        
        prey_fov_canvas = tk.Canvas(fov_frame, width=20, height=20, bg='white', highlightthickness=0)
        prey_fov_canvas.pack(side=tk.LEFT, padx=5)
        prey_fov_canvas.create_arc(2, 2, 18, 18, start=30, extent=240, outline='#5ac8fa', width=1, style=tk.ARC)
        ttk.Label(fov_frame, text="Prey (240°)", font=('Arial', 9)).pack(side=tk.LEFT)
        
        pred_fov_canvas = tk.Canvas(fov_frame, width=20, height=20, bg='white', highlightthickness=0)
        pred_fov_canvas.pack(side=tk.LEFT, padx=(15, 5))
        pred_fov_canvas.create_arc(2, 2, 18, 18, start=45, extent=180, outline='#ff6464', width=1, style=tk.ARC)
        ttk.Label(fov_frame, text="Predator (180°)", font=('Arial', 9)).pack(side=tk.LEFT)
        
        # Pheromone legend
        ttk.Label(stats_frame, text="Pheromones:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        pheromone_frame = ttk.Frame(stats_frame)
        pheromone_frame.pack(fill=tk.X, pady=2)
        
        danger_canvas = tk.Canvas(pheromone_frame, width=20, height=20, bg='white', highlightthickness=0)
        danger_canvas.pack(side=tk.LEFT, padx=5)
        danger_canvas.create_rectangle(2, 2, 18, 18, fill='#ff0000', outline='')
        ttk.Label(pheromone_frame, text="Danger", font=('Arial', 9)).pack(side=tk.LEFT)
        
        mating_canvas = tk.Canvas(pheromone_frame, width=20, height=20, bg='white', highlightthickness=0)
        mating_canvas.pack(side=tk.LEFT, padx=(15, 5))
        mating_canvas.create_rectangle(2, 2, 18, 18, fill='#00ff00', outline='')
        ttk.Label(pheromone_frame, text="Mating", font=('Arial', 9)).pack(side=tk.LEFT)
        
        food_frame = ttk.Frame(stats_frame)
        food_frame.pack(fill=tk.X, pady=2)
        
        food_canvas = tk.Canvas(food_frame, width=20, height=20, bg='white', highlightthickness=0)
        food_canvas.pack(side=tk.LEFT, padx=5)
        food_canvas.create_rectangle(2, 2, 18, 18, fill='#0000ff', outline='')
        ttk.Label(food_frame, text="Hunt Site", font=('Arial', 9)).pack(side=tk.LEFT)
        
        # Rightmost: Chart panel (pack last with side=tk.RIGHT)
        chart_frame = ttk.LabelFrame(content_frame, text="Statistics Over Time", padding="10")
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Create matplotlib figure with explicit initial size
        self._chart_dpi = 100.0
        try:
            self._chart_dpi = float(self.root.winfo_fpixels("1i"))
        except tk.TclError:
            pass
        # Set initial figure size (will be adjusted when widget resizes)
        # Use constrained_layout for better handling of dual y-axes
        self.fig = Figure(figsize=(10, 6), dpi=self._chart_dpi)
        
        # Make matplotlib recompute layout each draw (prevents clipping on resize)
        try:
            self.fig.set_layout_engine("tight")
        except Exception:
            # fallback for older matplotlib
            self.fig.set_constrained_layout(True)
        
        # Create canvas for matplotlib
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        chart_widget = self.chart_canvas.get_tk_widget()
        
        # Remove borders/highlight that cause off-by-few-px clipping
        chart_widget.configure(highlightthickness=0, bd=0)
        chart_widget.pack(fill=tk.BOTH, expand=True)
        
        # Use matplotlib's own resize handler (accounts for Tk canvas internals)
        chart_widget.bind("<Configure>", self.chart_canvas.resize)
        
        self._chart_widget = chart_widget
        self._chart_facecolor = self.root.cget("background")
        try:
            self._chart_facecolor = chart_widget.cget("background")
        except tk.TclError:
            pass
        
        # Initialize axes now
        self.ax = None
        self.ax2 = None
    
    def setup_config_tab(self, parent):
        """Build the configuration editor tab"""
        # Main container with canvas and scrollbar
        canvas_container = ttk.Frame(parent)
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for scrolling
        canvas = tk.Canvas(canvas_container, bg='white')
        scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Title
        title_frame = ttk.Frame(scrollable_frame)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(title_frame, text="Configuration Parameters", 
                 font=('Arial', 14, 'bold')).pack(side=tk.LEFT)
        
        # Save button
        ttk.Button(title_frame, text="💾 Save to config.py",
                  command=self.save_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(title_frame, text="🔄 Reload",
                  command=self.reload_config_ui).pack(side=tk.RIGHT, padx=5)
        
        # Store config entry widgets
        self.config_entries = {}
        
        # Get all config parameters grouped by category
        config_sections = self.get_config_sections()
        
        for section_name, params in config_sections:
            # Section frame
            section_frame = ttk.LabelFrame(scrollable_frame, text=section_name, padding="10")
            section_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Parameters in this section
            for param_name in params:
                if not hasattr(self.config, param_name):
                    continue
                    
                value = getattr(self.config, param_name)
                
                # Skip non-editable values
                if param_name.startswith('_') or callable(value) or isinstance(value, (dict, list)):
                    continue
                
                param_frame = ttk.Frame(section_frame)
                param_frame.pack(fill=tk.X, pady=2)
                
                # Label
                label = ttk.Label(param_frame, text=f"{param_name}:", width=30, anchor=tk.W)
                label.pack(side=tk.LEFT, padx=5)
                
                # Entry
                entry_var = tk.StringVar(value=str(value))
                entry = ttk.Entry(param_frame, textvariable=entry_var, width=20)
                entry.pack(side=tk.LEFT, padx=5)
                
                # Store reference
                self.config_entries[param_name] = (entry_var, type(value))
                
                # Current value display
                ttk.Label(param_frame, text=f"(type: {type(value).__name__})",
                         font=('Arial', 8), foreground='gray').pack(side=tk.LEFT, padx=5)
        
        # Bind mousewheel to canvas scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def get_config_sections(self):
        """Group config parameters by category based on comments in config.py"""
        sections = [
            ("Grid Settings", ['GRID_SIZE', 'FIELD_MIN', 'FIELD_MAX']),
            ("Population Settings", ['INITIAL_PREY_COUNT', 'INITIAL_PREDATOR_COUNT', 'MAX_PREY', 'MAX_PREDATORS']),
            ("Animal Behavior", ['PREDATOR_VISION_RANGE', 'PREY_VISION_RANGE', 'VISION_RANGE', 'MAX_VISIBLE_ANIMALS',
                                'PREY_FOV_DEG', 'PREDATOR_FOV_DEG', 'VISION_SHAPE',
                                'HUNGER_THRESHOLD', 'STARVATION_THRESHOLD', 'MATING_COOLDOWN']),
            ("Movement Speeds", ['PREDATOR_HUNGRY_MOVES', 'PREDATOR_NORMAL_MOVES', 'PREY_MOVES']),
            ("Mating Probabilities", ['MATING_PROBABILITY_PREY', 'MATING_PROBABILITY_PREDATOR']),
            ("Learning Rates", ['LEARNING_RATE_PREY', 'LEARNING_RATE_PREDATOR', 'GAMMA', 'ACTION_TEMPERATURE']),
            ("Rewards", ['SURVIVAL_REWARD', 'REPRODUCTION_REWARD', 'PREDATOR_EAT_REWARD', 'PREY_EVASION_REWARD',
                        'PREDATOR_APPROACH_REWARD', 'PREY_MATE_APPROACH_REWARD']),
            ("Penalties", ['EXTINCTION_PENALTY', 'DEATH_PENALTY', 'STARVATION_PENALTY', 'EATEN_PENALTY',
                          'EXHAUSTION_PENALTY', 'OLD_AGE_PENALTY', 'OVERPOPULATION_PENALTY']),
            ("Energy System", ['INITIAL_ENERGY', 'MAX_ENERGY', 'ENERGY_DECAY_RATE', 'MOVE_ENERGY_COST',
                              'MATING_ENERGY_COST', 'EATING_ENERGY_GAIN', 'REST_ENERGY_GAIN']),
            ("Age System", ['MAX_AGE', 'MATURITY_AGE']),
            ("Pheromone System", ['PHEROMONE_DECAY', 'PHEROMONE_DIFFUSION', 'DANGER_PHEROMONE_STRENGTH',
                                  'MATING_PHEROMONE_STRENGTH', 'PHEROMONE_SENSING_RANGE']),
            ("PPO Training", ['NUM_EPISODES', 'STEPS_PER_EPISODE', 'PPO_EPOCHS', 'PPO_CLIP_EPSILON',
                             'PPO_BATCH_SIZE', 'VALUE_LOSS_COEF', 'ENTROPY_COEF', 'MAX_GRAD_NORM', 'GAE_LAMBDA']),
            ("Curriculum Learning", ['CURRICULUM_ENABLED', 'STARVATION_ENABLED']),
        ]
        return sections
    
    def reload_config_ui(self):
        """Reload config values into UI from current config"""
        for param_name, (entry_var, _) in self.config_entries.items():
            if hasattr(self.config, param_name):
                value = getattr(self.config, param_name)
                entry_var.set(str(value))
        messagebox.showinfo("Reloaded", "Configuration values reloaded from current config.")
    
    def save_config(self):
        """Save edited config values to config.py file"""
        try:
            # Read current config file
            config_path = PROJECT_ROOT / 'src' / 'config.py'
            with open(config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Update values
            updated_lines = []
            for line in lines:
                updated = False
                for param_name, (entry_var, param_type) in self.config_entries.items():
                    # Match lines like: PARAM_NAME = value
                    if line.strip().startswith(f"{param_name} ="):
                        try:
                            # Get new value and convert to proper type
                            new_value_str = entry_var.get()
                            if param_type == bool:
                                new_value = new_value_str.lower() in ('true', '1', 'yes')
                            elif param_type == int:
                                new_value = int(float(new_value_str))  # Handle "10.0" -> 10
                            elif param_type == float:
                                new_value = float(new_value_str)
                            elif param_type == str:
                                new_value = new_value_str
                            else:
                                new_value = new_value_str
                            
                            # Preserve indentation and comments
                            indent = len(line) - len(line.lstrip())
                            comment_idx = line.find('#')
                            comment = line[comment_idx:] if comment_idx > 0 else '\n'
                            
                            # Format value
                            if isinstance(new_value, str) and param_type == str:
                                value_str = f'"{new_value}"'
                            elif isinstance(new_value, bool):
                                value_str = str(new_value)
                            else:
                                value_str = str(new_value)
                            
                            # Reconstruct line
                            new_line = f"{' ' * indent}{param_name} = {value_str}  {comment}"
                            updated_lines.append(new_line)
                            updated = True
                            
                            # Update runtime config
                            setattr(self.config, param_name, new_value)
                            break
                        except ValueError as e:
                            messagebox.showerror("Invalid Value", 
                                               f"Invalid value for {param_name}: {entry_var.get()}\n{e}")
                            return
                
                if not updated:
                    updated_lines.append(line)
            
            # Write back to file
            with open(config_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
            
            messagebox.showinfo("Saved", 
                              f"Configuration saved to {config_path}\n\n"
                              "Training scripts will use these values on next run.")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save configuration:\n{e}")

    def draw_grid(self):
        """Draw grid lines on canvas"""
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1:
            w = 700
            h = 700
        
        # Draw 10x10 grid
        for i in range(1, 10):
            x = i * w // 10
            y = i * h // 10
            self.canvas.create_line(x, 0, x, h, fill='#2d3561', tags='grid')
            self.canvas.create_line(0, y, w, y, fill='#2d3561', tags='grid')
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        # Get seed from input or generate random one
        seed_input = self.seed_var.get().strip()
        if seed_input:
            try:
                self.current_seed = int(seed_input)
            except ValueError:
                # Use hash of string as seed
                self.current_seed = hash(seed_input) % (2**31)
        else:
            # Generate new random seed only if field is empty
            self.current_seed = random.randint(0, 2**31 - 1)
        
        # Always update display to show the actual seed being used
        self.seed_var.set(str(self.current_seed))
        
        # Set random seeds for reproducibility
        random.seed(self.current_seed)
        torch.manual_seed(self.current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.current_seed)
        
        self.animals = self.create_population()
        self.pheromone_map = PheromoneMap(
            self.config.GRID_SIZE,
            decay_rate=self.config.PHEROMONE_DECAY,
            diffusion_rate=self.config.PHEROMONE_DIFFUSION
        )
        self.step_count = 0
        self.births = 0
        self.deaths = 0
        self.meals = 0
        self.paused = True
        self.play_btn.config(text="▶ Play")
        
        # Clear history
        self.history = {
            'step': [],
            'prey': [],
            'predators': [],
            'births': [],
            'deaths': [],
            'meals': []
        }
        
        self.update_stats()
        self.update_chart()
        self.render()
    
    def randomize_seed(self):
        """Generate new random seed and reset simulation"""
        self.seed_var.set("")  # Clear field to trigger new seed generation
        self.reset_simulation()
    
    def create_population(self):
        """Create initial population"""
        animals = []
        
        # Create prey
        for _ in range(self.config.INITIAL_PREY_COUNT):
            x = random.randint(self.config.FIELD_MIN, self.config.FIELD_MAX)
            y = random.randint(self.config.FIELD_MIN, self.config.FIELD_MAX)
            animal = Prey(x, y, "A", "#00ff00")
            animal.energy = self.config.INITIAL_ENERGY
            animals.append(animal)
        
        # Create predators
        for _ in range(self.config.INITIAL_PREDATOR_COUNT):
            x = random.randint(self.config.FIELD_MIN, self.config.FIELD_MAX)
            y = random.randint(self.config.FIELD_MIN, self.config.FIELD_MAX)
            animal = Predator(x, y, "B", "#ff0000")
            animal.energy = self.config.INITIAL_ENERGY
            animals.append(animal)
        
        return animals
    
    def toggle_pause(self):
        """Toggle simulation pause state"""
        self.paused = not self.paused
        self.play_btn.config(text="⏸ Pause" if not self.paused else "▶ Play")
        if not self.paused:
            self.run_step()
    
    def step_once(self):
        """Execute one simulation step"""
        if not self.paused:
            return
        self.simulate_one_step()
        self.update_stats()
        self.update_chart()
        self.render()
    
    def update_speed(self, value):
        """Update simulation speed (inverted: higher slider = faster)"""
        speed_level = int(float(value))
        # Convert 1-10 scale to delay: 1=200ms (slow), 10=0ms (no pause)
        if speed_level >= 10:
            self.speed = 1  # Minimum delay for tkinter
        else:
            self.speed = max(1, 201 - (speed_level * 22))
    
    def run_step(self):
        """Main simulation loop"""
        if self.update_id:
            self.root.after_cancel(self.update_id)
            self.update_id = None
        
        if not self.paused:
            self.simulate_one_step()
            self.update_stats()
            self.update_chart()
            self.render()
            
            # Check for end condition
            if not self.animals:
                self.paused = True
                self.play_btn.config(text="▶ Play")
                messagebox.showinfo("Simulation Ended", "All animals are dead!")
                return
            
            # Schedule next step
            self.update_id = self.root.after(self.speed, self.run_step)
    
    def simulate_one_step(self):
        """Advance simulation by one step"""
        animals_to_remove = []
        
        # Cache counts for observation features (matches training)
        prey_list = [a for a in self.animals if isinstance(a, Prey)]
        predator_list = [a for a in self.animals if isinstance(a, Predator)]
        self.config._prey_count = len(prey_list)
        self.config._pred_count = len(predator_list)
        
        # Age updates and old age deaths
        for animal in self.animals:
            animal.update_age()
            if animal.is_old(self.config):
                animals_to_remove.append(animal)
                self.deaths += 1
        
        # Movement phase (skip animals already marked for removal)
        active_animals = [a for a in self.animals if a not in animals_to_remove]
        for animal in active_animals:
            model = self.model_prey if isinstance(animal, Prey) else self.model_predator
            pos_before = (animal.x, animal.y)
            with torch.no_grad():
                animal.move_training(model, self.animals, self.config, self.pheromone_map)
            moved = (animal.x, animal.y) != pos_before
            
            animal.update_energy(self.config, moved)
            if animal.is_exhausted():
                animals_to_remove.append(animal)
                self.deaths += 1
        
        # Remove old/exhausted animals
        for animal in animals_to_remove:
            if animal in self.animals:
                self.animals.remove(animal)
        animals_to_remove.clear()
        
        # Predators eat and starvation checks (matches training order)
        for predator in [a for a in self.animals if isinstance(a, Predator)]:
            ate, _, eaten = predator.perform_eat(self.animals, self.config)
            if ate:
                self.meals += 1
                if eaten and eaten in self.animals:
                    self.animals.remove(eaten)
                    self.deaths += 1
            else:
                predator.steps_since_last_meal += 1
                if (self.config.STARVATION_ENABLED and
                        predator.steps_since_last_meal >= self.config.STARVATION_THRESHOLD):
                    animals_to_remove.append(predator)
                    self.deaths += 1
        
        for animal in animals_to_remove:
            if animal in self.animals:
                self.animals.remove(animal)
        animals_to_remove.clear()
        
        # Mating phase (spatial hash to match training)
        new_animals = []
        mated_animals = set()
        pos_map = defaultdict(list)
        for a in self.animals:
            pos_map[(a.x, a.y)].append(a)
        
        for animal1 in self.animals:
            if animal1.id in mated_animals or not animal1.can_reproduce(self.config):
                continue
            
            mated = False
            for dx in (-1, 0, 1):
                if mated:
                    break
                for dy in (-1, 0, 1):
                    nx = (animal1.x + dx) % self.config.GRID_SIZE
                    ny = (animal1.y + dy) % self.config.GRID_SIZE
                    for animal2 in pos_map[(nx, ny)]:
                        if animal2.id <= animal1.id:
                            continue
                        if animal2.id in mated_animals or not animal2.can_reproduce(self.config):
                            continue
                        if animal1.can_mate(animal2, self.config):
                            mating_prob = (self.config.MATING_PROBABILITY_PREY
                                           if animal1.name == "A"
                                           else self.config.MATING_PROBABILITY_PREDATOR)
                            if random.random() < mating_prob:
                                child_x = (animal1.x + animal2.x) // 2
                                child_y = (animal1.y + animal2.y) // 2
                                if isinstance(animal1, Prey):
                                    child = Prey(child_x, child_y, animal1.name, animal1.color,
                                                 {animal1.id, animal2.id})
                                else:
                                    child = Predator(child_x, child_y, animal1.name, animal1.color,
                                                     {animal1.id, animal2.id})
                                child.energy = self.config.INITIAL_ENERGY
                                new_animals.append(child)
                                self.births += 1
                                
                                animal1.energy -= self.config.MATING_ENERGY_COST
                                animal2.energy -= self.config.MATING_ENERGY_COST
                                animal1.move_away(self.config)
                                animal2.move_away(self.config)
                                animal1.mating_cooldown = self.config.MATING_COOLDOWN
                                animal2.mating_cooldown = self.config.MATING_COOLDOWN
                                animal1.num_children += 1
                                animal2.num_children += 1
                                
                                mated_animals.add(animal1.id)
                                mated_animals.add(animal2.id)
                                mated = True
                                break
        
        # Add new animals (separate capacity limits)
        prey_count = sum(1 for a in self.animals if isinstance(a, Prey))
        predator_count = sum(1 for a in self.animals if isinstance(a, Predator))
        new_prey = [a for a in new_animals if isinstance(a, Prey)]
        new_predators = [a for a in new_animals if isinstance(a, Predator)]
        
        if prey_count + len(new_prey) <= self.config.MAX_PREY:
            self.animals.extend(new_prey)
        else:
            available_prey_slots = max(0, self.config.MAX_PREY - prey_count)
            if available_prey_slots > 0:
                self.animals.extend(new_prey[:available_prey_slots])
        
        if predator_count + len(new_predators) <= self.config.MAX_PREDATORS:
            self.animals.extend(new_predators)
        else:
            available_predator_slots = max(0, self.config.MAX_PREDATORS - predator_count)
            if available_predator_slots > 0:
                self.animals.extend(new_predators[:available_predator_slots])
        
        # Cooldowns, survival time, and pheromones (matches training order)
        for animal in self.animals:
            if animal.mating_cooldown > 0:
                animal.mating_cooldown -= 1
            animal.survival_time += 1
            animal.deposit_pheromones(self.animals, self.pheromone_map, self.config)
        
        self.pheromone_map.update()
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
        
        # Add to history
        self.history['step'].append(self.step_count)
        self.history['prey'].append(prey_count)
        self.history['predators'].append(predator_count)
        self.history['births'].append(self.births)
        self.history['deaths'].append(self.deaths)
        self.history['meals'].append(self.meals)
    
    def update_chart(self):
        """Update statistics chart"""
        # Ensure widget has valid dimensions
        if not hasattr(self, '_chart_widget'):
            return
        
        width = self._chart_widget.winfo_width()
        height = self._chart_widget.winfo_height()
        if width <= 1 or height <= 1:
            # Widget not ready yet, schedule retry
            self.root.after(50, self.update_chart)
            return
        
        # Clear entire figure to prevent line multiplication
        self.fig.clear()
        
        # Re-apply layout engine after clear (some matplotlib versions reset it)
        try:
            self.fig.set_layout_engine("tight")
        except Exception:
            self.fig.set_constrained_layout(True)
        
        # Recreate axes with proper facecolor to avoid white margins
        bg = self._chart_facecolor
        self.fig.patch.set_facecolor(bg)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(bg)
        self.ax.set_xlabel('Step')
        self.ax.set_ylabel('Population Count')
        self.ax.set_title('Population & Events Over Time')
        self.ax.grid(True, alpha=0.3)
        
        if len(self.history['step']) > 0:
            steps = self.history['step']
            
            # Plot population lines on primary axis
            self.ax.plot(steps, self.history['prey'], label='Prey', color='#5ac8fa', linewidth=2)
            self.ax.plot(steps, self.history['predators'], label='Predators', color='#ff6464', linewidth=2)
            
            # Set Y axis to start from 0
            self.ax.set_ylim(bottom=0)
            self.ax.legend(loc='upper left')
            
            # Create secondary axis for events
            self.ax2 = self.ax.twinx()
            self.ax2.set_facecolor(bg)
            self.ax2.set_ylabel('Events Count', color='gray')
            self.ax2.plot(steps, self.history['births'], label='Births', color='#34c759', linestyle='--', linewidth=1.5)
            self.ax2.plot(steps, self.history['deaths'], label='Deaths', color='#ff9500', linestyle='--', linewidth=1.5)
            self.ax2.plot(steps, self.history['meals'], label='Meals', color='#af52de', linestyle='--', linewidth=1.5)
            self.ax2.tick_params(axis='y', labelcolor='gray')
            
            # Set secondary Y axis to start from 0
            self.ax2.set_ylim(bottom=0)
            self.ax2.legend(loc='upper right')
        else:
            # Show empty chart with axes configured
            self.ax.set_ylim(0, 100)
            self.ax.text(50, 50, 'Waiting for data...', 
                        ha='center', va='center', fontsize=12, color='gray', alpha=0.5)

        # Use draw_idle() to batch redraws and reduce flicker
        self.chart_canvas.draw_idle()
    
    def render(self):
        """Render animals on canvas"""
        # Clear previous animals (keep grid)
        self.canvas.delete('animal')
        self.canvas.delete('pheromone')
        
        # Get canvas dimensions
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1:
            w = 700
            h = 700
        
        # Calculate scale
        field_range = self.config.GRID_SIZE
        scale_x = w / field_range
        scale_y = h / field_range
        
        def to_canvas(x, y):
            cx = x * scale_x
            cy = y * scale_y
            return cx, cy
        
        # Draw pheromone heatmap (sample at lower resolution for performance)
        sample_rate = 2  # Sample every 2nd cell (finer detail)
        cell_size = scale_x * sample_rate
        
        for x in range(0, self.config.GRID_SIZE, sample_rate):
            for y in range(0, self.config.GRID_SIZE, sample_rate):
                # Get pheromone intensities
                danger = self.pheromone_map.get_pheromone(x, y, 'danger')
                mating = self.pheromone_map.get_pheromone(x, y, 'mating')
                food = self.pheromone_map.get_pheromone(x, y, 'food')
                
                # Draw each pheromone type if present (threshold 0.05 to show faint traces)
                if danger > 0.05:
                    # Use RGB with minimum brightness so color is always visible
                    intensity = min(255, int(danger * 255))
                    # Ensure minimum red of 80 so it doesn't fade to black
                    red = max(80, intensity)
                    color = f'#{red:02x}0000'
                    cx, cy = to_canvas(x, y)
                    self.canvas.create_rectangle(
                        cx, cy, cx + cell_size, cy + cell_size,
                        fill=color, outline='', tags='pheromone'
                    )
                
                if mating > 0.05:
                    intensity = min(255, int(mating * 255))
                    green = max(80, intensity)
                    color = f'#00{green:02x}00'
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
        
        def draw_fov_cone(cx, cy, animal, color):
            """Draw cone-shaped field of view"""
            vision_range = animal.get_vision_range(self.config)
            vision_radius = vision_range * scale_x
            fov_deg = animal.get_fov_deg(self.config)
            
            # Calculate heading angle in degrees
            # atan2 gives angle in standard math: counter-clockwise from positive X
            heading_angle = math.atan2(animal.heading_dy, animal.heading_dx)
            heading_deg = math.degrees(heading_angle)
            
            # Tkinter arc: 0° = East (right), 90° = North (up), counter-clockwise
            # Our canvas: Y increases downward, so we need to flip the Y component
            # Conversion: negate the angle from atan2
            center_angle = -heading_deg
            half_fov = fov_deg / 2
            start_angle = center_angle - half_fov
            extent_angle = fov_deg
            
            # Draw the vision cone as a filled arc
            self.canvas.create_arc(
                cx - vision_radius, cy - vision_radius,
                cx + vision_radius, cy + vision_radius,
                start=start_angle, extent=extent_angle,
                outline=color, width=1, style=tk.PIESLICE,
                fill='', tags='animal'
            )
            
            # Draw the two boundary lines of the FOV cone
            angle1 = math.radians(heading_deg - half_fov)
            angle2 = math.radians(heading_deg + half_fov)
            
            x1 = cx + vision_radius * math.cos(angle1)
            y1 = cy + vision_radius * math.sin(angle1)
            x2 = cx + vision_radius * math.cos(angle2)
            y2 = cy + vision_radius * math.sin(angle2)
            
            self.canvas.create_line(cx, cy, x1, y1, fill=color, width=1, dash=(2, 4), tags='animal')
            self.canvas.create_line(cx, cy, x2, y2, fill=color, width=1, dash=(2, 4), tags='animal')
        
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
            color = '#ff9500' if animal.steps_since_last_meal >= self.config.HUNGER_THRESHOLD else '#ff6464'
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
            color = '#ff9500' if animal.steps_since_last_meal >= self.config.HUNGER_THRESHOLD else '#ff6464'
            self.canvas.create_polygon(cx, cy-8, cx-8, cy+8, cx+8, cy+8,
                                      fill=color, outline='#ff3b30', width=2, tags='animal')
    
    def cleanup(self):
        """Cleanup before closing"""
        if self.update_id:
            self.root.after_cancel(self.update_id)


def main():
    """Run the demo"""
    print("\n" + "=" * 70)
    print("  LIFE GAME DEMO - Predator/Prey Simulation (Tkinter)")
    print("=" * 70)
    
    root = tk.Tk()
    app = LifeGameDemo(root)
    
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
