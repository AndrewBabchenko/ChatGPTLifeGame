"""
Config Tab - Configuration parameter editor
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path


class ConfigTab:
    """Tab for editing configuration parameters"""
    
    def __init__(self, parent, app):
        self.app = app
        self.frame = ttk.Frame(parent)
        self.config_entries = {}
        
        # Parameter descriptions/comments
        self.param_descriptions = {
            'GRID_SIZE': 'Size of the simulation grid (NxN)',
            'FIELD_MIN': 'Minimum coordinate value',
            'FIELD_MAX': 'Maximum coordinate value',
            'INITIAL_PREY_COUNT': 'Starting number of prey animals',
            'INITIAL_PREDATOR_COUNT': 'Starting number of predators',
            'MAX_PREY': 'Maximum allowed prey population',
            'MAX_PREDATORS': 'Maximum allowed predator population',
            'PREDATOR_VISION_RANGE': 'How far predators can see',
            'PREY_VISION_RANGE': 'How far prey can see',
            'VISION_RANGE': 'Default vision range',
            'MAX_VISIBLE_ANIMALS': 'Max animals in field of view',
            'PREY_FOV_DEG': 'Prey field of view in degrees',
            'PREDATOR_FOV_DEG': 'Predator field of view in degrees',
            'VISION_SHAPE': 'Shape of vision (cone/circle)',
            'HUNGER_THRESHOLD': 'Energy level when hungry',
            'STARVATION_THRESHOLD': 'Steps without food before death',
            'MATING_COOLDOWN': 'Steps between mating events',
            'PREDATOR_HUNGRY_MOVES': 'Movement speed when hungry',
            'PREDATOR_NORMAL_MOVES': 'Normal movement speed',
            'PREY_MOVES': 'Prey movement speed',
            'MATING_PROBABILITY_PREY': 'Chance of prey mating (0-1)',
            'MATING_PROBABILITY_PREDATOR': 'Chance of predator mating (0-1)',
            'LEARNING_RATE_PREY': 'Neural network learning rate for prey',
            'LEARNING_RATE_PREDATOR': 'Neural network learning rate for predators',
            'GAMMA': 'Discount factor for future rewards',
            'ACTION_TEMPERATURE': 'Exploration vs exploitation parameter',
            'SURVIVAL_REWARD': 'Reward for staying alive',
            'REPRODUCTION_REWARD': 'Reward for mating',
            'PREDATOR_EAT_REWARD': 'Reward for catching prey',
            'PREY_EVASION_REWARD': 'Reward for escaping predator',
            'PREDATOR_APPROACH_REWARD': 'Reward for getting closer to prey',
            'PREY_MATE_APPROACH_REWARD': 'Reward for approaching mate',
            'EXTINCTION_PENALTY': 'Penalty when species extinct',
            'DEATH_PENALTY': 'Penalty for dying',
            'STARVATION_PENALTY': 'Penalty for starving',
            'EATEN_PENALTY': 'Penalty when eaten',
            'EXHAUSTION_PENALTY': 'Penalty for low energy',
            'OLD_AGE_PENALTY': 'Penalty for aging',
            'OVERPOPULATION_PENALTY': 'Penalty for overcrowding',
            'INITIAL_ENERGY': 'Starting energy level',
            'MAX_ENERGY': 'Maximum energy capacity',
            'ENERGY_DECAY_RATE': 'Energy loss per step',
            'MOVE_ENERGY_COST': 'Energy cost for movement',
            'MATING_ENERGY_COST': 'Energy cost for mating',
            'EATING_ENERGY_GAIN': 'Energy gained from eating',
            'REST_ENERGY_GAIN': 'Energy gained from resting',
            'MAX_AGE': 'Maximum lifespan in steps',
            'MATURITY_AGE': 'Age when can reproduce',
            'PHEROMONE_DECAY': 'Pheromone dissipation rate (0-1)',
            'PHEROMONE_DIFFUSION': 'Pheromone spread rate (0-1)',
            'DANGER_PHEROMONE_STRENGTH': 'Intensity of danger signals',
            'MATING_PHEROMONE_STRENGTH': 'Intensity of mating signals',
            'PHEROMONE_SENSING_RANGE': 'Detection range for pheromones',
            'NUM_EPISODES': 'Training: Total episodes to run',
            'STEPS_PER_EPISODE': 'Training: Steps per episode',
            'PPO_EPOCHS': 'Training: PPO optimization epochs',
            'PPO_CLIP_EPSILON': 'Training: PPO clipping parameter',
            'PPO_BATCH_SIZE': 'Training: Batch size for updates',
            'VALUE_LOSS_COEF': 'Training: Value function loss weight',
            'ENTROPY_COEF': 'Training: Entropy bonus weight',
            'MAX_GRAD_NORM': 'Training: Gradient clipping threshold',
            'GAE_LAMBDA': 'Training: Advantage estimation parameter',
            'CURRICULUM_ENABLED': 'Training: Use curriculum learning',
            'STARVATION_ENABLED': 'Enable death by starvation',
            'GRASS_REGROW_INTERVAL': 'Steps between grass regrowth',
            'GRASS_ENERGY': 'Energy gained from eating grass',
            'PREY_HUNGER_THRESHOLD': 'Energy level when prey seeks grass',
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Build configuration editor UI"""
        # Main container with canvas and scrollbar
        canvas_container = ttk.Frame(self.frame)
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
        
        # Get all config parameters grouped by category
        config_sections = self.get_config_sections()
        
        for section_name, params in config_sections:
            # Section frame
            section_frame = ttk.LabelFrame(scrollable_frame, text=section_name, padding="10")
            section_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Parameters in this section
            for param_name in params:
                if not hasattr(self.app.config, param_name):
                    continue
                
                value = getattr(self.app.config, param_name)
                
                # Skip non-editable values
                if param_name.startswith('_') or callable(value) or isinstance(value, (dict, list)):
                    continue
                
                param_frame = ttk.Frame(section_frame)
                param_frame.pack(fill=tk.X, pady=2)
                
                # Label with parameter name
                label = ttk.Label(param_frame, text=f"{param_name}:", width=30, anchor=tk.W,
                                 font=('Arial', 9, 'bold'))
                label.pack(side=tk.LEFT, padx=5)
                
                # Entry
                entry_var = tk.StringVar(value=str(value))
                entry = ttk.Entry(param_frame, textvariable=entry_var, width=20)
                entry.pack(side=tk.LEFT, padx=5)
                
                # Store reference
                self.config_entries[param_name] = (entry_var, type(value))
                
                # Type and description
                description = self.param_descriptions.get(param_name, '')
                info_text = f"({type(value).__name__})"
                if description:
                    info_text = f"{description} - {info_text}"
                ttk.Label(param_frame, text=info_text,
                         font=('Arial', 8), foreground='#555').pack(side=tk.LEFT, padx=5)
        
        # Bind mousewheel to canvas scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def get_config_sections(self):
        """Group config parameters by category (excluding training parameters for demo)"""
        sections = [
            ("Grid Settings", ['GRID_SIZE', 'FIELD_MIN', 'FIELD_MAX']),
            ("Population Settings", ['INITIAL_PREY_COUNT', 'INITIAL_PREDATOR_COUNT', 'MAX_PREY', 'MAX_PREDATORS']),
            ("Animal Behavior", ['PREDATOR_VISION_RANGE', 'PREY_VISION_RANGE', 'VISION_RANGE', 'MAX_VISIBLE_ANIMALS',
                                'PREY_FOV_DEG', 'PREDATOR_FOV_DEG', 'VISION_SHAPE',
                                'HUNGER_THRESHOLD', 'STARVATION_THRESHOLD', 'MATING_COOLDOWN']),
            ("Movement Speeds", ['PREDATOR_HUNGRY_MOVES', 'PREDATOR_NORMAL_MOVES', 'PREY_MOVES']),
            ("Mating Probabilities", ['MATING_PROBABILITY_PREY', 'MATING_PROBABILITY_PREDATOR']),
            ("Energy System", ['INITIAL_ENERGY', 'MAX_ENERGY', 'ENERGY_DECAY_RATE', 'MOVE_ENERGY_COST',
                              'MATING_ENERGY_COST', 'EATING_ENERGY_GAIN', 'REST_ENERGY_GAIN']),
            ("Age System", ['MAX_AGE', 'MATURITY_AGE']),
            ("Pheromone System", ['PHEROMONE_DECAY', 'PHEROMONE_DIFFUSION', 'DANGER_PHEROMONE_STRENGTH',
                                  'MATING_PHEROMONE_STRENGTH', 'PHEROMONE_SENSING_RANGE']),
            ("Grass System", ['GRASS_REGROW_INTERVAL', 'GRASS_ENERGY', 'PREY_HUNGER_THRESHOLD']),
            ("General Settings", ['STARVATION_ENABLED']),
        ]
        return sections
    
    def reload_config_ui(self):
        """Reload config values into UI from current config"""
        for param_name, (entry_var, _) in self.config_entries.items():
            if hasattr(self.app.config, param_name):
                value = getattr(self.app.config, param_name)
                entry_var.set(str(value))
        messagebox.showinfo("Reloaded", "Configuration values reloaded from current config.")
    
    def save_config(self):
        """Save edited config values to config.py file"""
        try:
            from pathlib import Path
            PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
            config_path = PROJECT_ROOT / 'src' / 'config.py'
            
            with open(config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Update values
            updated_lines = []
            for line in lines:
                updated = False
                for param_name, (entry_var, param_type) in self.config_entries.items():
                    if line.strip().startswith(f"{param_name} ="):
                        try:
                            new_value_str = entry_var.get()
                            if param_type == bool:
                                new_value = new_value_str.lower() in ('true', '1', 'yes')
                            elif param_type == int:
                                new_value = int(float(new_value_str))
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
                            
                            new_line = f"{' ' * indent}{param_name} = {value_str}  {comment}"
                            updated_lines.append(new_line)
                            updated = True
                            
                            # Update runtime config
                            setattr(self.app.config, param_name, new_value)
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
