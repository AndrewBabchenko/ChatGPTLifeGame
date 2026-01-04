"""
Config Tab - View and edit simulation configuration with comments
"""
import tkinter as tk
from tkinter import ttk, messagebox
import re
from pathlib import Path

from .base_tab import BaseTab


class ConfigTab(BaseTab):
    """Configuration viewer/editor tab"""
    
    def setup_ui(self):
        """Setup configuration UI"""
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Simulation Configuration", 
                 font=('Arial', 14, 'bold')).pack(side=tk.LEFT)
        
        ttk.Button(control_frame, text="💾 Save to config.py", 
                  command=self.save_config).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(control_frame, text="🔄 Reload", 
                  command=self.reload_config).pack(side=tk.RIGHT)
        
        # Info label
        info_label = ttk.Label(main_frame,
                              text="⚠️ Changes will be saved to config.py and used by the next training run.",
                              font=('Arial', 10), foreground='darkorange')
        info_label.pack(fill=tk.X, pady=(0, 10))
        
        # Config content frame with scrolling
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='#f8f9fa')
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.config_frame = ttk.Frame(self.canvas)
        
        self.config_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.config_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # Also bind for Linux
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
        
        # Store config entry widgets and comments
        self.config_entries = {}
        self.config_comments = {}
        
        # Load initial config
        self.reload_config()
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def refresh(self):
        """Refresh config (no-op, config is manual refresh only)"""
        pass
    
    def parse_config_comments(self, config_path):
        """Parse comments from config.py file for each parameter"""
        comments = {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                # Look for parameter definitions: PARAM_NAME = value
                if '=' in line and not line.strip().startswith('#'):
                    parts = line.split('=')
                    if len(parts) >= 2:
                        param_name = parts[0].strip()
                        # Check if it's an uppercase parameter (config constant)
                        if param_name.isupper() or '_' in param_name:
                            # Look for inline comment
                            comment_idx = line.find('#')
                            if comment_idx > 0:
                                comment = line[comment_idx + 1:].strip()
                                if comment:
                                    comments[param_name] = comment
        except Exception as e:
            print(f"Warning: Could not parse config comments: {e}")
        
        return comments
    
    def get_config_sections(self):
        """Group config parameters by category - comprehensive list"""
        sections = [
            ("Grid Settings", ['GRID_SIZE', 'FIELD_MIN', 'FIELD_MAX']),
            ("Population Settings", ['INITIAL_PREY_COUNT', 'INITIAL_PREDATOR_COUNT', 
                                    'MAX_PREY', 'MAX_PREDATORS', 'MIN_PREY', 'MIN_PREDATORS']),
            ("Vision & Perception", ['PREDATOR_VISION_RANGE', 'PREY_VISION_RANGE', 'VISION_RANGE', 
                                    'MAX_VISIBLE_ANIMALS', 'PREY_FOV_DEG', 'PREDATOR_FOV_DEG', 
                                    'VISION_SHAPE', 'PHEROMONE_SENSING_RANGE']),
            ("Movement", ['PREDATOR_HUNGRY_MOVES', 'PREDATOR_NORMAL_MOVES', 'PREY_MOVES',
                         'MOVE_ENERGY_COST']),
            ("Energy System", ['INITIAL_ENERGY', 'MAX_ENERGY', 'ENERGY_DECAY_RATE',
                              'EATING_ENERGY_GAIN', 'REST_ENERGY_GAIN', 'HUNGER_THRESHOLD', 
                              'STARVATION_THRESHOLD']),
            ("Mating & Reproduction", ['MATING_COOLDOWN', 'MATING_ENERGY_COST', 
                                      'MATING_PROBABILITY_PREY', 'MATING_PROBABILITY_PREDATOR',
                                      'MATURITY_AGE', 'MAX_AGE']),
            ("Rewards", ['SURVIVAL_REWARD', 'REPRODUCTION_REWARD', 'PREDATOR_EAT_REWARD', 
                        'PREY_EVASION_REWARD', 'PREDATOR_APPROACH_REWARD', 
                        'PREY_MATE_APPROACH_REWARD', 'MATING_REWARD']),
            ("Penalties", ['EXTINCTION_PENALTY', 'DEATH_PENALTY', 'STARVATION_PENALTY', 
                          'EATEN_PENALTY', 'EXHAUSTION_PENALTY', 'OLD_AGE_PENALTY', 
                          'OVERPOPULATION_PENALTY']),
            ("Pheromone System", ['PHEROMONE_DECAY', 'PHEROMONE_DIFFUSION', 
                                 'DANGER_PHEROMONE_STRENGTH', 'MATING_PHEROMONE_STRENGTH']),
            ("PPO Training", ['NUM_EPISODES', 'STEPS_PER_EPISODE', 'PPO_EPOCHS', 
                             'PPO_CLIP_EPSILON', 'PPO_BATCH_SIZE', 'BATCH_SIZE',
                             'VALUE_LOSS_COEF', 'ENTROPY_COEF', 'MAX_GRAD_NORM', 'GAE_LAMBDA']),
            ("Learning Rates", ['LEARNING_RATE', 'LEARNING_RATE_PREY', 'LEARNING_RATE_PREDATOR', 
                               'GAMMA', 'ACTION_TEMPERATURE']),
            ("Network Architecture", ['HIDDEN_DIM', 'SELF_FEATURE_DIM', 'ANIMAL_FEATURE_DIM',
                                     'NUM_HEADS', 'NUM_LAYERS']),
            ("Curriculum & Features", ['CURRICULUM_ENABLED', 'STARVATION_ENABLED',
                                      'USE_ATTENTION', 'USE_LSTM']),
            ("Grass Settings", ['GRASS_PATCH_SIZE', 'GRASS_PATCH_DIAMETER',
                               'INITIAL_GRASS_DENSITY', 'GRASS_REGROWTH_RATE']),
        ]
        return sections
    
    def reload_config(self):
        """Reload configuration from file"""
        # Clear existing entries
        for widget in self.config_frame.winfo_children():
            widget.destroy()
        self.config_entries.clear()
        
        try:
            # Try to import SimulationConfig
            import sys
            project_root = str(Path(__file__).parent.parent.parent.parent.absolute())
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            # Reload config module to get fresh values
            import importlib
            from src import config as config_module
            importlib.reload(config_module)
            from src.config import SimulationConfig
            config = SimulationConfig()
            
            config_path = Path(project_root) / "src" / "config.py"
            self.config_comments = self.parse_config_comments(config_path)
            
            # Get config sections
            config_sections = self.get_config_sections()
            
            for section_name, params in config_sections:
                # Check if any params in this section exist
                existing_params = [p for p in params if hasattr(config, p)]
                if not existing_params:
                    continue
                
                # Section frame
                section_frame = ttk.LabelFrame(self.config_frame, text=section_name, padding="10")
                section_frame.pack(fill=tk.X, padx=10, pady=5)
                
                for param_name in existing_params:
                    value = getattr(config, param_name)
                    
                    # Skip non-editable values
                    if param_name.startswith('_') or callable(value) or isinstance(value, (dict, list)):
                        continue
                    
                    param_frame = ttk.Frame(section_frame)
                    param_frame.pack(fill=tk.X, pady=3)
                    
                    # Label with parameter name
                    ttk.Label(param_frame, text=f"{param_name}:", width=30, 
                             anchor=tk.W, font=('Arial', 10)).pack(side=tk.LEFT)
                    
                    # Entry for value
                    entry_var = tk.StringVar(value=str(value))
                    entry = ttk.Entry(param_frame, textvariable=entry_var, width=15)
                    entry.pack(side=tk.LEFT, padx=5)
                    # Store: entry_var, original_type, original_evaluated_value
                    self.config_entries[param_name] = (entry_var, type(value), value)
                    
                    # Type hint
                    type_hint = type(value).__name__
                    ttk.Label(param_frame, text=f"({type_hint})", 
                             font=('Arial', 9), foreground='gray').pack(side=tk.LEFT, padx=(0, 10))
                    
                    # Comment from config file
                    comment = self.config_comments.get(param_name, '')
                    if comment:
                        comment_label = ttk.Label(param_frame, text=f"💬 {comment}",
                                                 font=('Arial', 9), foreground='#555555',
                                                 wraplength=400)
                        comment_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            self.app.status_label.config(text="Config loaded", foreground="green")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_label = ttk.Label(self.config_frame, text=f"Error loading config: {e}",
                                   font=('Arial', 11), foreground='red')
            error_label.pack(pady=20)
            self.app.status_label.config(text=f"Config error: {e}", foreground="red")
    
    def save_config(self):
        """Save configuration changes to config.py"""
        try:
            import sys
            project_root = str(Path(__file__).parent.parent.parent.parent.absolute())
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            config_path = Path(project_root) / "src" / "config.py"
            
            if not config_path.exists():
                messagebox.showerror("Error", "Config file not found")
                return
            
            # Read current config file
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update values - only change if user actually modified the evaluated value
            changes = []
            for param_name, (entry_var, original_type, original_value) in self.config_entries.items():
                new_value_str = entry_var.get().strip()
                
                # Convert to appropriate type
                try:
                    if original_type == bool:
                        typed_value = new_value_str.lower() in ('true', '1', 'yes')
                    elif original_type == float:
                        typed_value = float(new_value_str)
                    elif original_type == int:
                        typed_value = int(float(new_value_str))  # Handle "1.0" -> 1
                    else:
                        typed_value = new_value_str
                except ValueError:
                    typed_value = new_value_str
                
                # Only update if the user changed the evaluated value
                if typed_value == original_value:
                    continue  # No change from original
                
                # Find and replace in content (preserve comments)
                pattern = rf'({param_name}\s*=\s*)([^#\n]+)(.*)'
                match = re.search(pattern, content)
                if match:
                    old_value = match.group(2).strip()
                    comment_part = match.group(3)  # Preserve inline comment
                    new_value_repr = repr(typed_value) if isinstance(typed_value, str) else str(typed_value)
                    
                    # Replace while preserving comment
                    replacement = rf'\g<1>{new_value_repr}{comment_part}'
                    content = re.sub(pattern, replacement, content, count=1)
                    changes.append(f"{param_name}: {original_value} → {typed_value}")
            
            if changes:
                # Write updated config
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                msg = f"Saved {len(changes)} changes:\n" + "\n".join(changes[:10])
                if len(changes) > 10:
                    msg += f"\n... and {len(changes) - 10} more"
                messagebox.showinfo("Config Saved", msg)
                self.app.status_label.config(text=f"Config saved ({len(changes)} changes)", foreground="green")
            else:
                messagebox.showinfo("No Changes", "No configuration changes detected")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")
            self.app.status_label.config(text=f"Save error: {e}", foreground="red")
