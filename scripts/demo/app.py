"""
Life Game Demo - Main Application
Shows trained Actor-Critic agents in action with modular tab interface
"""

import os
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import ctypes
import torch

# Fix blurry text on Windows with DPI scaling
if sys.platform == 'win32':
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except:
            pass

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SimulationConfig
from scripts.demo.tabs.simulation_tab import SimulationTab
from scripts.demo.tabs.chart_tab import ChartTab
from scripts.demo.tabs.evaluation_tab import EvaluationTab
from scripts.demo.tabs.config_tab import ConfigTab


class LifeGameDemo:
    """Main demo application with modular tabs"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Life Game - Predator/Prey Simulation")
        self.root.geometry("2200x900")
        
        # Configuration
        self.config = SimulationConfig()
        self.device = torch.device("cpu")
        
        # Shared state for all tabs
        self.prey_checkpoint_path = None
        self.predator_checkpoint_path = None
        
        # Build UI
        self.setup_ui()
    
    def setup_ui(self):
        """Build main UI with controls and notebook tabs"""
        # Top control frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(control_frame, text="Life Game Demo", 
                 font=('Arial', 16, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Prey model selector
        ttk.Label(control_frame, text="Prey Model:").pack(side=tk.LEFT, padx=(20, 5))
        self.prey_checkpoint_var = tk.StringVar()
        self.prey_checkpoint_combo = ttk.Combobox(control_frame, textvariable=self.prey_checkpoint_var,
                                            width=30, state='readonly')
        self.prey_checkpoint_combo.pack(side=tk.LEFT, padx=5)
        self.prey_checkpoint_combo.bind('<<ComboboxSelected>>', self.on_checkpoint_changed)
        
        # Predator model selector
        ttk.Label(control_frame, text="Predator Model:").pack(side=tk.LEFT, padx=(15, 5))
        self.predator_checkpoint_var = tk.StringVar()
        self.predator_checkpoint_combo = ttk.Combobox(control_frame, textvariable=self.predator_checkpoint_var,
                                            width=30, state='readonly')
        self.predator_checkpoint_combo.pack(side=tk.LEFT, padx=5)
        self.predator_checkpoint_combo.bind('<<ComboboxSelected>>', self.on_checkpoint_changed)
        
        ttk.Button(control_frame, text="🔄 Refresh",
                  command=self.refresh_checkpoints).pack(side=tk.LEFT, padx=5)
        
        # Populate checkpoint lists
        self.refresh_checkpoints()
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.simulation_tab = SimulationTab(self.notebook, self)
        self.notebook.add(self.simulation_tab.frame, text="Simulation")
        
        self.chart_tab = ChartTab(self.notebook, self)
        self.notebook.add(self.chart_tab.frame, text="Charts")
        
        self.evaluation_tab = EvaluationTab(self.notebook, self)
        self.notebook.add(self.evaluation_tab.frame, text="Evaluation Results")
        
        self.config_tab = ConfigTab(self.notebook, self)
        self.notebook.add(self.config_tab.frame, text="Configuration")
    
    def refresh_checkpoints(self):
        """Scan for available checkpoints"""
        checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
        if not checkpoint_dir.exists():
            self.prey_checkpoint_combo['values'] = ['No checkpoints found']
            self.predator_checkpoint_combo['values'] = ['No checkpoints found']
            return
        
        # Find all prey (model_A) checkpoints
        prey_checkpoints = []
        for file in sorted(checkpoint_dir.glob("*model_A*.pth")):
            prey_checkpoints.append(file.name)
        
        # Find all predator (model_B) checkpoints
        predator_checkpoints = []
        for file in sorted(checkpoint_dir.glob("*model_B*.pth")):
            predator_checkpoints.append(file.name)
        
        # Set combo values
        if prey_checkpoints:
            self.prey_checkpoint_combo['values'] = prey_checkpoints
            if not self.prey_checkpoint_var.get() or self.prey_checkpoint_var.get() not in prey_checkpoints:
                self.prey_checkpoint_combo.current(0)
                self.prey_checkpoint_path = checkpoint_dir / prey_checkpoints[0]
        else:
            self.prey_checkpoint_combo['values'] = ['No checkpoints found']
        
        if predator_checkpoints:
            self.predator_checkpoint_combo['values'] = predator_checkpoints
            if not self.predator_checkpoint_var.get() or self.predator_checkpoint_var.get() not in predator_checkpoints:
                self.predator_checkpoint_combo.current(0)
                self.predator_checkpoint_path = checkpoint_dir / predator_checkpoints[0]
        else:
            self.predator_checkpoint_combo['values'] = ['No checkpoints found']
        
        # Load initial models
        self.on_checkpoint_changed(None)
    
    def on_checkpoint_changed(self, event):
        """Handle checkpoint selection change"""
        checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
        
        # Update prey checkpoint path
        prey_selected = self.prey_checkpoint_var.get()
        if prey_selected and prey_selected != 'No checkpoints found':
            self.prey_checkpoint_path = checkpoint_dir / prey_selected
        
        # Update predator checkpoint path  
        predator_selected = self.predator_checkpoint_var.get()
        if predator_selected and predator_selected != 'No checkpoints found':
            self.predator_checkpoint_path = checkpoint_dir / predator_selected
        
        # Notify simulation tab to reload models
        if hasattr(self, 'simulation_tab') and self.prey_checkpoint_path and self.predator_checkpoint_path:
            self.simulation_tab.load_models(
                str(self.prey_checkpoint_path),
                str(self.predator_checkpoint_path)
            )
    
    def cleanup(self):
        """Cleanup before closing"""
        if hasattr(self, 'simulation_tab'):
            self.simulation_tab.cleanup()


def main():
    """Run the demo"""
    print("\n" + "=" * 70)
    print("  LIFE GAME DEMO - Predator/Prey Simulation")
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
