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
        self.checkpoint_path = None
        
        # Build UI
        self.setup_ui()
    
    def setup_ui(self):
        """Build main UI with controls and notebook tabs"""
        # Top control frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(control_frame, text="Life Game Demo", 
                 font=('Arial', 16, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Model selector
        ttk.Label(control_frame, text="Checkpoint:").pack(side=tk.LEFT, padx=(20, 5))
        self.checkpoint_var = tk.StringVar()
        self.checkpoint_combo = ttk.Combobox(control_frame, textvariable=self.checkpoint_var,
                                            width=25, state='readonly')
        self.checkpoint_combo.pack(side=tk.LEFT, padx=5)
        self.checkpoint_combo.bind('<<ComboboxSelected>>', self.on_checkpoint_changed)
        
        ttk.Button(control_frame, text="🔄 Refresh",
                  command=self.refresh_checkpoints).pack(side=tk.LEFT, padx=5)
        
        # Populate checkpoint list
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
            self.checkpoint_combo['values'] = ['No checkpoints found']
            return
        
        # Find all checkpoint files (both model_A and model_B pairs)
        checkpoints = []
        
        # Pattern: model_A_ppo_epX.pth (and corresponding model_B)
        for file in sorted(checkpoint_dir.glob("model_A_ppo*.pth")):
            base_name = file.stem
            # Check if there's a matching model_B
            model_b = checkpoint_dir / (base_name.replace("model_A", "model_B") + ".pth")
            if model_b.exists():
                # Extract episode number or use base name
                if "ep" in base_name:
                    ep_part = base_name.split("ep")[1]
                    try:
                        ep_num = int(ep_part)
                        label = f"Episode {ep_num}"
                    except:
                        label = base_name.replace("model_A_ppo_", "")
                else:
                    label = "Latest" if base_name == "model_A_ppo" else base_name.replace("model_A_ppo_", "")
                
                checkpoints.append((label, file))
        
        if not checkpoints:
            self.checkpoint_combo['values'] = ['No checkpoints found']
        else:
            labels = [label for label, _ in checkpoints]
            self.checkpoint_combo['values'] = labels
            
            # Select first checkpoint by default
            if not self.checkpoint_var.get() or self.checkpoint_var.get() not in labels:
                self.checkpoint_combo.current(0)
                self.on_checkpoint_changed(None)
    
    def on_checkpoint_changed(self, event):
        """Handle checkpoint selection change"""
        selected = self.checkpoint_var.get()
        
        # Find the corresponding file
        checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
        
        # Map label back to file
        if selected == "Latest":
            model_a_path = checkpoint_dir / "model_A_ppo.pth"
        elif selected.startswith("Episode "):
            try:
                ep_num = int(selected.split()[1])
                model_a_path = checkpoint_dir / f"model_A_ppo_ep{ep_num}.pth"
            except:
                return
        else:
            # Try to find by label
            model_a_path = checkpoint_dir / f"model_A_ppo_{selected}.pth"
        
        if model_a_path.exists():
            self.checkpoint_path = model_a_path
            # Notify simulation tab to reload models
            if hasattr(self, 'simulation_tab'):
                self.simulation_tab.load_models(str(model_a_path))
    
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
