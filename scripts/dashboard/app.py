"""
Training Dashboard Application - Main Application Class

Modular dashboard for monitoring RL training progress with multiple analysis tabs.
"""
import tkinter as tk
from tkinter import ttk, filedialog
import re
import sys
import ctypes
from pathlib import Path
from typing import Dict, Optional, Any

from .tabs import (
    TrainingControlTab,
    StabilityTab,
    EnvironmentTab,
    BehaviorsTab,
    TrendsTab,
    LogTab,
    ConfigTab,
    EvaluationTab,
)


class TrainingDashboardApp:
    """Main dashboard application class"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Life Game Training Dashboard")
        
        # Enable DPI awareness on Windows for crisp rendering
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
        
        # Larger window size for better readability
        self.root.geometry("1600x1000")
        self.root.minsize(1200, 800)
        
        # Data storage
        self.episode_data: Dict[int, Dict[str, Any]] = {}
        self.last_log_file: Optional[Path] = None
        self.auto_refresh_enabled = False
        self.refresh_interval = 5000  # ms
        
        # Setup UI
        self._setup_main_ui()
        self._setup_tabs()
        
        # Auto-find latest log
        self.auto_find_log()
        
        # Start refresh if auto-enabled
        self.root.after(1000, self._initial_refresh)
    
    @property
    def episodes(self):
        """Return sorted list of episode numbers"""
        return sorted(self.episode_data.keys())
    
    @property
    def metrics(self):
        """Return metrics dict with lists for each metric across all episodes"""
        result = {}
        eps = self.episodes
        if not eps:
            return result
        
        # Gather all metric keys from any episode
        all_keys = set()
        for ep_data in self.episode_data.values():
            all_keys.update(ep_data.keys())
        
        # Build lists for each metric
        for key in all_keys:
            result[key] = [self.episode_data.get(ep, {}).get(key, 0) for ep in eps]
        
        return result
    
    @property  
    def log_file(self):
        """Alias for last_log_file for backward compatibility"""
        return self.last_log_file
    
    def _setup_main_ui(self):
        """Setup main UI frame and status bar"""
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="5")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top control bar
        control_bar = ttk.Frame(self.main_frame)
        control_bar.pack(fill=tk.X, pady=(0, 5))
        
        # Auto-refresh toggle
        self.auto_refresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_bar, text="Auto-refresh", variable=self.auto_refresh_var,
                       command=self.toggle_auto_refresh).pack(side=tk.LEFT, padx=5)
        
        # Refresh interval
        ttk.Label(control_bar, text="Interval:").pack(side=tk.LEFT, padx=(10, 2))
        self.interval_var = tk.StringVar(value="5")
        interval_combo = ttk.Combobox(control_bar, textvariable=self.interval_var,
                                     values=["2", "5", "10", "30"], width=4)
        interval_combo.pack(side=tk.LEFT)
        interval_combo.bind('<<ComboboxSelected>>', self._update_interval)
        ttk.Label(control_bar, text="sec").pack(side=tk.LEFT, padx=(2, 10))
        
        # Manual refresh button
        ttk.Button(control_bar, text="⟳ Refresh Now", 
                  command=self.refresh_data).pack(side=tk.LEFT, padx=5)
        
        # Copy All Data button
        ttk.Button(control_bar, text="📋 Copy All (CSV)", 
                  command=self.copy_all_csv).pack(side=tk.LEFT, padx=15)
        
        # Log file display
        ttk.Label(control_bar, text="Log:").pack(side=tk.LEFT, padx=(20, 5))
        self.log_display = ttk.Label(control_bar, text="None", foreground='blue')
        self.log_display.pack(side=tk.LEFT)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready", foreground='gray')
        self.status_label.pack(side=tk.LEFT)
        
        self.last_episode_label = ttk.Label(status_frame, text="Last Episode: --", foreground='gray')
        self.last_episode_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.episodes_label = ttk.Label(status_frame, text="Episodes: 0", foreground='gray')
        self.episodes_label.pack(side=tk.RIGHT)
    
    def _setup_tabs(self):
        """Setup all dashboard tabs"""
        self.tabs = {}
        
        # Define tab order and classes
        tab_config = [
            ("Training", TrainingControlTab),
            ("Stability", StabilityTab),
            ("Environment", EnvironmentTab),
            ("Evaluation", EvaluationTab),  # New evaluation tab
            ("Behaviors", BehaviorsTab),
            ("Trends", TrendsTab),
            ("Log", LogTab),
            ("Config", ConfigTab),
        ]
        
        for name, TabClass in tab_config:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            self.tabs[name] = TabClass(frame, self)
    
    def copy_all_csv(self):
        """Copy all dashboard data to clipboard in CSV format"""
        import io
        csv_content = io.StringIO()
        
        # Header with timestamp
        csv_content.write(f"# Life Game Training Dashboard Export\n")
        csv_content.write(f"# Log file: {self.log_file or 'None'}\n")
        csv_content.write(f"# Episodes: {len(self.episodes)}\n\n")
        
        # === PPO STABILITY METRICS ===
        csv_content.write("# PPO STABILITY METRICS\n")
        csv_content.write("episode,kl_divergence,clip_fraction,entropy,policy_loss,value_loss\n")
        for i, ep in enumerate(self.episodes):
            kl = self.metrics.get('kl_divergence', [0] * len(self.episodes))[i] if i < len(self.metrics.get('kl_divergence', [])) else ''
            clip = self.metrics.get('clip_fraction', [0] * len(self.episodes))[i] if i < len(self.metrics.get('clip_fraction', [])) else ''
            ent = self.metrics.get('entropy', [0] * len(self.episodes))[i] if i < len(self.metrics.get('entropy', [])) else ''
            ploss = self.metrics.get('policy_loss', [0] * len(self.episodes))[i] if i < len(self.metrics.get('policy_loss', [])) else ''
            vloss = self.metrics.get('value_loss', [0] * len(self.episodes))[i] if i < len(self.metrics.get('value_loss', [])) else ''
            csv_content.write(f"{ep},{kl},{clip},{ent},{ploss},{vloss}\n")
        
        csv_content.write("\n")
        
        # === REWARDS ===
        csv_content.write("# REWARDS\n")
        csv_content.write("episode,reward_prey,reward_predator\n")
        for i, ep in enumerate(self.episodes):
            ra = self.metrics.get('reward_a', [0] * len(self.episodes))[i] if i < len(self.metrics.get('reward_a', [])) else ''
            rb = self.metrics.get('reward_b', [0] * len(self.episodes))[i] if i < len(self.metrics.get('reward_b', [])) else ''
            csv_content.write(f"{ep},{ra},{rb}\n")
        
        csv_content.write("\n")
        
        # === ENVIRONMENT METRICS ===
        csv_content.write("# ENVIRONMENT METRICS\n")
        csv_content.write("episode,final_prey,final_predator,meals,starvation_deaths\n")
        for i, ep in enumerate(self.episodes):
            prey = self.metrics.get('prey_final', [0] * len(self.episodes))[i] if i < len(self.metrics.get('prey_final', [])) else ''
            pred = self.metrics.get('pred_final', [0] * len(self.episodes))[i] if i < len(self.metrics.get('pred_final', [])) else ''
            meals = self.metrics.get('meals', [0] * len(self.episodes))[i] if i < len(self.metrics.get('meals', [])) else ''
            starv = self.metrics.get('starvation_deaths', [0] * len(self.episodes))[i] if i < len(self.metrics.get('starvation_deaths', [])) else ''
            csv_content.write(f"{ep},{prey},{pred},{meals},{starv}\n")
        
        content = csv_content.getvalue()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status_label.config(text=f"✓ Copied {len(self.episodes)} episodes to clipboard (CSV)", foreground="green")
        self.root.after(3000, lambda: self.status_label.config(text="Ready", foreground="gray"))
    
    def toggle_auto_refresh(self):
        """Toggle auto-refresh on/off"""
        self.auto_refresh_enabled = self.auto_refresh_var.get()
        if self.auto_refresh_enabled:
            self._schedule_refresh()
            self.status_label.config(text="Auto-refresh enabled", foreground="green")
        else:
            self.status_label.config(text="Auto-refresh disabled", foreground="gray")
    
    def _update_interval(self, event=None):
        """Update refresh interval"""
        try:
            self.refresh_interval = int(self.interval_var.get()) * 1000
        except ValueError:
            self.refresh_interval = 5000
    
    def _schedule_refresh(self):
        """Schedule next auto-refresh"""
        if self.auto_refresh_enabled:
            self.refresh_data()
            self.root.after(self.refresh_interval, self._schedule_refresh)
    
    def _initial_refresh(self):
        """Initial data refresh after startup"""
        if self.last_log_file and self.last_log_file.exists():
            self.refresh_data()
        else:
            # If no log found, just update tab displays
            self._refresh_all_tabs()
    
    def refresh_data(self):
        """Refresh all data from log file"""
        print(f"[App] refresh_data() called, log_file={self.last_log_file}")
        if self.last_log_file and self.last_log_file.exists():
            print(f"[App] Parsing log file...")
            self._parse_log_file()
            print(f"[App] Parsed {len(self.episode_data)} episodes")
            self._refresh_all_tabs()
            
            # Update status bar with episode counts
            total_episodes = len(self.episode_data)
            self.episodes_label.config(text=f"Episodes: {total_episodes}")
            
            # Update last episode label
            if total_episodes > 0:
                last_ep = max(self.episode_data.keys())
                self.last_episode_label.config(text=f"Last Episode: {last_ep}", foreground='blue')
            else:
                self.last_episode_label.config(text="Last Episode: --", foreground='gray')
    
    def _parse_log_file(self):
        """Parse training log file for metrics"""
        if not self.last_log_file or not self.last_log_file.exists():
            return
        
        try:
            with open(self.last_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse episode blocks - format: "Episode 1/150"
            episode_pattern = r'Episode\s+(\d+)/\d+'
            
            # Patterns matching actual log format
            metric_patterns = {
                # Rewards: "Rewards: Prey=1495.8, Predator=1645.5"
                'reward_a': r'Rewards:\s*Prey=([+-]?\d+\.?\d*)',
                'reward_b': r'Rewards:.*Predator=([+-]?\d+\.?\d*)',
                # PPO Diagnostics: "[PPO Diagnostics] KL: 0.001965, ClipFrac: 0.054"
                'kl_divergence': r'\[PPO Diagnostics\]\s*KL:\s*([+-]?\d+\.?\d*)',
                'clip_fraction': r'ClipFrac:\s*([+-]?\d+\.?\d*)',
                # Losses: "Losses: Policy(P=0.000/Pr=0.000), Value(P=5.364/Pr=3.738), Entropy(P=3.174/Pr=3.174)"
                'policy_loss_prey': r'Policy\(P=([+-]?\d+\.?\d*)',
                'policy_loss_pred': r'Policy\(P=[^/]+/Pr=([+-]?\d+\.?\d*)',
                'value_loss_prey': r'Value\(P=([+-]?\d+\.?\d*)',
                'value_loss_pred': r'Value\(P=[^/]+/Pr=([+-]?\d+\.?\d*)',
                'entropy_prey': r'Entropy\(P=([+-]?\d+\.?\d*)',
                'entropy_pred': r'Entropy\(P=[^/]+/Pr=([+-]?\d+\.?\d*)',
                # Final counts: "Final: Prey=27, Predators=3"
                'prey_count': r'Final:\s*Prey=(\d+)',
                'predator_count': r'Final:.*Predators=(\d+)',
                # Deaths/Births: "Births=5, Deaths=27, Meals=27"
                'births': r'Births=(\d+)',
                'deaths': r'Deaths=(\d+)',
                'meals': r'Meals=(\d+)',
                # Starvation: "Exhaustion=0, Old Age=0, Starvation=28"
                'starvation': r'Starvation=(\d+)',
                # Experiences: "Starting PPO update (Prey experiences=2008, Predator=9563)"
                'exp_prey': r'Prey experiences=(\d+)',
                'exp_pred': r'Predator=(\d+)\)\.\.\.',
            }
            
            # Find all episodes
            lines = content.split('\n')
            current_episode = None
            
            for line in lines:
                # Check for episode marker
                ep_match = re.search(episode_pattern, line)
                if ep_match:
                    current_episode = int(ep_match.group(1))
                    if current_episode not in self.episode_data:
                        self.episode_data[current_episode] = {}
                
                if current_episode is None:
                    continue
                
                # Extract metrics
                for metric, pattern in metric_patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        try:
                            value = float(match.group(1))
                            self.episode_data[current_episode][metric] = value
                        except ValueError:
                            pass
            
            # Compute combined metrics
            for ep_data in self.episode_data.values():
                # Kills = Meals (predators eating prey)
                if 'meals' in ep_data:
                    ep_data['kills'] = ep_data['meals']
                # Total animals = prey + predators
                if 'prey_count' in ep_data and 'predator_count' in ep_data:
                    ep_data['total_animals'] = ep_data['prey_count'] + ep_data['predator_count']
                # Combined policy loss
                if 'policy_loss_prey' in ep_data and 'policy_loss_pred' in ep_data:
                    ep_data['policy_loss'] = (ep_data['policy_loss_prey'] + ep_data['policy_loss_pred']) / 2
                # Combined value loss
                if 'value_loss_prey' in ep_data and 'value_loss_pred' in ep_data:
                    ep_data['value_loss'] = (ep_data['value_loss_prey'] + ep_data['value_loss_pred']) / 2
                # Combined entropy
                if 'entropy_prey' in ep_data and 'entropy_pred' in ep_data:
                    ep_data['entropy'] = (ep_data['entropy_prey'] + ep_data['entropy_pred']) / 2
            
        except Exception as e:
            self.status_label.config(text=f"Log parse error: {e}", foreground="red")
    
    def _refresh_all_tabs(self):
        """Refresh all tabs with new data"""
        print(f"[App] Refreshing {len(self.tabs)} tabs...")
        for tab_name, tab in self.tabs.items():
            try:
                print(f"[App] Refreshing tab: {tab_name}")
                tab.refresh()
            except Exception as e:
                print(f"Tab refresh error ({tab_name}): {e}")
                import traceback
                traceback.print_exc()
    
    def auto_find_log(self, clear_data: bool = False):
        """Auto-find the latest log file
        
        Args:
            clear_data: If True, clear existing episode data (for manual Find Latest button)
        """
        log_dir = Path("outputs/logs")
        if not log_dir.exists():
            return
        
        log_files = list(log_dir.glob("*.log"))
        if log_files:
            # Get most recently modified
            latest = max(log_files, key=lambda p: p.stat().st_mtime)
            
            # Only switch if it's a different file
            if self.last_log_file != latest:
                if clear_data:
                    self.episode_data.clear()
                self.last_log_file = latest
                self.log_display.config(text=latest.name)
                self.status_label.config(text=f"Found log: {latest.name}", foreground="blue")
                if clear_data:
                    self.refresh_data()


def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Configure ttk styles for larger fonts
    style = ttk.Style()
    try:
        style.theme_use('vista')  # Use vista theme for cleaner tabs on Windows
    except tk.TclError:
        try:
            style.theme_use('winnative')
        except tk.TclError:
            pass
    
    style.configure('TNotebook', tabposition='n')
    style.configure('TNotebook.Tab', font=('Arial', 11, 'bold'), padding=[15, 8])
    style.configure('TButton', font=('Arial', 11), padding=8)
    style.configure('TCheckbutton', font=('Arial', 11))
    style.configure('TLabel', font=('Arial', 11))
    style.configure('TLabelframe.Label', font=('Arial', 11, 'bold'))
    
    app = TrainingDashboardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
