"""
Trends Tab - Visualize training metrics over time with matplotlib
"""
import tkinter as tk
from tkinter import ttk

from .base_tab import BaseTab

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TrendsTab(BaseTab):
    """Training trends visualization tab"""
    
    def setup_ui(self):
        """Setup trends visualization UI"""
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Training Trends", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 15))
        
        if not HAS_MATPLOTLIB:
            ttk.Label(main_frame, text="Matplotlib not available. Install with: pip install matplotlib",
                     font=('Arial', 12), foreground='red').pack(pady=20)
            return
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Metrics:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.metric_vars = {}
        # Per-model metrics like Stability tab
        metrics = [('Rewards Prey', 'reward_a'), ('Rewards Pred', 'reward_b'),
                  ('Policy Loss Prey', 'policy_loss_prey'), ('Policy Loss Pred', 'policy_loss_pred'),
                  ('Value Loss Prey', 'value_loss_prey'), ('Value Loss Pred', 'value_loss_pred'),
                  ('Entropy Prey', 'entropy_prey'), ('Entropy Pred', 'entropy_pred'),
                  ('KL Div', 'kl_divergence'), ('Clip Frac', 'clip_fraction')]
        
        for name, key in metrics:
            var = tk.BooleanVar(value=(key in ['reward_a', 'reward_b']))
            self.metric_vars[key] = var
            ttk.Checkbutton(control_frame, text=name, variable=var,
                           command=self.update_charts).pack(side=tk.LEFT, padx=3)
        
        # Window size control
        ttk.Label(control_frame, text="Window:", font=('Arial', 10)).pack(side=tk.LEFT, padx=(20, 5))
        self.window_var = tk.StringVar(value="50")
        window_combo = ttk.Combobox(control_frame, textvariable=self.window_var, 
                                   values=["10", "25", "50", "100", "All"], width=6)
        window_combo.pack(side=tk.LEFT)
        window_combo.bind('<<ComboboxSelected>>', lambda e: self.update_charts())
        
        # Matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)  # Rewards
        self.ax2 = self.fig.add_subplot(212)  # Losses
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig.tight_layout(pad=3.0)
    
    def refresh(self):
        """Refresh charts with new data"""
        if not HAS_MATPLOTLIB:
            return
        if not hasattr(self, 'window_var'):
            return
        if not self.app.episode_data:
            self._clear_charts()
            return
        self.update_charts()
    
    def _clear_charts(self):
        """Clear all charts when no data"""
        if not HAS_MATPLOTLIB:
            return
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_title('Waiting for training data...', fontsize=12, color='gray')
        self.ax2.set_title('', fontsize=12)
        self.canvas.draw_idle()
    
    def update_charts(self):
        """Update trend charts"""
        if not HAS_MATPLOTLIB or not self.app.episode_data:
            return
        
        # Get window size
        window = self.window_var.get()
        if window == "All":
            episodes = sorted(self.app.episode_data.keys())
        else:
            try:
                n = int(window)
                all_eps = sorted(self.app.episode_data.keys())
                episodes = all_eps[-n:] if len(all_eps) > n else all_eps
            except ValueError:
                episodes = sorted(self.app.episode_data.keys())
        
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot rewards (top chart)
        reward_metrics = ['reward_a', 'reward_b']
        colors = {'reward_a': 'blue', 'reward_b': 'green', 
                 'policy_loss_prey': 'lightblue', 'policy_loss_pred': 'darkgreen',
                 'value_loss_prey': 'orange', 'value_loss_pred': 'brown',
                 'entropy_prey': 'purple', 'entropy_pred': 'magenta',
                 'kl_divergence': 'red', 'clip_fraction': 'gray'}
        labels = {'reward_a': 'Reward Prey (A)', 'reward_b': 'Reward Predator (B)',
                 'policy_loss_prey': 'Policy Loss Prey', 'policy_loss_pred': 'Policy Loss Predator',
                 'value_loss_prey': 'Value Loss Prey', 'value_loss_pred': 'Value Loss Predator',
                 'entropy_prey': 'Entropy Prey', 'entropy_pred': 'Entropy Predator',
                 'kl_divergence': 'KL Divergence', 'clip_fraction': 'Clip Fraction'}
        
        for metric in reward_metrics:
            if self.metric_vars.get(metric, tk.BooleanVar(value=False)).get():
                values = [self.app.episode_data[ep].get(metric) for ep in episodes]
                valid_eps = [ep for ep, v in zip(episodes, values) if v is not None]
                valid_vals = [v for v in values if v is not None]
                
                if valid_vals:
                    # Plot raw values
                    self.ax1.plot(valid_eps, valid_vals, color=colors.get(metric, 'gray'),
                                 label=labels.get(metric, metric), linewidth=1.5, alpha=0.6)
                    
                    # Add smoothed line (moving average) only if enough data
                    if len(valid_vals) > 5:
                        smoothed = self._smooth(valid_vals, window=5)
                        self.ax1.plot(valid_eps, smoothed, color=colors.get(metric, 'gray'),
                                     linestyle='--', alpha=0.9, linewidth=2,
                                     label=f'{labels.get(metric, metric)} (smoothed)')
        
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')
        self.ax1.set_title('Rewards Over Time')
        self.ax1.legend(loc='upper left', fontsize=8)
        self.ax1.grid(True, alpha=0.3)
        
        # Plot losses (bottom chart)
        loss_metrics = ['policy_loss_prey', 'policy_loss_pred', 'value_loss_prey', 'value_loss_pred',
                       'entropy_prey', 'entropy_pred', 'kl_divergence', 'clip_fraction']
        
        for metric in loss_metrics:
            if self.metric_vars.get(metric, tk.BooleanVar(value=False)).get():
                values = [self.app.episode_data[ep].get(metric) for ep in episodes]
                valid_eps = [ep for ep, v in zip(episodes, values) if v is not None]
                valid_vals = [v for v in values if v is not None]
                
                if valid_vals:
                    self.ax2.plot(valid_eps, valid_vals, color=colors.get(metric, 'gray'),
                                 label=labels.get(metric, metric), linewidth=1.5)
        
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Value')
        self.ax2.set_title('Training Metrics Over Time')
        self.ax2.legend(loc='upper left', fontsize=8)
        self.ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()
    
    @staticmethod
    def _smooth(values, window=5):
        """Simple moving average smoothing"""
        if len(values) < window:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            smoothed.append(sum(values[start:i+1]) / (i - start + 1))
        return smoothed
