"""
Training Dashboard App - Real-time GUI for monitoring PPO training

Features:
- Auto-refresh training logs
- Real-time PPO stability metrics with health indicators
- Trend graphs for all metrics
- Integrated behavior analysis
- Tkinter GUI with refresh button
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import subprocess
import sys
import ctypes
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class TrainingDashboardApp:
    """GUI Dashboard for training monitoring"""
    
    def __init__(self, root):
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
        self.episodes = []
        self.metrics = defaultdict(list)
        self.last_log_file = None
        self.auto_refresh_enabled = False
        self.training_process = None  # Track running training process
        self.training_log_handle = None  # Track log file handle
        
        # Health thresholds
        self.thresholds = {
            'kl': {'healthy': 0.03, 'warning': 0.05, 'bad': 0.1},
            'clip_frac': {'healthy': 0.20, 'warning': 0.35, 'bad': 0.5},
            'extreme_ratio_pct': {'healthy': 0.10, 'warning': 0.20, 'bad': 0.30},
            'entropy_min': 1.5,
            'ppo_kl': {'healthy': 0.03, 'warning': 0.05, 'bad': 0.1},
            'ppo_clip_frac': {'healthy': 0.20, 'warning': 0.35, 'bad': 0.5},
            'entropy': {'min_healthy': 1.5, 'min_warning': 1.0},  # Higher is better
            # Environment metrics (good ranges)
            'meals': {'good': 80, 'ok': 50},  # More meals = better predator hunting
            'starvation_deaths': {'good': 15, 'ok': 25, 'reverse': True},  # Less = better
            'prey_final': {'good': 15, 'ok': 5},  # More survivors = better prey evasion
            'predators_final': {'good': 15, 'ok': 8},  # More survivors = better hunting
            'value_loss': {'good': 5.0, 'ok': 10.0, 'reverse': True},  # Less = better
            'policy_loss': {'good': 0.05, 'ok': 0.2},  # Some loss needed for learning
        }
        
        self.setup_ui()
        self.auto_find_log()
    
    def setup_ui(self):
        """Create the GUI layout"""
        # Configure ttk styles for larger fonts
        style = ttk.Style()
        style.configure('TNotebook', tabposition='n')
        style.configure('TNotebook.Tab', font=('Arial', 13, 'bold'), padding=[20, 10])
        style.configure('TButton', font=('Arial', 12), padding=10)
        style.configure('TCheckbutton', font=('Arial', 12))
        style.configure('TLabel', font=('Arial', 12))
        style.configure('TLabelframe.Label', font=('Arial', 12, 'bold'))
        
        # Top control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # ttk.Label(control_frame, text="Training Dashboard", font=('Arial', 24, 'bold')).pack(side=tk.LEFT, padx=10)
        
        self.refresh_btn = ttk.Button(control_frame, text="🔄 Refresh", command=self.refresh_data)
        self.refresh_btn.pack(side=tk.LEFT, padx=5)
        
        self.auto_refresh_var = tk.BooleanVar(value=False)
        self.auto_check = ttk.Checkbutton(control_frame, text="Auto-refresh (10s)", 
                                          variable=self.auto_refresh_var,
                                          command=self.toggle_auto_refresh)
        self.auto_check.pack(side=tk.LEFT, padx=5)
        
        # Copy all data button
        ttk.Button(control_frame, text="📋 Copy All Data", 
                  command=self.copy_all_data).pack(side=tk.LEFT, padx=15)
        
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="gray", font=('Arial', 12))
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Main content area with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Training Control
        self.control_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.control_tab, text="Training Control")
        self.setup_training_control_tab()
        
        # Tab 2: PPO Stability
        self.stability_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stability_tab, text="PPO Stability")
        self.setup_stability_tab()
        
        # Tab 3: Environment
        self.env_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.env_tab, text="Environment")
        self.setup_environment_tab()
        
        # Tab 4: Behaviors
        self.behavior_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.behavior_tab, text="Behaviors")
        self.setup_behavior_tab()
        
        # Tab 5: Trends
        self.trends_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.trends_tab, text="Trends")
        self.setup_trends_tab()
        
        # Tab 6: Raw Log
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="Raw Log")
        self.setup_log_tab()
        
        # Tab 7: Configuration
        self.config_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.config_tab, text="Configuration")
        self.setup_config_tab()
        
        # Auto-load latest log on startup
        self.root.after(100, self.auto_find_log)
    
    def setup_stability_tab(self):
        """Setup PPO stability metrics"""
        # Add copy button at top
        button_frame = ttk.Frame(self.stability_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(button_frame, text="📊 Data source: Training log file", font=('Arial', 10), foreground='blue').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📋 Copy Content (All Episodes)", command=self.copy_stability_content).pack(side=tk.RIGHT)
        
        # Create grid for metrics
        metrics_frame = ttk.Frame(self.stability_tab, padding="10")
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stability_widgets = {}
        
        metrics = [
            ('KL Divergence', 'ppo_kl', '< 0.03 ✅  < 0.05 ⚠️  < 0.1 🔴',
             'Measures how much policy changed. Too high = unstable training, too low = no learning.'),
            ('Clip Fraction', 'ppo_clip_frac', '< 0.20 ✅  < 0.35 ⚠️  < 0.5 🔴',
             'Shows % of updates clipped by PPO. High values mean aggressive policy changes.'),
            ('Extreme Ratios', 'ppo_extreme_ratio_pct', '< 10% ✅  < 20% ⚠️  < 30% 🔴',
             'Actions with very high probability ratios. Indicates potential instability.'),
            ('Entropy', 'entropy', '> 1.5 for exploration',
             'Measures action randomness. Higher = more exploration, lower = more deterministic.'),
        ]
        
        for i, (label, key, threshold, explanation) in enumerate(metrics):
            frame = ttk.LabelFrame(metrics_frame, text=label, padding="10")
            frame.grid(row=i//2, column=i%2, padx=10, pady=10, sticky='nsew')
            
            # Explanation at the top
            explain_label = tk.Label(frame, text=explanation, font=('Arial', 10), fg='gray',
                                    wraplength=300, justify=tk.CENTER)
            explain_label.pack(pady=(0, 5))
            
            value_label = tk.Label(frame, text="--", font=('Arial', 36, 'bold'))
            value_label.pack()
            
            trend_label = tk.Label(frame, text="--", font=('Arial', 16))
            trend_label.pack()
            
            # Create colored threshold indicator
            threshold_frame = tk.Frame(frame)
            threshold_frame.pack(pady=5)
            
            if key == 'ppo_kl':
                tk.Label(threshold_frame, text="< 0.03", font=('Arial', 10, 'bold'), fg='green').pack(side=tk.LEFT, padx=3)
                tk.Label(threshold_frame, text="< 0.05", font=('Arial', 10, 'bold'), fg='darkorange').pack(side=tk.LEFT, padx=3)
                tk.Label(threshold_frame, text="< 0.1", font=('Arial', 10, 'bold'), fg='red').pack(side=tk.LEFT, padx=3)
            elif key == 'ppo_clip_frac':
                tk.Label(threshold_frame, text="< 0.20", font=('Arial', 10, 'bold'), fg='green').pack(side=tk.LEFT, padx=3)
                tk.Label(threshold_frame, text="< 0.35", font=('Arial', 10, 'bold'), fg='darkorange').pack(side=tk.LEFT, padx=3)
                tk.Label(threshold_frame, text="< 0.5", font=('Arial', 10, 'bold'), fg='red').pack(side=tk.LEFT, padx=3)
            elif key == 'ppo_extreme_ratio_pct':
                tk.Label(threshold_frame, text="< 10%", font=('Arial', 10, 'bold'), fg='green').pack(side=tk.LEFT, padx=3)
                tk.Label(threshold_frame, text="< 20%", font=('Arial', 10, 'bold'), fg='darkorange').pack(side=tk.LEFT, padx=3)
                tk.Label(threshold_frame, text="< 30%", font=('Arial', 10, 'bold'), fg='red').pack(side=tk.LEFT, padx=3)
            else:
                tk.Label(threshold_frame, text=threshold, font=('Arial', 10), fg='gray').pack()
            
            # Use Text widget for color-coded history - match background to frame
            history_text = tk.Text(frame, height=2, font=('Courier', 12), wrap=tk.WORD, 
                                  relief=tk.FLAT, borderwidth=0, state=tk.DISABLED,
                                  bg=self.root.cget('bg'))
            history_text.tag_configure('center', justify='center')
            history_text.pack()
            
            self.stability_widgets[key] = {
                'value': value_label,
                'trend': trend_label,
                'history': history_text
            }
        
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.columnconfigure(1, weight=1)
        metrics_frame.rowconfigure(0, weight=1)
        metrics_frame.rowconfigure(1, weight=1)
    
    def setup_environment_tab(self):
        """Setup environment metrics"""
        # Add copy button at top
        button_frame = ttk.Frame(self.env_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(button_frame, text="📊 Data source: Training log file", font=('Arial', 10), foreground='blue').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📋 Copy Content (All Episodes)", command=self.copy_environment_content).pack(side=tk.RIGHT)
        
        env_frame = ttk.Frame(self.env_tab, padding="10")
        env_frame.pack(fill=tk.BOTH, expand=True)
        
        self.env_widgets = {}
        
        metrics = [
            ('Meals/Episode', 'meals', 'Predator hunting success',
             'Number of prey eaten. Higher = better predator hunting skills.'),
            ('Starvation Deaths', 'starvation_deaths', 'Predator failures',
             'Predators dying from low energy. Lower = better energy management.'),
            ('Final Prey Count', 'prey_final', 'Prey survival',
             'Prey alive at episode end. Higher = better evasion skills.'),
            ('Final Predator Count', 'predators_final', 'Predator survival',
             'Predators alive at episode end. Higher = better hunting and energy management.'),
            ('Value Loss', 'value_loss', 'Should stabilize ↘',
             'Critic error in predicting returns. Should decrease and stabilize over time.'),
            ('Policy Loss', 'policy_loss', 'Update magnitude',
             'Policy gradient magnitude. Should be non-zero but not too large.'),
        ]
        
        for i, (label, key, desc, explanation) in enumerate(metrics):
            frame = ttk.LabelFrame(env_frame, text=label, padding="10")
            frame.grid(row=i//2, column=i%2, padx=10, pady=10, sticky='nsew')
            
            # Explanation at the top
            explain_label = tk.Label(frame, text=explanation, font=('Arial', 10), fg='gray',
                                    wraplength=300, justify=tk.CENTER)
            explain_label.pack(pady=(0, 5))
            
            value_label = tk.Label(frame, text="--", font=('Arial', 32, 'bold'))
            value_label.pack()
            
            trend_label = tk.Label(frame, text="--", font=('Arial', 15))
            trend_label.pack()
            
            # Create colored threshold indicator
            threshold_frame = tk.Frame(frame)
            threshold_frame.pack(pady=5)
            
            if key == 'meals':
                tk.Label(threshold_frame, text="≥80 Good", font=('Arial', 10, 'bold'), fg='green').pack(side=tk.LEFT, padx=3)
                tk.Label(threshold_frame, text="≥50 OK", font=('Arial', 10, 'bold'), fg='darkorange').pack(side=tk.LEFT, padx=3)
                tk.Label(threshold_frame, text="<50 Low", font=('Arial', 10, 'bold'), fg='red').pack(side=tk.LEFT, padx=3)
            elif key == 'starvation_deaths':
                tk.Label(threshold_frame, text="≤15 Good", font=('Arial', 10, 'bold'), fg='green').pack(side=tk.LEFT, padx=3)
                tk.Label(threshold_frame, text="≤25 OK", font=('Arial', 10, 'bold'), fg='darkorange').pack(side=tk.LEFT, padx=3)
                tk.Label(threshold_frame, text=">25 High", font=('Arial', 10, 'bold'), fg='red').pack(side=tk.LEFT, padx=3)
            elif key in ['prey_final', 'predators_final']:
                tk.Label(threshold_frame, text="≥15 Good", font=('Arial', 10, 'bold'), fg='green').pack(side=tk.LEFT, padx=3)
                tk.Label(threshold_frame, text="≥5/8 OK", font=('Arial', 10, 'bold'), fg='darkorange').pack(side=tk.LEFT, padx=3)
                tk.Label(threshold_frame, text="<5/8 Low", font=('Arial', 10, 'bold'), fg='red').pack(side=tk.LEFT, padx=3)
            elif key == 'value_loss':
                tk.Label(threshold_frame, text="<5 Good", font=('Arial', 10, 'bold'), fg='green').pack(side=tk.LEFT, padx=3)
                tk.Label(threshold_frame, text="<10 OK", font=('Arial', 10, 'bold'), fg='darkorange').pack(side=tk.LEFT, padx=3)
                tk.Label(threshold_frame, text=">10 High", font=('Arial', 10, 'bold'), fg='red').pack(side=tk.LEFT, padx=3)
            else:
                tk.Label(threshold_frame, text=desc, font=('Arial', 10), fg='gray').pack()
            
            # Use Text widget for color-coded history - match background to frame
            history_text = tk.Text(frame, height=2, font=('Courier', 12), wrap=tk.WORD,
                                  relief=tk.FLAT, borderwidth=0, state=tk.DISABLED,
                                  bg=self.root.cget('bg'))
            history_text.tag_configure('center', justify='center')
            history_text.pack()
            
            self.env_widgets[key] = {
                'value': value_label,
                'trend': trend_label,
                'history': history_text
            }
        
        env_frame.columnconfigure(0, weight=1)
        env_frame.columnconfigure(1, weight=1)
        env_frame.rowconfigure(0, weight=1)
        env_frame.rowconfigure(1, weight=1)
        env_frame.rowconfigure(2, weight=1)
    
    def setup_behavior_tab(self):
        """Setup behavioral evaluation display"""
        # Add buttons at top
        button_frame = ttk.Frame(self.behavior_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="🔬 Run Behavior Analysis", command=self.run_behavior_analysis).pack(side=tk.LEFT, padx=5)
        self.behavior_status = ttk.Label(button_frame, text="Not run yet", foreground='gray', font=('Arial', 11))
        self.behavior_status.pack(side=tk.LEFT, padx=10)
        ttk.Label(button_frame, text="📊 Data source: Model checkpoints", font=('Arial', 10), foreground='blue').pack(side=tk.LEFT, padx=15)
        ttk.Button(button_frame, text="📋 Copy Content (All Episodes)", command=self.copy_behavior_content).pack(side=tk.RIGHT)
        
        # Explanation and legend
        info_frame = ttk.Frame(self.behavior_tab)
        info_frame.pack(fill=tk.X, padx=10, pady=(5, 0))
        
        # Main description
        description = ("Behavioral metrics show percentage of correct actions in test scenarios (100 samples per metric).")
        ttk.Label(info_frame, text=description, font=('Arial', 10, 'bold'), foreground='#2c3e50').pack(anchor=tk.W, pady=(0,5))
        
        # Metrics descriptions
        metrics_frame = ttk.Frame(info_frame)
        metrics_frame.pack(fill=tk.X, pady=(0,5))
        
        pred_metrics = ("Predator: Hunting (approaching prey), Hunger (seeking food when low energy), Mating (approaching same species), "
                       "Selective Hunt (prioritizing closer prey), "
                       "Energy Mgmt (hunting urgency increases when hungry)")
        ttk.Label(metrics_frame, text=pred_metrics, font=('Arial', 9), foreground='#7d3c98', wraplength=1400, justify=tk.LEFT).pack(anchor=tk.W)
        
        prey_metrics = ("Prey: Evasion (fleeing from predators), Mating (approaching same species), "
                       "Multi-Target (fleeing from nearest threat), "
                       "Flocking (grouping with others), Threat Assess (reacting more to near vs far threats)")
        ttk.Label(metrics_frame, text=prey_metrics, font=('Arial', 9), foreground='#229954', wraplength=1400, justify=tk.LEFT).pack(anchor=tk.W, pady=(2,0))
        
        # Legend frame with color indicators
        legend_frame = ttk.Frame(info_frame)
        legend_frame.pack(fill=tk.X)
        
        ttk.Label(legend_frame, text="Indicators:", font=('Arial', 10, 'bold'), foreground='#2c3e50').pack(side=tk.LEFT)
        
        # Excellent indicator
        excellent_frame = tk.Frame(legend_frame, bg='#d4edda', relief=tk.RIDGE, borderwidth=1)
        excellent_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(excellent_frame, text="  ✅ ≥80% (Excellent)  ", bg='#d4edda', font=('Arial', 9)).pack()
        
        # Good indicator
        good_frame = tk.Frame(legend_frame, bg='#fff3cd', relief=tk.RIDGE, borderwidth=1)
        good_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(good_frame, text="  ⚠️ 60-79% (Good)  ", bg='#fff3cd', font=('Arial', 9)).pack()
        
        # Poor indicator
        poor_frame = tk.Frame(legend_frame, bg='#f8d7da', relief=tk.RIDGE, borderwidth=1)
        poor_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(poor_frame, text="  ❌ <60% (Poor)  ", bg='#f8d7da', font=('Arial', 9)).pack()
        
        # Table container
        table_container = ttk.Frame(self.behavior_tab)
        table_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Fixed header at top
        self.behavior_header_frame = tk.Frame(table_container, bg='white')
        self.behavior_header_frame.pack(fill=tk.X)
        
        # Create scrollable content area
        scroll_frame = ttk.Frame(table_container)
        scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbars
        self.behavior_canvas = tk.Canvas(scroll_frame, bg='white', highlightthickness=0)
        vsb = ttk.Scrollbar(scroll_frame, orient="vertical", command=self.behavior_canvas.yview)
        hsb = ttk.Scrollbar(scroll_frame, orient="horizontal", command=self.behavior_canvas.xview)
        
        self.behavior_canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.behavior_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas for data
        self.behavior_table_frame = tk.Frame(self.behavior_canvas, bg='white')
        self.behavior_canvas.create_window((0, 0), window=self.behavior_table_frame, anchor='nw')
        
        # Update scroll region when content changes
        self.behavior_table_frame.bind('<Configure>', lambda e: self.behavior_canvas.configure(scrollregion=self.behavior_canvas.bbox('all')))
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            self.behavior_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.behavior_canvas.bind('<Enter>', lambda e: self.behavior_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.behavior_canvas.bind('<Leave>', lambda e: self.behavior_canvas.unbind_all("<MouseWheel>"))
    
    def setup_training_control_tab(self):
        """Setup training control and monitoring"""
        # Main container
        main_frame = ttk.Frame(self.control_tab, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Training Control & Monitoring", 
                               font=('Arial', 18, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # === Training Status Section ===
        status_frame = ttk.LabelFrame(main_frame, text="Training Status", padding="15")
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        status_inner = ttk.Frame(status_frame)
        status_inner.pack(fill=tk.X)
        
        ttk.Label(status_inner, text="Status:", font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        self.training_status_label = ttk.Label(status_inner, text="Not Running", 
                                               font=('Arial', 12), foreground='gray')
        self.training_status_label.pack(side=tk.LEFT, padx=(0, 20))
        
        self.training_indicator = tk.Canvas(status_inner, width=20, height=20, highlightthickness=0)
        self.training_indicator.pack(side=tk.LEFT)
        self.training_indicator.create_oval(2, 2, 18, 18, fill='gray', outline='darkgray', tags='indicator')
        
        # Process ID display
        self.pid_label = ttk.Label(status_inner, text="PID: N/A", font=('Arial', 10), foreground='gray')
        self.pid_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # === Training Control Section ===
        control_frame = ttk.LabelFrame(main_frame, text="Training Control", padding="15")
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        button_row = ttk.Frame(control_frame)
        button_row.pack(fill=tk.X)
        
        self.start_training_btn = ttk.Button(button_row, text="▶ Start Training", 
                                            command=self.start_training, 
                                            style='Accent.TButton')
        self.start_training_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_training_btn = ttk.Button(button_row, text="⏹ Stop Training", 
                                           command=self.stop_training, state=tk.DISABLED)
        self.stop_training_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_row, text="🔄 Check Status", 
                  command=self.check_training_status).pack(side=tk.LEFT)
        
        # Info text
        info_text = ("Training runs in an independent process. If the dashboard crashes, "
                    "training will continue running in the background.")
        info_label = ttk.Label(control_frame, text=info_text, font=('Arial', 10), 
                              foreground='blue', wraplength=700)
        info_label.pack(pady=(10, 0))
        
        # === Log File Selection Section ===
        log_frame = ttk.LabelFrame(main_frame, text="Log File Selection", padding="15")
        log_frame.pack(fill=tk.X, pady=(0, 15))
        
        log_row1 = ttk.Frame(log_frame)
        log_row1.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(log_row1, text="Current Log:", font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        self.current_log_label = ttk.Label(log_row1, text="None", font=('Arial', 11), 
                                          foreground='blue')
        self.current_log_label.pack(side=tk.LEFT)
        
        log_row2 = ttk.Frame(log_frame)
        log_row2.pack(fill=tk.X)
        
        ttk.Button(log_row2, text="📂 Select Log File", 
                  command=self.select_log_file).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(log_row2, text="🔍 Auto-Find Latest", 
                  command=self.auto_find_log).pack(side=tk.LEFT)
        
        log_info = ("The dashboard will automatically switch to the new log file when training starts. "
                   "You can also manually select a different log file to view.")
        ttk.Label(log_frame, text=log_info, font=('Arial', 10), 
                 foreground='gray', wraplength=700).pack(pady=(10, 0))
        
        # Start periodic status check
        self.check_training_status()
        self.root.after(2000, self.periodic_status_check)
    
    def setup_trends_tab(self):
        """Setup trends visualization with line charts"""
        # Control panel at top
        control_frame = ttk.Frame(self.trends_tab, padding="10")
        control_frame.pack(fill=tk.X)
        
        ttk.Label(control_frame, text="📈 Metric Trends Across Episodes", 
                 font=('Arial', 16, 'bold')).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(control_frame, text="🔄 Update Charts", 
                  command=self.update_trend_charts).pack(side=tk.RIGHT, padx=5)
        
        # Main content with scrollbar
        main_frame = ttk.Frame(self.trends_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with vertical scrollbar for charts
        canvas = tk.Canvas(main_frame, bg='white')
        vsb = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollable frame inside canvas
        scrollable_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        
        # === PPO METRICS SECTION ===
        ppo_section = ttk.LabelFrame(scrollable_frame, text="PPO Stability Metrics", padding="15")
        ppo_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # PPO chart canvas
        self.ppo_fig = Figure(figsize=(14, 8), dpi=100)
        self.ppo_canvas_widget = FigureCanvasTkAgg(self.ppo_fig, master=ppo_section)
        self.ppo_canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store metric keys for chart update
        self.ppo_metrics = ['ppo_kl', 'ppo_clip_frac', 'ppo_extreme_ratio_pct', 'entropy']
        
        # === ENVIRONMENT METRICS SECTION ===
        env_section = ttk.LabelFrame(scrollable_frame, text="Environment Metrics", padding="15")
        env_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Environment chart canvas
        self.env_fig = Figure(figsize=(14, 8), dpi=100)
        self.env_canvas_widget = FigureCanvasTkAgg(self.env_fig, master=env_section)
        self.env_canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store metric keys for chart update
        self.env_metrics = ['prey_final', 'predators_final', 'meals', 'starvation_deaths', 'policy_loss', 'value_loss']
        
        # Update scroll region
        scrollable_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox('all'))
        
        # Bind mouse wheel
        def _on_trends_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind('<Enter>', lambda e: canvas.bind_all("<MouseWheel>", _on_trends_mousewheel))
        canvas.bind('<Leave>', lambda e: canvas.unbind_all("<MouseWheel>"))
    
    def update_trend_charts(self):
        """Redraw charts with all metrics"""
        if not self.episodes:
            return
        
        # === PPO METRICS CHART ===
        self.ppo_fig.clear()
        
        for idx, key in enumerate(self.ppo_metrics):
            ax = self.ppo_fig.add_subplot(2, 2, idx + 1)
            
            if key in self.metrics and self.metrics[key]:
                values = self.metrics[key]
                episodes = self.episodes[:len(values)]
                
                ax.plot(episodes, values, marker='o', linewidth=2, markersize=4)
                ax.set_xlabel('Episode', fontsize=10)
                ax.set_ylabel(key.replace('_', ' ').title(), fontsize=10)
                ax.set_title(key.replace('_', ' ').title(), fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Force integer x-axis ticks
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                
                # Add threshold lines for relevant metrics
                if key == 'ppo_kl':
                    ax.axhline(y=0.03, color='green', linestyle='--', alpha=0.5, label='Healthy')
                    ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Warning')
                    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Bad')
                    ax.legend(fontsize=8)
                elif key == 'ppo_clip_frac':
                    ax.axhline(y=0.20, color='green', linestyle='--', alpha=0.5, label='Healthy')
                    ax.axhline(y=0.35, color='orange', linestyle='--', alpha=0.5, label='Warning')
                    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Bad')
                    ax.legend(fontsize=8)
                elif key == 'ppo_extreme_ratio_pct':
                    ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='<10% Good')
                    ax.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='<20% Warning')
                    ax.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='<30% Bad')
                    ax.legend(fontsize=8)
                elif key == 'entropy':
                    ax.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='Min Healthy')
                    ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='Min Warning')
                    ax.legend(fontsize=8)
        
        self.ppo_fig.tight_layout()
        self.ppo_canvas_widget.draw()
        
        # === ENVIRONMENT METRICS CHART ===
        self.env_fig.clear()
        
        n_plots = len(self.env_metrics)
        rows = (n_plots + 2) // 3  # 3 columns max
        cols = min(n_plots, 3)
        
        for idx, key in enumerate(self.env_metrics):
            ax = self.env_fig.add_subplot(rows, cols, idx + 1)
            
            if key in self.metrics and self.metrics[key]:
                values = self.metrics[key]
                episodes = self.episodes[:len(values)]
                
                ax.plot(episodes, values, marker='o', linewidth=2, markersize=4)
                ax.set_xlabel('Episode', fontsize=10)
                ax.set_ylabel(key.replace('_', ' ').title(), fontsize=10)
                ax.set_title(key.replace('_', ' ').title(), fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Force integer x-axis ticks
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                
                # Add reference lines for key metrics
                if key in ['prey_final', 'predators_final']:
                    ax.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='≥15 Good')
                    ax.axhline(y=5 if key == 'prey_final' else 8, color='orange', 
                              linestyle='--', alpha=0.5, label='OK')
                    ax.legend(fontsize=8)
                elif key == 'value_loss':
                    ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='<5 Good')
                    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='<10 OK')
                    ax.legend(fontsize=8)
        
        self.env_fig.tight_layout()
        self.env_canvas_widget.draw()
    
    def setup_log_tab(self):
        """Setup raw log viewer"""
        control_frame = ttk.Frame(self.log_tab)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_frame, text="Raw training log:", font=('Arial', 12)).pack(side=tk.LEFT, padx=5)
        
        self.log_file_label = ttk.Label(control_frame, text="No file", foreground='gray', font=('Arial', 11))
        self.log_file_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="📊 Data source: Training log file", font=('Arial', 10), foreground='blue').pack(side=tk.LEFT, padx=15)
        ttk.Button(control_frame, text="📋 Copy Log Content", command=self.copy_log_content).pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="Reload Log", command=self.reload_raw_log).pack(side=tk.RIGHT, padx=5)
        
        self.raw_log_text = scrolledtext.ScrolledText(self.log_tab, height=30, font=('Courier', 10), wrap=tk.NONE)
        self.raw_log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def setup_config_tab(self):
        """Setup configuration editor tab"""
        from pathlib import Path
        import sys
        
        # Ensure src is in path to import config
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from src.config import SimulationConfig
        self.config = SimulationConfig()
        self.config_path = project_root / 'src' / 'config.py'
        
        # Parse comments from config file
        self.config_comments = self.parse_config_comments()
        
        # Main container with canvas and scrollbar
        canvas_container = ttk.Frame(self.config_tab)
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
        
        # Title and buttons
        title_frame = ttk.Frame(scrollable_frame)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(title_frame, text="Configuration Parameters", 
                 font=('Arial', 16, 'bold')).pack(side=tk.LEFT)
        
        ttk.Button(title_frame, text="💾 Save to config.py",
                  command=self.save_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(title_frame, text="🔄 Reload",
                  command=self.reload_config_ui).pack(side=tk.RIGHT, padx=5)
        
        # Info label
        info_label = ttk.Label(scrollable_frame, 
                              text="⚠️ Changes will be saved to config.py and used by the next training run.",
                              font=('Arial', 10), foreground='darkorange')
        info_label.pack(padx=10, pady=(0, 10))
        
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
                entry = ttk.Entry(param_frame, textvariable=entry_var, width=15)
                entry.pack(side=tk.LEFT, padx=5)
                
                # Store reference
                self.config_entries[param_name] = (entry_var, type(value))
                
                # Type indicator
                ttk.Label(param_frame, text=f"({type(value).__name__})",
                         font=('Arial', 9), foreground='gray').pack(side=tk.LEFT, padx=5)
                
                # Comment from config file on same line
                comment = self.config_comments.get(param_name, '')
                if comment:
                    ttk.Label(param_frame, text=f"💬 {comment}", 
                             font=('Arial', 9), foreground='#555555',
                             wraplength=500, justify=tk.LEFT).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Bind mousewheel to canvas scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def parse_config_comments(self):
        """Parse comments from config.py file for each parameter"""
        comments = {}
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                # Look for parameter definitions: PARAM_NAME = value
                if '=' in line and not line.strip().startswith('#'):
                    # Extract parameter name
                    parts = line.split('=')
                    if len(parts) >= 2:
                        param_name = parts[0].strip()
                        # Check if it's an uppercase parameter (config constant)
                        if param_name.isupper():
                            # Look for inline comment
                            comment_idx = line.find('#')
                            if comment_idx > 0:
                                comment = line[comment_idx+1:].strip()
                                if comment:
                                    comments[param_name] = comment
        except Exception as e:
            print(f"Warning: Could not parse config comments: {e}")
        
        return comments
    
    def get_config_sections(self):
        """Group config parameters by category"""
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
        """Reload config values into UI from current config file"""
        try:
            # Re-import config to get fresh values
            import importlib
            from src import config as config_module
            importlib.reload(config_module)
            from src.config import SimulationConfig
            self.config = SimulationConfig()
            
            # Update UI
            for param_name, (entry_var, _) in self.config_entries.items():
                if hasattr(self.config, param_name):
                    value = getattr(self.config, param_name)
                    entry_var.set(str(value))
            
            messagebox.showinfo("Reloaded", "Configuration values reloaded from config.py")
        except Exception as e:
            messagebox.showerror("Reload Error", f"Failed to reload config:\n{e}")
    
    def save_config(self):
        """Save edited config values to config.py file"""
        try:
            # Read current config file
            with open(self.config_path, 'r', encoding='utf-8') as f:
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
                            new_value_str = entry_var.get().strip()
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
            with open(self.config_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
            
            messagebox.showinfo("Configuration Saved", 
                              f"Changes saved to:\n{self.config_path}\n\n"
                              "✅ Next training run will use these values.\n\n"
                              "💡 If training is running, restart it to apply changes.")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save configuration:\n{e}")
    
    def auto_find_log(self):
        """Find most recent training log"""
        log_dir = Path("outputs/logs")
        print(f"Looking for logs in: {log_dir.absolute()}")
        
        if not log_dir.exists():
            msg = f"Directory not found: {log_dir.absolute()}"
            print(msg)
            self.status_label.config(text="No logs dir", foreground="red")
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(tk.END, msg)
            return
        
        log_files = sorted(log_dir.glob("training_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"Found {len(log_files)} log files")
        
        if log_files:
            self.last_log_file = log_files[0]
            print(f"Using log: {self.last_log_file}")
            self.status_label.config(text=f"Log: {self.last_log_file.name}", foreground="blue")
            # Update Training Control tab if it exists
            if hasattr(self, 'current_log_label'):
                self.current_log_label.config(text=self.last_log_file.name)
            self.refresh_data()
        else:
            msg = f"No training_*.log files in {log_dir.absolute()}"
            print(msg)
            self.status_label.config(text="No training logs", foreground="red")
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(tk.END, msg)
    
    def start_training(self):
        """Start training in independent process"""
        if self.training_process and self.training_process.poll() is None:
            self.status_label.config(text="Training already running!", foreground="orange")
            return
        
        try:
            # Use PowerShell to run training script
            script_path = Path("scripts/train.py").absolute()
            if not script_path.exists():
                self.status_label.config(text="Training script not found!", foreground="red")
                self.training_output.insert(tk.END, f"Error: {script_path} not found\n")
                return
            
            # Get Python from current environment
            venv_python = Path(".venv_rocm/Scripts/python.exe").absolute()
            if not venv_python.exists():
                # Try alternative paths
                venv_python = Path("venv/Scripts/python.exe").absolute()
                if not venv_python.exists():
                    venv_python = "python"  # Use system python as fallback
            
            # Create timestamped log file
            log_dir = Path("outputs/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"training_{timestamp}.log"
            
            # Get path to tee wrapper script
            tee_script = Path(__file__).parent / "tee_output.py"
            
            # Launch training in NEW CONSOLE WINDOW with tee-like output (both console AND log)
            if sys.platform == 'win32':
                # Windows: Use tee wrapper to show output in console AND write to log file
                self.training_process = subprocess.Popen(
                    [str(venv_python), str(tee_script), str(log_file), str(venv_python), str(script_path)],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # Unix: Use terminal emulator with tee command
                self.training_process = subprocess.Popen(
                    ['x-terminal-emulator', '-e', 'sh', '-c', 
                     f'{venv_python} {script_path} | tee {log_file}']
                )
            
            # No log handle needed - tee wrapper handles it
            self.training_log_handle = None
            
            # Update UI
            self.training_status_label.config(text="Running", foreground='green')
            self.training_indicator.itemconfig('indicator', fill='green')
            self.pid_label.config(text=f"PID: {self.training_process.pid}", foreground='green')
            self.start_training_btn.config(state=tk.DISABLED)
            self.stop_training_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Training started in console window!", foreground="green")
            
            # Switch to the new log file
            self.last_log_file = log_file
            self.current_log_label.config(text=log_file.name)
            
            # Enable auto-refresh to monitor progress
            if not self.auto_refresh_enabled:
                self.auto_refresh_var.set(True)
                self.toggle_auto_refresh()
            
            # Show info message
            messagebox.showinfo("Training Started", 
                              f"Training started in new console window.\n\n"
                              f"Log file: {log_file}\n\n"
                              f"Monitor progress using the other dashboard tabs.")
            
        except Exception as e:
            self.status_label.config(text=f"Failed to start: {e}", foreground="red")
            messagebox.showerror("Error", f"Failed to start training: {e}")
    
    def stop_training(self):
        """Stop the running training process"""
        if not self.training_process or self.training_process.poll() is not None:
            self.status_label.config(text="No training running", foreground="gray")
            return
        
        try:
            self.training_process.terminate()
            self.training_process.wait(timeout=5)
            messagebox.showinfo("Training Stopped", "Training has been stopped.")
        except subprocess.TimeoutExpired:
            self.training_process.kill()
            messagebox.showwarning("Force Stop", "Training process was force killed.")
        except Exception as e:
            messagebox.showerror("Error", f"Error stopping training: {e}")
        
        # Close log file handle if open
        if hasattr(self, 'training_log_handle') and self.training_log_handle:
            try:
                self.training_log_handle.close()
            except:
                pass
            self.training_log_handle = None
        
        self.training_process = None
        self.update_training_status(False)
    
    def check_training_status(self):
        """Check if training is currently running"""
        is_running = self.training_process and self.training_process.poll() is None
        self.update_training_status(is_running)
        
        # If process finished, read final output and close log handle
        if self.training_process and self.training_process.poll() is not None:
            return_code = self.training_process.returncode
            self.training_output.insert(tk.END, f"\n\nTraining finished with exit code: {return_code}\n")
            if return_code == 0:
                self.training_output.insert(tk.END, "Training completed successfully!\n")
            else:
                self.training_output.insert(tk.END, f"Training failed. Check the log file for details.\n")
            
            # Close log file handle
            if hasattr(self, 'training_log_handle') and self.training_log_handle:
                try:
                    self.training_log_handle.close()
                except:
                    pass
                self.training_log_handle = None
            
            self.training_process = None
    
    def update_training_status(self, is_running):
        """Update UI based on training status"""
        if is_running:
            self.training_status_label.config(text="Running", foreground='green')
            self.training_indicator.itemconfig('indicator', fill='green')
            self.start_training_btn.config(state=tk.DISABLED)
            self.stop_training_btn.config(state=tk.NORMAL)
            if self.training_process:
                self.pid_label.config(text=f"PID: {self.training_process.pid}", foreground='green')
        else:
            self.training_status_label.config(text="Not Running", foreground='gray')
            self.training_indicator.itemconfig('indicator', fill='gray')
            self.pid_label.config(text="PID: N/A", foreground='gray')
            self.start_training_btn.config(state=tk.NORMAL)
            self.stop_training_btn.config(state=tk.DISABLED)
    
    def periodic_status_check(self):
        """Periodically check training status"""
        self.check_training_status()
        self.root.after(2000, self.periodic_status_check)
    
    def monitor_log_file(self):
        """Monitor log file for new content and display in output area"""
        if not self.training_process or self.training_process.poll() is not None:
            return
        
        if not self.last_log_file or not self.last_log_file.exists():
            # Schedule next check
            self.root.after(500, self.monitor_log_file)
            return
        
        try:
            # Read last 20 lines from log file
            with open(self.last_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if lines:
                    # Get last 20 lines
                    recent_lines = lines[-20:]
                    
                    # Update output area with recent lines
                    self.training_output.delete(1.0, tk.END)
                    self.training_output.insert(tk.END, f"=== Last 20 lines from {self.last_log_file.name} ===\n\n")
                    self.training_output.insert(tk.END, ''.join(recent_lines))
                    self.training_output.see(tk.END)
        except Exception as e:
            pass  # Ignore read errors (file might be locked)
        
        # Schedule next check
        if self.training_process and self.training_process.poll() is None:
            self.root.after(1000, self.monitor_log_file)
    
    def select_log_file(self):
        """Open file dialog to select a log file"""
        from tkinter import filedialog
        
        log_dir = Path("outputs/logs")
        if not log_dir.exists():
            log_dir = Path(".")
        
        filename = filedialog.askopenfilename(
            title="Select Training Log File",
            initialdir=log_dir,
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        
        if filename:
            self.last_log_file = Path(filename)
            self.current_log_label.config(text=self.last_log_file.name)
            self.status_label.config(text=f"Log: {self.last_log_file.name}", foreground="blue")
            self.refresh_data()
        
        if log_files:
            self.last_log_file = log_files[0]
            print(f"Using log: {self.last_log_file}")
            self.status_label.config(text=f"Log: {self.last_log_file.name}", foreground="blue")
            self.refresh_data()
        else:
            msg = f"No training_*.log files in {log_dir.absolute()}"
            print(msg)
            self.status_label.config(text="No training logs", foreground="red")
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(tk.END, msg)
    
    def parse_log_file(self, log_file):
        """Parse training log and extract metrics"""
        self.episodes = []
        self.metrics = defaultdict(list)
        
        if not Path(log_file).exists():
            print(f"Log file not found: {log_file}")
            return
        
        # Detect encoding (UTF-16 or UTF-8)
        with open(log_file, 'rb') as f:
            raw_start = f.read(100)
            encoding = 'utf-16' if b'\x00' in raw_start[:50] else 'utf-8'
        
        print(f"Parsing log file: {log_file} (encoding: {encoding})")
        current_episode = None
        current_metrics = {}
        line_count = 0
        
        with open(log_file, 'r', encoding=encoding, errors='ignore') as f:
            for line in f:
                line_count += 1
                # Episode number
                match = re.search(r'\[.*?\] Episode (\d+)/\d+', line)
                if match:
                    if current_episode is not None and current_metrics:
                        self.episodes.append(current_episode)
                        for key, val in current_metrics.items():
                            self.metrics[key].append(val)
                        print(f"  Added episode {current_episode} with {len(current_metrics)} metrics")
                    current_episode = int(match.group(1))
                    current_metrics = {}
                    print(f"  Started parsing episode {current_episode}")
                    continue
                
                # PPO Diagnostics
                if '[PPO Diagnostics]' in line:
                    kl_match = re.search(r'KL: ([\d.]+)', line)
                    clip_match = re.search(r'ClipFrac: ([\d.]+)', line)
                    if kl_match and 'ppo_kl' not in current_metrics:  # Take first prey value
                        current_metrics['ppo_kl'] = float(kl_match.group(1))
                    if clip_match and 'ppo_clip_frac' not in current_metrics:
                        current_metrics['ppo_clip_frac'] = float(clip_match.group(1))
                
                # Extreme ratios warning
                if 'extreme' in line.lower() and 'ratio' in line.lower():
                    match = re.search(r'([\d.]+)%', line)
                    if match:
                        current_metrics['ppo_extreme_ratio_pct'] = float(match.group(1)) / 100.0
                
                # Losses
                if 'Losses:' in line:
                    # Format: Policy(P=x/Pr=y), Value(P=x/Pr=y), Entropy(P=x/Pr=y)
                    p_match = re.search(r'Policy\(P=([-\d.]+)/Pr=([-\d.]+)\)', line)
                    v_match = re.search(r'Value\(P=([-\d.]+)/Pr=([-\d.]+)\)', line)
                    e_match = re.search(r'Entropy\(P=([-\d.]+)/Pr=([-\d.]+)\)', line)
                    
                    if p_match:
                        current_metrics['policy_loss'] = (abs(float(p_match.group(1))) + abs(float(p_match.group(2)))) / 2
                    if v_match:
                        current_metrics['value_loss'] = (float(v_match.group(1)) + float(v_match.group(2))) / 2
                    if e_match:
                        current_metrics['entropy'] = (float(e_match.group(1)) + float(e_match.group(2))) / 2
                
                # Final counts
                if 'Final:' in line:
                    prey_match = re.search(r'Prey=(\d+)', line)
                    pred_match = re.search(r'Predators=(\d+)', line)
                    if prey_match:
                        current_metrics['prey_final'] = int(prey_match.group(1))
                    if pred_match:
                        current_metrics['predators_final'] = int(pred_match.group(1))
                
                # Births/Deaths/Meals
                if 'Meals=' in line:
                    meals_match = re.search(r'Meals=(\d+)', line)
                    if meals_match:
                        current_metrics['meals'] = int(meals_match.group(1))
                
                if 'Starvation=' in line:
                    starve_match = re.search(r'Starvation=(\d+)', line)
                    if starve_match:
                        current_metrics['starvation_deaths'] = int(starve_match.group(1))
        
        # Add last episode
        if current_episode is not None and current_metrics:
            self.episodes.append(current_episode)
            for key, val in current_metrics.items():
                self.metrics[key].append(val)
            print(f"  Added episode {current_episode} with {len(current_metrics)} metrics")
        
        print(f"Parsed {line_count} lines, found {len(self.episodes)} episodes")
        print(f"Metrics collected: {list(self.metrics.keys())}")
    
    def refresh_data(self):
        """Refresh all dashboard data"""
        if not self.last_log_file:
            self.auto_find_log()
            return
        
        self.status_label.config(text="Refreshing...", foreground="orange")
        self.root.update()
        
        # Parse log
        self.parse_log_file(self.last_log_file)
        
        # Update all displays
        self.update_stability_display()
        self.update_environment_display()
        self.update_trend_charts()  # Update line charts
        self.reload_raw_log()
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_label.config(
            text=f"Updated: {timestamp} | Episodes: {len(self.episodes)}", 
            foreground="green"
        )
    
    def update_stability_display(self):
        """Update stability metrics tab"""
        for key, widgets in self.stability_widgets.items():
            if key not in self.metrics or not self.metrics[key]:
                widgets['value'].config(text="--", fg='gray')
                widgets['trend'].config(text="No data")
                history_widget = widgets['history']
                history_widget.config(state=tk.NORMAL)
                history_widget.delete('1.0', tk.END)
                history_widget.config(state=tk.DISABLED)
                continue
            
            values = self.metrics[key]
            current = values[-1]
            
            # Format value
            if key == 'ppo_extreme_ratio_pct':
                val_text = f"{current*100:.1f}%"
            else:
                val_text = f"{current:.4f}"
            
            # Color based on health
            color = self.get_health_color(key, current)
            widgets['value'].config(text=val_text, fg=color)
            
            # Trend
            trend = self.get_trend_arrow(values)
            widgets['trend'].config(text=trend)
            
            # Color-coded history (last 5)
            history_widget = widgets['history']
            history_widget.config(state=tk.NORMAL)
            history_widget.delete('1.0', tk.END)
            
            history = values[-5:]
            for idx, val in enumerate(history):
                if key == 'ppo_extreme_ratio_pct':
                    text = f"{val*100:.1f}%"
                else:
                    text = f"{val:.4f}"
                
                color = self.get_health_color(key, val)
                history_widget.insert(tk.END, text, (f'color_{color}_{idx}', 'center'))
                history_widget.tag_config(f'color_{color}_{idx}', foreground=color)
                
                if idx < len(history) - 1:
                    history_widget.insert(tk.END, " \u2192 ", 'center')
            
            history_widget.config(state=tk.DISABLED)
    
    def update_environment_display(self):
        """Update environment metrics tab"""
        for key, widgets in self.env_widgets.items():
            if key not in self.metrics or not self.metrics[key]:
                widgets['value'].config(text="--", fg='gray')
                widgets['trend'].config(text="No data")
                history_widget = widgets['history']
                history_widget.config(state=tk.NORMAL)
                history_widget.delete('1.0', tk.END)
                history_widget.config(state=tk.DISABLED)
                continue
            
            values = self.metrics[key]
            current = values[-1]
            
            # Format value
            if isinstance(current, float):
                val_text = f"{current:.3f}"
            else:
                val_text = f"{current}"
            
            # Color based on metric type
            color = self.get_environment_color(key, current)
            widgets['value'].config(text=val_text, fg=color)
            
            # Trend
            trend = self.get_trend_arrow(values)
            widgets['trend'].config(text=trend)
            
            # Color-coded history (last 5)
            history_widget = widgets['history']
            history_widget.config(state=tk.NORMAL)
            history_widget.delete('1.0', tk.END)
            
            history = values[-5:]
            for idx, val in enumerate(history):
                # Use same format as current value for consistency
                if isinstance(current, float):
                    text = f"{val:.3f}"
                else:
                    text = f"{val}"
                
                color = self.get_environment_color(key, val)
                history_widget.insert(tk.END, text, (f'color_{color}_{idx}', 'center'))
                history_widget.tag_config(f'color_{color}_{idx}', foreground=color)
                
                if idx < len(history) - 1:
                    history_widget.insert(tk.END, " \u2192 ", 'center')
            
            history_widget.config(state=tk.DISABLED)
    
    def get_health_color(self, metric, value):
        """Get color based on metric health"""
        # Handle special entropy case (higher is better, but check minimum)
        if metric == 'entropy':
            if value >= self.thresholds.get('entropy', {}).get('min_healthy', 1.5):
                return 'green'
            elif value >= self.thresholds.get('entropy', {}).get('min_warning', 1.0):
                return 'darkorange'
            else:
                return 'red'
        
        if metric not in self.thresholds:
            return 'black'
        
        thresh = self.thresholds[metric]
        if 'healthy' in thresh:
            if value <= thresh['healthy']:
                return 'green'
            elif value <= thresh['warning']:
                return 'darkorange'
            elif value <= thresh.get('bad', float('inf')):
                return 'red'
            else:
                return 'darkred'
        return 'black'
    
    def get_environment_color(self, metric, value):
        """Get color for environment metrics (performance indicators)"""
        if metric not in self.thresholds:
            return 'black'
        
        thresh = self.thresholds[metric]
        is_reverse = thresh.get('reverse', False)  # True if lower is better
        
        if is_reverse:
            # Lower is better (e.g., starvation deaths, value loss)
            if value <= thresh['good']:
                return 'green'
            elif value <= thresh['ok']:
                return 'darkorange'
            else:
                return 'red'
        else:
            # Higher is better (e.g., meals, survivors)
            if value >= thresh['good']:
                return 'green'
            elif value >= thresh['ok']:
                return 'darkorange'
            else:
                return 'red'
    
    def get_trend_arrow(self, values, window=3):
        """Get trend arrow"""
        if len(values) < window:
            return "—"
        
        recent = values[-window:]
        if recent[-1] > recent[0] * 1.1:
            return "↗ Rising"
        elif recent[-1] < recent[0] * 0.9:
            return "↘ Falling"
        else:
            return "→ Stable"
    
    def run_behavior_analysis(self):
        """Run integrated behavior analysis on all available checkpoints"""
        self.behavior_status.config(text="Analyzing...", foreground="orange")
        self.status_label.config(text="Running behavior analysis...", foreground="orange")
        self.root.update()
        
        try:
            # Add project root to Python path
            import os
            project_root = str(Path(__file__).parent.parent.absolute())
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from src.config import SimulationConfig
            from src.models.actor_critic_network import ActorCriticNetwork
            
            config = SimulationConfig()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Find all checkpoints
            ckpt_dir = Path("outputs/checkpoints")
            if not ckpt_dir.exists():
                self.behavior_status.config(text="No checkpoints found", foreground="red")
                return
            
            # model_B = predator, model_A = prey
            pred_ckpts = sorted(ckpt_dir.glob("model_B_ppo_ep*.pth"))
            prey_ckpts = sorted(ckpt_dir.glob("model_A_ppo_ep*.pth"))
            
            if not pred_ckpts and not prey_ckpts:
                self.behavior_status.config(text="No checkpoints", foreground="red")
                return
            
            # Clear existing table data (grid widgets)
            for widget in self.behavior_table_frame.winfo_children():
                widget.destroy()
            
            # Organize by episode: {episode: {pred_hunting, pred_hunger, pred_mating, prey_evasion, prey_mating}}
            episode_data = {}
            
            # Process predator checkpoints (model_B)
            for ckpt_path in pred_ckpts:
                ep_match = re.search(r'ep(\d+)', ckpt_path.stem)
                if ep_match:
                    episode = int(ep_match.group(1))
                    if episode not in episode_data:
                        episode_data[episode] = {}
                    
                    model = ActorCriticNetwork(config).to(device)
                    state = torch.load(ckpt_path, map_location=device, weights_only=False)
                    model.load_state_dict(state, strict=True)
                    
                    hunt = self.evaluate_hunting(model, config, device, samples=100)
                    hunger = self.evaluate_hunger_response(model, config, device, samples=100)
                    mate_pred = self.evaluate_mating_behavior(model, config, device, is_predator=True, samples=100)
                    selective = self.evaluate_selective_hunting(model, config, device, samples=100)
                    energy_mgmt = self.evaluate_energy_management(model, config, device, samples=100)
                    
                    episode_data[episode]['pred_hunting'] = self.format_behavior_score(hunt['rate'])
                    episode_data[episode]['pred_hunger'] = self.format_behavior_score(hunger['rate'])
                    episode_data[episode]['pred_mating'] = self.format_behavior_score(mate_pred['rate'])
                    episode_data[episode]['pred_selective'] = self.format_behavior_score(selective['rate'])
                    episode_data[episode]['pred_energy_mgmt'] = self.format_behavior_score(energy_mgmt['rate'])
            
            # Process prey checkpoints (model_A)
            for ckpt_path in prey_ckpts:
                ep_match = re.search(r'ep(\d+)', ckpt_path.stem)
                if ep_match:
                    episode = int(ep_match.group(1))
                    if episode not in episode_data:
                        episode_data[episode] = {}
                    
                    model = ActorCriticNetwork(config).to(device)
                    state = torch.load(ckpt_path, map_location=device, weights_only=False)
                    model.load_state_dict(state, strict=True)
                    
                    evade = self.evaluate_evasion(model, config, device, samples=100)
                    mate_prey = self.evaluate_mating_behavior(model, config, device, is_predator=False, samples=100)
                    multi_target = self.evaluate_multi_target_handling(model, config, device, samples=100)
                    flocking = self.evaluate_flocking(model, config, device, samples=100)
                    threat_assess = self.evaluate_threat_assessment(model, config, device, samples=100)
                    
                    episode_data[episode]['prey_evasion'] = self.format_behavior_score(evade['rate'])
                    episode_data[episode]['prey_mating'] = self.format_behavior_score(mate_prey['rate'])
                    episode_data[episode]['prey_multi_target'] = self.format_behavior_score(multi_target['rate'])
                    episode_data[episode]['prey_flocking'] = self.format_behavior_score(flocking['rate'])
                    episode_data[episode]['prey_threat_assess'] = self.format_behavior_score(threat_assess['rate'])
            
            # Populate table with individual cell coloring
            # Clear existing widgets
            for widget in self.behavior_table_frame.winfo_children():
                widget.destroy()
            for widget in self.behavior_header_frame.winfo_children():
                widget.destroy()
            
            # Define colors
            colors = {
                'excellent': '#d4edda',  # Green
                'good': '#fff3cd',       # Yellow
                'poor': '#f8d7da'        # Red
            }
            
            col_width = 140
            
            # Create header
            headers = ["Episode", "Pred: Hunting", "Pred: Hunger", "Pred: Mating",
                      "Pred: Selective Hunt", "Pred: Energy Mgmt",
                      "Prey: Evasion", "Prey: Mating",
                      "Prey: Multi-Target", "Prey: Flocking", "Prey: Threat Assess"]
            for col, header_text in enumerate(headers):
                header_label = tk.Label(self.behavior_header_frame, text=header_text, 
                                       font=('Arial', 11, 'bold'), bg='#e9ecef', 
                                       relief=tk.RIDGE, borderwidth=1, width=17, padx=10, pady=8)
                header_label.grid(row=0, column=col, sticky='ew')
            
            # Data rows sorted by episode (reverse order - latest first)
            for row_idx, episode in enumerate(sorted(episode_data.keys(), reverse=True)):
                data = episode_data[episode]
                
                # Episode number cell
                episode_label = tk.Label(self.behavior_table_frame, text=str(episode),
                                        font=('Arial', 11, 'bold'), bg='white',
                                        relief=tk.RIDGE, borderwidth=1, width=17, padx=10, pady=6)
                episode_label.grid(row=row_idx, column=0, sticky='ew')
                
                # Each behavior cell with individual coloring
                for col_idx, key in enumerate(['pred_hunting', 'pred_hunger', 'pred_mating',
                                               'pred_selective', 'pred_energy_mgmt',
                                               'prey_evasion', 'prey_mating',
                                               'prey_multi_target', 'prey_flocking', 'prey_threat_assess'], start=1):
                    value_str = data.get(key, 'N/A')
                    
                    # Determine cell color based on score
                    bg_color = 'white'
                    if '%' in value_str:
                        try:
                            score = int(value_str.split()[-1].rstrip('%'))
                            if score >= 80:
                                bg_color = colors['excellent']
                            elif score >= 60:
                                bg_color = colors['good']
                            else:
                                bg_color = colors['poor']
                        except (ValueError, IndexError):
                            pass
                    
                    # Create cell label
                    cell_label = tk.Label(self.behavior_table_frame, text=value_str,
                                         font=('Arial', 11), bg=bg_color,
                                         relief=tk.RIDGE, borderwidth=1, width=17, padx=10, pady=6)
                    cell_label.grid(row=row_idx, column=col_idx, sticky='ew')
            
            self.behavior_status.config(text=f"Analyzed {len(episode_data)} episodes", foreground="green")
            self.status_label.config(text="Behavior analysis complete", foreground="green")
            
        except Exception as e:
            import traceback
            # Clear table and show error
            for widget in self.behavior_table_frame.winfo_children():
                widget.destroy()
            error_label = tk.Label(self.behavior_table_frame, text=f"Error: {str(e)}", 
                                  font=('Arial', 11), fg='red')
            error_label.grid(row=0, column=0)
            self.behavior_status.config(text="Error - see status bar", foreground="red")
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")
            print(f"Behavior analysis error:\n{traceback.format_exc()}")
    
    @staticmethod
    def format_behavior_score(rate):
        """Format behavior score with emoji indicator"""
        if rate >= 80:
            emoji = "\u2705"
        elif rate >= 60:
            emoji = "\u26a0\ufe0f"
        else:
            emoji = "\u274c"
        return f"{emoji} {rate:.0f}%"
    
    # Behavior evaluation helper methods
    @staticmethod
    def make_vis_batch(B, N, targets, device="cpu"):
        """Create visible animals batch with multiple targets"""
        vis = torch.zeros(B, N, 8, device=device)
        vis[:, :, 7] = 0.0  # all padding by default
        
        for i, target in enumerate(targets):
            if i >= N:
                break
            vis[:, i, 7] = 1.0  # is_present
            vis[:, i, 0] = target['dx']
            vis[:, i, 1] = target['dy']
            vis[:, i, 2] = float((target['dx']**2 + target['dy']**2) ** 0.5)
            vis[:, i, 3] = 1.0 if target.get('is_predator', False) else 0.0
            vis[:, i, 4] = 1.0 if target.get('is_prey', False) else 0.0
        
        return vis
    
    @staticmethod
    def get_action_dirs():
        """Get normalized action direction vectors"""
        dirs = torch.tensor([
            [0.0, -1.0],[1.0, -1.0],[1.0, 0.0],[1.0, 1.0],
            [0.0,  1.0],[-1.0, 1.0],[-1.0,0.0],[-1.0,-1.0]
        ], dtype=torch.float32)
        dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
        return dirs
    
    @torch.no_grad()
    def evaluate_hunting(self, model, config, device, samples=50):
        """Test predator hunting: moving toward prey"""
        model.eval()
        obs = torch.zeros(1, 34, device=device)
        obs[0, 4] = 1.0  # is_predator
        obs[0, 5] = 0.8  # high energy
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        approach_count = 0
        
        for _ in range(samples):
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(2, 10)
            dx = dist * np.cos(angle)
            dy = dist * np.sin(angle)
            
            targets = [{'dx': dx, 'dy': dy, 'is_prey': True}]
            vis = self.make_vis_batch(1, N, targets, device)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            target_vec = torch.tensor([[dx, dy]], device=device, dtype=torch.float32)
            target_vec = target_vec / (target_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            dot = (action_vec @ target_vec.T).item()
            
            if dot > 0:
                approach_count += 1
        
        return {'rate': (approach_count / samples) * 100}
    
    @torch.no_grad()
    def evaluate_evasion(self, model, config, device, samples=50):
        """Test prey evasion: moving away from predators"""
        model.eval()
        obs = torch.zeros(1, 34, device=device)
        obs[0, 3] = 1.0  # is_prey
        obs[0, 5] = 0.8  # high energy
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        evade_count = 0
        
        for _ in range(samples):
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(2, 10)
            dx = dist * np.cos(angle)
            dy = dist * np.sin(angle)
            
            targets = [{'dx': dx, 'dy': dy, 'is_predator': True}]
            vis = self.make_vis_batch(1, N, targets, device)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            target_vec = torch.tensor([[dx, dy]], device=device, dtype=torch.float32)
            target_vec = target_vec / (target_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            dot = (action_vec @ target_vec.T).item()
            
            if dot < 0:
                evade_count += 1
        
        return {'rate': (evade_count / samples) * 100}
    
    @torch.no_grad()
    def evaluate_mating_behavior(self, model, config, device, is_predator=True, samples=50):
        """Test mating: approaching same-species when high energy"""
        model.eval()
        obs = torch.zeros(1, 34, device=device)
        if is_predator:
            obs[0, 4] = 1.0  # is_predator
        else:
            obs[0, 3] = 1.0  # is_prey
        obs[0, 5] = 0.9  # high energy
        obs[0, 6] = 0.5  # mature age
        
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        approach_count = 0
        
        for _ in range(samples):
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(1, 5)
            dx = dist * np.cos(angle)
            dy = dist * np.sin(angle)
            
            targets = [{'dx': dx, 'dy': dy, 'is_predator': is_predator, 'is_prey': not is_predator}]
            vis = self.make_vis_batch(1, N, targets, device)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            target_vec = torch.tensor([[dx, dy]], device=device, dtype=torch.float32)
            target_vec = target_vec / (target_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            dot = (action_vec @ target_vec.T).item()
            
            if dot > 0.3:
                approach_count += 1
        
        return {'rate': (approach_count / samples) * 100}
    
    @torch.no_grad()
    def evaluate_hunger_response(self, model, config, device, samples=50):
        """Test hunger response: predator approaching prey when low energy"""
        model.eval()
        obs = torch.zeros(1, 34, device=device)
        obs[0, 4] = 1.0  # is_predator
        obs[0, 5] = 0.2  # LOW energy
        
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        hunt_count = 0
        
        for _ in range(samples):
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(2, 10)
            dx = dist * np.cos(angle)
            dy = dist * np.sin(angle)
            
            targets = [{'dx': dx, 'dy': dy, 'is_prey': True}]
            vis = self.make_vis_batch(1, N, targets, device)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            target_vec = torch.tensor([[dx, dy]], device=device, dtype=torch.float32)
            target_vec = target_vec / (target_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            dot = (action_vec @ target_vec.T).item()
            
            if dot > 0:
                hunt_count += 1
        
        return {'rate': (hunt_count / samples) * 100}
    
    @torch.no_grad()
    def evaluate_action_diversity(self, model, config, device, samples=200):
        """Test action diversity: entropy of action distribution (higher = more diverse)"""
        model.eval()
        obs = torch.zeros(1, 34, device=device)
        obs[0, 3] = 1.0  # is_prey
        obs[0, 5] = 0.8  # high energy
        N = config.MAX_VISIBLE_ANIMALS
        vis = self.make_vis_batch(1, N, [], device)
        
        action_counts = torch.zeros(8)
        
        for _ in range(samples):
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            action_counts[action] += 1
        
        # Calculate entropy (0 = all same action, log2(8)≈2.08 = perfectly uniform)
        probs = action_counts / action_counts.sum()
        probs = probs[probs > 0]  # Remove zeros
        entropy = -(probs * torch.log2(probs)).sum().item()
        max_entropy = np.log2(8)  # 2.08 for 8 actions
        diversity_pct = (entropy / max_entropy) * 100
        
        return {'rate': diversity_pct}
    
    @torch.no_grad()
    def evaluate_energy_conservation(self, model, config, device, samples=50):
        """Test energy conservation: prey resting when low energy and no threats"""
        model.eval()
        obs = torch.zeros(1, 34, device=device)
        obs[0, 3] = 1.0  # is_prey
        obs[0, 5] = 0.15  # LOW energy
        N = config.MAX_VISIBLE_ANIMALS
        vis = self.make_vis_batch(1, N, [], device)  # No threats nearby
        
        rest_count = 0
        
        for _ in range(samples):
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            if action == 8:  # Rest action
                rest_count += 1
        
        return {'rate': (rest_count / samples) * 100}
    
    @torch.no_grad()
    def evaluate_selective_hunting(self, model, config, device, samples=50):
        """Test selective hunting: predators choosing closer prey over distant"""
        model.eval()
        obs = torch.zeros(1, 34, device=device)
        obs[0, 4] = 1.0  # is_predator
        obs[0, 5] = 0.6  # moderate energy
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        chose_closer = 0
        
        for _ in range(samples):
            # Two prey: one close, one far
            angle_close = np.random.uniform(0, 2*np.pi)
            angle_far = np.random.uniform(0, 2*np.pi)
            
            dist_close = np.random.uniform(2, 4)
            dist_far = np.random.uniform(7, 10)
            
            dx_close = dist_close * np.cos(angle_close)
            dy_close = dist_close * np.sin(angle_close)
            dx_far = dist_far * np.cos(angle_far)
            dy_far = dist_far * np.sin(angle_far)
            
            targets = [
                {'dx': dx_close, 'dy': dy_close, 'is_prey': True},
                {'dx': dx_far, 'dy': dy_far, 'is_prey': True}
            ]
            vis = self.make_vis_batch(1, N, targets, device)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            # Check which target the action aligns with more
            close_vec = torch.tensor([[dx_close, dy_close]], device=device, dtype=torch.float32)
            far_vec = torch.tensor([[dx_far, dy_far]], device=device, dtype=torch.float32)
            close_vec = close_vec / (close_vec.norm() + 1e-8)
            far_vec = far_vec / (far_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            
            dot_close = (action_vec @ close_vec.T).item()
            dot_far = (action_vec @ far_vec.T).item()
            
            if dot_close > dot_far:
                chose_closer += 1
        
        return {'rate': (chose_closer / samples) * 100}
    
    @torch.no_grad()
    def evaluate_multi_target_handling(self, model, config, device, samples=50):
        """Test multi-target handling: prey fleeing from nearest predator when multiple threats"""
        model.eval()
        obs = torch.zeros(1, 34, device=device)
        obs[0, 3] = 1.0  # is_prey
        obs[0, 5] = 0.8  # high energy
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        fled_nearest = 0
        
        for _ in range(samples):
            # Two predators: one close, one far
            angle_close = np.random.uniform(0, 2*np.pi)
            angle_far = np.random.uniform(0, 2*np.pi)
            
            dist_close = np.random.uniform(2, 4)
            dist_far = np.random.uniform(7, 10)
            
            dx_close = dist_close * np.cos(angle_close)
            dy_close = dist_close * np.sin(angle_close)
            dx_far = dist_far * np.cos(angle_far)
            dy_far = dist_far * np.sin(angle_far)
            
            targets = [
                {'dx': dx_close, 'dy': dy_close, 'is_predator': True},
                {'dx': dx_far, 'dy': dy_far, 'is_predator': True}
            ]
            vis = self.make_vis_batch(1, N, targets, device)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            # Check if fleeing from nearest (negative dot product)
            close_vec = torch.tensor([[dx_close, dy_close]], device=device, dtype=torch.float32)
            close_vec = close_vec / (close_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            
            dot_close = (action_vec @ close_vec.T).item()
            
            if dot_close < -0.3:  # Fleeing (negative dot)
                fled_nearest += 1
        
        return {'rate': (fled_nearest / samples) * 100}
    
    @torch.no_grad()
    def evaluate_edge_avoidance(self, model, config, device, samples=50):
        """Test edge avoidance: animals turning away from boundaries"""
        model.eval()
        obs = torch.zeros(1, 34, device=device)
        obs[0, 3] = 1.0  # is_prey
        obs[0, 5] = 0.8  # high energy
        obs[0, 0] = 0.85  # Near edge (x close to 1.0)
        obs[0, 1] = 0.5   # Center y
        N = config.MAX_VISIBLE_ANIMALS
        vis = self.make_vis_batch(1, N, [], device)
        action_dirs = self.get_action_dirs().to(device)
        
        avoided_edge = 0
        
        for _ in range(samples):
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            # Check if action moves away from edge (negative x direction)
            action_vec = action_dirs[action:action+1]
            if action_vec[0, 0].item() < 0:  # Moving left (away from right edge)
                avoided_edge += 1
        
        return {'rate': (avoided_edge / samples) * 100}
    
    @torch.no_grad()
    def evaluate_flocking(self, model, config, device, samples=50):
        """Test flocking: prey moving toward same-species when predator nearby"""
        model.eval()
        obs = torch.zeros(1, 34, device=device)
        obs[0, 3] = 1.0  # is_prey
        obs[0, 5] = 0.7  # good energy
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        flocked = 0
        
        for _ in range(samples):
            # Predator far, prey nearby
            angle_pred = np.random.uniform(0, 2*np.pi)
            angle_prey = np.random.uniform(0, 2*np.pi)
            
            dx_pred = 9 * np.cos(angle_pred)
            dy_pred = 9 * np.sin(angle_pred)
            dx_prey = 3 * np.cos(angle_prey)
            dy_prey = 3 * np.sin(angle_prey)
            
            targets = [
                {'dx': dx_pred, 'dy': dy_pred, 'is_predator': True},
                {'dx': dx_prey, 'dy': dy_prey, 'is_prey': True}
            ]
            vis = self.make_vis_batch(1, N, targets, device)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            # Check if moving toward prey
            prey_vec = torch.tensor([[dx_prey, dy_prey]], device=device, dtype=torch.float32)
            prey_vec = prey_vec / (prey_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            dot = (action_vec @ prey_vec.T).item()
            
            if dot > 0.3:
                flocked += 1
        
        return {'rate': (flocked / samples) * 100}
    
    @torch.no_grad()
    def evaluate_energy_management(self, model, config, device, samples=50):
        """Test energy management: hunting urgency increases as energy drops"""
        model.eval()
        obs_high = torch.zeros(1, 34, device=device)
        obs_high[0, 4] = 1.0  # is_predator
        obs_high[0, 5] = 0.8  # HIGH energy
        
        obs_low = torch.zeros(1, 34, device=device)
        obs_low[0, 4] = 1.0  # is_predator
        obs_low[0, 5] = 0.15  # LOW energy
        
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        hunt_high = 0
        hunt_low = 0
        
        for _ in range(samples):
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(4, 8)
            dx = dist * np.cos(angle)
            dy = dist * np.sin(angle)
            
            targets = [{'dx': dx, 'dy': dy, 'is_prey': True}]
            vis = self.make_vis_batch(1, N, targets, device)
            
            target_vec = torch.tensor([[dx, dy]], device=device, dtype=torch.float32)
            target_vec = target_vec / (target_vec.norm() + 1e-8)
            
            # High energy behavior
            _, move_probs_high, _ = model.forward(obs_high, vis)
            action_high = torch.argmax(move_probs_high, dim=1).item()
            action_vec_high = action_dirs[action_high:action_high+1]
            if (action_vec_high @ target_vec.T).item() > 0:
                hunt_high += 1
            
            # Low energy behavior
            _, move_probs_low, _ = model.forward(obs_low, vis)
            action_low = torch.argmax(move_probs_low, dim=1).item()
            action_vec_low = action_dirs[action_low:action_low+1]
            if (action_vec_low @ target_vec.T).item() > 0:
                hunt_low += 1
        
        # Return the difference: low energy should hunt MORE than high energy
        urgency_increase = (hunt_low - hunt_high) / samples * 100
        # Normalize to 0-100% (positive increase is good)
        normalized = 50 + urgency_increase  # 50% = no difference, 100% = always hunts when low
        return {'rate': max(0, min(100, normalized))}
    
    @torch.no_grad()
    def evaluate_threat_assessment(self, model, config, device, samples=50):
        """Test threat assessment: prey reacts more strongly to near vs far predators"""
        model.eval()
        obs = torch.zeros(1, 34, device=device)
        obs[0, 3] = 1.0  # is_prey
        obs[0, 5] = 0.8  # high energy
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        evade_near = 0
        evade_far = 0
        
        for _ in range(samples):
            angle = np.random.uniform(0, 2*np.pi)
            
            # Near threat
            dx_near = 2.5 * np.cos(angle)
            dy_near = 2.5 * np.sin(angle)
            targets_near = [{'dx': dx_near, 'dy': dy_near, 'is_predator': True}]
            vis_near = self.make_vis_batch(1, N, targets_near, device)
            
            _, move_probs_near, _ = model.forward(obs, vis_near)
            action_near = torch.argmax(move_probs_near, dim=1).item()
            
            target_vec_near = torch.tensor([[dx_near, dy_near]], device=device, dtype=torch.float32)
            target_vec_near = target_vec_near / (target_vec_near.norm() + 1e-8)
            action_vec_near = action_dirs[action_near:action_near+1]
            if (action_vec_near @ target_vec_near.T).item() < 0:
                evade_near += 1
            
            # Far threat
            dx_far = 9 * np.cos(angle)
            dy_far = 9 * np.sin(angle)
            targets_far = [{'dx': dx_far, 'dy': dy_far, 'is_predator': True}]
            vis_far = self.make_vis_batch(1, N, targets_far, device)
            
            _, move_probs_far, _ = model.forward(obs, vis_far)
            action_far = torch.argmax(move_probs_far, dim=1).item()
            
            target_vec_far = torch.tensor([[dx_far, dy_far]], device=device, dtype=torch.float32)
            target_vec_far = target_vec_far / (target_vec_far.norm() + 1e-8)
            action_vec_far = action_dirs[action_far:action_far+1]
            if (action_vec_far @ target_vec_far.T).item() < 0:
                evade_far += 1
        
        # Return the difference: should evade near threats more than far
        assessment_diff = (evade_near - evade_far) / samples * 100
        # Normalize to 0-100% (50% = equal response, 100% = perfect discrimination)
        normalized = 50 + assessment_diff
        return {'rate': max(0, min(100, normalized))}
    
    def reload_raw_log(self):
        """Reload and display raw log file"""
        if not self.last_log_file or not Path(self.last_log_file).exists():
            self.raw_log_text.delete(1.0, tk.END)
            self.raw_log_text.insert(tk.END, "No log file loaded")
            self.log_file_label.config(text="No file", foreground='red')
            return
        
        try:
            # Detect encoding
            with open(self.last_log_file, 'rb') as f:
                raw_start = f.read(100)
                encoding = 'utf-16' if b'\x00' in raw_start[:50] else 'utf-8'
            
            # Read full log
            with open(self.last_log_file, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            self.raw_log_text.delete(1.0, tk.END)
            self.raw_log_text.insert(tk.END, content)
            self.log_file_label.config(text=self.last_log_file.name, foreground='blue')
            
        except Exception as e:
            self.raw_log_text.delete(1.0, tk.END)
            self.raw_log_text.insert(tk.END, f"Error reading log file:\n{str(e)}")
            self.log_file_label.config(text="Error", foreground='red')
    
    def toggle_auto_refresh(self):
        """Toggle auto-refresh"""
        self.auto_refresh_enabled = self.auto_refresh_var.get()
        if self.auto_refresh_enabled:
            self.auto_refresh_loop()
    
    def auto_refresh_loop(self):
        """Auto-refresh loop"""
        if self.auto_refresh_enabled:
            self.refresh_data()
            self.root.after(10000, self.auto_refresh_loop)  # 10 seconds
    
    def copy_all_data(self):
        """Copy all dashboard data to clipboard (PPO, Environment, Behaviors)"""
        content = "="*100 + "\n"
        content += "LIFE GAME TRAINING DASHBOARD - COMPLETE DATA EXPORT\n"
        content += "="*100 + "\n\n"
        
        # === PPO STABILITY METRICS ===
        content += "\n" + "="*100 + "\n"
        content += "PPO STABILITY METRICS - ALL EPISODES\n"
        content += "="*100 + "\n\n"
        
        # Current PPO values
        content += "LATEST VALUES:\n" + "-"*50 + "\n"
        for metric_name, key in [('KL Divergence', 'ppo_kl'), ('Clip Fraction', 'ppo_clip_frac'), 
                                  ('Extreme Ratios', 'ppo_extreme_ratio_pct'), ('Entropy', 'entropy')]:
            if key in self.stability_widgets:
                value = self.stability_widgets[key]['value'].cget('text')
                trend = self.stability_widgets[key]['trend'].cget('text')
                content += f"{metric_name}: {value}  |  {trend}\n"
        
        content += "\n" + "FULL HISTORY:\n" + "-"*50 + "\n"
        for i, ep in enumerate(self.episodes):
            content += f"Episode {ep}:\n"
            for metric_name, key in [('KL Divergence', 'ppo_kl'), ('Clip Fraction', 'ppo_clip_frac'), 
                                      ('Extreme Ratios', 'ppo_extreme_ratio_pct'), ('Entropy', 'entropy')]:
                if key in self.metrics and i < len(self.metrics[key]):
                    val = self.metrics[key][i]
                    if isinstance(val, float):
                        content += f"  {metric_name}: {val:.6f}\n"
                    else:
                        content += f"  {metric_name}: {val}\n"
            content += "\n"
        
        # === ENVIRONMENT METRICS ===
        content += "\n" + "="*100 + "\n"
        content += "ENVIRONMENT METRICS - ALL EPISODES\n"
        content += "="*100 + "\n\n"
        
        # Current Environment values
        content += "LATEST VALUES:\n" + "-"*50 + "\n"
        for metric_name, key in [('Meals/Episode', 'meals'), ('Starvation Deaths', 'starvation_deaths'),
                                  ('Final Prey Count', 'prey_final'), ('Final Predator Count', 'predators_final'),
                                  ('Value Loss', 'value_loss'), ('Policy Loss', 'policy_loss')]:
            if key in self.env_widgets:
                value = self.env_widgets[key]['value'].cget('text')
                trend = self.env_widgets[key]['trend'].cget('text')
                content += f"{metric_name}: {value}  |  {trend}\n"
        
        content += "\n" + "FULL HISTORY:\n" + "-"*50 + "\n"
        for i, ep in enumerate(self.episodes):
            content += f"Episode {ep}:\n"
            for metric_name, key in [('Meals/Episode', 'meals'), ('Starvation Deaths', 'starvation_deaths'),
                                      ('Final Prey Count', 'prey_final'), ('Final Predator Count', 'predators_final'),
                                      ('Value Loss', 'value_loss'), ('Policy Loss', 'policy_loss')]:
                if key in self.metrics and i < len(self.metrics[key]):
                    val = self.metrics[key][i]
                    if isinstance(val, float):
                        content += f"  {metric_name}: {val:.4f}\n"
                    else:
                        content += f"  {metric_name}: {val}\n"
            content += "\n"
        
        # === BEHAVIORAL ANALYSIS ===
        content += "\n" + "="*100 + "\n"
        content += "BEHAVIORAL ANALYSIS\n"
        content += "="*100 + "\n\n"
        content += f"{'Episode':<10} {'Pred:Hunt':<12} {'Pred:Hunger':<12} {'Pred:Mate':<12} "
        content += f"{'Pred:Div':<12} {'Pred:Select':<12} {'Pred:EnerMgmt':<12} "
        content += f"{'Prey:Evade':<12} {'Prey:Mate':<12} {'Prey:EnerCons':<12} {'Prey:MultiT':<12} "
        content += f"{'Prey:Edge':<12} {'Prey:Flock':<12} {'Prey:Threat':<12}\n"
        content += "-"*180 + "\n"
        
        # Extract behavioral data from grid
        row_count = 0
        for widget in self.behavior_table_frame.winfo_children():
            info = widget.grid_info()
            if info and info['row'] > 0:
                row_count = max(row_count, info['row'])
        
        for row in range(1, row_count + 1):
            row_data = []
            for col in range(11):
                for widget in self.behavior_table_frame.winfo_children():
                    info = widget.grid_info()
                    if info and info['row'] == row and info['column'] == col:
                        row_data.append(widget.cget('text'))
                        break
            if len(row_data) >= 6:
                while len(row_data) < 11:
                    row_data.append('N/A')
                content += f"{row_data[0]:<10} {row_data[1]:<12} {row_data[2]:<12} {row_data[3]:<12} {row_data[4]:<12} {row_data[5]:<12} "
                content += f"{row_data[6]:<12} {row_data[7]:<12} {row_data[8]:<12} {row_data[9]:<12} {row_data[10]:<12}\n"
        
        content += "\n" + "="*100 + "\n"
        content += "END OF DATA EXPORT\n"
        content += "="*100 + "\n"
        
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status_label.config(text=f"Copied ALL data ({len(self.episodes)} episodes)!", foreground="green")
        self.root.after(3000, lambda: self.status_label.config(text="Ready", foreground="gray"))
    
    def copy_stability_content(self):
        """Copy PPO Stability tab content to clipboard (ALL episodes)"""
        content = "PPO STABILITY METRICS - ALL EPISODES\n" + "="*70 + "\n\n"
        
        # Current values
        for metric_name, key in [('KL Divergence', 'ppo_kl'), ('Clip Fraction', 'ppo_clip_frac'), 
                                  ('Extreme Ratios', 'ppo_extreme_ratio_pct'), ('Entropy', 'entropy')]:
            if key in self.stability_widgets:
                value = self.stability_widgets[key]['value'].cget('text')
                trend = self.stability_widgets[key]['trend'].cget('text')
                content += f"{metric_name}: {value}  |  {trend}\n"
        
        content += "\n" + "="*70 + "\nFULL HISTORY (ALL EPISODES)\n" + "="*70 + "\n\n"
        
        # All episodes data
        for i, ep in enumerate(self.episodes):
            content += f"Episode {ep}:\n"
            for metric_name, key in [('KL Divergence', 'ppo_kl'), ('Clip Fraction', 'ppo_clip_frac'), 
                                      ('Extreme Ratios', 'ppo_extreme_ratio_pct'), ('Entropy', 'entropy')]:
                if key in self.metrics and i < len(self.metrics[key]):
                    val = self.metrics[key][i]
                    if isinstance(val, float):
                        content += f"  {metric_name}: {val:.6f}\n"
                    else:
                        content += f"  {metric_name}: {val}\n"
            content += "\n"
        
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status_label.config(text=f"Copied PPO Stability ({len(self.episodes)} episodes)!", foreground="green")
        self.root.after(3000, lambda: self.status_label.config(text="Ready", foreground="gray"))
    
    def copy_environment_content(self):
        """Copy Environment tab content to clipboard (ALL episodes)"""
        content = "ENVIRONMENT METRICS - ALL EPISODES\n" + "="*70 + "\n\n"
        
        # Current values
        for metric_name, key in [('Meals/Episode', 'meals'), ('Starvation Deaths', 'starvation_deaths'),
                                  ('Final Prey Count', 'prey_final'), ('Final Predator Count', 'predators_final'),
                                  ('Value Loss', 'value_loss'), ('Policy Loss', 'policy_loss')]:
            if key in self.env_widgets:
                value = self.env_widgets[key]['value'].cget('text')
                trend = self.env_widgets[key]['trend'].cget('text')
                content += f"{metric_name}: {value}  |  {trend}\n"
        
        content += "\n" + "="*70 + "\nFULL HISTORY (ALL EPISODES)\n" + "="*70 + "\n\n"
        
        # All episodes data
        for i, ep in enumerate(self.episodes):
            content += f"Episode {ep}:\n"
            for metric_name, key in [('Meals/Episode', 'meals'), ('Starvation Deaths', 'starvation_deaths'),
                                      ('Final Prey Count', 'prey_final'), ('Final Predator Count', 'predators_final'),
                                      ('Value Loss', 'value_loss'), ('Policy Loss', 'policy_loss')]:
                if key in self.metrics and i < len(self.metrics[key]):
                    val = self.metrics[key][i]
                    if isinstance(val, float):
                        content += f"  {metric_name}: {val:.4f}\n"
                    else:
                        content += f"  {metric_name}: {val}\n"
            content += "\n"
        
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status_label.config(text=f"Copied Environment ({len(self.episodes)} episodes)!", foreground="green")
        self.root.after(3000, lambda: self.status_label.config(text="Ready", foreground="gray"))
    
    def copy_behavior_content(self):
        """Copy Behaviors tab content to clipboard"""
        content = "BEHAVIORAL ANALYSIS\n" + "="*130 + "\n\n"
        content += f"{'Episode':<10} {'Pred:Hunt':<12} {'Pred:Hunger':<12} {'Pred:Mate':<12} "
        content += f"{'Pred:Select':<12} {'Pred:EnerMgmt':<12} "
        content += f"{'Prey:Evade':<12} {'Prey:Mate':<12} {'Prey:MultiT':<12} "
        content += f"{'Prey:Flock':<12} {'Prey:Threat':<12}\n"
        content += "-"*130 + "\n"
        
        # Extract data from grid labels
        row_count = 0
        for widget in self.behavior_table_frame.winfo_children():
            info = widget.grid_info()
            if info and info['row'] > 0:  # Skip header row
                row_count = max(row_count, info['row'])
        
        # Collect data by rows
        for row in range(1, row_count + 1):
            row_data = []
            for col in range(11):
                for widget in self.behavior_table_frame.winfo_children():
                    info = widget.grid_info()
                    if info and info['row'] == row and info['column'] == col:
                        row_data.append(widget.cget('text'))
                        break
            if len(row_data) >= 6:  # At least the original columns
                # Pad with N/A if missing new columns
                while len(row_data) < 11:
                    row_data.append('N/A')
                content += f"{row_data[0]:<10} {row_data[1]:<12} {row_data[2]:<12} {row_data[3]:<12} {row_data[4]:<12} {row_data[5]:<12} "
                content += f"{row_data[6]:<12} {row_data[7]:<12} {row_data[8]:<12} {row_data[9]:<12} {row_data[10]:<12}\n"
        
        if row_count > 0:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.status_label.config(text="Copied behavior analysis!", foreground="green")
            self.root.after(3000, lambda: self.status_label.config(text="Ready", foreground="gray"))
    
    def copy_log_content(self):
        """Copy Raw Log tab content to clipboard"""
        content = self.raw_log_text.get(1.0, tk.END).strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.status_label.config(text="Copied Raw Log content!", foreground="green")
            self.root.after(3000, lambda: self.status_label.config(text="Ready", foreground="gray"))


def main():
    root = tk.Tk()
    app = TrainingDashboardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
