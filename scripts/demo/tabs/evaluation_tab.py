"""
Evaluation Tab - Display current simulation statistics with evaluation metrics
"""

import tkinter as tk
from tkinter import ttk
import math


class EvaluationTab:
    """Tab for displaying current simulation statistics"""
    
    def __init__(self, parent, app):
        self.app = app
        self.frame = ttk.Frame(parent)
        self.summary_boxes = {}
        
        # Tracking for evaluation metrics
        self.detection_events = 0
        self.capture_events = 0
        self.escape_events = 0
        self.capture_times = []
        self.chase_start_distances = {}  # (pred_id, prey_id) -> distance at detection
        self.chase_step_counts = {}  # (pred_id, prey_id) -> steps since detection
        
        self.setup_ui()
    
    def setup_ui(self):
        """Build statistics display UI"""
        # Top title
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Label(control_frame, text="Current Simulation Statistics",
                 font=('Arial', 14, 'bold')).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="🔄 Refresh",
                  command=self.refresh_display).pack(side=tk.RIGHT, padx=5)
        
        # Summary section
        summary_frame = ttk.LabelFrame(self.frame, text="Statistics", padding="10")
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.create_summary_section(summary_frame)
        
        # Auto-refresh every 2 seconds
        self.frame.after(2000, self.auto_refresh)
    
    def create_summary_section(self, parent):
        """Create summary boxes"""
        grid_frame = ttk.Frame(parent)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Evaluation metrics matching dashboard
        summaries = [
            ('Current Step', 'step_count', ''),
            ('Current Prey', 'prey_count', ''),
            ('Current Predators', 'predator_count', ''),
            ('Total Births', 'births', ''),
            ('Total Deaths', 'deaths', ''),
            ('Total Meals', 'meals', ''),
            ('Grass Eaten', 'grass_eaten', ''),
            ('Detection Events', 'detection_events', ''),
            ('Capture %', 'capture_rate', '%'),
            ('Escape %', 'escape_rate', '%'),
            ('Avg Capture Time', 'avg_capture_time', ' steps'),
            ('Meals/Pred', 'meals_per_pred', ''),
        ]
        
        row = 0
        col = 0
        for title, key, suffix in summaries:
            self.create_summary_box(grid_frame, title, key, suffix, row, col)
            col += 1
            if col >= 4:
                col = 0
                row += 1
    
    def create_summary_box(self, parent, title, key, suffix, row, col):
        """Create a summary indicator box"""
        frame = ttk.Frame(parent, relief='solid', borderwidth=1)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
        
        ttk.Label(frame, text=title, font=('Arial', 9)).pack(pady=(5, 0))
        value_label = ttk.Label(frame, text="--", font=('Arial', 14, 'bold'))
        value_label.pack(pady=(0, 5))
        
        self.summary_boxes[key] = (value_label, suffix)
        
        # Configure column weights
        parent.grid_columnconfigure(col, weight=1)
    
    def refresh_display(self):
        """Update display with current simulation stats"""
        if not hasattr(self.app, 'simulation_tab'):
            return
        
        sim_tab = self.app.simulation_tab
        
        # Get current counts
        prey_count = len([a for a in sim_tab.animals if a.name == "A"])
        predator_count = len([a for a in sim_tab.animals if a.name == "B"])
        
        # Calculate capture rate and escape rate
        capture_rate = 0.0
        escape_rate = 0.0
        if self.detection_events > 0:
            capture_rate = self.capture_events / self.detection_events
            escape_rate = self.escape_events / self.detection_events
        
        # Calculate average capture time
        avg_capture_time = 0.0
        if self.capture_times:
            avg_capture_time = sum(self.capture_times) / len(self.capture_times)
        
        # Calculate meals per predator
        meals_per_pred = 0.0
        if predator_count > 0:
            meals_per_pred = sim_tab.meals / predator_count
        
        # Update all summary boxes
        display_values = {
            'step_count': sim_tab.step_count,
            'prey_count': prey_count,
            'predator_count': predator_count,
            'births': sim_tab.births,
            'deaths': sim_tab.deaths,
            'meals': sim_tab.meals,
            'grass_eaten': sim_tab.grass_eaten,
            'detection_events': self.detection_events,
            'capture_rate': capture_rate,
            'escape_rate': escape_rate,
            'avg_capture_time': avg_capture_time,
            'meals_per_pred': meals_per_pred,
        }
        
        for key, (label, suffix) in self.summary_boxes.items():
            value = display_values.get(key, 0)
            if suffix == '%':
                text = f"{value * 100:.1f}%"
            elif suffix == ' steps':
                text = f"{value:.1f}{suffix}"
            elif suffix == '/step':
                text = f"{value:.3f}{suffix}"
            else:
                text = f"{value:.0f}{suffix}"
            label.config(text=text, foreground='blue')
    
    def track_detection(self, predator, prey):
        """Track when a predator detects a prey"""
        chase_key = (predator.id, prey.id)
        if chase_key not in self.chase_start_distances:
            self.detection_events += 1
            # Calculate distance
            dx = predator.x - prey.x
            dy = predator.y - prey.y
            distance = math.sqrt(dx*dx + dy*dy)
            self.chase_start_distances[chase_key] = distance
            self.chase_step_counts[chase_key] = 0
    
    def track_capture(self, predator, prey):
        """Track when a predator captures prey"""
        chase_key = (predator.id, prey.id)
        if chase_key in self.chase_step_counts:
            self.capture_events += 1
            self.capture_times.append(self.chase_step_counts[chase_key])
            # Clean up
            del self.chase_start_distances[chase_key]
            del self.chase_step_counts[chase_key]
    
    def track_escape(self, predator, prey, chase_key=None):
        """Track when prey escapes from predator"""
        if chase_key is None and predator and prey:
            chase_key = (predator.id, prey.id)
        
        if chase_key and chase_key in self.chase_step_counts:
            self.escape_events += 1
            # Clean up
            if chase_key in self.chase_start_distances:
                del self.chase_start_distances[chase_key]
            if chase_key in self.chase_step_counts:
                del self.chase_step_counts[chase_key]
    
    def update_chase_steps(self):
        """Increment step counter for all active chases"""
        for chase_key in list(self.chase_step_counts.keys()):
            self.chase_step_counts[chase_key] += 1
    
    def reset_metrics(self):
        """Reset all tracking metrics"""
        self.detection_events = 0
        self.capture_events = 0
        self.escape_events = 0
        self.capture_times = []
        self.chase_start_distances = {}
        self.chase_step_counts = {}
    
    def auto_refresh(self):
        """Auto-refresh display periodically"""
        self.refresh_display()
        self.frame.after(2000, self.auto_refresh)
