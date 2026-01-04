"""
Chart Tab - Statistics visualization over time
"""

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class ChartTab:
    """Tab for displaying simulation statistics charts"""
    
    def __init__(self, parent, app):
        self.app = app
        self.frame = ttk.Frame(parent)
        
        # History tracking
        self.history = {
            'step': [],
            'prey': [],
            'predators': [],
            'births': [],
            'deaths': [],
            'meals': [],
            'grass_eaten': []
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Build chart UI"""
        # Chart frame
        chart_frame = ttk.LabelFrame(self.frame, text="Statistics Over Time", padding="10")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self._chart_dpi = 100.0
        try:
            self._chart_dpi = float(self.app.root.winfo_fpixels("1i"))
        except tk.TclError:
            pass
        
        self.fig = Figure(figsize=(12, 8), dpi=self._chart_dpi)
        
        try:
            self.fig.set_layout_engine("tight")
        except Exception:
            self.fig.set_constrained_layout(True)
        
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        chart_widget = self.chart_canvas.get_tk_widget()
        
        chart_widget.configure(highlightthickness=0, bd=0)
        chart_widget.pack(fill=tk.BOTH, expand=True)
        
        chart_widget.bind("<Configure>", self.chart_canvas.resize)
        
        self._chart_widget = chart_widget
        self._chart_facecolor = self.app.root.cget("background")
        try:
            self._chart_facecolor = chart_widget.cget("background")
        except tk.TclError:
            pass
        
        self.ax = None
        self.ax2 = None
        
        # Initialize chart
        self.frame.after(100, self.initialize_chart)
    
    def initialize_chart(self):
        """Initialize empty chart"""
        self.update_chart(0, 0, 0, 0, 0, 0, 0)
    
    def reset(self):
        """Reset chart history"""
        self.history = {
            'step': [],
            'prey': [],
            'predators': [],
            'births': [],
            'deaths': [],
            'meals': [],
            'grass_eaten': []
        }
        self.update_chart(0, 0, 0, 0, 0, 0, 0)
    
    def update_chart(self, step, prey_count, predator_count, births, deaths, meals, grass_eaten):
        """Update chart with new data"""
        # Ensure widget has valid dimensions
        if not hasattr(self, '_chart_widget'):
            return
        
        width = self._chart_widget.winfo_width()
        height = self._chart_widget.winfo_height()
        if width <= 1 or height <= 1:
            self.frame.after(50, lambda: self.update_chart(step, prey_count, predator_count, births, deaths, meals, grass_eaten))
            return
        
        # Add to history
        self.history['step'].append(step)
        self.history['prey'].append(prey_count)
        self.history['predators'].append(predator_count)
        self.history['births'].append(births)
        self.history['deaths'].append(deaths)
        self.history['meals'].append(meals)
        self.history['grass_eaten'].append(grass_eaten)
        
        # Clear and redraw
        self.fig.clear()
        
        try:
            self.fig.set_layout_engine("tight")
        except Exception:
            self.fig.set_constrained_layout(True)
        
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
            
            self.ax.plot(steps, self.history['prey'], label='Prey', color='#5ac8fa', linewidth=2)
            self.ax.plot(steps, self.history['predators'], label='Predators', color='#ff6464', linewidth=2)
            
            self.ax.set_ylim(bottom=0)
            self.ax.legend(loc='upper left')
            
            self.ax2 = self.ax.twinx()
            self.ax2.set_facecolor(bg)
            self.ax2.set_ylabel('Events Count', color='gray')
            self.ax2.plot(steps, self.history['births'], label='Births', color='#34c759', linestyle='--', linewidth=1.5)
            self.ax2.plot(steps, self.history['deaths'], label='Deaths', color='#ff9500', linestyle='--', linewidth=1.5)
            self.ax2.plot(steps, self.history['meals'], label='Meals', color='#af52de', linestyle='--', linewidth=1.5)
            self.ax2.plot(steps, self.history['grass_eaten'], label='Grass Eaten', color='#2d5016', linestyle=':', linewidth=1.5)
            self.ax2.tick_params(axis='y', labelcolor='gray')
            
            self.ax2.set_ylim(bottom=0)
            self.ax2.legend(loc='upper right')
        else:
            self.ax.set_ylim(0, 100)
            self.ax.text(50, 50, 'Waiting for data...',
                        ha='center', va='center', fontsize=12, color='gray', alpha=0.5)
        
        self.chart_canvas.draw_idle()
