"""
Base tab class for dashboard modules
"""
import tkinter as tk
from tkinter import ttk
from abc import ABC, abstractmethod


class BaseTab(ABC):
    """Base class for all dashboard tabs"""
    
    def __init__(self, parent: ttk.Frame, app):
        """
        Initialize tab
        
        Args:
            parent: Parent ttk.Frame (notebook tab)
            app: Main TrainingDashboardApp instance for accessing shared state
        """
        self.parent = parent
        self.app = app
        self.root = app.root
        self.widgets = {}
        self.setup_ui()  # Call setup_ui automatically
        
    @abstractmethod
    def setup_ui(self):
        """Setup the tab UI - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def refresh(self):
        """Refresh tab data - must be implemented by subclasses"""
        pass
    
    def get_trend_arrow(self, values, window=3):
        """Get trend arrow for a list of values"""
        if len(values) < window:
            return "—"
        
        recent = values[-window:]
        if recent[-1] > recent[0] * 1.1:
            return "↗ Rising"
        elif recent[-1] < recent[0] * 0.9:
            return "↘ Falling"
        else:
            return "→ Stable"
