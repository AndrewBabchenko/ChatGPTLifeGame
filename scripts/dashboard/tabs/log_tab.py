"""
Log Tab - View raw training log file
"""
import tkinter as tk
from tkinter import ttk, scrolledtext

from .base_tab import BaseTab


class LogTab(BaseTab):
    """Raw log file viewer tab"""
    
    def setup_ui(self):
        """Setup log viewer UI"""
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Raw Training Log", 
                 font=('Arial', 14, 'bold')).pack(side=tk.LEFT)
        
        ttk.Button(control_frame, text="� Copy Log", 
                  command=self.copy_log).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(control_frame, text="�🔄 Reload", 
                  command=self.reload_log).pack(side=tk.RIGHT, padx=(10, 0))
        
        self.lines_var = tk.StringVar(value="500")
        ttk.Label(control_frame, text="Lines:").pack(side=tk.RIGHT, padx=(10, 5))
        lines_combo = ttk.Combobox(control_frame, textvariable=self.lines_var,
                                  values=["100", "250", "500", "1000", "All"], width=6)
        lines_combo.pack(side=tk.RIGHT)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(main_frame, font=('Courier', 9),
                                                  wrap=tk.NONE, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Horizontal scrollbar
        h_scroll = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, 
                                command=self.log_text.xview)
        h_scroll.pack(fill=tk.X)
        self.log_text.configure(xscrollcommand=h_scroll.set)
    
    def refresh(self):
        """Refresh log view"""
        if not hasattr(self, 'log_text'):
            return
        self.reload_log()
    
    def reload_log(self):
        """Reload log file content"""
        if not self.app.last_log_file or not self.app.last_log_file.exists():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, "No log file selected or file not found.")
            self.log_text.config(state=tk.DISABLED)
            return
        
        try:
            lines_setting = self.lines_var.get()
            
            with open(self.app.last_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                if lines_setting == "All":
                    content = f.read()
                else:
                    try:
                        n_lines = int(lines_setting)
                        all_lines = f.readlines()
                        content = ''.join(all_lines[-n_lines:])
                    except ValueError:
                        content = f.read()
            
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, content)
            self.log_text.see(tk.END)  # Scroll to bottom
            self.log_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, f"Error reading log file: {e}")
            self.log_text.config(state=tk.DISABLED)
    
    def copy_log(self):
        """Copy log content to clipboard"""
        content = self.log_text.get(1.0, tk.END).strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.app.status_label.config(text="✓ Copied log content to clipboard", foreground="green")
            self.root.after(3000, lambda: self.app.status_label.config(text="Ready", foreground="gray"))
        else:
            self.app.status_label.config(text="No log content to copy", foreground="red")
            self.root.after(3000, lambda: self.app.status_label.config(text="Ready", foreground="gray"))
