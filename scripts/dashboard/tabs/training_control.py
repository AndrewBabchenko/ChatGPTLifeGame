"""
Training Control Tab - Start/stop training, select log files
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from .base_tab import BaseTab


# Phase configurations
PHASE_INFO = {
    1: {"name": "Pure Hunt/Evade", "config": "config_phase1", "color": "#4CAF50"},
    2: {"name": "Add Starvation", "config": "config_phase2", "color": "#2196F3"},
    3: {"name": "Add Reproduction", "config": "config_phase3", "color": "#FF9800"},
    4: {"name": "Full Ecosystem", "config": "config", "color": "#9C27B0"},
}


def load_phase_config(phase_num, force_reload=True):
    """Load and return the config module for a given phase.
    
    Args:
        phase_num: Phase number (1-4)
        force_reload: If True, forces re-reading from disk (not cache)
    """
    import importlib
    import sys
    
    config_name = PHASE_INFO[phase_num]["config"]
    module_name = f"src.{config_name}"
    
    if force_reload:
        # Remove from sys.modules to force fresh import
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        # Also invalidate import caches to ensure we read from disk
        importlib.invalidate_caches()
    
    try:
        module = importlib.import_module(module_name)
        if force_reload and module_name in sys.modules:
            # Double-check with reload to get latest file content
            module = importlib.reload(module)
        return module
    except ImportError as e:
        print(f"Error loading phase {phase_num} config: {e}")
        return None


class TrainingControlTab(BaseTab):
    """Training control and monitoring tab"""
    
    def __init__(self, parent: ttk.Frame, app):
        self.training_process = None
        self.training_log_handle = None
        self.current_phase = None
        super().__init__(parent, app)  # This will call setup_ui()
    
    def setup_ui(self):
        """Setup training control UI"""
        main_frame = ttk.Frame(self.parent, padding="20")
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
        
        self.pid_label = ttk.Label(status_inner, text="PID: N/A", font=('Arial', 10), foreground='gray')
        self.pid_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # === Training Control Section ===
        control_frame = ttk.LabelFrame(main_frame, text="Training Control", padding="15")
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        button_row = ttk.Frame(control_frame)
        button_row.pack(fill=tk.X)
        
        self.start_training_btn = ttk.Button(button_row, text="▶ Start Training", 
                                            command=self.start_training)
        self.start_training_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_training_btn = ttk.Button(button_row, text="⏹ Stop Training", 
                                           command=self.stop_training, state=tk.DISABLED)
        self.stop_training_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_row, text="🔄 Check Status", 
                  command=self.check_training_status).pack(side=tk.LEFT)
        
        info_text = ("Training runs in an independent process. If the dashboard crashes, "
                    "training will continue running in the background.")
        info_label = ttk.Label(control_frame, text=info_text, font=('Arial', 10), 
                              foreground='blue', wraplength=700)
        info_label.pack(pady=(10, 0))
        
        # === PHASE TRAINING SECTION ===
        phase_frame = ttk.LabelFrame(main_frame, text="Curriculum Phase Training", padding="15")
        phase_frame.pack(fill=tk.X, pady=(0, 15))
        
        phase_info_label = ttk.Label(phase_frame, 
            text="Start training for a specific curriculum phase. Each phase builds on the previous.",
            font=('Arial', 10), foreground='gray')
        phase_info_label.pack(pady=(0, 10))
        
        # Phase buttons row
        phase_buttons_frame = ttk.Frame(phase_frame)
        phase_buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.phase_buttons = {}
        for phase_num in [1, 2, 3, 4]:
            info = PHASE_INFO[phase_num]
            btn_frame = ttk.Frame(phase_buttons_frame)
            btn_frame.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
            
            btn = tk.Button(btn_frame, 
                           text=f"Phase {phase_num}\n{info['name']}", 
                           command=lambda p=phase_num: self.start_phase_training(p),
                           font=('Arial', 10, 'bold'),
                           bg=info['color'],
                           fg='white',
                           activebackground=info['color'],
                           activeforeground='white',
                           width=18, height=2)
            btn.pack(fill=tk.X)
            self.phase_buttons[phase_num] = btn
        
        # Reload button row (prominent)
        reload_frame = ttk.Frame(phase_frame)
        reload_frame.pack(fill=tk.X, pady=(10, 5))
        
        self.reload_btn = ttk.Button(reload_frame, text="🔄 Reload Config Files", 
                                     command=self.reload_phase_info)
        self.reload_btn.pack(side=tk.LEFT)
        
        ttk.Label(reload_frame, text="← Click after editing config files to see updated settings",
                 font=('Arial', 9), foreground='gray').pack(side=tk.LEFT, padx=(10, 0))
        
        # Phase status display
        self.phase_status_frame = ttk.Frame(phase_frame)
        self.phase_status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.phase_info_display = ttk.Label(self.phase_status_frame, 
            text="Click 'Reload Config Files' to see current phase settings", 
            font=('Arial', 10), foreground='gray')
        self.phase_info_display.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # === Log File Selection Section ===
        log_frame = ttk.LabelFrame(main_frame, text="Log File Selection", padding="15")
        log_frame.pack(fill=tk.X, pady=(0, 15))
        
        log_row1 = ttk.Frame(log_frame)
        log_row1.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(log_row1, text="Current Log:", font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        self.current_log_label = ttk.Label(log_row1, text="None", font=('Arial', 11), foreground='blue')
        self.current_log_label.pack(side=tk.LEFT)
        
        log_row2 = ttk.Frame(log_frame)
        log_row2.pack(fill=tk.X)
        
        ttk.Button(log_row2, text="📂 Select Log File", 
                  command=self.select_log_file).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(log_row2, text="🔍 Auto-Find Latest", 
                  command=lambda: self.app.auto_find_log(clear_data=True)).pack(side=tk.LEFT)
        
        log_info = ("The dashboard will automatically switch to the new log file when training starts. "
                   "You can also manually select a different log file to view.")
        ttk.Label(log_frame, text=log_info, font=('Arial', 10), 
                 foreground='gray', wraplength=700).pack(pady=(10, 0))
        
        # Training output area
        output_frame = ttk.LabelFrame(main_frame, text="Training Output (last 20 lines)", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        from tkinter import scrolledtext
        self.training_output = scrolledtext.ScrolledText(output_frame, height=10, font=('Courier', 9))
        self.training_output.pack(fill=tk.BOTH, expand=True)
        
        # Start periodic status check
        self.check_training_status()
        self.root.after(2000, self.periodic_status_check)
    
    def refresh(self):
        """Refresh tab - update log label"""
        if self.app.last_log_file:
            self.current_log_label.config(text=self.app.last_log_file.name)
    
    def reload_phase_info(self):
        """Reload phase configurations from disk and update display"""
        from datetime import datetime
        
        info_lines = []
        details = []
        for phase_num in [1, 2, 3, 4]:
            config = load_phase_config(phase_num, force_reload=True)
            if config:
                prey_ckpt = getattr(config, 'LOAD_PREY_CHECKPOINT', None)
                pred_ckpt = getattr(config, 'LOAD_PREDATOR_CHECKPOINT', None)
                save_prefix = getattr(config, 'SAVE_CHECKPOINT_PREFIX', f'phase{phase_num}')
                episodes = getattr(config.SimulationConfig, 'NUM_EPISODES', 'N/A')
                
                # Short display
                if prey_ckpt and pred_ckpt:
                    load_short = 'From prev'
                elif prey_ckpt or pred_ckpt:
                    load_short = 'Partial'
                else:
                    load_short = 'Fresh'
                info_lines.append(f"P{phase_num}: {episodes}ep ({load_short})")
                
                # Detailed info for output
                details.append(f"Phase {phase_num}: {episodes} episodes, prey={prey_ckpt or 'None'}, pred={pred_ckpt or 'None'}, save={save_prefix}")
        
        # Update display label
        self.phase_info_display.config(
            text=" | ".join(info_lines) if info_lines else "Could not load configs",
            foreground='blue' if info_lines else 'red'
        )
        
        # Show details in training output
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_output.insert(tk.END, f"\n[{timestamp}] Config files reloaded from disk:\n")
        for detail in details:
            self.training_output.insert(tk.END, f"  {detail}\n")
        self.training_output.see(tk.END)
        
        self.app.status_label.config(text="✓ Phase configs reloaded from disk", foreground="green")
    
    def start_phase_training(self, phase_num):
        """Start training for a specific curriculum phase"""
        if self.training_process and self.training_process.poll() is None:
            self.app.status_label.config(text="Training already running!", foreground="orange")
            return
        
        self.current_phase = phase_num
        phase_info = PHASE_INFO[phase_num]
        
        # Load phase config to show details
        config = load_phase_config(phase_num)
        if not config:
            messagebox.showerror("Error", f"Could not load config for Phase {phase_num}")
            return
        
        prey_ckpt = getattr(config, 'LOAD_PREY_CHECKPOINT', None)
        pred_ckpt = getattr(config, 'LOAD_PREDATOR_CHECKPOINT', None)
        save_prefix = getattr(config, 'SAVE_CHECKPOINT_PREFIX', f'phase{phase_num}')
        
        # Confirm with user
        msg = f"Start Phase {phase_num}: {phase_info['name']}?\n\n"
        msg += f"• Prey checkpoint: {prey_ckpt or 'Fresh start'}\n"
        msg += f"• Predator checkpoint: {pred_ckpt or 'Fresh start'}\n"
        msg += f"• Save prefix: {save_prefix}\n"
        msg += f"• Episodes: {getattr(config.SimulationConfig, 'NUM_EPISODES', 50)}\n\n"
        msg += "Continue?"
        
        if not messagebox.askyesno("Start Phase Training", msg):
            return
        
        try:
            # Use run_phase.py script with phase argument
            run_phase_script = Path("scripts/run_phase.py").absolute()
            if not run_phase_script.exists():
                messagebox.showerror("Error", f"Phase runner script not found: {run_phase_script}")
                return
            
            # Get Python from ROCm environment first (for GPU training), fall back to .venv
            venv_python = Path(".venv_rocm/Scripts/python.exe").absolute()
            if not venv_python.exists():
                venv_python = Path(".venv/Scripts/python.exe").absolute()
                if not venv_python.exists():
                    venv_python = "python"
            
            # Create timestamped log file
            log_dir = Path("outputs/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"phase{phase_num}_{timestamp}.log"
            
            # Path: scripts/dashboard/tabs/training_control.py -> scripts/tee_output.py
            tee_script = Path(__file__).parent.parent.parent / "tee_output.py"
            
            if sys.platform == 'win32':
                self.training_process = subprocess.Popen(
                    [str(venv_python), str(tee_script), str(log_file), 
                     str(venv_python), str(run_phase_script), "--phase", str(phase_num)],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                self.training_process = subprocess.Popen(
                    ['x-terminal-emulator', '-e', 'sh', '-c', 
                     f'{venv_python} {run_phase_script} --phase {phase_num} | tee {log_file}']
                )
            
            # Update UI
            self.training_status_label.config(
                text=f"Phase {phase_num}: {phase_info['name']}", foreground='green')
            self.training_indicator.itemconfig('indicator', fill=phase_info['color'])
            self.pid_label.config(text=f"PID: {self.training_process.pid}", foreground='green')
            self.start_training_btn.config(state=tk.DISABLED)
            self.stop_training_btn.config(state=tk.NORMAL)
            
            # Disable all phase buttons while training
            for btn in self.phase_buttons.values():
                btn.config(state=tk.DISABLED)
            
            self.app.status_label.config(
                text=f"Phase {phase_num} training started!", foreground="green")
            
            # Clear old data and switch to new log file
            self.app.episode_data.clear()
            self.app.last_log_file = log_file
            self.current_log_label.config(text=log_file.name)
            self.app.log_display.config(text=log_file.name)  # Also update top bar
            
            # Show status in training output
            self.training_output.insert(tk.END, f"\n[Training] Phase {phase_num} started\n")
            self.training_output.insert(tk.END, f"[Training] Log file: {log_file}\n")
            self.training_output.insert(tk.END, f"[Training] Waiting for output...\n")
            self.training_output.see(tk.END)
            
            # Enable auto-refresh
            if not self.app.auto_refresh_enabled:
                self.app.auto_refresh_var.set(True)
                self.app.toggle_auto_refresh()
            
            # Wait for log file to be created, then refresh
            self._wait_for_log_file(log_file)
            
            messagebox.showinfo("Phase Training Started", 
                              f"Phase {phase_num} ({phase_info['name']}) started.\n\n"
                              f"Log file: {log_file}\n\n"
                              f"Dashboard will auto-refresh every {self.app.refresh_interval // 1000} seconds.")
            
        except Exception as e:
            self.app.status_label.config(text=f"Failed to start: {e}", foreground="red")
            messagebox.showerror("Error", f"Failed to start phase training: {e}")
    
    def start_training(self):
        """Start training in independent process"""
        if self.training_process and self.training_process.poll() is None:
            self.app.status_label.config(text="Training already running!", foreground="orange")
            return
        
        try:
            script_path = Path("scripts/train.py").absolute()
            if not script_path.exists():
                self.app.status_label.config(text="Training script not found!", foreground="red")
                self.training_output.insert(tk.END, f"Error: {script_path} not found\n")
                return
            
            # Get Python from ROCm environment first (for GPU training), fall back to .venv
            venv_python = Path(".venv_rocm/Scripts/python.exe").absolute()
            if not venv_python.exists():
                venv_python = Path(".venv/Scripts/python.exe").absolute()
                if not venv_python.exists():
                    venv_python = "python"
            
            # Create timestamped log file
            log_dir = Path("outputs/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"training_{timestamp}.log"
            
            # Path: scripts/dashboard/tabs/training_control.py -> scripts/tee_output.py
            tee_script = Path(__file__).parent.parent.parent / "tee_output.py"
            
            if sys.platform == 'win32':
                self.training_process = subprocess.Popen(
                    [str(venv_python), str(tee_script), str(log_file), str(venv_python), str(script_path)],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                self.training_process = subprocess.Popen(
                    ['x-terminal-emulator', '-e', 'sh', '-c', 
                     f'{venv_python} {script_path} | tee {log_file}']
                )
            
            # Update UI
            self.training_status_label.config(text="Running", foreground='green')
            self.training_indicator.itemconfig('indicator', fill='green')
            self.pid_label.config(text=f"PID: {self.training_process.pid}", foreground='green')
            self.start_training_btn.config(state=tk.DISABLED)
            self.stop_training_btn.config(state=tk.NORMAL)
            self.app.status_label.config(text="Training started!", foreground="green")
            
            # Clear old data and switch to new log file
            self.app.episode_data.clear()
            self.app.last_log_file = log_file
            self.current_log_label.config(text=log_file.name)
            self.app.log_display.config(text=log_file.name)  # Also update top bar
            
            # Show status in training output
            self.training_output.insert(tk.END, f"\n[Training] Started at {datetime.now().strftime('%H:%M:%S')}\n")
            self.training_output.insert(tk.END, f"[Training] Log file: {log_file}\n")
            self.training_output.insert(tk.END, f"[Training] Waiting for output...\n")
            self.training_output.see(tk.END)
            
            # Enable auto-refresh
            if not self.app.auto_refresh_enabled:
                self.app.auto_refresh_var.set(True)
                self.app.toggle_auto_refresh()
            
            # Wait for log file to be created, then refresh
            self._wait_for_log_file(log_file)
            
            messagebox.showinfo("Training Started", 
                              f"Training started in new console window.\n\n"
                              f"Log file: {log_file}\n\n"
                              f"Dashboard will auto-refresh every {self.app.refresh_interval // 1000} seconds.")
            
        except Exception as e:
            self.app.status_label.config(text=f"Failed to start: {e}", foreground="red")
            messagebox.showerror("Error", f"Failed to start training: {e}")
    
    def stop_training(self):
        """Stop the running training process"""
        if not self.training_process or self.training_process.poll() is not None:
            self.app.status_label.config(text="No training running", foreground="gray")
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
        
        self.training_process = None
        self.update_training_status(False)
    
    def check_training_status(self):
        """Check if training is currently running"""
        is_running = self.training_process and self.training_process.poll() is None
        self.update_training_status(is_running)
        
        if self.training_process and self.training_process.poll() is not None:
            return_code = self.training_process.returncode
            self.training_output.insert(tk.END, f"\n\nTraining finished with exit code: {return_code}\n")
            self.training_process = None
    
    def update_training_status(self, is_running):
        """Update UI based on training status"""
        if is_running:
            if self.current_phase:
                phase_info = PHASE_INFO[self.current_phase]
                self.training_status_label.config(
                    text=f"Phase {self.current_phase}: {phase_info['name']}", foreground='green')
                self.training_indicator.itemconfig('indicator', fill=phase_info['color'])
            else:
                self.training_status_label.config(text="Running", foreground='green')
                self.training_indicator.itemconfig('indicator', fill='green')
            self.start_training_btn.config(state=tk.DISABLED)
            self.stop_training_btn.config(state=tk.NORMAL)
            if self.training_process:
                self.pid_label.config(text=f"PID: {self.training_process.pid}", foreground='green')
            # Disable phase buttons while training
            for btn in self.phase_buttons.values():
                btn.config(state=tk.DISABLED)
        else:
            self.training_status_label.config(text="Not Running", foreground='gray')
            self.training_indicator.itemconfig('indicator', fill='gray')
            self.pid_label.config(text="PID: N/A", foreground='gray')
            self.start_training_btn.config(state=tk.NORMAL)
            self.stop_training_btn.config(state=tk.DISABLED)
            self.current_phase = None
            # Re-enable phase buttons
            for btn in self.phase_buttons.values():
                btn.config(state=tk.NORMAL)
    
    def _wait_for_log_file(self, log_file: Path, attempts: int = 0, max_attempts: int = 15):
        """Wait for log file to be created and have content, then refresh"""
        if attempts >= max_attempts:
            self.training_output.insert(tk.END, f"[Warning] Log file wait timeout after {max_attempts}s\n")
            self.training_output.see(tk.END)
            return
        
        if log_file.exists() and log_file.stat().st_size > 0:
            # File exists and has content - refresh now
            self.training_output.insert(tk.END, f"[Training] Log file detected, refreshing...\n")
            self.training_output.see(tk.END)
            self.app.refresh_data()
        else:
            # Keep waiting
            self.root.after(1000, lambda: self._wait_for_log_file(log_file, attempts + 1, max_attempts))
    
    def periodic_status_check(self):
        """Periodically check training status"""
        self.check_training_status()
        self.root.after(2000, self.periodic_status_check)
    
    def select_log_file(self):
        """Open file dialog to select a log file"""
        log_dir = Path("outputs/logs")
        if not log_dir.exists():
            log_dir = Path(".")
        
        filename = filedialog.askopenfilename(
            title="Select Training Log File",
            initialdir=log_dir,
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        
        if filename:
            # Clear old data and switch to new log
            self.app.episode_data.clear()
            self.app.last_log_file = Path(filename)
            self.current_log_label.config(text=self.app.last_log_file.name)
            self.app.status_label.config(text=f"Log: {self.app.last_log_file.name}", foreground="blue")
            self.app.refresh_data()
