"""
Evaluation Tab - Display eval_checkpoints.py results with learning curves
"""
import tkinter as tk
from tkinter import ttk, messagebox
import json
import subprocess
import sys
from pathlib import Path
from threading import Thread

from .base_tab import BaseTab

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class EvaluationTab(BaseTab):
    """Checkpoint evaluation results tab"""
    
    def setup_ui(self):
        """Setup evaluation UI"""
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Checkpoint Evaluation Results", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="⟳ Reload Results", 
                  command=self.load_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="▶ Run Evaluation", 
                  command=self.run_evaluation).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="✕ Clear Results", 
                  command=self.clear_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="📋 Copy CSV", 
                  command=self.copy_csv).pack(side=tk.LEFT, padx=(0, 10))
        
        self.eval_status = ttk.Label(control_frame, text="Click 'Reload' to load results", 
                                    font=('Arial', 10), foreground='gray')
        self.eval_status.pack(side=tk.LEFT, padx=(10, 0))
        
        # Progress bar for evaluation
        self.eval_progress = ttk.Progressbar(control_frame, mode='determinate', length=200, maximum=100)
        self.eval_progress.pack(side=tk.RIGHT, padx=(10, 0))
        self.eval_progress.pack_forget()  # Hidden by default
        
        # Progress label showing percentage
        self.eval_progress_label = ttk.Label(control_frame, text="", font=('Arial', 9))
        self.eval_progress_label.pack(side=tk.RIGHT, padx=(5, 0))
        self.eval_progress_label.pack_forget()  # Hidden by default
        
        # Info about evaluation
        info_text = ("Evaluation runs actual environment simulations to measure learned behaviors. "
                    "Results show how hunting/evasion success rates change across training episodes.")
        ttk.Label(main_frame, text=info_text, font=('Arial', 9), 
                 foreground='gray', wraplength=800).pack(pady=(0, 10))
        
        # Create notebook for sub-tabs
        self.eval_notebook = ttk.Notebook(main_frame)
        self.eval_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary tab
        self.summary_frame = ttk.Frame(self.eval_notebook, padding="10")
        self.eval_notebook.add(self.summary_frame, text="Summary")
        self._setup_summary_tab()
        
        # Learning Curves tab
        self.curves_frame = ttk.Frame(self.eval_notebook, padding="10")
        self.eval_notebook.add(self.curves_frame, text="Learning Curves")
        self._setup_curves_tab()
        
        # Details tab
        self.details_frame = ttk.Frame(self.eval_notebook, padding="10")
        self.eval_notebook.add(self.details_frame, text="Detailed Results")
        self._setup_details_tab()
        
        # Store results
        self.eval_results = None
    
    def _setup_summary_tab(self):
        """Setup summary view"""
        # Key metrics grid
        metrics_frame = ttk.LabelFrame(self.summary_frame, text="Key Learning Metrics", padding="15")
        metrics_frame.pack(fill=tk.X, pady=(0, 15))
        
        for i in range(4):
            metrics_frame.columnconfigure(i, weight=1, uniform='col')
        
        # Row 1: Predator metrics
        self._create_summary_box(metrics_frame, "Predator Capture Rate", "pred_capture", 0, 0,
                                "Success rate when predator detects prey")
        self._create_summary_box(metrics_frame, "Capture Rate Change", "pred_capture_change", 0, 1,
                                "Change from first to last checkpoint")
        self._create_summary_box(metrics_frame, "Avg Capture Time", "pred_capture_time", 0, 2,
                                "Steps to catch prey after detection")
        self._create_summary_box(metrics_frame, "Meals per Predator", "pred_meals", 0, 3,
                                "Average successful hunts per predator")
        
        # Row 2: Prey metrics
        self._create_summary_box(metrics_frame, "Prey Escape Rate", "prey_escape", 1, 0,
                                "Success rate when prey detects predator")
        self._create_summary_box(metrics_frame, "Escape Rate Change", "prey_escape_change", 1, 1,
                                "Change from first to last checkpoint")
        self._create_summary_box(metrics_frame, "Avg Distance Gain", "prey_dist_gain", 1, 2,
                                "Distance gained after 5 steps of evasion")
        self._create_summary_box(metrics_frame, "Starvation Rate", "prey_starvation", 1, 3,
                                "Deaths from energy depletion vs total")
        
        # Progress assessment
        progress_frame = ttk.LabelFrame(self.summary_frame, text="Learning Progress Assessment", padding="15")
        progress_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.progress_label = ttk.Label(progress_frame, text="No evaluation data loaded yet.",
                                       font=('Arial', 11), wraplength=700)
        self.progress_label.pack()
        
        # Recommendation
        self.recommendation_label = ttk.Label(progress_frame, text="",
                                             font=('Arial', 10, 'italic'), foreground='blue',
                                             wraplength=700)
        self.recommendation_label.pack(pady=(10, 0))
    
    def _create_summary_box(self, parent, title, key, row, col, description=""):
        """Create a summary metric box"""
        frame = ttk.Frame(parent, padding="10")
        frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
        
        ttk.Label(frame, text=title, font=('Arial', 10, 'bold')).pack()
        value_label = ttk.Label(frame, text="--", font=('Arial', 14))
        value_label.pack()
        self.widgets[f'{key}_value'] = value_label
        
        if description:
            desc_label = ttk.Label(frame, text=description, font=('Arial', 8), 
                                  foreground='gray', wraplength=150)
            desc_label.pack(pady=(5, 0))
    
    def _setup_curves_tab(self):
        """Setup learning curves visualization"""
        if not HAS_MATPLOTLIB:
            ttk.Label(self.curves_frame, text="Matplotlib not available for charts.",
                     font=('Arial', 12), foreground='red').pack(pady=20)
            return
        
        # Control panel
        control_frame = ttk.Frame(self.curves_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Metrics to show:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.curve_vars = {}
        for name, key in [('Capture Rate', 'capture'), ('Escape Rate', 'escape'), 
                         ('Capture Time', 'time'), ('Distance Gain', 'dist')]:
            var = tk.BooleanVar(value=(key in ['capture', 'escape']))
            self.curve_vars[key] = var
            ttk.Checkbutton(control_frame, text=name, variable=var,
                           command=self._update_curves).pack(side=tk.LEFT, padx=5)
        
        # Matplotlib figure
        self.curve_fig = Figure(figsize=(10, 5), dpi=100)
        self.curve_ax = self.curve_fig.add_subplot(111)
        
        self.curve_canvas = FigureCanvasTkAgg(self.curve_fig, master=self.curves_frame)
        self.curve_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _setup_details_tab(self):
        """Setup detailed results table"""
        # Add description frame at top
        desc_frame = ttk.Frame(self.details_frame)
        desc_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create 3-column layout for descriptions
        col1 = ttk.Frame(desc_frame)
        col1.pack(side=tk.LEFT, padx=10)
        col2 = ttk.Frame(desc_frame)
        col2.pack(side=tk.LEFT, padx=10)
        col3 = ttk.Frame(desc_frame)
        col3.pack(side=tk.LEFT, padx=10)
        
        descriptions = [
            ("Capture %", "Predator success rate"),
            ("Escape %", "Prey success rate"),
            ("Capture Time", "Steps to catch prey"),
            ("Dist+1/+5", "Distance gain after 1/5 steps"),
            ("Final Prey/Pred", "Final population counts"),
            ("Meals/Pred", "Hunts per predator"),
            ("Detection Events", "Times detected each other"),
            ("Total Deaths", "Sum of all deaths"),
            ("Starvation", "Energy depletion deaths")
        ]
        
        for i, (label, desc) in enumerate(descriptions):
            col = col1 if i < 3 else (col2 if i < 6 else col3)
            txt = f"• {label} - {desc}"
            lbl = ttk.Label(col, text=txt, font=('Arial', 9), foreground='#555')
            lbl.pack(anchor='w', pady=1)
        
        # Table frame
        table_frame = ttk.Frame(self.details_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('checkpoint', 'capture_rate', 'escape_rate', 'capture_time', 
                  'dist_gain_1', 'dist_gain_5', 'final_prey', 'final_pred', 'meals_per_pred',
                  'prey_detections', 'pred_detections', 'prey_deaths', 'pred_deaths',
                  'prey_starvation', 'pred_starvation')
        self.details_tree = ttk.Treeview(table_frame, columns=columns, 
                                        show='headings', height=15)
        
        headers = ['Checkpoint', 'Capture %', 'Escape %', 'Time', 'Dist+1', 'Dist+5',
                  'Final Prey', 'Final Pred', 'Meals/Pred', 'Prey Detect', 'Pred Detect',
                  'Prey Deaths', 'Pred Deaths', 'Prey Starv', 'Pred Starv']
        widths = [90, 80, 80, 60, 70, 70, 80, 80, 80, 85, 85, 85, 85, 80, 80]
        
        for col, header, width in zip(columns, headers, widths):
            self.details_tree.heading(col, text=header)
            self.details_tree.column(col, width=width, anchor='center')
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, 
                                 command=self.details_tree.yview)
        self.details_tree.configure(yscrollcommand=scrollbar.set)
        
        self.details_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def refresh(self):
        """Refresh tab"""
        # Check if widgets exist
        if not hasattr(self, 'eval_results'):
            return
        
        # Auto-load results if not loaded
        if self.eval_results is None:
            self.load_results()
    
    def load_results(self):
        """Load evaluation results from JSON file"""
        results_path = Path(__file__).parent.parent.parent.parent / "outputs" / "eval_results" / "eval_summary.json"
        
        if not results_path.exists():
            self.eval_status.config(text="No results found - run evaluation first", foreground="orange")
            return
        
        try:
            with open(results_path, 'r') as f:
                raw_data = json.load(f)
            
            # Convert new format (results array) to old format (checkpoint_metrics dict)
            # New format: {"results": [{"checkpoint_episode": 1, "prey_escape_rate_mean": 0.9, ...}, ...]}
            # Old format: {"checkpoint_metrics": {"ep_1": {"prey_escape_rate": 0.9, ...}, ...}}
            if 'results' in raw_data:
                checkpoint_metrics = {}
                for result in raw_data['results']:
                    ep = result['checkpoint_episode']
                    # Remove _mean/_total suffixes and convert to old format
                    metrics = {}
                    for key, value in result.items():
                        if key.endswith('_mean'):
                            metrics[key[:-5]] = value  # Remove _mean
                        elif key.endswith('_total'):
                            metrics[key[:-6]] = value  # Remove _total
                        elif key not in ['episodes', 'checkpoint_episode', 'num_eval_episodes', 'steps_per_episode']:
                            metrics[key] = value
                    checkpoint_metrics[f'ep_{ep}'] = metrics
                self.eval_results = {'checkpoint_metrics': checkpoint_metrics}
            else:
                # Old format, use as-is
                self.eval_results = raw_data
            
            self._populate_summary()
            self._update_curves()
            self._populate_details()
            
            n_checkpoints = len(self.eval_results.get('checkpoint_metrics', {}))
            self.eval_status.config(text=f"Loaded {n_checkpoints} checkpoint results", foreground="green")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.eval_status.config(text=f"Error loading results: {e}", foreground="red")
    
    def _populate_summary(self):
        """Populate summary metrics"""
        if not self.eval_results:
            return
        
        ckpts = self.eval_results.get('checkpoint_metrics', {})
        if not ckpts:
            return
        
        # Get first and last checkpoint metrics
        episodes = sorted([int(k.split('_')[-1]) for k in ckpts.keys() if k.startswith('ep_')])
        if len(episodes) < 2:
            return
        
        first_ep = episodes[0]
        last_ep = episodes[-1]
        first_data = ckpts.get(f'ep_{first_ep}', {})
        last_data = ckpts.get(f'ep_{last_ep}', {})
        
        # Predator metrics
        cap_first = first_data.get('predator_capture_rate', 0) * 100
        cap_last = last_data.get('predator_capture_rate', 0) * 100
        cap_change = cap_last - cap_first
        
        self.widgets['pred_capture_value'].config(text=f"{cap_last:.1f}%")
        self.widgets['pred_capture_change_value'].config(
            text=f"{'+' if cap_change >= 0 else ''}{cap_change:.1f}%",
            foreground='green' if cap_change > 0 else ('red' if cap_change < 0 else 'black')
        )
        
        cap_time = last_data.get('predator_time_to_capture_median', 0)
        self.widgets['pred_capture_time_value'].config(text=f"{cap_time:.1f} steps")
        
        meals = last_data.get('predator_meals_per_alive', 0)
        self.widgets['pred_meals_value'].config(text=f"{meals:.2f}")
        
        # Prey metrics
        esc_first = first_data.get('prey_escape_rate', 0) * 100
        esc_last = last_data.get('prey_escape_rate', 0) * 100
        esc_change = esc_last - esc_first
        
        self.widgets['prey_escape_value'].config(text=f"{esc_last:.1f}%")
        self.widgets['prey_escape_change_value'].config(
            text=f"{'+' if esc_change >= 0 else ''}{esc_change:.1f}%",
            foreground='green' if esc_change > 0 else ('red' if esc_change < 0 else 'black')
        )
        
        dist_gain = last_data.get('prey_dist_gain_5', 0)
        self.widgets['prey_dist_gain_value'].config(text=f"{dist_gain:.2f}")
        
        starv_rate = last_data.get('prey_starvation_deaths', 0) / max(1, last_data.get('prey_deaths', 1)) * 100
        self.widgets['prey_starvation_value'].config(text=f"{starv_rate:.1f}%")
        
        # Progress assessment
        self._update_progress_assessment(cap_first, cap_last, esc_first, esc_last, cap_change, esc_change)
    
    def _update_progress_assessment(self, cap_first, cap_last, esc_first, esc_last, cap_change, esc_change):
        """Update progress assessment text"""
        assessment = []
        
        # Predator learning
        if cap_change > 20:
            assessment.append("✅ Strong predator learning: Capture rate improved significantly!")
        elif cap_change > 5:
            assessment.append("📈 Moderate predator learning: Capture rate is improving.")
        elif cap_change < -5:
            assessment.append("⚠️ Predator regression: Capture rate has decreased.")
        else:
            assessment.append("➡️ Predator: Minimal capture rate change.")
        
        # Prey learning
        if esc_change > 20:
            assessment.append("✅ Strong prey learning: Escape rate improved significantly!")
        elif esc_change > 5:
            assessment.append("📈 Moderate prey learning: Escape rate is improving.")
        elif esc_change < -5:
            assessment.append("⚠️ Prey regression: Escape rate has decreased.")
        else:
            assessment.append("➡️ Prey: Minimal escape rate change.")
        
        # Overall
        if cap_change > 10 and esc_change > 10:
            assessment.append("\n🎯 Both species are learning! Arms race dynamics emerging.")
        elif cap_change > 10 and esc_change < 0:
            assessment.append("\n⚔️ Predators dominating - prey needs more training.")
        elif cap_change < 0 and esc_change > 10:
            assessment.append("\n🛡️ Prey dominating - predators need more training.")
        
        self.progress_label.config(text="\n".join(assessment))
        
        # Recommendation
        if cap_last < 20 and esc_last < 40:
            rec = "💡 Recommendation: Continue training - both species need more episodes to develop strategies."
        elif cap_last > 50:
            rec = "💡 Recommendation: Consider adjusting reward balance if predators are too dominant."
        elif esc_last > 80:
            rec = "💡 Recommendation: Consider adjusting reward balance if prey are too successful at escaping."
        else:
            rec = "💡 Recommendation: Training appears balanced. Continue monitoring metrics."
        
        self.recommendation_label.config(text=rec)
    
    def _update_curves(self):
        """Update learning curves chart"""
        if not HAS_MATPLOTLIB or not self.eval_results:
            return
        
        self.curve_ax.clear()
        
        ckpts = self.eval_results.get('checkpoint_metrics', {})
        if not ckpts:
            return
        
        # Extract data
        episodes = []
        capture_rates = []
        escape_rates = []
        capture_times = []
        dist_gains = []
        
        for key in sorted(ckpts.keys(), key=lambda k: int(k.split('_')[-1]) if k.startswith('ep_') else 0):
            if not key.startswith('ep_'):
                continue
            ep = int(key.split('_')[-1])
            data = ckpts[key]
            
            episodes.append(ep)
            capture_rates.append(data.get('predator_capture_rate', 0) * 100)
            escape_rates.append(data.get('prey_escape_rate', 0) * 100)
            capture_times.append(data.get('predator_time_to_capture_median', 0))
            dist_gains.append(data.get('prey_dist_gain_5', 0))
        
        if not episodes:
            return
        
        # Plot selected metrics
        if self.curve_vars.get('capture', tk.BooleanVar()).get():
            self.curve_ax.plot(episodes, capture_rates, 'r-o', label='Capture Rate %', linewidth=2, markersize=6)
        
        if self.curve_vars.get('escape', tk.BooleanVar()).get():
            self.curve_ax.plot(episodes, escape_rates, 'g-s', label='Escape Rate %', linewidth=2, markersize=6)
        
        if self.curve_vars.get('time', tk.BooleanVar()).get():
            # Normalize capture time to similar scale
            max_time = max(capture_times) if capture_times else 1
            norm_times = [t / max_time * 100 for t in capture_times]
            self.curve_ax.plot(episodes, norm_times, 'b-^', label=f'Capture Time (norm)', linewidth=2, markersize=6)
        
        if self.curve_vars.get('dist', tk.BooleanVar()).get():
            # Normalize distance gain
            max_dist = max(abs(d) for d in dist_gains) if dist_gains else 1
            norm_dists = [(d / max_dist * 50) + 50 for d in dist_gains]  # Center around 50
            self.curve_ax.plot(episodes, norm_dists, 'm-d', label='Dist Gain (norm)', linewidth=2, markersize=6)
        
        self.curve_ax.set_xlabel('Checkpoint Episode', fontsize=11)
        self.curve_ax.set_ylabel('Rate / Normalized Value', fontsize=11)
        self.curve_ax.set_title('Learning Progress Over Training', fontsize=12, fontweight='bold')
        self.curve_ax.legend(loc='best', fontsize=9)
        self.curve_ax.grid(True, alpha=0.3)
        self.curve_ax.set_ylim(0, 105)
        
        self.curve_fig.tight_layout()
        self.curve_canvas.draw()
    
    def _populate_details(self):
        """Populate detailed results table"""
        # Clear existing
        for item in self.details_tree.get_children():
            self.details_tree.delete(item)
        
        if not self.eval_results:
            return
        
        ckpts = self.eval_results.get('checkpoint_metrics', {})
        
        for key in sorted(ckpts.keys(), key=lambda k: int(k.split('_')[-1]) if k.startswith('ep_') else 0):
            if not key.startswith('ep_'):
                continue
            
            ep = key.split('_')[-1]
            data = ckpts[key]
            
            self.details_tree.insert('', 'end', values=(
                f"ep {ep}",
                f"{data.get('predator_capture_rate', 0) * 100:.1f}%",
                f"{data.get('prey_escape_rate', 0) * 100:.1f}%",
                f"{data.get('predator_time_to_capture_median', 0):.1f}",
                f"{data.get('prey_dist_gain_1', 0):.2f}",
                f"{data.get('prey_dist_gain_5', 0):.2f}",
                f"{data.get('final_prey_count', 0):.0f}",
                f"{data.get('final_predator_count', 0):.0f}",
                f"{data.get('predator_meals_per_alive', 0):.2f}",
                data.get('prey_detection_events', 0),
                data.get('predator_detection_events', 0),
                data.get('prey_deaths', 0),
                data.get('predator_deaths', 0),
                data.get('prey_starvation_deaths', 0),
                data.get('predator_starvation_deaths', 0)
            ))
    
    def clear_results(self):
        """Clear existing evaluation results"""
        results_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "eval_results"
        if not results_dir.exists():
            messagebox.showinfo("No Results", "No evaluation results found.")
            return
        
        json_files = list(results_dir.glob("*.json"))
        if not json_files:
            messagebox.showinfo("No Results", "No evaluation results found.")
            return
        
        if messagebox.askyesno("Clear Results", 
                              f"Delete {len(json_files)} evaluation result files?\n\nThis cannot be undone."):
            try:
                for f in json_files:
                    f.unlink()
                self.eval_results = None
                self.eval_status.config(text="Results cleared", foreground="gray")
                # Clear displays
                for widget_key in list(self.widgets.keys()):
                    if '_value' in widget_key:
                        self.widgets[widget_key].config(text="--")
                if hasattr(self, 'progress_label'):
                    self.progress_label.config(text="No evaluation data loaded yet.")
                messagebox.showinfo("Results Cleared", f"Deleted {len(json_files)} result files.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear results: {e}")
    
    def run_evaluation(self):
        """Run eval_checkpoints.py in separate console window"""
        # Check for existing results and offer to clear them
        results_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "eval_results"
        existing_results = list(results_dir.glob("*.json")) if results_dir.exists() else []
        
        if existing_results:
            response = messagebox.askyesnocancel(
                "Existing Results Found",
                f"Found {len(existing_results)} existing evaluation results.\n\n"
                "Yes = Clear old results and start fresh\n"
                "No = Keep old results and add new ones\n"
                "Cancel = Don't run evaluation"
            )
            if response is None:  # Cancel
                return
            elif response:  # Yes - clear old results
                for f in existing_results:
                    f.unlink()
                self.eval_status.config(text="Old results cleared", foreground="gray")
        
        if messagebox.askyesno("Run Evaluation", 
                              "This will run checkpoint evaluation in a separate window.\n\nContinue?"):
            self.eval_status.config(text="Running evaluation in separate window...", foreground="orange")
            
            # Show progress bar and label
            self.eval_progress_label.config(text="0%")
            self.eval_progress_label.pack(side=tk.RIGHT, padx=(5, 0))
            self.eval_progress['value'] = 0
            self.eval_progress.pack(side=tk.RIGHT, padx=(10, 0))
            self.root.update()
            
            def run_eval():
                import re
                import os
                import time
                try:
                    script_path = Path(__file__).parent.parent.parent / "eval_checkpoints.py"
                    project_root = Path(__file__).parent.parent.parent.parent
                    results_dir = project_root / "outputs" / "eval_results"
                    
                    # Count checkpoints to evaluate - support both old and new formats
                    ckpt_dir = project_root / "outputs" / "checkpoints"
                    
                    # Old format: model_A_ppo_ep*.pth, model_B_ppo_ep*.pth
                    # New format: {prefix}_ep*_model_A.pth, {prefix}_ep*_model_B.pth
                    pred_ckpts = list(ckpt_dir.glob("model_B_ppo_ep*.pth"))
                    pred_ckpts.extend(ckpt_dir.glob("*_ep*_model_B.pth"))
                    prey_ckpts = list(ckpt_dir.glob("model_A_ppo_ep*.pth"))
                    prey_ckpts.extend(ckpt_dir.glob("*_ep*_model_A.pth"))
                    
                    # Find episodes that have both models
                    pred_eps = set()
                    prey_eps = set()
                    for p in pred_ckpts:
                        match = re.search(r'ep(\d+)', p.stem)
                        if match:
                            pred_eps.add(int(match.group(1)))
                    for p in prey_ckpts:
                        match = re.search(r'ep(\d+)', p.stem)
                        if match:
                            prey_eps.add(int(match.group(1)))
                    common_eps = sorted(pred_eps & prey_eps)
                    
                    if not common_eps:
                        self.root.after(0, lambda: self._eval_complete(False, "No checkpoint pairs found"))
                        return
                    
                    # Use all available episode checkpoints
                    episodes_to_eval = common_eps
                    total_checkpoints = len(episodes_to_eval)
                    
                    print(f"Will evaluate {total_checkpoints} checkpoints: {episodes_to_eval}")
                    
                    # Get Python from ROCm environment first (for GPU), fall back to .venv
                    venv_python = Path(".venv_rocm/Scripts/python.exe").absolute()
                    if not venv_python.exists():
                        venv_python = Path(".venv/Scripts/python.exe").absolute()
                        if not venv_python.exists():
                            venv_python = sys.executable
                    
                    # Build command with all episode numbers
                    cmd = [str(venv_python), str(script_path), "--episodes"] + [str(ep) for ep in episodes_to_eval]
                    
                    # Start process in separate console window
                    if sys.platform == 'win32':
                        process = subprocess.Popen(
                            cmd,
                            cwd=str(project_root),
                            creationflags=subprocess.CREATE_NEW_CONSOLE
                        )
                    else:
                        process = subprocess.Popen(
                            ['x-terminal-emulator', '-e'] + cmd,
                            cwd=str(project_root)
                        )
                    
                    # Poll for completed checkpoints by counting result files
                    completed_last = 0
                    while process.poll() is None:
                        time.sleep(0.5)
                        # Count eval_ep*.json files (excluding eval_summary.json)
                        completed = len(list(results_dir.glob("eval_ep*.json")))
                        if completed > completed_last:
                            completed_last = completed
                            pct = int(completed / total_checkpoints * 100)
                            self.root.after(0, lambda c=completed, t=total_checkpoints, p=pct: 
                                          self._update_progress(p, c, t))
                    
                    # Final update to 100% if successful
                    if process.returncode == 0:
                        final_completed = len(list(results_dir.glob("eval_ep*.json")))
                        self.root.after(0, lambda: self._update_progress(100, final_completed, total_checkpoints))
                        time.sleep(0.2)
                        self.root.after(0, lambda: self._eval_complete(True))
                    else:
                        self.root.after(0, lambda: self._eval_complete(False, "Process exited with error"))
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    self.root.after(0, lambda: self._eval_complete(False, "Evaluation timed out"))
                except Exception as e:
                    self.root.after(0, lambda: self._eval_complete(False, str(e)))
            
            Thread(target=run_eval, daemon=True).start()
    
    def _update_progress(self, percent, completed=None, total=None):
        """Update progress bar from main thread"""
        self.eval_progress['value'] = percent
        if completed is not None and total is not None:
            self.eval_progress_label.config(text=f"{completed}/{total} ({percent}%)")
        else:
            self.eval_progress_label.config(text=f"{percent}%")
    
    def _eval_complete(self, success, error=None):
        """Handle evaluation completion"""
        # Hide progress bar and label
        self.eval_progress['value'] = 0
        self.eval_progress.pack_forget()
        self.eval_progress_label.pack_forget()
        
        if success:
            self.eval_status.config(text="Evaluation complete! Reloading results...", foreground="green")
            self.load_results()
        else:
            self.eval_status.config(text=f"Evaluation failed: {error}", foreground="red")
            messagebox.showerror("Evaluation Error", f"Evaluation failed:\n{error}")
    
    def copy_csv(self):
        """Copy evaluation results to clipboard in CSV format"""
        import io
        
        if not self.eval_results:
            self.app.status_label.config(text="No evaluation data to copy. Load results first.", foreground="red")
            self.root.after(3000, lambda: self.app.status_label.config(text="Ready", foreground="gray"))
            return
        
        csv = io.StringIO()
        ckpts = self.eval_results.get('checkpoint_metrics', {})
        
        csv.write("# Checkpoint Evaluation Results\n")
        csv.write("episode,predator_capture_rate,prey_escape_rate,time_to_capture_median,")
        csv.write("prey_dist_gain_1,prey_dist_gain_5,prey_deaths,predator_deaths\n")
        
        for key in sorted(ckpts.keys(), key=lambda k: int(k.split('_')[-1]) if k.startswith('ep_') else 0):
            if not key.startswith('ep_'):
                continue
            ep = key.split('_')[-1]
            data = ckpts[key]
            csv.write(f"{ep},")
            csv.write(f"{data.get('predator_capture_rate', 0):.4f},")
            csv.write(f"{data.get('prey_escape_rate', 0):.4f},")
            csv.write(f"{data.get('predator_time_to_capture_median', 0):.2f},")
            csv.write(f"{data.get('prey_dist_gain_1', 0):.2f},")
            csv.write(f"{data.get('prey_dist_gain_5', 0):.2f},")
            csv.write(f"{data.get('prey_deaths', 0)},")
            csv.write(f"{data.get('predator_deaths', 0)}\n")
        
        self.root.clipboard_clear()
        self.root.clipboard_append(csv.getvalue())
        self.app.status_label.config(text=f"✓ Copied evaluation results ({len(ckpts)} checkpoints)", foreground="green")
        self.root.after(3000, lambda: self.app.status_label.config(text="Ready", foreground="gray"))
