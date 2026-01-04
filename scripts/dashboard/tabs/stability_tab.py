"""
PPO Stability Tab - Monitor PPO training metrics for stability (separate per model)
"""
import tkinter as tk
from tkinter import ttk

from .base_tab import BaseTab


class StabilityTab(BaseTab):
    """PPO stability monitoring tab with separate Prey/Predator metrics"""
    
    def setup_ui(self):
        """Setup stability monitoring UI"""
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="PPO Training Stability Monitor", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Copy button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(button_frame, text="📡 Data source: Training log", font=('Arial', 9), foreground='blue').pack(side=tk.LEFT)
        ttk.Button(button_frame, text="📋 Copy CSV", command=self.copy_csv).pack(side=tk.RIGHT)
        
        # Shared metrics frame (KL, Clip - these are from PPO update, typically combined)
        shared_frame = ttk.LabelFrame(main_frame, text="PPO Update Metrics (Combined)", padding="10")
        shared_frame.pack(fill=tk.X, pady=(0, 10))
        
        for i in range(4):
            shared_frame.columnconfigure(i, weight=1, uniform='col')
        
        self._create_indicator(shared_frame, "KL Divergence", "kl_div", 0, 0,
                              "Target ~0.01, >0.03 = unstable")
        self._create_indicator(shared_frame, "Clip Fraction", "clip_fraction", 0, 1,
                              "Should stay < 0.2")
        self._create_indicator(shared_frame, "Experiences (Prey)", "exp_prey", 0, 2,
                              "Prey training samples")
        self._create_indicator(shared_frame, "Experiences (Pred)", "exp_pred", 0, 3,
                              "Predator training samples")
        
        # Two-column frame for Prey and Predator
        models_frame = ttk.Frame(main_frame)
        models_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        models_frame.columnconfigure(0, weight=1)
        models_frame.columnconfigure(1, weight=1)
        
        # ===== PREY (Model A) =====
        prey_frame = ttk.LabelFrame(models_frame, text="🐰 Prey (Model A)", padding="10")
        prey_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        for i in range(3):
            prey_frame.columnconfigure(i, weight=1, uniform='col')
        
        self._create_indicator(prey_frame, "Policy Loss", "policy_loss_prey", 0, 0,
                              "Watch for wild swings")
        self._create_indicator(prey_frame, "Value Loss", "value_loss_prey", 0, 1,
                              "Gradual decrease good")
        self._create_indicator(prey_frame, "Entropy", "entropy_prey", 0, 2,
                              "Higher = exploration")
        
        # Prey metrics table
        prey_table_frame = ttk.Frame(prey_frame)
        prey_table_frame.grid(row=1, column=0, columnspan=3, sticky='nsew', pady=(10, 0))
        
        self.prey_tree = self._create_metrics_tree(prey_table_frame)
        
        # ===== PREDATOR (Model B) =====
        pred_frame = ttk.LabelFrame(models_frame, text="🦁 Predator (Model B)", padding="10")
        pred_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        
        for i in range(3):
            pred_frame.columnconfigure(i, weight=1, uniform='col')
        
        self._create_indicator(pred_frame, "Policy Loss", "policy_loss_pred", 0, 0,
                              "Watch for wild swings")
        self._create_indicator(pred_frame, "Value Loss", "value_loss_pred", 0, 1,
                              "Gradual decrease good")
        self._create_indicator(pred_frame, "Entropy", "entropy_pred", 0, 2,
                              "Higher = exploration")
        
        # Predator metrics table
        pred_table_frame = ttk.Frame(pred_frame)
        pred_table_frame.grid(row=1, column=0, columnspan=3, sticky='nsew', pady=(10, 0))
        
        self.pred_tree = self._create_metrics_tree(pred_table_frame)
        
        # Health summary
        summary_frame = ttk.LabelFrame(main_frame, text="Training Health Summary", padding="10")
        summary_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.health_label = ttk.Label(summary_frame, text="No data yet - waiting for metrics...",
                                     font=('Arial', 11), foreground='gray', wraplength=900)
        self.health_label.pack()
    
    def _create_metrics_tree(self, parent):
        """Create a compact metrics treeview"""
        columns = ('metric', 'current', 'mean', 'trend')
        tree = ttk.Treeview(parent, columns=columns, show='headings', height=4)
        
        col_widths = {'metric': 100, 'current': 80, 'mean': 80, 'trend': 50}
        for col in columns:
            tree.heading(col, text=col.title())
            tree.column(col, width=col_widths[col], anchor='center')
        
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        return tree
    
    def _create_indicator(self, parent, title, key, row, col, tooltip):
        """Create a stability indicator widget"""
        frame = ttk.Frame(parent, padding="5")
        frame.grid(row=row, column=col, padx=3, pady=3, sticky='nsew')
        
        ttk.Label(frame, text=title, font=('Arial', 10, 'bold')).pack()
        
        # Current value (large, color-coded)
        value_label = ttk.Label(frame, text="--", font=('Arial', 12, 'bold'))
        value_label.pack()
        self.widgets[f'{key}_value'] = value_label
        
        # Last 5 values (smaller, color-coded)
        history_label = ttk.Label(frame, text="", font=('Arial', 8))
        history_label.pack()
        self.widgets[f'{key}_history'] = history_label
        
        ttk.Label(frame, text=tooltip, font=('Arial', 8), foreground='gray', 
                 wraplength=120).pack()
    
    def copy_csv(self):
        """Copy stability metrics to clipboard in CSV format (per-model)"""
        import io
        csv = io.StringIO()
        
        csv.write("# PPO Stability Metrics (Per Model)\n")
        csv.write("episode,kl_divergence,clip_fraction,")
        csv.write("policy_loss_prey,value_loss_prey,entropy_prey,")
        csv.write("policy_loss_pred,value_loss_pred,entropy_pred\n")
        
        for ep, data in sorted(self.app.episode_data.items()):
            csv.write(f"{ep},")
            csv.write(f"{data.get('kl_divergence', '')},")
            csv.write(f"{data.get('clip_fraction', '')},")
            csv.write(f"{data.get('policy_loss_prey', '')},")
            csv.write(f"{data.get('value_loss_prey', '')},")
            csv.write(f"{data.get('entropy_prey', '')},")
            csv.write(f"{data.get('policy_loss_pred', '')},")
            csv.write(f"{data.get('value_loss_pred', '')},")
            csv.write(f"{data.get('entropy_pred', '')}\n")
        
        self.root.clipboard_clear()
        self.root.clipboard_append(csv.getvalue())
        n_eps = len(self.app.episode_data)
        self.app.status_label.config(text=f"✓ Copied stability metrics ({n_eps} episodes)", foreground="green")
        self.root.after(3000, lambda: self.app.status_label.config(text="Ready", foreground="gray"))

    def refresh(self):
        """Refresh stability metrics display"""
        # Check if widgets exist first
        if not hasattr(self, 'prey_tree') or not hasattr(self, 'pred_tree'):
            return
        
        if not self.app.episode_data:
            self._clear_display()
            return
        
        # Get latest episode with actual data (not just max episode number)
        latest_ep = None
        latest = None
        for ep in sorted(self.app.episode_data.keys(), reverse=True):
            ep_data = self.app.episode_data[ep]
            # Check if this episode has any meaningful data
            if any(ep_data.get(key) is not None for key in ['kl_divergence', 'policy_loss_prey', 'policy_loss_pred']):
                latest_ep = ep
                latest = ep_data
                break
        
        if latest is None:
            return
        
        # Debug: print what data we have
        print(f"[Stability] Refreshing with episode {latest_ep}, keys: {list(latest.keys())[:5]}...")
        
        # Update shared indicators
        self._update_indicator('kl_div', latest.get('kl_divergence'),
                              good_range=(0, 0.015), warn_range=(0.015, 0.03))
        self._update_indicator('clip_fraction', latest.get('clip_fraction'), 
                              good_range=(0, 0.15), warn_range=(0.15, 0.25))
        self._update_indicator('exp_prey', latest.get('exp_prey'),
                              good_range=(0, 999999), warn_range=(0, 999999), is_neutral=True, as_int=True)
        self._update_indicator('exp_pred', latest.get('exp_pred'),
                              good_range=(0, 999999), warn_range=(0, 999999), is_neutral=True, as_int=True)
        
        # Update Prey indicators
        self._update_indicator('policy_loss_prey', latest.get('policy_loss_prey'),
                              good_range=(-0.1, 0.1), warn_range=(-0.5, 0.5))
        self._update_indicator('value_loss_prey', latest.get('value_loss_prey'),
                              good_range=(0, 5), warn_range=(5, 15))
        self._update_indicator('entropy_prey', latest.get('entropy_prey'),
                              good_range=(0.5, 4), warn_range=(0.2, 0.5))
        
        # Update Predator indicators
        self._update_indicator('policy_loss_pred', latest.get('policy_loss_pred'),
                              good_range=(-0.1, 0.1), warn_range=(-0.5, 0.5))
        self._update_indicator('value_loss_pred', latest.get('value_loss_pred'),
                              good_range=(0, 5), warn_range=(5, 15))
        self._update_indicator('entropy_pred', latest.get('entropy_pred'),
                              good_range=(0.5, 4), warn_range=(0.2, 0.5))
        
        # Update per-model tables
        self._update_model_table(self.prey_tree, 'prey')
        self._update_model_table(self.pred_tree, 'pred')
        
        # Update health summary
        self._update_health_summary(latest)
    
    def _update_indicator(self, key, value, good_range, warn_range, is_neutral=False, as_int=False):
        """Update an indicator with value and status color"""
        value_label = self.widgets.get(f'{key}_value')
        history_label = self.widgets.get(f'{key}_history')
        
        if value_label is None:
            return
        
        if value is None:
            value_label.config(text="--", foreground='gray')
            if history_label:
                history_label.config(text="")
            return
        
        # Determine color for value
        def get_color(val):
            if is_neutral:
                return 'blue'
            elif good_range[0] <= val <= good_range[1]:
                return 'green'
            elif warn_range[0] <= val <= warn_range[1]:
                return 'orange'
            else:
                return 'red'
        
        # Format and color current value
        if as_int:
            value_label.config(text=f"{int(value):,}", foreground=get_color(value))
        elif abs(value) < 0.001:
            value_label.config(text=f"{value:.6f}", foreground=get_color(value))
        else:
            value_label.config(text=f"{value:.4f}", foreground=get_color(value))
        
        # Get last 5 values for this metric
        if history_label:
            # Extract metric name from key (e.g., 'kl_div' -> 'kl_divergence')
            metric_map = {
                'kl_div': 'kl_divergence',
                'clip_fraction': 'clip_fraction',
                'exp_prey': 'exp_prey',
                'exp_pred': 'exp_pred',
                'policy_loss_prey': 'policy_loss_prey',
                'policy_loss_pred': 'policy_loss_pred',
                'value_loss_prey': 'value_loss_prey',
                'value_loss_pred': 'value_loss_pred',
                'entropy_prey': 'entropy_prey',
                'entropy_pred': 'entropy_pred'
            }
            
            metric_key = metric_map.get(key, key)
            history_values = []
            
            for ep in sorted(self.app.episode_data.keys(), reverse=True):
                val = self.app.episode_data[ep].get(metric_key)
                if val is not None:
                    history_values.append(val)
                if len(history_values) >= 6:  # Current + 5 previous
                    break
            
            # Skip the current value (first one) and take next 5
            if len(history_values) > 1:
                history_text = "Last 5: "
                for val in history_values[1:6]:
                    if as_int:
                        history_text += f"{int(val):,}  "
                    elif abs(val) < 0.001:
                        history_text += f"{val:.4f}  "
                    else:
                        history_text += f"{val:.3f}  "
                history_label.config(text=history_text.strip(), foreground=get_color(history_values[1]))
            else:
                history_label.config(text="")
    
    def _update_model_table(self, tree, model_suffix):
        """Update a model's metrics treeview"""
        # Clear existing items
        for item in tree.get_children():
            tree.delete(item)
        
        if not self.app.episode_data:
            return
        
        # Metrics for this model
        metrics = [
            (f'policy_loss_{model_suffix}', 'Policy Loss'),
            (f'value_loss_{model_suffix}', 'Value Loss'),
            (f'entropy_{model_suffix}', 'Entropy'),
        ]
        
        for metric_key, display_name in metrics:
            values = [ep.get(metric_key) for ep in self.app.episode_data.values() 
                     if ep.get(metric_key) is not None]
            if not values:
                continue
            
            current = values[-1]
            mean = sum(values) / len(values)
            trend = self.get_trend_arrow(values)
            
            tree.insert('', 'end', values=(
                display_name,
                f"{current:.4f}",
                f"{mean:.4f}",
                trend
            ))
    
    def _update_health_summary(self, latest):
        """Update the training health summary (per-model)"""
        issues = []
        
        # Shared issues
        clip = latest.get('clip_fraction')
        if clip is not None and clip > 0.2:
            issues.append(f"⚠ High clip fraction ({clip:.3f}) - consider reducing learning rate")
        
        kl = latest.get('kl_divergence')
        if kl is not None and kl > 0.025:
            issues.append(f"⚠ High KL divergence ({kl:.4f}) - policy updating too fast")
        
        # Prey-specific
        entropy_prey = latest.get('entropy_prey')
        if entropy_prey is not None and entropy_prey < 0.5:
            issues.append(f"🐰 Prey entropy low ({entropy_prey:.3f}) - may be converging prematurely")
        
        vloss_prey = latest.get('value_loss_prey')
        if vloss_prey is not None and vloss_prey > 15:
            issues.append(f"🐰 Prey value loss high ({vloss_prey:.2f}) - critic struggling")
        
        # Predator-specific
        entropy_pred = latest.get('entropy_pred')
        if entropy_pred is not None and entropy_pred < 0.5:
            issues.append(f"🦁 Predator entropy low ({entropy_pred:.3f}) - may be converging prematurely")
        
        vloss_pred = latest.get('value_loss_pred')
        if vloss_pred is not None and vloss_pred > 15:
            issues.append(f"🦁 Predator value loss high ({vloss_pred:.2f}) - critic struggling")
        
        if issues:
            self.health_label.config(
                text="Issues detected:\n" + "\n".join(issues),
                foreground='orange'
            )
        else:
            self.health_label.config(
                text="✓ Both models training stably",
                foreground='green'
            )
    
    def _clear_display(self):
        """Clear all display values when no data is available"""
        # Clear all indicators to "--"
        for key in ['kl_div', 'clip_fraction', 'exp_prey', 'exp_pred',
                    'policy_loss_prey', 'value_loss_prey', 'entropy_prey',
                    'policy_loss_pred', 'value_loss_pred', 'entropy_pred']:
            value_label = self.widgets.get(f'{key}_value')
            history_label = self.widgets.get(f'{key}_history')
            if value_label:
                value_label.config(text="--", foreground='gray')
            if history_label:
                history_label.config(text="")
        
        # Clear tables
        if hasattr(self, 'prey_tree'):
            for item in self.prey_tree.get_children():
                self.prey_tree.delete(item)
        if hasattr(self, 'pred_tree'):
            for item in self.pred_tree.get_children():
                self.pred_tree.delete(item)
        
        # Clear health summary
        if hasattr(self, 'health_label'):
            self.health_label.config(text="No data - waiting for training...", foreground='gray')
