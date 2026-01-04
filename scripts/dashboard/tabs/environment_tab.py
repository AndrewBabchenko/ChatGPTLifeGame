"""
Environment Tab - Monitor environment and population statistics
"""
import tkinter as tk
from tkinter import ttk

from .base_tab import BaseTab


class EnvironmentTab(BaseTab):
    """Environment statistics monitoring tab"""
    
    def setup_ui(self):
        """Setup environment stats UI"""
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Environment & Population Statistics", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # Copy button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(button_frame, text="📡 Data source: Training log", font=('Arial', 9), foreground='blue').pack(side=tk.LEFT)
        ttk.Button(button_frame, text="📋 Copy CSV", command=self.copy_csv).pack(side=tk.RIGHT)
        
        # Population frame
        pop_frame = ttk.LabelFrame(main_frame, text="Population Overview", padding="15")
        pop_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Create 4-column grid for population stats
        for i in range(4):
            pop_frame.columnconfigure(i, weight=1, uniform='col')
        
        # Population stats
        self._create_stat_box(pop_frame, "Total Animals", "total_animals", 0, 0)
        self._create_stat_box(pop_frame, "Predators (A)", "predator_count", 0, 1)
        self._create_stat_box(pop_frame, "Prey (B)", "prey_count", 0, 2)
        self._create_stat_box(pop_frame, "Grass Patches", "grass_count", 0, 3)
        
        # Energy and survival stats
        energy_frame = ttk.LabelFrame(main_frame, text="Energy & Survival", padding="15")
        energy_frame.pack(fill=tk.X, pady=(0, 15))
        
        for i in range(4):
            energy_frame.columnconfigure(i, weight=1, uniform='col')
        
        self._create_stat_box(energy_frame, "Avg Predator Energy", "avg_predator_energy", 0, 0)
        self._create_stat_box(energy_frame, "Avg Prey Energy", "avg_prey_energy", 0, 1)
        self._create_stat_box(energy_frame, "Kills This Episode", "kills", 0, 2)
        self._create_stat_box(energy_frame, "Deaths This Episode", "deaths", 0, 3)
        
        # Rewards frame
        rewards_frame = ttk.LabelFrame(main_frame, text="Rewards (Current Episode)", padding="15")
        rewards_frame.pack(fill=tk.X, pady=(0, 15))
        
        for i in range(3):
            rewards_frame.columnconfigure(i, weight=1, uniform='col')
        
        self._create_stat_box(rewards_frame, "Predator Reward", "predator_reward", 0, 0, is_reward=True)
        self._create_stat_box(rewards_frame, "Prey Reward", "prey_reward", 0, 1, is_reward=True)
        self._create_stat_box(rewards_frame, "Combined Reward", "total_reward", 0, 2, is_reward=True)
        
        # Detailed population history
        history_frame = ttk.LabelFrame(main_frame, text="Episode History (Last 20)", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('episode', 'predators', 'prey', 'kills', 'deaths', 'reward_a', 'reward_b')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=8)
        
        headers = ['Episode', 'Predators', 'Prey', 'Kills', 'Deaths', 'Reward A', 'Reward B']
        widths = [80, 80, 80, 80, 80, 100, 100]
        
        for col, header, width in zip(columns, headers, widths):
            self.history_tree.heading(col, text=header)
            self.history_tree.column(col, width=width, anchor='center')
        
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_stat_box(self, parent, title, key, row, col, is_reward=False):
        """Create a statistics display box"""
        frame = ttk.Frame(parent, padding="10")
        frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
        
        ttk.Label(frame, text=title, font=('Arial', 10, 'bold')).pack()
        
        # Current value (large, color-coded)
        value_label = ttk.Label(frame, text="--", font=('Arial', 16, 'bold'))
        value_label.pack()
        self.widgets[f'{key}_value'] = value_label
        
        if is_reward:
            # Trend indicator
            trend_label = ttk.Label(frame, text="", font=('Arial', 10))
            trend_label.pack()
            self.widgets[f'{key}_trend'] = trend_label
        
        # Last 5 values (smaller)
        history_label = ttk.Label(frame, text="", font=('Arial', 8))
        history_label.pack()
        self.widgets[f'{key}_history'] = history_label
    
    def refresh(self):
        """Refresh environment display"""
        # Check if widgets exist first
        if not hasattr(self, 'history_tree'):
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
            if any(ep_data.get(key) is not None for key in ['prey_count', 'predator_count', 'reward_a', 'reward_b']):
                latest_ep = ep
                latest = ep_data
                break
        
        if latest is None:
            return
        
        # Debug: print what data we have
        print(f"[Environment] Refreshing with episode {latest_ep}, total_animals={latest.get('total_animals')}")
        
        # Update population stats
        self._update_stat('total_animals', latest.get('total_animals'))
        self._update_stat('predator_count', latest.get('predator_count'))
        self._update_stat('prey_count', latest.get('prey_count'))
        self._update_stat('grass_count', latest.get('grass_count'))
        
        # Update energy stats
        self._update_stat('avg_predator_energy', latest.get('avg_predator_energy'), is_float=True)
        self._update_stat('avg_prey_energy', latest.get('avg_prey_energy'), is_float=True)
        self._update_stat('kills', latest.get('kills'))
        self._update_stat('deaths', latest.get('deaths'))
        
        # Update rewards with trends
        self._update_reward_stat('predator_reward', 'reward_a')
        self._update_reward_stat('prey_reward', 'reward_b')
        self._update_combined_reward()
        
        # Update history table
        self._update_history_table()
    
    def _update_stat(self, key, value, is_float=False):
        """Update a stat display"""
        label = self.widgets.get(f'{key}_value')
        history_label = self.widgets.get(f'{key}_history')
        
        if label is None:
            print(f"[Environment] Warning: Widget {key}_value not found")
            return
        
        if value is None:
            label.config(text="--", foreground='gray')
            if history_label:
                history_label.config(text="")
        elif is_float:
            label.config(text=f"{value:.2f}", foreground='blue')
            # Show last 5 values
            if history_label:
                self._update_history(key, history_label, is_float=True)
        else:
            label.config(text=str(value), foreground='blue')
            # Show last 5 values
            if history_label:
                self._update_history(key, history_label, is_float=False)
    
    def _update_history(self, key, history_label, is_float=False):
        """Update history display with last 5 values"""
        # Map widget key to data key
        key_map = {
            'total_animals': 'total_animals',
            'predator_count': 'predator_count',
            'prey_count': 'prey_count',
            'grass_count': 'grass_count',
            'avg_predator_energy': 'avg_predator_energy',
            'avg_prey_energy': 'avg_prey_energy',
            'kills': 'kills',
            'deaths': 'deaths'
        }
        
        data_key = key_map.get(key, key)
        history_values = []
        
        for ep in sorted(self.app.episode_data.keys(), reverse=True):
            val = self.app.episode_data[ep].get(data_key)
            if val is not None:
                history_values.append(val)
            if len(history_values) >= 6:  # Current + 5 previous
                break
        
        # Skip current value and show last 5
        if len(history_values) > 1:
            history_text = "Last 5: "
            for val in history_values[1:6]:
                if is_float:
                    history_text += f"{val:.1f}  "
                else:
                    history_text += f"{int(val)}  "
            history_label.config(text=history_text.strip(), foreground='gray')
        else:
            history_label.config(text="")
    
    def _update_reward_stat(self, widget_key, data_key):
        """Update a reward stat with trend"""
        values = [ep.get(data_key) for ep in self.app.episode_data.values() if ep.get(data_key) is not None]
        
        value_label = self.widgets.get(f'{widget_key}_value')
        trend_label = self.widgets.get(f'{widget_key}_trend')
        history_label = self.widgets.get(f'{widget_key}_history')
        
        if not values:
            if value_label:
                value_label.config(text="--", foreground='gray')
            if history_label:
                history_label.config(text="")
            return
        
        current = values[-1]
        if value_label:
            value_label.config(text=f"{current:.1f}")
            # Color based on sign
            if current > 0:
                value_label.config(foreground='green')
            elif current < 0:
                value_label.config(foreground='red')
            else:
                value_label.config(foreground='black')
        
        if trend_label:
            trend = self.get_trend_arrow(values)
            trend_label.config(text=trend)
        
        # Show last 5 values
        if history_label and len(values) > 1:
            history_text = "Last 5: "
            for val in values[-6:-1][::-1]:  # Last 5 excluding current
                history_text += f"{val:.1f}  "
            history_label.config(text=history_text.strip(), foreground='gray')
    
    def _update_combined_reward(self):
        """Update combined reward display"""
        values_a = [ep.get('reward_a') for ep in self.app.episode_data.values() if ep.get('reward_a') is not None]
        values_b = [ep.get('reward_b') for ep in self.app.episode_data.values() if ep.get('reward_b') is not None]
        
        value_label = self.widgets.get('total_reward_value')
        trend_label = self.widgets.get('total_reward_trend')
        history_label = self.widgets.get('total_reward_history')
        
        if not values_a or not values_b:
            if value_label:
                value_label.config(text="--", foreground='gray')
            if history_label:
                history_label.config(text="")
            return
        
        # Calculate combined (average of both)
        combined = [(a + b) / 2 for a, b in zip(values_a, values_b)]
        current = combined[-1]
        
        if value_label:
            value_label.config(text=f"{current:.1f}")
            if current > 0:
                value_label.config(foreground='green')
            elif current < 0:
                value_label.config(foreground='red')
            else:
                value_label.config(foreground='black')
        
        if trend_label:
            trend = self.get_trend_arrow(combined)
            trend_label.config(text=trend)
        
        # Show last 5 values
        if history_label and len(combined) > 1:
            history_text = "Last 5: "
            for val in combined[-6:-1][::-1]:  # Last 5 excluding current
                history_text += f"{val:.1f}  "
            history_label.config(text=history_text.strip(), foreground='gray')
    
    def _update_history_table(self):
        """Update episode history table"""
        # Clear existing
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # Get last 20 episodes
        episodes = sorted(self.app.episode_data.keys())[-20:]
        
        for ep in episodes:
            data = self.app.episode_data[ep]
            self.history_tree.insert('', 'end', values=(
                ep,
                data.get('predator_count', '--'),
                data.get('prey_count', '--'),
                data.get('kills', '--'),
                data.get('deaths', '--'),
                f"{data.get('reward_a', 0):.3f}" if data.get('reward_a') is not None else '--',
                f"{data.get('reward_b', 0):.3f}" if data.get('reward_b') is not None else '--'
            ))
    
    def copy_csv(self):
        """Copy environment metrics to clipboard in CSV format"""
        import io
        csv = io.StringIO()
        
        csv.write("# Environment & Reward Metrics\n")
        csv.write("episode,reward_prey,reward_predator,final_prey,final_predator,meals,starvation_deaths\n")
        
        for i, ep in enumerate(self.app.episodes):
            ra = self.app.metrics.get('reward_a', [])[i] if i < len(self.app.metrics.get('reward_a', [])) else ''
            rb = self.app.metrics.get('reward_b', [])[i] if i < len(self.app.metrics.get('reward_b', [])) else ''
            prey = self.app.metrics.get('prey_final', [])[i] if i < len(self.app.metrics.get('prey_final', [])) else ''
            pred = self.app.metrics.get('pred_final', [])[i] if i < len(self.app.metrics.get('pred_final', [])) else ''
            meals = self.app.metrics.get('meals', [])[i] if i < len(self.app.metrics.get('meals', [])) else ''
            starv = self.app.metrics.get('starvation_deaths', [])[i] if i < len(self.app.metrics.get('starvation_deaths', [])) else ''
            csv.write(f"{ep},{ra},{rb},{prey},{pred},{meals},{starv}\n")
        
        self.root.clipboard_clear()
        self.root.clipboard_append(csv.getvalue())
        self.app.status_label.config(text=f"✓ Copied environment metrics ({len(self.app.episodes)} episodes)", foreground="green")
        self.root.after(3000, lambda: self.app.status_label.config(text="Ready", foreground="gray"))
    
    def _clear_display(self):
        """Clear all display values when no data is available"""
        # Clear all stat boxes to "--"
        for key in ['total_animals', 'predator_count', 'prey_count', 'grass_count',
                    'avg_predator_energy', 'avg_prey_energy', 'kills', 'deaths',
                    'predator_reward', 'prey_reward', 'combined_reward']:
            value_label = self.widgets.get(f'{key}_value')
            history_label = self.widgets.get(f'{key}_history')
            if value_label:
                value_label.config(text="--", foreground='gray')
            if history_label:
                history_label.config(text="")
        
        # Clear history table
        if hasattr(self, 'history_tree'):
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
