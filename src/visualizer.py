"""
Optimized game field visualizer for the Life Game simulation
Fast rendering with fixed layout
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import numpy as np
from typing import List, Dict
import sys
sys.path.insert(0, 'src')
from animal import Animal


class GameFieldVisualizer:
    """Optimized game field display for visualization with statistics and controls"""
    
    def __init__(self, config):
        self.config = config
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.suptitle('Life Game - Predator-Prey Ecosystem', 
                         fontsize=14, fontweight='bold')
        
        # Create tight grid layout - 3 rows, 3 columns
        gs = self.fig.add_gridspec(3, 3, hspace=0.25, wspace=0.3, 
                                  left=0.05, right=0.95, top=0.92, bottom=0.08)
        
        # Main game field (left side) - spans all 3 rows, 2 columns
        self.ax_main = self.fig.add_subplot(gs[:, :2])
        self.ax_main.set_xlim(-1, config.GRID_SIZE + 1)
        self.ax_main.set_ylim(-1, config.GRID_SIZE + 1)
        self.ax_main.set_facecolor('#2d5016')  # Dark green game field
        self.ax_main.set_aspect('equal')
        
        # Remove chart elements - make it a game field
        self.ax_main.set_xticks([])
        self.ax_main.set_yticks([])
        for spine in self.ax_main.spines.values():
            spine.set_visible(False)
        
        # Add subtle grid pattern for game aesthetic
        for i in range(0, config.GRID_SIZE, 10):
            self.ax_main.axhline(i, color='#3d6d25', linewidth=0.5, alpha=0.3)
            self.ax_main.axvline(i, color='#3d6d25', linewidth=0.5, alpha=0.3)
        
        # Population graph (top right)
        self.ax_pop = self.fig.add_subplot(gs[0, 2])
        self.ax_pop.set_title('Population Over Time', fontsize=9, fontweight='bold', pad=3)
        self.ax_pop.set_xlabel('Step', fontsize=7)
        self.ax_pop.set_ylabel('Count', fontsize=7)
        self.ax_pop.tick_params(labelsize=6)
        self.ax_pop.grid(True, alpha=0.3)
        
        # Live statistics panel (middle right)
        self.ax_stats = self.fig.add_subplot(gs[1, 2])
        self.ax_stats.axis('off')
        self.ax_stats.set_title('Live Statistics', fontsize=9, fontweight='bold', pad=3)
        
        # Legend panel (bottom right)
        self.ax_legend = self.fig.add_subplot(gs[2, 2])
        self.ax_legend.axis('off')
        
        # Data for population tracking
        self.prey_history = []
        self.predator_history = []
        self.step_history = []
        
        # Simulation control state
        self.paused = False
        self.restart_requested = False
        
        # Cached artists for better performance
        self.prey_scatter = None
        self.pred_scatter = None
        self.hungry_scatter = None
        
        # Create legend and buttons
        self._create_legend()
        self._create_buttons()
        
        plt.ion()  # Interactive mode
        plt.show()
    
    def _create_legend(self):
        """Create visual legend for animal types"""
        legend_elements = [
            mpatches.Patch(facecolor='#00ff00', edgecolor='white', label='Prey (Well-fed)', linewidth=1.5),
            mpatches.Patch(facecolor='#ff4444', edgecolor='white', label='Predator (Normal)', linewidth=1.5),
            mpatches.Patch(facecolor='#880000', edgecolor='white', label='Predator (Hungry)', linewidth=1.5),
        ]
        self.ax_legend.legend(handles=legend_elements, loc='center', 
                             frameon=True, fontsize=7.5, ncol=1)
        self.ax_legend.set_title('Legend', fontsize=8, fontweight='bold', loc='center', pad=2)
    
    def _create_buttons(self):
        """Create interactive control buttons"""
        # Pause/Resume button
        ax_pause = plt.axes([0.38, 0.015, 0.10, 0.035])
        self.btn_pause = Button(ax_pause, 'Pause', color='lightblue', hovercolor='skyblue')
        self.btn_pause.on_clicked(self._toggle_pause)
        
        # Restart button
        ax_restart = plt.axes([0.52, 0.015, 0.10, 0.035])
        self.btn_restart = Button(ax_restart, 'Restart', color='lightcoral', hovercolor='salmon')
        self.btn_restart.on_clicked(self._request_restart)
    
    def _toggle_pause(self, event):
        """Toggle simulation pause state"""
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.label.set_text('Resume')
            self.btn_pause.color = 'lightgreen'
        else:
            self.btn_pause.label.set_text('Pause')
            self.btn_pause.color = 'lightblue'
        self.fig.canvas.draw_idle()
    
    def _request_restart(self, event):
        """Request simulation restart"""
        self.restart_requested = True
    
    def update(self, animals: List[Animal], step: int, stats: Dict = None):
        """Update all visualization components - OPTIMIZED"""
        # Count populations
        prey_count = sum(1 for a in animals if not a.predator)
        predator_count = sum(1 for a in animals if a.predator)
        
        # Update history
        self.prey_history.append(prey_count)
        self.predator_history.append(predator_count)
        self.step_history.append(step)
        
        # === OPTIMIZED MAIN FIELD RENDERING ===
        self.ax_main.clear()
        self.ax_main.set_xlim(-1, self.config.GRID_SIZE + 1)
        self.ax_main.set_ylim(-1, self.config.GRID_SIZE + 1)
        self.ax_main.set_facecolor('#2d5016')
        self.ax_main.set_aspect('equal')
        self.ax_main.set_xticks([])
        self.ax_main.set_yticks([])
        for spine in self.ax_main.spines.values():
            spine.set_visible(False)
        
        # Add grid pattern
        for i in range(0, self.config.GRID_SIZE, 10):
            self.ax_main.axhline(i, color='#3d6d25', linewidth=0.5, alpha=0.3)
            self.ax_main.axvline(i, color='#3d6d25', linewidth=0.5, alpha=0.3)
        
        # Title as game HUD - NO EMOJI
        title_text = f'STEP: {step} | ANIMALS: {len(animals)} | PREY: {prey_count} | PREDATORS: {predator_count}'
        self.ax_main.text(self.config.GRID_SIZE / 2, self.config.GRID_SIZE + 3, 
                         title_text,
                         fontsize=10, fontweight='bold', ha='center', color='white',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        
        # Separate animals by type for batch rendering
        prey_x, prey_y = [], []
        pred_x, pred_y = [], []
        hungry_x, hungry_y = [], []
        
        for animal in animals:
            if not animal.predator:
                prey_x.append(animal.x)
                prey_y.append(animal.y)
            else:
                if animal.steps_since_last_meal >= self.config.HUNGER_THRESHOLD:
                    hungry_x.append(animal.x)
                    hungry_y.append(animal.y)
                else:
                    pred_x.append(animal.x)
                    pred_y.append(animal.y)
        
        # Draw all animals at once using scatter (FAST!)
        if prey_x:
            self.ax_main.scatter(prey_x, prey_y, c='#00ff00', s=40, alpha=0.85, 
                               edgecolors='#88ff88', linewidth=1.5, marker='o')
        if pred_x:
            self.ax_main.scatter(pred_x, pred_y, c='#ff4444', s=50, alpha=0.85, 
                               edgecolors='#ffaaaa', linewidth=1.5, marker='^')
        if hungry_x:
            self.ax_main.scatter(hungry_x, hungry_y, c='#880000', s=60, alpha=0.9, 
                               edgecolors='#ff0000', linewidth=2, marker='^')
        
        # Update population graph - only every 5 steps for performance
        if step % 5 == 0:
            self.ax_pop.clear()
            self.ax_pop.set_title('Population Over Time', fontsize=9, fontweight='bold', pad=3)
            self.ax_pop.set_xlabel('Step', fontsize=7)
            self.ax_pop.set_ylabel('Count', fontsize=7)
            self.ax_pop.tick_params(labelsize=6)
            self.ax_pop.grid(True, alpha=0.3)
            
            if len(self.step_history) > 1:
                self.ax_pop.plot(self.step_history, self.prey_history, 'g-', 
                               linewidth=1.5, label='Prey')
                self.ax_pop.plot(self.step_history, self.predator_history, 'r-', 
                               linewidth=1.5, label='Predators')
                self.ax_pop.legend(loc='upper right', fontsize=6)
                self.ax_pop.fill_between(self.step_history, self.prey_history, alpha=0.2, color='green')
                self.ax_pop.fill_between(self.step_history, self.predator_history, alpha=0.2, color='red')
        
        # Update statistics panel
        self._update_stats(animals, step, prey_count, predator_count, stats)
        
        # Refresh display - reduced pause time for better performance
        plt.pause(0.0001)
    
    def _update_stats(self, animals: List[Animal], step: int, 
                     prey_count: int, predator_count: int, stats: Dict):
        """Update the statistics text panel"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Calculate additional statistics
        if animals:
            avg_prey_survival = np.mean([a.survival_time for a in animals if not a.predator]) if prey_count > 0 else 0
            avg_pred_survival = np.mean([a.survival_time for a in animals if a.predator]) if predator_count > 0 else 0
            avg_hunger = np.mean([a.steps_since_last_meal for a in animals if a.predator]) if predator_count > 0 else 0
        else:
            avg_prey_survival = avg_pred_survival = avg_hunger = 0
        
        # Create stats text - NO EMOJI
        stats_text = f"""CURRENT STATUS
==================
Step: {step}

POPULATION
  Prey: {prey_count}
  Predators: {predator_count}
  Ratio: {prey_count/max(predator_count,1):.1f}:1

AVERAGES
  Prey Survival: {avg_prey_survival:.1f}
  Pred Survival: {avg_pred_survival:.1f}
  Pred Hunger: {avg_hunger:.1f}
"""
        
        if stats:
            stats_text += f"\nEVENTS\n  Births: {stats.get('total_births', 0)}\n  Deaths: {stats.get('total_deaths', 0)}\n  Meals: {stats.get('total_meals', 0)}"
        
        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=7.5, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def show_final_stats(self, stats: Dict):
        """Display final statistics in a dedicated window"""
        fig_stats = plt.figure(figsize=(10, 7))
        fig_stats.suptitle('Simulation Summary', fontsize=14, fontweight='bold')
        
        # Create subplots for different statistics
        gs = fig_stats.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
        
        # Population history
        ax1 = fig_stats.add_subplot(gs[0, :])
        ax1.plot(self.step_history, self.prey_history, 'g-', linewidth=2, label='Prey')
        ax1.plot(self.step_history, self.predator_history, 'r-', linewidth=2, label='Predators')
        ax1.set_title('Population Dynamics', fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Population')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(self.step_history, self.prey_history, alpha=0.2, color='green')
        ax1.fill_between(self.step_history, self.predator_history, alpha=0.2, color='red')
        
        # Statistics text (left middle)
        ax2 = fig_stats.add_subplot(gs[1, 0])
        ax2.axis('off')
        summary_text = f"""FINAL STATISTICS
========================
Total Steps: {stats['total_steps']}
Duration: {stats['duration']:.2f}s

POPULATION
  Final Prey: {stats['final_prey']}
  Final Predators: {stats['final_predators']}
  Peak Prey: {stats['peak_prey']}
  Peak Predators: {stats['peak_predators']}
  Min Prey: {stats['min_prey']}

EVENTS
  Total Births: {stats['total_births']}
  Total Deaths: {stats['total_deaths']}
  Predator Meals: {stats['total_meals']}
"""
        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Ratio over time (right middle)
        ax3 = fig_stats.add_subplot(gs[1, 1])
        ratios = [p/max(pr, 1) for p, pr in zip(self.prey_history, self.predator_history)]
        ax3.plot(self.step_history, ratios, 'b-', linewidth=2)
        ax3.axhline(y=10, color='g', linestyle='--', alpha=0.5, label='Healthy (10:1)')
        ax3.set_title('Prey:Predator Ratio', fontweight='bold')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Population distribution (bottom)
        ax4 = fig_stats.add_subplot(gs[2, :])
        categories = ['Prey\n(Start)', 'Predators\n(Start)', 'Prey\n(Final)', 'Predators\n(Final)']
        values = [self.prey_history[0], self.predator_history[0], 
                 stats['final_prey'], stats['final_predators']]
        colors = ['lightgreen', 'lightcoral', 'green', 'red']
        bars = ax4.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
        ax4.set_title('Population Comparison', fontweight='bold')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        plt.show()
