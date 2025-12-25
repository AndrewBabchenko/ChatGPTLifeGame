"""
Game field visualizer for the Life Game simulation
Displays animals on a game board (not a chart)
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle
import numpy as np
from typing import List, Dict
from animal import Animal


class GameFieldVisualizer:
    """Game field display for visualization with statistics and controls"""
    
    def __init__(self, config):
        self.config = config
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle('🎮 Life Game - Predator-Prey Ecosystem 🎮', 
                         fontsize=18, fontweight='bold')
        
        # Create grid layout with 5 rows for better spacing
        gs = self.fig.add_gridspec(5, 3, hspace=0.35, wspace=0.4, 
                                  left=0.05, right=0.95, top=0.93, bottom=0.08)
        
        # Main game field (left side, larger) - spans 4 rows
        self.ax_main = self.fig.add_subplot(gs[:4, :2])
        self.ax_main.set_xlim(-1, config.GRID_SIZE + 1)
        self.ax_main.set_ylim(-1, config.GRID_SIZE + 1)
        self.ax_main.set_facecolor('#2d5016')  # Dark green game field
        self.ax_main.set_aspect('equal')
        
        # Remove chart elements - make it a game field
        self.ax_main.set_xticks([])
        self.ax_main.set_yticks([])
        self.ax_main.spines['top'].set_visible(False)
        self.ax_main.spines['right'].set_visible(False)
        self.ax_main.spines['bottom'].set_visible(False)
        self.ax_main.spines['left'].set_visible(False)
        
        # Add grid pattern for game aesthetic
        for i in range(0, config.GRID_SIZE, 10):
            self.ax_main.axhline(i, color='#3d6d25', linewidth=0.5, alpha=0.3)
            self.ax_main.axvline(i, color='#3d6d25', linewidth=0.5, alpha=0.3)
        
        # Population graph (top right)
        self.ax_pop = self.fig.add_subplot(gs[0, 2])
        self.ax_pop.set_title('📊 Population Over Time', fontsize=10, fontweight='bold')
        self.ax_pop.set_xlabel('Step', fontsize=8)
        self.ax_pop.set_ylabel('Count', fontsize=8)
        self.ax_pop.grid(True, alpha=0.3)
        
        # Live statistics panel (middle right)
        self.ax_stats = self.fig.add_subplot(gs[1:3, 2])
        self.ax_stats.axis('off')
        self.ax_stats.set_title('📈 Live Statistics', fontsize=10, fontweight='bold')
        
        # Legend panel (bottom right)
        self.ax_legend = self.fig.add_subplot(gs[3, 2])
        self.ax_legend.axis('off')
        
        # Control buttons panel (bottom row)
        self.ax_controls = self.fig.add_subplot(gs[4, :])
        self.ax_controls.axis('off')
        
        # Data for population tracking
        self.prey_history = []
        self.predator_history = []
        self.step_history = []
        
        # Simulation control state
        self.paused = False
        self.restart_requested = False
        
        # Create legend and buttons
        self._create_legend()
        self._create_buttons()
        
        plt.ion()  # Interactive mode
        plt.show()
    
    def _create_legend(self):
        """Create visual legend for animal types"""
        legend_elements = [
            mpatches.Patch(facecolor='#00ff00', edgecolor='white', label='🐰 Prey (Well-fed)', linewidth=2),
            mpatches.Patch(facecolor='#ff4444', edgecolor='white', label='🦊 Predator (Normal)', linewidth=2),
            mpatches.Patch(facecolor='#880000', edgecolor='white', label='🐺 Predator (Hungry)', linewidth=2),
        ]
        self.ax_legend.legend(handles=legend_elements, loc='center', 
                             frameon=True, fontsize=9, ncol=1)
        self.ax_legend.set_title('🏷️ Legend', fontsize=10, fontweight='bold', loc='center', pad=5)
    
    def _create_buttons(self):
        """Create interactive control buttons"""
        # Pause/Resume button
        ax_pause = plt.axes([0.35, 0.02, 0.12, 0.04])
        self.btn_pause = Button(ax_pause, '⏸️ Pause', color='lightblue', hovercolor='skyblue')
        self.btn_pause.on_clicked(self._toggle_pause)
        
        # Restart button
        ax_restart = plt.axes([0.53, 0.02, 0.12, 0.04])
        self.btn_restart = Button(ax_restart, '🔄 Restart', color='lightcoral', hovercolor='salmon')
        self.btn_restart.on_clicked(self._request_restart)
    
    def _toggle_pause(self, event):
        """Toggle simulation pause state"""
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.label.set_text('▶️ Resume')
            self.btn_pause.color = 'lightgreen'
        else:
            self.btn_pause.label.set_text('⏸️ Pause')
            self.btn_pause.color = 'lightblue'
        self.fig.canvas.draw_idle()
    
    def _request_restart(self, event):
        """Request simulation restart"""
        self.restart_requested = True
    
    def update(self, animals: List[Animal], step: int, stats: Dict = None):
        """Update all visualization components"""
        # Count populations
        prey_count = sum(1 for a in animals if not a.predator)
        predator_count = sum(1 for a in animals if a.predator)
        
        # Update history
        self.prey_history.append(prey_count)
        self.predator_history.append(predator_count)
        self.step_history.append(step)
        
        # Clear and redraw game field
        self.ax_main.clear()
        self.ax_main.set_xlim(-1, self.config.GRID_SIZE + 1)
        self.ax_main.set_ylim(-1, self.config.GRID_SIZE + 1)
        self.ax_main.set_facecolor('#2d5016')  # Dark green game field
        self.ax_main.set_aspect('equal')
        
        # Remove chart elements
        self.ax_main.set_xticks([])
        self.ax_main.set_yticks([])
        self.ax_main.spines['top'].set_visible(False)
        self.ax_main.spines['right'].set_visible(False)
        self.ax_main.spines['bottom'].set_visible(False)
        self.ax_main.spines['left'].set_visible(False)
        
        # Add grid pattern
        for i in range(0, self.config.GRID_SIZE, 10):
            self.ax_main.axhline(i, color='#3d6d25', linewidth=0.5, alpha=0.3)
            self.ax_main.axvline(i, color='#3d6d25', linewidth=0.5, alpha=0.3)
        
        # Add title as game HUD
        self.ax_main.text(self.config.GRID_SIZE / 2, self.config.GRID_SIZE + 3, 
                         f'⏱️ STEP: {step} | 🐾 ANIMALS: {len(animals)} | 🐰 PREY: {prey_count} | 🦊 PREDATORS: {predator_count}',
                         fontsize=11, fontweight='bold', ha='center', color='white',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        
        # Draw animals as game sprites (circles with glow effect)
        for animal in animals:
            color = animal.display_color(self.config)
            
            # Determine sprite appearance
            if not animal.predator:
                # Prey - green circles
                sprite_color = '#00ff00'
                edge_color = '#88ff88'
                size = 0.6
                marker = 'o'
            else:
                # Predator - red triangles
                if animal.steps_since_last_meal >= self.config.HUNGER_THRESHOLD:
                    sprite_color = '#880000'  # Hungry - dark red
                    edge_color = '#ff0000'
                    size = 0.8
                else:
                    sprite_color = '#ff4444'  # Normal - bright red
                    edge_color = '#ffaaaa'
                    size = 0.7
                marker = '^'
            
            # Draw sprite with glow effect
            circle = Circle((animal.x, animal.y), size, 
                          facecolor=sprite_color, 
                          edgecolor=edge_color, 
                          linewidth=2, 
                          alpha=0.9)
            self.ax_main.add_patch(circle)
        
        # Update population graph
        self.ax_pop.clear()
        self.ax_pop.set_title('📊 Population Over Time', fontsize=10, fontweight='bold')
        self.ax_pop.set_xlabel('Step', fontsize=8)
        self.ax_pop.set_ylabel('Count', fontsize=8)
        self.ax_pop.grid(True, alpha=0.3)
        
        if len(self.step_history) > 1:
            self.ax_pop.plot(self.step_history, self.prey_history, 'g-', 
                           linewidth=2, label='Prey')
            self.ax_pop.plot(self.step_history, self.predator_history, 'r-', 
                           linewidth=2, label='Predators')
            self.ax_pop.legend(loc='upper right', fontsize=8)
            self.ax_pop.fill_between(self.step_history, self.prey_history, alpha=0.3, color='green')
            self.ax_pop.fill_between(self.step_history, self.predator_history, alpha=0.3, color='red')
        
        # Update statistics panel
        self._update_stats(animals, step, prey_count, predator_count, stats)
        
        # Refresh display
        plt.pause(0.001)
    
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
        
        # Create stats text with emoji
        stats_text = f"""📊 CURRENT STATUS
━━━━━━━━━━━━━━━━━━━
⏱️  Step: {step}

🐾 POPULATION
  🐰 Prey: {prey_count}
  🦊 Predators: {predator_count}
  📐 Ratio: {prey_count/max(predator_count,1):.1f}:1

📈 AVERAGES
  🐰 Prey Survival: {avg_prey_survival:.1f}
  🦊 Pred Survival: {avg_pred_survival:.1f}
  🍖 Pred Hunger: {avg_hunger:.1f}
"""
        
        if stats:
            stats_text += f"\n📋 EVENTS\n  👶 Births: {stats.get('total_births', 0)}\n  💀 Deaths: {stats.get('total_deaths', 0)}\n  🍽️  Meals: {stats.get('total_meals', 0)}"
        
        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=9, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def show_final_stats(self, stats: Dict):
        """Display final statistics in a dedicated window"""
        fig_stats = plt.figure(figsize=(10, 8))
        fig_stats.suptitle('🏆 Simulation Summary', fontsize=16, fontweight='bold')
        
        # Create subplots for different statistics
        gs = fig_stats.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
                fontsize=10, verticalalignment='top', fontfamily='monospace',
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
