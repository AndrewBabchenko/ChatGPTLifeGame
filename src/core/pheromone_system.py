"""
Pheromone Trail System
Animals can leave scent markers that persist in the environment
Enables indirect communication and territorial behavior
"""

import numpy as np
from typing import Tuple, List
import time


class PheromoneMap:
    """
    2D grid of pheromone concentrations
    Pheromones decay over time and diffuse to adjacent cells
    """
    def __init__(self, grid_size: int, decay_rate: float = 0.95, diffusion_rate: float = 0.1):
        self.grid_size = grid_size
        self.decay_rate = decay_rate  # How fast pheromones fade
        self.diffusion_rate = diffusion_rate  # How much pheromones spread
        
        # Separate maps for different pheromone types
        self.danger_map = np.zeros((grid_size, grid_size), dtype=np.float32)  # Warning signals
        self.mating_map = np.zeros((grid_size, grid_size), dtype=np.float32)  # Mating signals
        self.territory_map = np.zeros((grid_size, grid_size), dtype=np.float32)  # Territory markers
        self.food_map = np.zeros((grid_size, grid_size), dtype=np.float32)  # Food/prey locations
        
        self.last_update = time.time()
    
    def deposit_pheromone(self, x: int, y: int, pheromone_type: str, strength: float = 1.0):
        """
        Deposit pheromone at a location
        
        Args:
            x, y: Grid coordinates
            pheromone_type: 'danger', 'mating', 'territory', or 'food'
            strength: Intensity of pheromone (0.0 to 1.0)
        """
        x = int(x) % self.grid_size
        y = int(y) % self.grid_size
        
        if pheromone_type == 'danger':
            self.danger_map[x, y] = min(1.0, self.danger_map[x, y] + strength)
        elif pheromone_type == 'mating':
            self.mating_map[x, y] = min(1.0, self.mating_map[x, y] + strength)
        elif pheromone_type == 'territory':
            self.territory_map[x, y] = min(1.0, self.territory_map[x, y] + strength)
        elif pheromone_type == 'food':
            self.food_map[x, y] = min(1.0, self.food_map[x, y] + strength)
    
    def get_pheromone(self, x: int, y: int, pheromone_type: str) -> float:
        """Get pheromone concentration at location"""
        x = int(x) % self.grid_size
        y = int(y) % self.grid_size
        
        if pheromone_type == 'danger':
            return float(self.danger_map[x, y])
        elif pheromone_type == 'mating':
            return float(self.mating_map[x, y])
        elif pheromone_type == 'territory':
            return float(self.territory_map[x, y])
        elif pheromone_type == 'food':
            return float(self.food_map[x, y])
        return 0.0
    
    def get_local_pheromones(self, x: int, y: int, radius: int = 2) -> dict:
        """
        Get pheromone concentrations in local area
        Returns average and max within radius
        """
        x = int(x) % self.grid_size
        y = int(y) % self.grid_size
        
        x_min = max(0, x - radius)
        x_max = min(self.grid_size, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.grid_size, y + radius + 1)
        
        danger_patch = self.danger_map[x_min:x_max, y_min:y_max]
        mating_patch = self.mating_map[x_min:x_max, y_min:y_max]
        territory_patch = self.territory_map[x_min:x_max, y_min:y_max]
        food_patch = self.food_map[x_min:x_max, y_min:y_max]
        
        return {
            'danger_avg': float(danger_patch.mean()),
            'danger_max': float(danger_patch.max()),
            'mating_avg': float(mating_patch.mean()),
            'mating_max': float(mating_patch.max()),
            'territory_avg': float(territory_patch.mean()),
            'territory_max': float(territory_patch.max()),
            'food_avg': float(food_patch.mean()),
            'food_max': float(food_patch.max())
        }
    
    def get_gradient(self, x: int, y: int, pheromone_type: str) -> Tuple[float, float]:
        """
        Get pheromone gradient direction (points toward higher concentration)
        Returns (dx, dy) normalized direction vector
        """
        x = int(x) % self.grid_size
        y = int(y) % self.grid_size
        
        # Select map
        if pheromone_type == 'danger':
            pmap = self.danger_map
        elif pheromone_type == 'mating':
            pmap = self.mating_map
        elif pheromone_type == 'territory':
            pmap = self.territory_map
        elif pheromone_type == 'food':
            pmap = self.food_map
        else:
            return 0.0, 0.0
        
        # Compute gradient
        dx = pmap[(x + 1) % self.grid_size, y] - pmap[(x - 1) % self.grid_size, y]
        dy = pmap[x, (y + 1) % self.grid_size] - pmap[x, (y - 1) % self.grid_size]
        
        # Normalize
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
        
        return float(dx), float(dy)
    
    def update(self):
        """
        Update pheromone maps: apply decay and diffusion
        Call once per simulation step
        """
        # Decay all pheromones
        self.danger_map *= self.decay_rate
        self.mating_map *= self.decay_rate
        self.territory_map *= self.decay_rate
        self.food_map *= self.decay_rate
        
        # Diffusion (blur with neighbors)
        if self.diffusion_rate > 0:
            self.danger_map = self._diffuse(self.danger_map)
            self.mating_map = self._diffuse(self.mating_map)
            self.territory_map = self._diffuse(self.territory_map)
            self.food_map = self._diffuse(self.food_map)
    
    def _diffuse(self, pmap: np.ndarray) -> np.ndarray:
        """Apply diffusion to a pheromone map"""
        diffused = pmap.copy()
        
        # Simple 4-neighbor diffusion
        diffused += np.roll(pmap, 1, axis=0) * self.diffusion_rate
        diffused += np.roll(pmap, -1, axis=0) * self.diffusion_rate
        diffused += np.roll(pmap, 1, axis=1) * self.diffusion_rate
        diffused += np.roll(pmap, -1, axis=1) * self.diffusion_rate
        
        # Normalize
        diffused /= (1.0 + 4.0 * self.diffusion_rate)
        
        return diffused
    
    def visualize(self, pheromone_type: str) -> np.ndarray:
        """Get pheromone map for visualization"""
        if pheromone_type == 'danger':
            return self.danger_map.copy()
        elif pheromone_type == 'mating':
            return self.mating_map.copy()
        elif pheromone_type == 'territory':
            return self.territory_map.copy()
        elif pheromone_type == 'food':
            return self.food_map.copy()
        return np.zeros((self.grid_size, self.grid_size))
    
    def reset(self):
        """Clear all pheromones"""
        self.danger_map.fill(0)
        self.mating_map.fill(0)
        self.territory_map.fill(0)
        self.food_map.fill(0)
