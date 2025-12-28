"""
Pheromone Trail System
Animals can leave scent markers that persist in the environment
Enables indirect communication and territorial behavior

COORDINATE CONVENTION: Uses [x, y] indexing consistently (not NumPy's [row, col]).
- x: horizontal position (0 to grid_size-1)
- y: vertical position (0 to grid_size-1)
- All methods (deposit, get, gradient, visualize) use this convention

SENSING MODEL: Pheromones are omnidirectional (like smell).
- Not limited by animal heading/FOV (unlike vision)
- All methods are toroidal (wrap at edges)
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
        
        # Performance caches
        self._offset_cache = {}     # radius -> offsets array (avoid repeated arange)
        self._sensory_cache = {}    # (x,y,radius,mag_scale,version) -> dict (per-step cache)
        self._version = 0           # increments on update/reset (invalidates cache)
    
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
    
    def _offsets(self, radius: int) -> np.ndarray:
        """Get cached offset array for given radius (avoid repeated arange allocation)"""
        r = int(radius)
        off = self._offset_cache.get(r)
        if off is None:
            off = np.arange(-r, r + 1, dtype=np.int32)
            self._offset_cache[r] = off
        return off
    
    def get_local_pheromones(self, x: int, y: int, radius: int = 2) -> dict:
        """
        Get pheromone concentrations in local area (toroidal)
        Returns average and max within radius
        
        Uses toroidal wrapping so animals sense across grid edges.
        Optimized with cached offsets and np.ix_ (no meshgrid allocation).
        """
        x = int(x) % self.grid_size
        y = int(y) % self.grid_size
        
        # Get cached offsets (avoid repeated arange)
        off = self._offsets(radius)
        
        xi = (x + off) % self.grid_size
        yi = (y + off) % self.grid_size
        
        # Use np.ix_ instead of meshgrid (same result, less allocation)
        idx = np.ix_(xi, yi)
        
        danger_patch = self.danger_map[idx]
        mating_patch = self.mating_map[idx]
        territory_patch = self.territory_map[idx]
        food_patch = self.food_map[idx]
        
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
    
    def _get_gradient_raw(self, x: int, y: int, pheromone_type: str) -> float:
        """
        Get raw gradient magnitude (before normalization) for NN feature.
        Helps NN distinguish strong gradients from noise.
        """
        x = int(x) % self.grid_size
        y = int(y) % self.grid_size
        
        # Select map
        if pheromone_type == 'danger':
            pmap = self.danger_map
        elif pheromone_type == 'mating':
            pmap = self.mating_map
        elif pheromone_type == 'food':
            pmap = self.food_map
        else:
            return 0.0
        
        # Compute gradient (toroidal)
        dx = pmap[(x + 1) % self.grid_size, y] - pmap[(x - 1) % self.grid_size, y]
        dy = pmap[x, (y + 1) % self.grid_size] - pmap[x, (y - 1) % self.grid_size]
        
        # Return magnitude
        magnitude = np.sqrt(dx**2 + dy**2)
        return float(magnitude)
    
    def get_gradient(self, x: int, y: int, pheromone_type: str) -> Tuple[float, float]:
        """
        Get pheromone gradient direction (points toward higher concentration)
        Returns (dx, dy) normalized direction vector
        Uses central difference (1-cell) for gradient computation.
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
        
        # Compute gradient (toroidal)
        dx = pmap[(x + 1) % self.grid_size, y] - pmap[(x - 1) % self.grid_size, y]
        dy = pmap[x, (y + 1) % self.grid_size] - pmap[x, (y - 1) % self.grid_size]
        
        # Normalize
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
        
        return float(dx), float(dy)
    
    def _grad_dxdy(self, pmap: np.ndarray, x: int, y: int) -> Tuple[float, float]:
        """Compute gradient dx/dy at position (shared by direction and magnitude)"""
        g = self.grid_size
        dx = float(pmap[(x + 1) % g, y] - pmap[(x - 1) % g, y])
        dy = float(pmap[x, (y + 1) % g] - pmap[x, (y - 1) % g])
        return dx, dy
    
    def get_sensory_input(self, x: int, y: int, radius: int = 2, mag_scale: float = None) -> dict:
        """
        Get complete pheromone sensory input for neural network.
        Returns local concentrations (avg/max) AND gradient directions + magnitudes.
        
        This provides both:
        - Intensity information (how strong is the signal)
        - Directional cues (which way is it stronger)
        - Gradient magnitude (how steep is the slope - helps distinguish signal from noise)
        
        Args:
            x, y: position to sense from
            radius: sensing radius in cells
            mag_scale: scale factor for normalizing gradient magnitudes to [0,1].
                      Default 0.5 means gradients >0.5 saturate at 1.0.
        
        Recommended for NN state vectors to enable pheromone-following behavior.
        Optimized: computes gradients once, caches per-step (pheromones only change on update()).
        """
        x = int(x) % self.grid_size
        y = int(y) % self.grid_size
        
        if mag_scale is None:
            mag_scale = 0.5
        
        # Per-step cache (pheromones don't change within a step)
        key = (x, y, int(radius), float(mag_scale), self._version)
        cached = self._sensory_cache.get(key)
        if cached is not None:
            return cached
        
        # Get local concentrations
        local = self.get_local_pheromones(x, y, radius)
        
        # Danger gradient (compute dx/dy once)
        ddx, ddy = self._grad_dxdy(self.danger_map, x, y)
        dmag = (ddx*ddx + ddy*ddy) ** 0.5
        if dmag > 0.0:
            ddirx, ddiry = ddx / dmag, ddy / dmag
        else:
            ddirx, ddiry = 0.0, 0.0
        
        # Mating gradient
        mdx, mdy = self._grad_dxdy(self.mating_map, x, y)
        mmag = (mdx*mdx + mdy*mdy) ** 0.5
        if mmag > 0.0:
            mdirx, mdiry = mdx / mmag, mdy / mmag
        else:
            mdirx, mdiry = 0.0, 0.0
        
        # Food gradient
        fdx, fdy = self._grad_dxdy(self.food_map, x, y)
        fmag = (fdx*fdx + fdy*fdy) ** 0.5
        if fmag > 0.0:
            fdirx, fdiry = fdx / fmag, fdy / fmag
        else:
            fdirx, fdiry = 0.0, 0.0
        
        # Territory gradient (not used in NN features - set to 0.0 for compatibility)
        # Skip computation to save cycles (no animal uses territory gradients)
        tdirx, tdiry = 0.0, 0.0
        
        out = {
            # Local concentrations (8 values)
            'danger_avg': local['danger_avg'],
            'danger_max': local['danger_max'],
            'mating_avg': local['mating_avg'],
            'mating_max': local['mating_max'],
            'territory_avg': local['territory_avg'],
            'territory_max': local['territory_max'],
            'food_avg': local['food_avg'],
            'food_max': local['food_max'],
            
            # Gradient directions (8 values: 4 types × 2 components)
            'danger_grad_x': ddirx,
            'danger_grad_y': ddiry,
            'mating_grad_x': mdirx,
            'mating_grad_y': mdiry,
            'food_grad_x': fdirx,
            'food_grad_y': fdiry,
            'territory_grad_x': tdirx,  # 0.0 (not computed, not used)
            'territory_grad_y': tdiry,  # 0.0 (not computed, not used)
            
            # Gradient magnitudes (3 values: danger/mating/food only, normalized to [0,1])
            'danger_grad_mag': min(1.0, dmag / mag_scale),
            'mating_grad_mag': min(1.0, mmag / mag_scale),
            'food_grad_mag': min(1.0, fmag / mag_scale),
        }
        
        # Cache for this step (invalidated on update/reset)
        self._sensory_cache[key] = out
        return out
    
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
        
        # Clamp to valid range [0,1] to prevent numerical drift
        np.clip(self.danger_map, 0.0, 1.0, out=self.danger_map)
        np.clip(self.mating_map, 0.0, 1.0, out=self.mating_map)
        np.clip(self.territory_map, 0.0, 1.0, out=self.territory_map)
        np.clip(self.food_map, 0.0, 1.0, out=self.food_map)
        
        # Invalidate sensory cache (pheromones changed)
        self._version += 1
        self._sensory_cache.clear()
    
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
        
        # Invalidate sensory cache (pheromones changed)
        self._version += 1
        self._sensory_cache.clear()
