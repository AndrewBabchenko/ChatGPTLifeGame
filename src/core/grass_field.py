"""
Grass field with binary occupancy per cell.
Starts fully grassed; empty cells regrow on a fixed interval.
"""

import numpy as np


class GrassField:
    """Binary grass grid with toroidal indexing."""

    def __init__(self, grid_size: int, regrow_interval: int) -> None:
        self.grid_size = int(grid_size)
        self.regrow_interval = max(1, int(regrow_interval))
        self.grid = np.ones((self.grid_size, self.grid_size), dtype=bool)

    def reset_full(self) -> None:
        """Refill all cells with grass."""
        self.grid.fill(True)

    def has_grass(self, x: int, y: int) -> bool:
        xi = int(x) % self.grid_size
        yi = int(y) % self.grid_size
        return bool(self.grid[xi, yi])

    def consume(self, x: int, y: int) -> bool:
        """Consume grass at a cell if present. Returns True if consumed."""
        xi = int(x) % self.grid_size
        yi = int(y) % self.grid_size
        if self.grid[xi, yi]:
            self.grid[xi, yi] = False
            return True
        return False

    def step_regrow(self, step: int) -> None:
        """Patchy regrowth to create scarcity and encourage travel."""
        if step <= 0:
            return
        if step % self.regrow_interval != 0:
            return

        # Regrow only a fraction of empty cells each tick
        # (tune this; lower = more scarcity)
        regrow_prob = 0.05

        empty = ~self.grid
        if not empty.any():
            return

        rnd = np.random.rand(self.grid_size, self.grid_size)
        self.grid[empty & (rnd < regrow_prob)] = True

    def reset(self) -> None:
        self.reset_full()
