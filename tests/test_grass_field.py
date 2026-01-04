import numpy as np

from src.config import SimulationConfig
from src.core.grass_field import GrassField

def test_grass_regrows_after_interval():
    """Test that grass regrows probabilistically after interval."""
    config = SimulationConfig()
    grid_size = config.GRID_SIZE
    regrow_interval = config.GRASS_REGROW_INTERVAL
    field = GrassField(grid_size=grid_size, regrow_interval=regrow_interval)

    # Consume all grass
    field.grid.fill(False)
    assert field.has_grass(0, 0) is False
    
    # Not yet regrown at step before interval
    field.step_regrow(step=regrow_interval - 1)
    assert field.grid.sum() == 0, "No grass should regrow before interval"

    # Set seed for determinism
    np.random.seed(42)
    
    # Regrows probabilistically on interval boundary
    field.step_regrow(step=regrow_interval)
    
    # With regrow_prob=0.05, some cells should regrow (but not all)
    grass_count = field.grid.sum()
    total_cells = grid_size * grid_size
    assert grass_count > 0, "Some grass should regrow after interval"
    assert grass_count < total_cells * 0.2, "Not all grass should regrow (probabilistic)"
    
    print(f"✓ Grass regrows probabilistically: {grass_count}/{total_cells} cells after interval")
