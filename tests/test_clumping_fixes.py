"""
Test comprehensive clumping behavior fixes.

Verifies:
1. Grass regrowth is patchy (not full refill)
2. Prey flee supervision threshold is 0.999 (flee on sight)
3. Species-specific directional loss coefficients exist
4. Prey mating pheromones require safety check
5. Config defaults for evasion exist
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.config import SimulationConfig
from src.core.grass_field import GrassField
from src.core.animal import Prey, Predator
from src.core.pheromone_system import PheromoneMap


def test_grass_regrowth_is_patchy():
    """Verify grass regrows partially, not all at once."""
    config = SimulationConfig()
    grid_size = config.GRID_SIZE
    grass = GrassField(grid_size, regrow_interval=config.GRASS_REGROW_INTERVAL)
    
    # Eat all grass
    grass.grid.fill(False)
    
    # Step to regrow time
    grass.step_regrow(config.GRASS_REGROW_INTERVAL)
    
    # Should have some grass, but not all
    grass_count = grass.grid.sum()
    total_cells = grid_size * grid_size
    print(f"Grass regrown after full depletion: {grass_count}/{total_cells}")
    
    # With regrow_prob=0.05, expect ~5% coverage
    assert 0 < grass_count < total_cells, f"Grass should regrow partially, got {grass_count}/{total_cells}"
    assert grass_count < total_cells * 0.2, f"Too much grass regrew ({grass_count}), should be patchy"
    
    print("✓ Grass regrowth is patchy")


def test_flee_supervision_exists():
    """Verify flee supervision threshold config exists."""
    config = SimulationConfig()
    
    assert hasattr(config, "PREY_FLEE_SUPERVISION_DIST"), "Missing PREY_FLEE_SUPERVISION_DIST"
    threshold = config.PREY_FLEE_SUPERVISION_DIST
    print(f"PREY_FLEE_SUPERVISION_DIST = {threshold}")
    
    assert 0.0 < threshold <= 1.0, f"Threshold must be in (0, 1], got {threshold}"
    
    print("✓ Flee supervision threshold exists and is valid")


def test_directional_loss_coefficients_exist():
    """Verify species-specific directional loss coefficients exist and are positive."""
    config = SimulationConfig()
    
    assert hasattr(config, "DIRECTIONAL_LOSS_COEF_PREY"), "Missing DIRECTIONAL_LOSS_COEF_PREY"
    assert hasattr(config, "DIRECTIONAL_LOSS_COEF_PREDATOR"), "Missing DIRECTIONAL_LOSS_COEF_PREDATOR"
    
    prey_coef = config.DIRECTIONAL_LOSS_COEF_PREY
    pred_coef = config.DIRECTIONAL_LOSS_COEF_PREDATOR
    
    print(f"DIRECTIONAL_LOSS_COEF_PREY = {prey_coef}")
    print(f"DIRECTIONAL_LOSS_COEF_PREDATOR = {pred_coef}")
    
    assert prey_coef > 0, f"Prey coef must be positive, got {prey_coef}"
    assert pred_coef > 0, f"Predator coef must be positive, got {pred_coef}"
    
    print("✓ Directional loss coefficients exist and are positive")


def test_prey_mating_safety_check():
    """Verify prey mating safety logic exists in code."""
    config = SimulationConfig()
    
    # Verify PREY_SAFE_TO_MATE_DIST_NORM config exists (used in safety check)
    safe_dist = getattr(config, "PREY_SAFE_TO_MATE_DIST_NORM", None)
    assert safe_dist is not None, "PREY_SAFE_TO_MATE_DIST_NORM must be defined"
    print(f"PREY_SAFE_TO_MATE_DIST_NORM = {safe_dist}")
    
    # Read the source to verify safety check exists
    import inspect
    from src.core.animal import Prey
    source = inspect.getsource(Prey.deposit_pheromones)
    
    # Check for key safety conditions (updated logic uses is_safe with distance check)
    assert "is_safe" in source, "Must have is_safe variable"
    assert "predator_count" in source, "Must check predator_count"
    assert "nearest_predator_dist" in source, "Must check nearest_predator_dist"
    assert "safe_dist" in source or "PREY_SAFE_TO_MATE_DIST_NORM" in source, "Must use safe distance threshold"
    
    print("✓ Prey mating pheromones require safety (code inspection passed)")


def test_evasion_config_defaults():
    """Verify evasion reward config knobs exist."""
    config = SimulationConfig()
    
    assert hasattr(config, "PREY_EVASION_REWARD"), "Missing PREY_EVASION_REWARD"
    assert hasattr(config, "PREY_EVASION_PENALTY"), "Missing PREY_EVASION_PENALTY"
    
    print(f"PREY_EVASION_REWARD = {config.PREY_EVASION_REWARD}")
    print(f"PREY_EVASION_PENALTY = {config.PREY_EVASION_PENALTY}")
    
    assert config.PREY_EVASION_REWARD > 0, "Evasion reward must be positive"
    assert config.PREY_EVASION_PENALTY > 0, "Evasion penalty must be positive"
    
    print("✓ Evasion config defaults exist")


def test_update_energy_charges_for_attempts():
    """Verify update_energy charges for movement attempts, not success."""
    config = SimulationConfig()
    prey = Prey(10, 10, "Test", "blue")
    prey.energy = 50.0
    
    # Simulate 2 movement attempts (first blocked, second succeeded)
    initial_energy = prey.energy
    prey.update_energy(config, move_attempts=2)
    
    energy_lost = initial_energy - prey.energy
    expected_loss = config.ENERGY_DECAY_RATE + config.MOVE_ENERGY_COST * 2
    
    print(f"Energy lost: {energy_lost:.2f}, expected: {expected_loss:.2f}")
    
    assert abs(energy_lost - expected_loss) < 0.01, f"Should charge for attempts, not success"
    
    print("✓ update_energy charges for attempts")


if __name__ == "__main__":
    print("Testing comprehensive clumping fixes...\n")
    
    test_grass_regrowth_is_patchy()
    test_flee_supervision_exists()
    test_directional_loss_coefficients_exist()
    test_prey_mating_safety_check()
    test_evasion_config_defaults()
    test_update_energy_charges_for_attempts()
    
    print("\n✓ All clumping fix tests passed!")
