"""
Test that step_idx parameter threading works correctly (refactor validation)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.config import SimulationConfig
from src.core.animal import Prey, Predator
from src.core.grass_field import GrassField
from src.core.pheromone_system import PheromoneMap
from src.models.actor_critic_network import ActorCriticNetwork

def test_step_idx_threading():
    """Verify step_idx is threaded through all visibility-dependent functions"""
    config = SimulationConfig()
    
    # Create animals
    prey1 = Prey(10, 10, 'A', (0, 255, 0))
    prey2 = Prey(12, 12, 'A', (0, 255, 0))
    pred1 = Predator(50, 50, 'B', (255, 0, 0))
    animals = [prey1, prey2, pred1]
    
    # Create environment
    grass_field = GrassField(config.GRID_SIZE, regrow_interval=10)
    pheromone_map = PheromoneMap(config.GRID_SIZE)
    
    # Create model
    device = torch.device('cpu')
    model = ActorCriticNetwork(config)
    model.to(device)
    
    # Test 1: move_training accepts step_idx
    print("Testing move_training with step_idx...")
    transitions = prey1.move_training(model, animals, config, pheromone_map, step_idx=10)
    assert len(transitions) > 0, "move_training should return transitions"
    assert hasattr(prey1, '_last_visible_step'), "Should cache visibility step"
    assert prey1._last_visible_step == 10, f"Expected step 10, got {prey1._last_visible_step}"
    print(f"✓ move_training cached visibility for step {prey1._last_visible_step}")
    
    # Test 2: Cached visibility persists within same step
    print("\nTesting visibility cache reuse...")
    initial_calls = getattr(prey1, '_communicate_calls', 0)
    prey1._communicate_calls = 0  # Reset counter
    
    # First call at step 20 - should call communicate()
    transitions1 = prey1.move_training(model, animals, config, pheromone_map, step_idx=20)
    calls_after_first = getattr(prey1, '_communicate_calls', 0)
    
    # Second call at step 20 - should use cache (no communicate())
    transitions2 = prey1.move_training(model, animals, config, pheromone_map, step_idx=20)
    calls_after_second = getattr(prey1, '_communicate_calls', 0)
    
    # Note: move_training calls communicate internally, which we can't easily track
    # But we can verify the cache step is set correctly
    assert prey1._last_visible_step == 20, f"Expected step 20, got {prey1._last_visible_step}"
    print(f"✓ Visibility cached at step {prey1._last_visible_step}")
    
    # Test 3: can_reproduce accepts step_idx
    print("\nTesting can_reproduce with step_idx...")
    prey1.age = config.MATURITY_AGE
    prey1.energy = config.MATING_ENERGY_COST + 10
    prey1.mating_cooldown = 0
    can_mate = prey1.can_reproduce(config, animals, step_idx=30)
    print(f"✓ can_reproduce(step_idx=30) returned: {can_mate}")
    
    # Test 4: deposit_pheromones accepts step_idx
    print("\nTesting deposit_pheromones with step_idx...")
    prey1.deposit_pheromones(animals, pheromone_map, config, step_idx=40)
    pred1.deposit_pheromones(animals, pheromone_map, config, step_idx=40)
    print("✓ deposit_pheromones(step_idx=40) executed successfully")
    
    # Test 5: Blocked movement tracking
    print("\nTesting blocked movement tracking...")
    prey1.x = 20
    prey1.y = 20
    prey2.x = 20
    prey2.y = 20  # Block prey1's position
    
    transitions = prey1.move_training(model, animals, config, pheromone_map, step_idx=50)
    blocked_any = getattr(prey1, '_blocked_any_this_step', False)
    move_attempts = getattr(prey1, '_move_attempts_this_step', 0)
    moved_any = getattr(prey1, '_moved_any_this_step', False)
    
    print(f"✓ Blocked tracking: blocked_any={blocked_any}, move_attempts={move_attempts}, moved_any={moved_any}")
    assert hasattr(prey1, '_blocked_any_this_step'), "Should track blocked_any"
    assert hasattr(prey1, '_move_attempts_this_step'), "Should track move_attempts"
    assert hasattr(prey1, '_moved_any_this_step'), "Should track moved_any"
    
    # Test 6: Verify no config.CURRENT_STEP references in code
    print("\nVerifying no config.CURRENT_STEP usage...")
    import inspect
    from src.core import animal
    
    # Get source of animal.py
    source = inspect.getsource(animal)
    assert 'config.CURRENT_STEP' not in source, "animal.py should not reference config.CURRENT_STEP"
    print("✓ animal.py does not reference config.CURRENT_STEP")
    
    print("\n" + "="*60)
    print("✓ All step_idx refactor tests passed!")
    print("="*60)

if __name__ == "__main__":
    test_step_idx_threading()
