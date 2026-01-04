"""
Test: Turning actually changes visibility (core hierarchical-policy contract)
Validates turn → heading/FOV changes → communicate() changes
"""
from src.config import SimulationConfig
from src.core.animal import Predator, Prey


def test_turning_changes_visibility_cone_fov():
    """
    Validates that the hierarchical turn→move policy is meaningful:
    - Turning changes heading
    - Heading changes which animals are visible (FOV cone)
    - This is the core contract of why we have a turn policy
    """
    config = SimulationConfig()
    config.PREDATOR_VISION_RANGE = 20
    config.PREDATOR_FOV_DEG = 60  # narrow so turning matters

    predator = Predator(50, 50, "B", "#ff0000")
    prey = Prey(55, 50, "A", "#00ff00")  # due east, within range

    # Force predator heading North (0)
    predator.heading_idx = 0
    predator.heading_dx, predator.heading_dy = predator.DIRECTIONS[predator.heading_idx]

    animals = [predator, prey]

    # Initially: heading N, prey is E -> outside narrow 60° cone => not visible
    vis0 = predator.communicate(animals, config)
    assert not any(row[4] >= 0.5 for row in vis0), "Prey should NOT be visible before turning"

    # Turn right twice: N -> NE -> E
    predator.apply_turn_action(predator.TURN_RIGHT)
    vis1 = predator.communicate(animals, config)
    assert not any(row[4] >= 0.5 for row in vis1), "Prey should still NOT be visible at NE with narrow cone"

    predator.apply_turn_action(predator.TURN_RIGHT)
    vis2 = predator.communicate(animals, config)

    # Now prey should be visible
    assert any(row[4] >= 0.5 for row in vis2), "Prey should be visible after turning to face East"
    first_present = next(row for row in vis2 if row[4] >= 0.5)
    
    print("✓ Turning correctly changes FOV and visibility")


if __name__ == "__main__":
    test_turning_changes_visibility_cone_fov()
