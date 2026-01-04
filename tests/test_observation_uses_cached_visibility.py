"""
Test: get_enhanced_input(..., visible_animals=...) must NOT call communicate()
This validates the "avoid redundant scan" refactor
"""
import torch
import pytest

from src.config import SimulationConfig
from src.core.animal import Prey


def test_get_enhanced_input_uses_cached_visible_list(monkeypatch):
    """
    Validates that when visible_animals is provided, get_enhanced_input doesn't call communicate()
    This is critical for the performance optimization of avoiding redundant world scans
    """
    config = SimulationConfig()

    a = Prey(10, 10, "A", "#00ff00")

    # Fixed-size padding list as communicate() would return
    visible = [[0.0]*9 for _ in range(config.MAX_VISIBLE_ANIMALS)]
    visible[0][8] = 1.0  # is_present
    visible[0][3] = 1.0  # is_predator
    visible[0][4] = 0.0  # is_prey
    visible[0][0] = 0.2  # dx_norm
    visible[0][1] = 0.0  # dy_norm
    visible[0][2] = 0.3  # dist_norm

    # If communicate() gets called, that's a bug for this test
    def boom(*args, **kwargs):
        raise AssertionError("communicate() was called even though visible_animals was provided!")

    monkeypatch.setattr(a, "communicate", boom)

    obs = a.get_enhanced_input(
        animals=[a],              # doesn't matter here
        config=config,
        pheromone_map=None,
        visible_animals=visible   # cached list
    )

    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (1, config.SELF_FEATURE_DIM)
    print("✓ get_enhanced_input correctly uses cached visibility (no redundant communicate())")


if __name__ == "__main__":
    test_get_enhanced_input_uses_cached_visible_list(pytest.MonkeyPatch())
