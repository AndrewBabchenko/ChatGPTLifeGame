import torch

from src.config import SimulationConfig
from src.core.grass_field import GrassField
from src.core.animal import Prey, Predator


def test_visible_slot_includes_grass_flag():
    cfg = SimulationConfig
    cfg.GRASS_FIELD = GrassField(cfg.GRID_SIZE, cfg.GRASS_REGROW_INTERVAL)
    prey = Prey(0, 0, "A", "#00ff00")
    predator = Predator(0, cfg.GRID_SIZE - 1, "B", "#ff0000")  # directly in front via wrap-around

    # Force deterministic heading to face north (predator at y=1 wraps but within wide FOV)
    prey.heading_idx = 0
    prey.heading_dx, prey.heading_dy = prey.DIRECTIONS[prey.heading_idx]

    visible = prey.communicate([prey, predator], cfg)
    present_slots = [row for row in visible if row[8] >= 0.5]
    assert any(row[3] >= 0.5 for row in present_slots), "Predator should be visible"

    # Grass map is carried in the self observation tail (independent of animal slots)
    obs = prey.get_enhanced_input([prey, predator], cfg, None, visible_animals=visible)
    grass_vec = obs[0, 34:]  # flattened grass patch

    radius = cfg.PREY_VISION_RANGE
    diam = cfg.GRASS_PATCH_DIAMETER
    dx = 0
    dy = -1  # predator cell wraps to north
    idx = (dy + radius) * diam + (dx + radius)
    assert grass_vec[idx].item() == 1.0

    # After consuming grass at predator cell, flag should drop to 0 on recompute
    cfg.GRASS_FIELD.consume(predator.x, predator.y)
    visible_after = prey.communicate([prey, predator], cfg)
    obs_after = prey.get_enhanced_input([prey, predator], cfg, None, visible_animals=visible_after)
    grass_vec_after = obs_after[0, 34:]
    assert grass_vec_after[idx].item() == 0.0
