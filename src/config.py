"""
Configuration settings for the Life Game simulation
"""


class SimulationConfig:
    """Configuration class for all simulation parameters"""
    # Grid settings
    GRID_SIZE = 100
    FIELD_MIN = 20
    FIELD_MAX = 80
    
    # Population settings
    INITIAL_PREY_COUNT = 120
    INITIAL_PREDATOR_COUNT = 10
    MAX_ANIMALS = 400
    
    # Animal behavior
    VISION_RANGE = 8
    MAX_VISIBLE_ANIMALS = 15
    HUNGER_THRESHOLD = 30
    STARVATION_THRESHOLD = 60
    MATING_COOLDOWN = 15
    
    # Movement
    PREDATOR_HUNGRY_MOVES = 2
    PREDATOR_NORMAL_MOVES = 1
    PREY_MOVES = 1
    
    # Mating probabilities
    MATING_PROBABILITY_PREY = 0.9
    MATING_PROBABILITY_PREDATOR = 0.15
