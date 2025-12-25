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
    
    # Reinforcement Learning settings
    LEARNING_RATE_PREY = 0.001
    LEARNING_RATE_PREDATOR = 0.001
    GAMMA = 0.99
    
    # Reward structure
    SURVIVAL_REWARD = 0.2
    REPRODUCTION_REWARD = 10.0
    EXTINCTION_PENALTY = -1000.0
    PREDATOR_EAT_REWARD = 15.0
    OVERPOPULATION_PENALTY = -10.0
    PREY_EVASION_REWARD = 5.0  # Reward for prey staying away from predators
    PREDATOR_APPROACH_REWARD = 2.0  # Reward for predators approaching prey
    
    # Advanced features
    # Energy system
    INITIAL_ENERGY = 100.0
    MAX_ENERGY = 100.0
    ENERGY_DECAY_RATE = 0.5  # Energy lost per step
    MOVE_ENERGY_COST = 1.0  # Additional cost per move
    MATING_ENERGY_COST = 15.0  # Energy cost to mate
    EATING_ENERGY_GAIN = 30.0  # Energy gained by predator when eating
    REST_ENERGY_GAIN = 2.0  # Energy gained when resting (not moving)
    
    # Age system
    MAX_AGE = 1000  # Maximum lifespan
    MATURITY_AGE = 50  # Age when can start mating
    EXPERIENCE_GAIN_RATE = 0.1  # How fast animals learn
    
    # Pheromone system
    PHEROMONE_DECAY = 0.95  # How fast pheromones fade (per step)
    PHEROMONE_DIFFUSION = 0.1  # How much pheromones spread
    DANGER_PHEROMONE_STRENGTH = 0.8  # Strength when leaving danger signal
    MATING_PHEROMONE_STRENGTH = 0.6  # Strength when seeking mates
    PHEROMONE_SENSING_RANGE = 5  # How far animals sense pheromones
    
    # PPO Training parameters
    PPO_EPOCHS = 4  # Number of epochs per update
    PPO_CLIP_EPSILON = 0.2  # PPO clipping parameter
    PPO_BATCH_SIZE = 64  # Mini-batch size
    VALUE_LOSS_COEF = 0.5  # Coefficient for value loss
    ENTROPY_COEF = 0.01  # Coefficient for entropy bonus
    MAX_GRAD_NORM = 0.5  # Gradient clipping
    GAE_LAMBDA = 0.95  # GAE lambda parameter
