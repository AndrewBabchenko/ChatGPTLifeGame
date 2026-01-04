"""Configuration settings for Life Game - PHASE 2: Add Energy & Starvation

CURRICULUM PHASE 2 OF 4
=======================
Goal: Add survival pressure - predators must hunt to survive.
- ENABLED: Starvation deaths (predators die if they don't eat)
- ENABLED: Energy system (movement costs energy)
- DISABLED: Mating (population still fixed)
- DISABLED: Grass foraging rewards

Builds on Phase 1 hunting/evasion. Now predators learn hunting is NECESSARY.
Run this for ~50 episodes after Phase 1, before moving to Phase 3.

Usage:
    from src.config_phase2 import SimulationConfig
"""

# ============================================================================
# PHASE MANAGEMENT - Configure checkpoint loading and phase transitions
# ============================================================================
PHASE_NUMBER = 2
PHASE_NAME = "Add Starvation"

# Checkpoint loading: Load from Phase 1
LOAD_PREY_CHECKPOINT = "outputs/checkpoints/phase1_ep192_model_A.pth"
LOAD_PREDATOR_CHECKPOINT = "outputs/checkpoints/phase1_ep181_model_B.pth"

# Checkpoint save prefix
SAVE_CHECKPOINT_PREFIX = "phase2"

# Early stopping
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_EPISODES = 30


def _build_grass_patch_offsets(radius: int, diameter: int):
    """Precompute grass patch offsets without relying on class-scope comprehensions."""
    return [
        (dx, dy, (dy + radius) * diameter + (dx + radius))
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]


class SimulationConfig:
    """Phase 2 Configuration - Hunt/Evade + Starvation (No Mating)"""
    
    # Store base config values (prevents curriculum leakage)
    _BASE = {}  # Will be populated after class definition
    
    # Step counter for cache validation (incremented by training/demo loops)
    CURRENT_STEP = 0  # Used to validate visibility cache freshness
    
    # Debug assertions (enable for development, disable for production)
    DEBUG_ASSERTIONS = False
    
    # ============================================================================
    # GRID SETTINGS
    # ============================================================================
    GRID_SIZE = 100
    FIELD_MIN = 5
    FIELD_MAX = 95
    
    # ============================================================================
    # POPULATION SETTINGS - STILL FIXED (NO BIRTHS)
    # ============================================================================
    INITIAL_PREY_COUNT = 40
    INITIAL_PREDATOR_COUNT = 20
    MAX_PREY = 200
    MAX_PREDATORS = 40
    
    # ============================================================================
    # ANIMAL BEHAVIOR - VISION TUNED FOR ENCOUNTERS
    # ============================================================================
    PREDATOR_VISION_RANGE = 8
    PREY_VISION_RANGE = 5
    VISION_RANGE = 5
    MAX_VISIBLE_ANIMALS = 24
    
    # Field of view angles (degrees)
    PREY_FOV_DEG = 240
    PREDATOR_FOV_DEG = 180
    VISION_SHAPE = "circle"
    
    # Hunger and starvation thresholds
    HUNGER_THRESHOLD = 80
    
    # PHASE 2: ENABLE STARVATION - predators must hunt to survive
    STARVATION_THRESHOLD = 150  # Generous threshold for learning
    
    MATING_COOLDOWN = 30
    
    # ============================================================================
    # MOVEMENT SPEEDS
    # ============================================================================
    PREDATOR_HUNGRY_MOVES = 2
    PREDATOR_NORMAL_MOVES = 1
    PREY_MOVES = 1
    
    # ============================================================================
    # MATING PROBABILITIES - PHASE 2: STILL DISABLED
    # ============================================================================
    MATING_PROBABILITY_PREY = 0.0  # NO MATING - population stays fixed
    MATING_PROBABILITY_PREDATOR = 0.0  # NO MATING - population stays fixed
    
    # ============================================================================
    # REINFORCEMENT LEARNING SETTINGS
    # ============================================================================
    LEARNING_RATE_PREY = 0.00008
    LEARNING_RATE_PREDATOR = 0.0001
    
    GAMMA = 0.97
    
    # Observation contract version (must match Animal.OBS_VERSION)
    OBS_VERSION = 5
    OBS_HISTORY_LEN = 10  # Number of self-state frames stacked (1 = no memory)
    
    # Validate history length
    assert 1 <= OBS_HISTORY_LEN <= 50, f"OBS_HISTORY_LEN={OBS_HISTORY_LEN} must be between 1 and 50"

    # Grass sensing dimensions
    GRASS_PATCH_DIAMETER = 17
    GRASS_PATCH_SIZE = 289
    GRASS_PATCH_OFFSETS = _build_grass_patch_offsets(PREY_VISION_RANGE, GRASS_PATCH_DIAMETER)
    GRASS_PATCH_ZERO = [0.0] * GRASS_PATCH_SIZE
    BASE_SELF_FEATURE_DIM = 34 + GRASS_PATCH_SIZE
    SELF_FEATURE_DIM = BASE_SELF_FEATURE_DIM * OBS_HISTORY_LEN
    
    # Action temperature for exploration
    ACTION_TEMPERATURE = 1  
    TURN_STRAIGHT_BIAS = 1.0
    CHASE_OVERRIDE_IN_INFERENCE = False
    
    # ============================================================================
    # REWARD STRUCTURE - PHASE 2: ADD STARVATION PRESSURE
    # ============================================================================
    # Positive rewards
    SURVIVAL_REWARD = 0.2
    REPRODUCTION_REWARD = 0.0  # DISABLED - no mating rewards
    
    # Predator rewards - HUNTING FOCUS
    PREDATOR_EAT_REWARD = 30.0
    PREDATOR_APPROACH_REWARD = 0.8
    PREDATOR_DETECTION_BONUS = 0.05  # Bonus when prey becomes visible
    PREDATOR_VISIBLE_REWARD = 0.01  # Small reward per step with prey in view
    PREDATOR_COVERAGE_BONUS = 0.01  # Reward for moving to a new tile when no prey visible
    PREDATOR_REVISIT_PENALTY = -0.005  # Penalty for revisiting recent tiles
    PREDATOR_IDLE_PENALTY = -0.01  # Penalty for not moving when no prey visible
    PREDATOR_COVERAGE_WINDOW = 12  # Recent-step window for coverage tracking
    
    # Prey rewards
    PREY_EVASION_SCALE_CELLS = 8 * 0.5
    PREY_EVASION_PENALTY_DIST_CELLS = 8 * 0.6
    PREY_MATE_APPROACH_REWARD = 0.0  # DISABLED - no mating
    PREY_MATE_APPROACH_SCALE_CELLS = 8 * 0.4
    PREY_MATE_SAFE_DIST_CELLS = 8 * 0.8
    
    # Negative penalties - ADD STARVATION
    EXTINCTION_PENALTY = -100.0
    DEATH_PENALTY = -5.0
    STARVATION_PENALTY = -5.0  # ENABLED - penalty for starving
    EATEN_PENALTY = -20.0
    EXHAUSTION_PENALTY = -7.5  # ENABLED - penalty for exhaustion
    OLD_AGE_PENALTY = -2.0
    OVERPOPULATION_PENALTY = 0.0  # DISABLED - no population dynamics yet
    
    # Mating behavior thresholds (not used in Phase 2)
    PREY_MATING_ENERGY_THRESHOLD = 60.0
    PREY_SAFE_MATING_DISTANCE = 15.0
    PREY_FLEE_SUPERVISION_DIST = 0.999
    PREY_MATING_SAFE_STEPS = 10
    
    # Safe mating rules
    PREY_SAFE_TO_MATE_DIST_NORM = 0.95
    PREY_BLOCK_MATING_IF_THREAT = True
    
    # Threat presence penalty
    PREY_THREAT_PRESENCE_PENALTY = -0.15
    PREY_THREAT_PRESENCE_POWER = 1.5
    PREY_THREAT_VISIBLE_EPS = 0.999
    
    # Blocked movement penalty
    PREY_BLOCKED_UNDER_THREAT_PENALTY = -0.5
    
    # Grass regrowth
    GRASS_REGROW_PROB = 0.05
    
    # Evasion reward shaping
    PREY_EVASION_REWARD = 2.5
    PREY_EVASION_PENALTY = 1.0
    PREY_EVASION_SCALE_CELLS = 8 * 0.5
    PREY_EVASION_PENALTY_DIST_CELLS = 8 * 0.6
    
    # Directional loss coefficients
    DIRECTIONAL_LOSS_COEF_PREY = 2.0
    DIRECTIONAL_LOSS_COEF_PREDATOR = 2.0
    
    # ============================================================================
    # ENERGY SYSTEM - ENABLED (survival pressure)
    # ============================================================================
    INITIAL_ENERGY = 100.0
    MAX_ENERGY = 100.0
    
    ENERGY_DECAY_RATE = 0.2  # ENABLED - base metabolism drain
    MOVE_ENERGY_COST = 0.2  # ENABLED - movement costs energy
    
    MATING_ENERGY_COST = 0.0  # DISABLED - no mating
    EATING_ENERGY_GAIN = 50.0  # Energy from successful hunt
    REST_ENERGY_GAIN = 0.5  # Can recover by resting

    # Grass feeding - STILL DISABLED
    PREY_HUNGER_THRESHOLD = 80.0
    GRASS_ENERGY = 1.0
    GRASS_EAT_REWARD = 0.0  # DISABLED - no foraging reward
    GRASS_HUNGER_PENALTY = 0.0  # DISABLED - no hunger penalty for prey
    GRASS_REGROW_INTERVAL = 30
    
    # ============================================================================
    # AGE SYSTEM
    # ============================================================================
    MAX_AGE = 1000
    MATURITY_AGE = 40
    EXPERIENCE_GAIN_RATE = 0.1
    
    # ============================================================================
    # PHEROMONE SYSTEM
    # ============================================================================
    PHEROMONE_DECAY = 0.95
    PHEROMONE_DIFFUSION = 0.1
    DANGER_PHEROMONE_STRENGTH = 0.8
    MATING_PHEROMONE_STRENGTH = 0.0  # DISABLED - no mating pheromones
    PHEROMONE_SENSING_RANGE = 5
    
    # ============================================================================
    # PPO TRAINING PARAMETERS
    # ============================================================================
    NUM_EPISODES = 50  # Phase 2 duration
    STEPS_PER_EPISODE = 300
    
    PPO_EPOCHS = 6
    PPO_CLIP_EPSILON = 0.15
    PPO_BATCH_SIZE = 2048
    
    VALUE_LOSS_COEF = 0.25
    ENTROPY_COEF = 0.04
    MAX_GRAD_NORM = 0.3
    GAE_LAMBDA = 0.92
    
    # ============================================================================
    # CURRICULUM LEARNING CONFIGURATION
    # ============================================================================
    CURRICULUM_ENABLED = False
    
    CURRICULUM_STAGES = []
    
    CURRENT_CURRICULUM_STAGE = 0
    STARVATION_ENABLED = True  # PHASE 2: Starvation enabled
    
    # ============================================================================
    # CURRICULUM LEARNING METHODS
    # ============================================================================
    
    @classmethod
    def get_curriculum_stage(cls, episode):
        """Get the curriculum stage configuration for a given episode number."""
        if not cls.CURRICULUM_ENABLED:
            return None, -1
        for idx, stage in enumerate(cls.CURRICULUM_STAGES):
            start_ep, end_ep = stage['episodes']
            if end_ep is None:
                if episode >= start_ep:
                    return stage, idx
            elif start_ep <= episode <= end_ep:
                return stage, idx
        return cls.CURRICULUM_STAGES[-1], len(cls.CURRICULUM_STAGES) - 1
    
    @classmethod
    def apply_curriculum_stage(cls, episode):
        """Apply curriculum stage settings for the given episode."""
        if not cls.CURRICULUM_ENABLED:
            return None
        stage, stage_idx = cls.get_curriculum_stage(episode)
        if stage is None:
            return None
        cls.CURRENT_CURRICULUM_STAGE = stage_idx
        for k, v in cls._BASE.items():
            setattr(cls, k, v)
        overrides = stage.get('overrides', {})
        for key, value in overrides.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
        return stage
    
    @classmethod
    def get_curriculum_info(cls, episode):
        """Get human-readable curriculum stage information for logging."""
        if not cls.CURRICULUM_ENABLED:
            return ""
        stage, stage_idx = cls.get_curriculum_stage(episode)
        if stage is None:
            return ""
        start_ep, end_ep = stage['episodes']
        end_str = str(end_ep) if end_ep is not None else "inf"
        info = f"\n{'='*70}\n"
        info += f"CURRICULUM STAGE {stage_idx + 1}/{len(cls.CURRICULUM_STAGES)}: {stage['name']}\n"
        info += f"Episodes: {start_ep}-{end_str}\n"
        info += f"Description: {stage['description']}\n"
        overrides = stage.get('overrides', {})
        if overrides:
            info += f"Active Overrides:\n"
            for key, value in overrides.items():
                info += f"  - {key}: {value}\n"
        info += f"{'='*70}\n"
        return info


# Populate _BASE with all uppercase class attributes (base config values)
SimulationConfig._BASE = {k: v for k, v in vars(SimulationConfig).items() if k.isupper()}
