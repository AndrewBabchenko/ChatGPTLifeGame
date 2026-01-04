"""Configuration settings for Life Game - PHASE 1: Pure Hunt/Evade

CURRICULUM PHASE 1 OF 4
=======================
Goal: Learn core hunting and evasion behaviors ONLY.
- NO starvation deaths (predators can't die from hunger)
- NO mating (population stays fixed)
- NO grass foraging rewards (prey focus on evasion)

This is the foundation phase. Predators learn to chase, prey learn to flee.
Run this for ~50 episodes before moving to Phase 2.

Usage:
    from src.config_phase1 import SimulationConfig
"""

# ============================================================================
# PHASE MANAGEMENT - Configure checkpoint loading and phase transitions
# ============================================================================
PHASE_NUMBER = 1
PHASE_NAME = "Pure Hunt/Evade"

# Checkpoint loading: Set to None for fresh start, or specify explicit checkpoint files
# Example: "outputs/checkpoints/phase0_ep10_model_A.pth"
LOAD_PREY_CHECKPOINT = None  # Phase 1 starts fresh
LOAD_PREDATOR_CHECKPOINT = None  # Phase 1 starts fresh

# Checkpoint save prefix (will save as {prefix}_model_A.pth and {prefix}_model_B.pth)
SAVE_CHECKPOINT_PREFIX = "phase1"  # Saves to outputs/checkpoints/phase1_*

# Early stopping: Move to next phase if metric plateaus for N episodes
EARLY_STOP_PATIENCE = 15  # Episodes without improvement before early stop
EARLY_STOP_MIN_EPISODES = 30  # Minimum episodes before early stopping allowed


def _build_grass_patch_offsets(radius: int, diameter: int):
    """Precompute grass patch offsets without relying on class-scope comprehensions."""
    return [
        (dx, dy, (dy + radius) * diameter + (dx + radius))
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]


class SimulationConfig:
    """Phase 1 Configuration - Pure Hunt/Evade (No Starvation, No Mating)"""
    
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
    # POPULATION SETTINGS - FIXED (NO BIRTHS/DEATHS FROM STARVATION)
    # ============================================================================
    INITIAL_PREY_COUNT = 40  # More targets for predators to learn hunting
    INITIAL_PREDATOR_COUNT = 20  # Fewer predators = more encounters per predator
    MAX_PREY = 200
    MAX_PREDATORS = 40
    
    # ============================================================================
    # ANIMAL BEHAVIOR - VISION TUNED FOR ENCOUNTERS
    # ============================================================================
    PREDATOR_VISION_RANGE = 8  # High visibility for learning
    PREY_VISION_RANGE = 5
    VISION_RANGE = 5
    MAX_VISIBLE_ANIMALS = 24
    
    # Field of view angles (degrees)
    PREY_FOV_DEG = 240
    PREDATOR_FOV_DEG = 180
    VISION_SHAPE = "circle"
    
    # Hunger and starvation thresholds
    HUNGER_THRESHOLD = 80
    
    # PHASE 1: DISABLE STARVATION - predators can't die from hunger
    STARVATION_THRESHOLD = 99999  # Effectively infinite - no starvation deaths
    
    MATING_COOLDOWN = 30
    
    # ============================================================================
    # MOVEMENT SPEEDS
    # ============================================================================
    PREDATOR_HUNGRY_MOVES = 2
    PREDATOR_NORMAL_MOVES = 1
    PREY_MOVES = 1
    
    # ============================================================================
    # MATING PROBABILITIES - PHASE 1: DISABLED
    # ============================================================================
    MATING_PROBABILITY_PREY = 0.0  # NO MATING - population stays fixed
    MATING_PROBABILITY_PREDATOR = 0.0  # NO MATING - population stays fixed
    
    # ============================================================================
    # REINFORCEMENT LEARNING SETTINGS
    # ============================================================================
    LEARNING_RATE_PREY = 0.0001
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
    # REWARD STRUCTURE - PHASE 1: HUNT/EVADE ONLY
    # ============================================================================
    # Positive rewards - CORE BEHAVIORS ONLY
    SURVIVAL_REWARD = 0.2  # Keep alive bonus
    REPRODUCTION_REWARD = 0.0  # DISABLED - no mating rewards
    
    # Predator rewards - HUNTING FOCUS
    PREDATOR_EAT_REWARD = 30.0  # Main predator objective
    PREDATOR_APPROACH_REWARD = 0.8  # Dense gradient for chasing
    # Predator search shaping (set to 0.0 to disable per phase)
    PREDATOR_DETECTION_BONUS = 0.05  # Bonus when prey becomes visible
    PREDATOR_VISIBLE_REWARD = 0.01  # Small reward per step with prey in view
    PREDATOR_COVERAGE_BONUS = 0.01  # Reward for new tiles when no prey visible
    PREDATOR_REVISIT_PENALTY = -0.005  # Penalty for revisiting recent tiles
    PREDATOR_IDLE_PENALTY = -0.01  # Penalty for not moving when no prey visible
    PREDATOR_COVERAGE_WINDOW = 12  # Recent-step window for coverage tracking
    
    # Prey rewards
    PREY_EVASION_SCALE_CELLS = 8 * 0.5
    PREY_EVASION_PENALTY_DIST_CELLS = 8 * 0.6
    PREY_MATE_APPROACH_REWARD = 0.0  # DISABLED - no mating
    PREY_MATE_APPROACH_SCALE_CELLS = 8 * 0.4
    PREY_MATE_SAFE_DIST_CELLS = 8 * 0.8
    
    # Negative penalties - SIMPLIFIED
    EXTINCTION_PENALTY = -100.0
    DEATH_PENALTY = -5.0  # Only from being eaten (no starvation)
    STARVATION_PENALTY = 0.0  # DISABLED - no starvation
    EATEN_PENALTY = -20.0  # Main penalty for prey
    EXHAUSTION_PENALTY = 0.0  # DISABLED - simplify energy
    OLD_AGE_PENALTY = -2.0
    OVERPOPULATION_PENALTY = 0.0  # DISABLED - no population dynamics
    
    # Mating behavior thresholds (not used in Phase 1)
    PREY_MATING_ENERGY_THRESHOLD = 60.0
    PREY_SAFE_MATING_DISTANCE = 15.0
    PREY_FLEE_SUPERVISION_DIST = 0.999
    PREY_MATING_SAFE_STEPS = 10
    
    # Safe mating rules
    PREY_SAFE_TO_MATE_DIST_NORM = 0.95
    PREY_BLOCK_MATING_IF_THREAT = True
    
    # Threat presence penalty - ACTIVE (core evasion behavior)
    PREY_THREAT_PRESENCE_PENALTY = -0.15
    PREY_THREAT_PRESENCE_POWER = 1.5
    PREY_THREAT_VISIBLE_EPS = 0.999
    
    # Blocked movement penalty
    PREY_BLOCKED_UNDER_THREAT_PENALTY = -0.5
    
    # Grass regrowth
    GRASS_REGROW_PROB = 0.05
    
    # Evasion reward shaping - CORE BEHAVIOR
    PREY_EVASION_REWARD = 2.5  # Main prey objective
    PREY_EVASION_PENALTY = 1.0
    PREY_EVASION_SCALE_CELLS = 8 * 0.5
    PREY_EVASION_PENALTY_DIST_CELLS = 8 * 0.6
    
    # Directional loss coefficients
    DIRECTIONAL_LOSS_COEF_PREY = 2.0
    DIRECTIONAL_LOSS_COEF_PREDATOR = 2.0
    
    # ============================================================================
    # ENERGY SYSTEM - SIMPLIFIED (no exhaustion deaths)
    # ============================================================================
    INITIAL_ENERGY = 100.0
    MAX_ENERGY = 100.0
    
    ENERGY_DECAY_RATE = 0.0  # DISABLED - no energy drain
    MOVE_ENERGY_COST = 0.0  # DISABLED - free movement
    
    MATING_ENERGY_COST = 0.0  # DISABLED
    EATING_ENERGY_GAIN = 50.0  # Still gain energy from eating (for future phases)
    REST_ENERGY_GAIN = 0.0

    # Grass feeding - DISABLED
    PREY_HUNGER_THRESHOLD = 80.0
    GRASS_ENERGY = 1.0
    GRASS_EAT_REWARD = 0.0  # DISABLED - no foraging reward
    GRASS_HUNGER_PENALTY = 0.0  # DISABLED - no hunger penalty
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
    NUM_EPISODES = 200  # Phase 1 duration
    STEPS_PER_EPISODE = 300
    
    PPO_EPOCHS = 6
    PPO_CLIP_EPSILON = 0.15
    PPO_BATCH_SIZE = 2048
    
    VALUE_LOSS_COEF = 0.25
    ENTROPY_COEF = 0.04  # High exploration for learning basics
    MAX_GRAD_NORM = 0.3
    GAE_LAMBDA = 0.92
    
    # ============================================================================
    # CURRICULUM LEARNING CONFIGURATION
    # ============================================================================
    CURRICULUM_ENABLED = False
    
    CURRICULUM_STAGES = []
    
    CURRENT_CURRICULUM_STAGE = 0
    STARVATION_ENABLED = False  # PHASE 1: Starvation disabled
    
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
