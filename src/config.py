"""
Configuration settings for the Life Game simulation

This file contains all tunable parameters for the predator-prey ecosystem simulation.
Animals learn through PPO reinforcement learning to hunt, evade, and reproduce.
"""


class SimulationConfig:
    """Configuration class for all simulation parameters"""
    
    # Store base config values (prevents curriculum leakage)
    _BASE = {}  # Will be populated after class definition
    
    # Step counter for cache validation (incremented by training/demo loops)
    CURRENT_STEP = 0  # Used to validate visibility cache freshness
    
    # ============================================================================
    # GRID SETTINGS
    # ============================================================================
    GRID_SIZE = 100  # Size of simulation world (100x100 = 10,000 positions)
    FIELD_MIN = 5  # Minimum spawn coordinate (creates buffer from edges)
    FIELD_MAX = 95  # Maximum spawn coordinate (usable area: 60x60 = 3,600 positions)
    
    # ============================================================================
    # POPULATION SETTINGS
    # ============================================================================
    INITIAL_PREY_COUNT = 50  # Starting prey population (species A, green)
    INITIAL_PREDATOR_COUNT = 30  # Starting predator population (INCREASED from 20 for more data)
    MAX_PREY = 200  # Hard population cap for prey (REDUCED for speed)
    MAX_PREDATORS = 60  # Hard population cap for predators (REDUCED for speed)
    # Note: Species have separate capacity limits so they don't compete for the same slots
    
    # ============================================================================
    # ANIMAL BEHAVIOR
    # ============================================================================
    PREDATOR_VISION_RANGE = 5  # Detection radius for predators (INCREASED - must be > prey for hunting success)
    PREY_VISION_RANGE = 8  # Detection radius for prey (REDUCED - prevents over-evasion)
    VISION_RANGE = 10  # Default/fallback (kept for compatibility)
    MAX_VISIBLE_ANIMALS = 24  # Maximum animals processed by neural network (sorted by distance)
    
    # Field of view angles (degrees)
    PREY_FOV_DEG = 240  # Prey have wide peripheral vision (240° = -120° to +120°)
    PREDATOR_FOV_DEG = 180  # Predators have forward-focused vision (180° = -90° to +90°)
    VISION_SHAPE = "circle"  # Vision uses circular range boundary (not square)
    
    # Hunger and starvation thresholds
    HUNGER_THRESHOLD = 80  # Energy level when predators switch to hungry mode (enter aggressive hunting)
                               # Predators are hungry when energy < 80 (80% of time after first ~40 steps)
    
    STARVATION_THRESHOLD = 100  # Steps without eating before predator dies (BALANCED - requires 2 hunts per episode)
                               # At 120: Predators must eat at least twice to survive 200 steps
                               # This gives learning time while still making hunting critical
    
    MATING_COOLDOWN = 30  # Steps required between mating attempts (INCREASED to 30 - prevent population explosions)
    
    # ============================================================================
    # MOVEMENT SPEEDS
    # ============================================================================
    PREDATOR_HUNGRY_MOVES = 2  # Tiles per step when energy < HUNGER_THRESHOLD (4x faster than prey)
    PREDATOR_NORMAL_MOVES = 1  # Tiles per step when energy >= HUNGER_THRESHOLD (2x faster than prey)
    PREY_MOVES = 1  # Tiles per step (baseline speed)
    
    # ============================================================================
    # MATING PROBABILITIES
    # ============================================================================
    MATING_PROBABILITY_PREY = 0.05  # 5% chance per step when conditions met (REDUCED for speed/control)
    MATING_PROBABILITY_PREDATOR = 0.1  # 10% chance per step when conditions met (REDUCED for speed/control)
    # Conditions: energy >= MATING_ENERGY_COST, age >= MATURITY_AGE, cooldown expired, partner adjacent
    
    # ============================================================================
    # REINFORCEMENT LEARNING SETTINGS
    # ============================================================================
    LEARNING_RATE_PREY = 0.006  # Adam optimizer (INCREASED from 5e-5 to unfreeze learning)
    LEARNING_RATE_PREDATOR = 0.0003  # Adam optimizer (INCREASED from 1e-4 to unfreeze learning)
                                    # Target: KL 0.003-0.03, ClipFrac 0.05-0.30
    
    GAMMA = 0.99  # Discount factor for future rewards (0=myopic, 1=far-sighted)
                  # At 0.99: reward in 100 steps valued at 36.6% of immediate reward
                  # Encourages long-term planning in 200-step episodes
    
    # Observation contract version (must match Animal.OBS_VERSION)
    OBS_VERSION = 2  # 34 features (added gradient magnitudes in v2)
    
    # Action temperature for exploration (higher = more random)
    ACTION_TEMPERATURE = 1.0  
    
    # Chase override (predator hardcoded hunting behavior)
    # Set to False when evaluating learned behavior
    CHASE_OVERRIDE_IN_INFERENCE = False  # Turn off to let NN control hunting
    
    # ============================================================================
    # REWARD STRUCTURE (scaled down 10x to reduce return variance)
    # ============================================================================
    # Positive rewards
    SURVIVAL_REWARD = 0.2  # Reward per step alive (encourages staying alive)
    REPRODUCTION_REWARD = 0.2  # Bonus for successful mating
    PREDATOR_EAT_REWARD = 10.0  # Reward for catching prey (main predator objective)
    PREY_EVASION_REWARD = 5  #0.5       # Reward per step for maintaining distance from predators
    PREDATOR_APPROACH_REWARD = 0.01  # Reward per step for closing distance to prey
    PREY_MATE_APPROACH_REWARD = 0.000  # Reward per step for prey approaching mates when safe (STEP 2 - dense shaping)
    
    # Negative penalties (punish unsuccessful behavior, scaled down 10x)
    EXTINCTION_PENALTY = -100.0  # Massive penalty if species goes extinct
    DEATH_PENALTY = -5.0  # Penalty when animal dies (any cause)
    STARVATION_PENALTY = -10.0  # Extra penalty for predator dying from starvation
    EATEN_PENALTY = -30.0  # Heavy penalty for prey being eaten
    EXHAUSTION_PENALTY = -7.5  # Penalty for running out of energy
    OLD_AGE_PENALTY = -2.0  # Small penalty for dying of old age (natural)
    OVERPOPULATION_PENALTY = -1.0  # Penalty per step when exceeding reserved capacity
    
    # Mating behavior thresholds
    PREY_MATING_ENERGY_THRESHOLD = 60.0  # Minimum energy for prey to seek mates (prevents desperate mating)
    PREY_SAFE_MATING_DISTANCE = 15.0     # Distance from nearest predator to consider "safe" for mating
    PREY_FLEE_SUPERVISION_DIST = 0.35    # Normalized distance threshold for flee supervision (only flee if predator this close)
    
    # ============================================================================
    # ENERGY SYSTEM
    # ============================================================================
    INITIAL_ENERGY = 100.0  # Starting energy for all animals
    MAX_ENERGY = 100.0  # Energy cap (eating when full provides no benefit)
    
    # Energy costs and gains
    ENERGY_DECAY_RATE = 0.2  # Base metabolism drain per step (even when resting)
    MOVE_ENERGY_COST = 0.2  # Additional drain per step when moving (REDUCED to 0.2 - prey can run longer)
                              # Total drain when moving: 0.4/step = 250 steps to drain 100 energy
    
    MATING_ENERGY_COST = 5.0  # Energy spent when mating succeeds (REDUCED to 5.0 - easier to trigger mating)
    EATING_ENERGY_GAIN = 80.0  # Energy gained by predator when eating prey (INCREASED from 50.0)
    REST_ENERGY_GAIN = 2.0  # Energy gained per step when not moving (allows recovery)
    
    # ============================================================================
    # AGE SYSTEM
    # ============================================================================
    MAX_AGE = 1000  # Maximum lifespan in steps (die of old age at 1000 steps = 5 episodes)
    MATURITY_AGE = 20  # Age when reproduction becomes possible (REDUCED to 10 - faster mating opportunities)
    EXPERIENCE_GAIN_RATE = 0.1  # (Currently unused - reserved for future age-based learning boost)
    
    # ============================================================================
    # PHEROMONE SYSTEM
    # ============================================================================
    PHEROMONE_DECAY = 0.95  # Fade rate per step (0.95^10 = 59.9% after 10 steps)
    PHEROMONE_DIFFUSION = 0.1  # Spread rate to adjacent 8 cells (creates gradients)
    DANGER_PHEROMONE_STRENGTH = 0.8  # Initial intensity when prey flee predators
    MATING_PHEROMONE_STRENGTH = 0.6  # Initial intensity when seeking mates
    PHEROMONE_SENSING_RANGE = 5  # Detection radius for pheromone gradients
    
    # ============================================================================
    # PPO TRAINING PARAMETERS
    # ============================================================================
    NUM_EPISODES = 20  # Total training episodes
    STEPS_PER_EPISODE = 200  # Steps per episode
    
    PPO_EPOCHS = 8  # Training passes per episode (INCREASED from 1 to enable learning)
    PPO_CLIP_EPSILON = 0.1  # Policy update limiter (REDUCED from 0.2 for tighter control)
    PPO_BATCH_SIZE = 4096  # Mini-batch size (INCREASED from 1024 to reduce optimizer steps per episode)
    
    VALUE_LOSS_COEF = 0.1  # Weight of value loss in total loss function (REDUCED from 0.5 to prevent critic dominance)
                             # total_loss = policy_loss + 0.1*value_loss + 0.03*entropy_loss
    
    ENTROPY_COEF = 0.01  # Exploration bonus
    MAX_GRAD_NORM = 0.5  # Gradient clipping threshold (prevents training instability)
    GAE_LAMBDA = 0.95  # Generalized Advantage Estimation parameter (bias-variance tradeoff)
    
    # ============================================================================
    # CURRICULUM LEARNING CONFIGURATION
    # ============================================================================
    # Progressive difficulty stages to improve predator hunting skills
    # Training gradually transitions from easy (abundant prey, no death) to realistic conditions
    
    CURRICULUM_ENABLED = False  # Set to True to enable prey mating curriculum
    
    # Define curriculum stages (list of dicts with episode ranges and overrides)
    CURRICULUM_STAGES = [
        {
            'name': 'Stage 1: Prey Mating Curriculum (reduced predator pressure)',
            'episodes': (1, 5),  # Episodes 1-5 for prey to learn mating behavior
            'description': 'Reduce predator pressure so prey can explore mating without constant threat',
            'overrides': {
                'INITIAL_PREDATOR_COUNT': 20,        # Reduced from 30 (STEP 5)
                'PREDATOR_HUNGRY_MOVES': 2,          # Reduced from 3 (STEP 5 - slower predators)
                'PREY_VISION_RANGE': 9,              # Increased from 7 (STEP 5 - better perception)
                'REPRODUCTION_REWARD': 2.0,          # Keep mating incentive
                'MATING_PROBABILITY_PREY': 0.15,     # Keep standard probability
            }
        },
        {
            'name': 'Stage 2: Normal Configuration',
            'episodes': (6, None),  # Episode 6 onwards (None = unlimited)
            'description': 'Restore full predator pressure with learned mating behavior.',
            'overrides': {
                # Empty = restore all base config values (no overrides)
            }
        }
    ]
    
    # Current stage tracking (will be updated during training)
    CURRENT_CURRICULUM_STAGE = 0  # Index into CURRICULUM_STAGES
    
    # Helper flag to disable starvation death (used by Stage 1)
    STARVATION_ENABLED = True  # Set to False to prevent predators from dying of starvation
    
    # ============================================================================
    # CURRICULUM LEARNING METHODS
    # ============================================================================
    
    @classmethod
    def get_curriculum_stage(cls, episode):
        """
        Get the curriculum stage configuration for a given episode number.
        
        Args:
            episode (int): Current episode number (1-indexed)
            
        Returns:
            dict: Stage configuration with name, description, and parameter overrides
            int: Stage index (0-indexed)
        """
        if not cls.CURRICULUM_ENABLED:
            return None, -1
        
        for idx, stage in enumerate(cls.CURRICULUM_STAGES):
            start_ep, end_ep = stage['episodes']
            if end_ep is None:  # Last stage (unlimited)
                if episode >= start_ep:
                    return stage, idx
            elif start_ep <= episode <= end_ep:
                return stage, idx
        
        # If no stage matches, return the last stage
        return cls.CURRICULUM_STAGES[-1], len(cls.CURRICULUM_STAGES) - 1
    
    @classmethod
    def apply_curriculum_stage(cls, episode):
        """
        Apply curriculum stage settings for the given episode.
        Updates class attributes with stage-specific overrides.
        
        Args:
            episode (int): Current episode number (1-indexed)
            
        Returns:
            dict: Applied stage configuration (or None if curriculum disabled)
        """
        if not cls.CURRICULUM_ENABLED:
            return None
        
        stage, stage_idx = cls.get_curriculum_stage(episode)
        
        if stage is None:
            return None
        
        # Update current stage tracker
        cls.CURRENT_CURRICULUM_STAGE = stage_idx
        
        # CRITICAL: Reset to base values first (prevents curriculum leakage)
        for k, v in cls._BASE.items():
            setattr(cls, k, v)
        
        # Apply overrides from this stage
        overrides = stage.get('overrides', {})
        for key, value in overrides.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
        
        return stage
    
    @classmethod
    def get_curriculum_info(cls, episode):
        """
        Get human-readable curriculum stage information for logging.
        
        Args:
            episode (int): Current episode number (1-indexed)
            
        Returns:
            str: Formatted string with stage info, or empty string if disabled
        """
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
