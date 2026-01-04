"""
Checkpoint Evaluation: Measure what models have learned through environment rollouts

This module evaluates trained checkpoints by running real environment simulations
and tracking event-based metrics (prey escapes, predator hunts) to quantify learning.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import SimulationConfig
from src.core.animal import Prey, Predator, Animal
from src.core.grass_field import GrassField
from src.core.pheromone_system import PheromoneMap
from src.models.actor_critic_network import ActorCriticNetwork


@dataclass
class PreyDetectionEvent:
    """Track when prey detects predator and attempts escape"""
    step: int
    prey_id: int
    predator_id: int
    initial_distance: float
    # Outcomes (filled in later)
    alive_after_T: bool = False
    distance_after_1: Optional[float] = None
    distance_after_5: Optional[float] = None
    escape_success: bool = False


@dataclass
class PredatorDetectionEvent:
    """Track when predator detects prey and attempts capture"""
    step: int
    predator_id: int
    prey_id: int
    initial_distance: float
    # Outcomes (filled in later)
    capture_success: bool = False
    steps_to_capture: Optional[int] = None


@dataclass
class EvalMetrics:
    """Aggregate metrics from evaluation episode"""
    episode: int
    checkpoint_episode: int
    
    # Population metrics
    final_prey_count: int
    final_predator_count: int
    prey_deaths: int
    predator_deaths: int
    prey_births: int
    predator_births: int
    
    # Prey escape metrics
    prey_detection_events: int
    prey_escape_rate: float
    prey_dist_gain_1_mean: float
    prey_dist_gain_5_mean: float
    
    # Predator hunt metrics
    predator_detection_events: int
    predator_capture_rate: float
    predator_time_to_capture_median: float
    
    # Survival metrics
    prey_starvation_deaths: int
    predator_starvation_deaths: int
    predator_meals_total: int
    predator_meals_per_alive: float


def load_checkpoint(checkpoint_path: str, config: SimulationConfig, device: torch.device) -> ActorCriticNetwork:
    """Load a model checkpoint"""
    model = ActorCriticNetwork(config)
    model.to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()  # Set to evaluation mode
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return model


def create_animals(config: SimulationConfig) -> List[Animal]:
    """Create initial population of animals"""
    animals = []
    
    # Create prey
    for i in range(config.INITIAL_PREY_COUNT):
        x = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        y = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        prey = Prey(x, y, 'A', (0, 255, 0))
        prey.energy = config.INITIAL_ENERGY
        animals.append(prey)
    
    # Create predators
    for i in range(config.INITIAL_PREDATOR_COUNT):
        x = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        y = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        predator = Predator(x, y, 'B', (255, 0, 0))
        predator.energy = config.INITIAL_ENERGY
        animals.append(predator)
    
    return animals


def compute_toroidal_distance(x1: int, y1: int, x2: int, y2: int, grid_size: int) -> float:
    """Compute toroidal distance between two points"""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    # Wrap around if shorter
    if dx > grid_size / 2:
        dx = grid_size - dx
    if dy > grid_size / 2:
        dy = grid_size - dy
    
    return np.sqrt(dx**2 + dy**2)


def get_nearest_target(animal: Animal, others: List[Animal], config: SimulationConfig, 
                       target_type: type) -> Tuple[Optional[Animal], float]:
    """Find nearest animal of target_type"""
    nearest = None
    min_dist = float('inf')
    
    for other in others:
        if isinstance(other, target_type) and other.id != animal.id:
            dist = compute_toroidal_distance(animal.x, animal.y, other.x, other.y, config.GRID_SIZE)
            if dist < min_dist:
                min_dist = dist
                nearest = other
    
    return nearest, min_dist


def run_eval_episode(prey_model: ActorCriticNetwork, 
                     predator_model: ActorCriticNetwork,
                     config: SimulationConfig,
                     device: torch.device,
                     steps: int = 200,
                     seed: int = 42,
                     deterministic: bool = True,
                     tracking_horizon: int = 10) -> Tuple[EvalMetrics, List[PreyDetectionEvent], List[PredatorDetectionEvent]]:
    """
    Run single evaluation episode with event tracking
    
    Args:
        prey_model: Trained prey model
        predator_model: Trained predator model
        config: Simulation config
        device: Torch device
        steps: Number of steps to simulate
        seed: Random seed for reproducibility
        deterministic: Use argmax for action selection (vs sampling)
        tracking_horizon: Steps to track after detection event
    
    Returns:
        metrics, prey_events, predator_events
    """
    # Set seeds for deterministic evaluation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create environment
    animals = create_animals(config)
    grass_field = GrassField(config.GRID_SIZE, regrow_interval=10)
    grass_field.regrow_prob = getattr(config, 'GRASS_REGROW_PROB', 0.05)
    pheromone_map = PheromoneMap(config.GRID_SIZE)
    
    # Tracking state
    prey_events: List[PreyDetectionEvent] = []
    predator_events: List[PredatorDetectionEvent] = []
    prey_in_threat = set()  # Track which prey are currently aware of threats
    predator_in_hunt = set()  # Track which predators are currently hunting
    
    # Active event tracking (for real-time distance measurements)
    active_prey_events = {}  # {event_idx: event_object}
    active_pred_events = {}  # {predator_id: event_object}
    
    # Statistics
    stats = {
        'prey_deaths': 0,
        'predator_deaths': 0,
        'prey_births': 0,
        'predator_births': 0,
        'prey_starvation_deaths': 0,
        'predator_starvation_deaths': 0,
        'predator_meals': 0
    }
    
    print(f"Starting eval episode (seed={seed}, steps={steps})")
    
    for step_idx in range(steps):
        grass_field.step_regrow(step_idx)
        animals_to_remove = []
        
        # Progress logging
        if (step_idx + 1) % 50 == 0:
            prey_count = sum(1 for a in animals if isinstance(a, Prey))
            pred_count = sum(1 for a in animals if isinstance(a, Predator))
            print(f"  Step {step_idx + 1}/{steps}: {len(animals)} animals (Prey={prey_count}, Pred={pred_count})")
        
        # Age animals and check for death
        for animal in animals:
            animal.update_age()
            
            if animal.is_old(config):
                animals_to_remove.append(animal)
                if isinstance(animal, Prey):
                    stats['prey_deaths'] += 1
                else:
                    stats['predator_deaths'] += 1
        
        # Remove old animals
        for animal in animals_to_remove:
            animals.remove(animal)
        animals_to_remove.clear()
        
        # === MOVEMENT PHASE ===
        for animal in animals:
            # Select appropriate model
            model = prey_model if isinstance(animal, Prey) else predator_model
            
            # Get observation using same path as training
            visible = animal.communicate(animals, config)
            obs = animal.get_enhanced_input(animals, config, pheromone_map, visible_animals=visible)
            
            # obs might be 1D or 2D depending on implementation, ensure it's (1, obs_dim)
            if obs.dim() == 1:
                obs_tensor = obs.unsqueeze(0).to(device)  # (1, obs_dim)
            else:
                obs_tensor = obs.to(device)  # Already (1, obs_dim)
            
            # Prepare visible animals tensor
            # communicate() already returns List[List[float]] (feature vectors)
            vis_slots = visible[:config.MAX_VISIBLE_ANIMALS]
            # Pad to max slots if needed
            while len(vis_slots) < config.MAX_VISIBLE_ANIMALS:
                vis_slots.append([0.0] * 9)
            vis_tensor = torch.tensor([vis_slots], dtype=torch.float32, device=device)  # (1, MAX_VISIBLE_ANIMALS, 9)
            
            # Get action from model
            with torch.no_grad():
                turn_logits, move_logits, _ = model(obs_tensor, vis_tensor)
                
                if deterministic:
                    turn_action = turn_logits.argmax(dim=1).item()
                    move_action = move_logits.argmax(dim=1).item()
                else:
                    turn_probs = torch.softmax(turn_logits, dim=1)
                    move_probs = torch.softmax(move_logits, dim=1)
                    turn_action = torch.multinomial(turn_probs, 1).item()
                    move_action = torch.multinomial(move_probs, 1).item()
            
            # Apply turn action using animal's method
            animal.apply_turn_action(turn_action)
            
            # Apply move action using proper action logic
            # This respects the model's learned policy and handles chase mechanics
            new_x, new_y = animal._apply_action_logic(move_action, animals, config, is_training=False)
            
            # Species-aware collision: predators can move onto prey tiles (and kill immediately)
            can_move = True
            immediate_kill = None
            
            if isinstance(animal, Predator):
                # Check if target has prey
                prey_at_target = next((a for a in animals if isinstance(a, Prey) and a.x == new_x and a.y == new_y), None)
                if prey_at_target:
                    # Predator catches prey - allow move and mark for immediate kill
                    immediate_kill = prey_at_target
                else:
                    # Check for other predators blocking
                    can_move = not animal._position_occupied(animals, new_x, new_y)
            else:
                # Prey: blocked by any animal
                can_move = not animal._position_occupied(animals, new_x, new_y)
            
            if can_move:
                animal.x = new_x
                animal.y = new_y
                
                # Immediate kill if predator caught prey
                if immediate_kill:
                    animal.energy = min(animal.energy + config.EATING_ENERGY_GAIN, config.MAX_ENERGY)
                    animal.steps_since_last_meal = 0
                    animals_to_remove.append(immediate_kill)
                    stats['prey_deaths'] += 1
                    stats['predator_meals'] += 1
                    
                    # Record kill event for tracking
                    if not hasattr(animal, '_kills_this_episode'):
                        animal._kills_this_episode = []
                    animal._kills_this_episode.append({
                        'step': step_idx,
                        'prey_id': immediate_kill.id,
                        'predator_id': animal.id
                    })
            
            # Update energy
            move_attempts = animal.get_move_count(config)
            animal.update_energy(config, move_attempts=move_attempts)
            
            # Check exhaustion
            if animal.is_exhausted():
                animals_to_remove.append(animal)
                if isinstance(animal, Prey):
                    stats['prey_deaths'] += 1
                    stats['prey_starvation_deaths'] += 1
                else:
                    stats['predator_deaths'] += 1
                    stats['predator_starvation_deaths'] += 1
        
        # Remove exhausted animals
        for animal in animals_to_remove:
            if animal in animals:
                animals.remove(animal)
        animals_to_remove.clear()
        
        # === EVENT DETECTION PHASE with real-time tracking ===
        # Build current population lists
        prey_list = [a for a in animals if isinstance(a, Prey)]
        predator_list = [a for a in animals if isinstance(a, Predator)]
        
        # A) Prey detection events
        new_prey_in_threat = set()
        for prey in prey_list:
            visible = prey.communicate(animals, config)
            vis_info = prey.summarize_visible(visible)
            
            if vis_info["predator_count"] > 0:
                new_prey_in_threat.add(prey.id)
                
                # New detection event?
                if prey.id not in prey_in_threat:
                    nearest_pred, dist = get_nearest_target(prey, predator_list, config, Predator)
                    if nearest_pred:
                        event = PreyDetectionEvent(
                            step=step_idx,
                            prey_id=prey.id,
                            predator_id=nearest_pred.id,
                            initial_distance=dist
                        )
                        event_idx = len(prey_events)
                        prey_events.append(event)
                        active_prey_events[event_idx] = {
                            'event': event,
                            'start_step': step_idx,
                            'nearest_pred_id': nearest_pred.id
                        }
        
        # Update active prey events with real-time distance tracking
        events_to_close = []
        for event_idx, tracking in list(active_prey_events.items()):
            event = tracking['event']
            prey = next((a for a in animals if a.id == event.prey_id), None)
            
            if prey is None:
                # Prey died - close event as failed
                event.alive_after_T = False
                event.escape_success = False
                events_to_close.append(event_idx)
            else:
                # Prey still alive - track distance at t+1 and t+5
                pred = next((a for a in animals if a.id == tracking['nearest_pred_id']), None)
                if pred:
                    steps_elapsed = step_idx - tracking['start_step']
                    current_dist = compute_toroidal_distance(prey.x, prey.y, pred.x, pred.y, config.GRID_SIZE)
                    
                    if steps_elapsed == 1 and event.distance_after_1 is None:
                        event.distance_after_1 = current_dist
                    elif steps_elapsed == 5 and event.distance_after_5 is None:
                        event.distance_after_5 = current_dist
                
                    # Check if tracking horizon reached
                    if steps_elapsed >= tracking_horizon:
                        event.alive_after_T = True
                        event.escape_success = True
                        events_to_close.append(event_idx)
        
        # Remove closed events
        for event_idx in events_to_close:
            active_prey_events.pop(event_idx, None)
        
        prey_in_threat = new_prey_in_threat
        
        # B) Predator detection events
        new_predator_in_hunt = set()
        for predator in predator_list:
            visible = predator.communicate(animals, config)
            vis_info = predator.summarize_visible(visible)
            
            if vis_info["prey_count"] > 0:
                new_predator_in_hunt.add(predator.id)
                
                # New detection event?
                if predator.id not in predator_in_hunt:
                    nearest_prey, dist = get_nearest_target(predator, prey_list, config, Prey)
                    if nearest_prey:
                        event = PredatorDetectionEvent(
                            step=step_idx,
                            predator_id=predator.id,
                            prey_id=nearest_prey.id,
                            initial_distance=dist
                        )
                        predator_events.append(event)
                        active_pred_events[predator.id] = {
                            'event': event,
                            'start_step': step_idx,
                            'target_prey_id': nearest_prey.id
                        }
        
        # Update active predator events - check for successful kills
        events_to_close = []
        for pred_id, tracking in list(active_pred_events.items()):
            event = tracking['event']
            prey = next((a for a in animals if a.id == tracking['target_prey_id']), None)
            
            if prey is None:
                # Prey died - successful capture
                event.capture_success = True
                event.steps_to_capture = step_idx - tracking['start_step']
                events_to_close.append(pred_id)
            else:
                # Check if tracking horizon exceeded
                steps_elapsed = step_idx - tracking['start_step']
                if steps_elapsed >= tracking_horizon:
                    event.capture_success = False
                    event.steps_to_capture = None
                    events_to_close.append(pred_id)
        
        # Remove closed events
        for pred_id in events_to_close:
            active_pred_events.pop(pred_id, None)
        
        predator_in_hunt = new_predator_in_hunt
        
        # === EATING PHASE (Grass only - predator kills handled immediately in movement) ===
        for animal in animals:
            if isinstance(animal, Prey):
                if grass_field.has_grass(animal.x, animal.y):
                    if grass_field.consume(animal.x, animal.y):
                        animal.energy = min(animal.energy + config.GRASS_ENERGY, config.MAX_ENERGY)
        
        # Remove killed prey (already marked during movement phase)
        for animal in animals_to_remove:
            if animal in animals:
                animals.remove(animal)
        animals_to_remove.clear()
        
        # Update predator hunger
        for animal in animals:
            if isinstance(animal, Predator):
                animal.update_post_action(config)
                # Check starvation
                if animal.steps_since_last_meal >= config.STARVATION_THRESHOLD:
                    animals_to_remove.append(animal)
                    stats['predator_deaths'] += 1
                    stats['predator_starvation_deaths'] += 1
        
        # Remove starved predators
        for animal in animals_to_remove:
            if animal in animals:
                animals.remove(animal)
        animals_to_remove.clear()
        
        # === REPRODUCTION PHASE ===
        # Match training's mating logic (3x3 grid proximity check)
        mated_animals = set()
        
        # Build position map for faster neighbor lookup
        pos_map = defaultdict(list)
        for a in animals:
            pos_map[(a.x, a.y)].append(a)
        
        for animal1 in animals:
            if animal1.id in mated_animals:
                continue
            if animal1.mating_cooldown > 0:
                animal1.mating_cooldown -= 1
                continue
            if animal1.energy < config.MATING_ENERGY_COST:
                continue
            
            # Check neighboring cells (3x3 grid centered on animal1)
            mated = False
            for dx in (-1, 0, 1):
                if mated:
                    break
                for dy in (-1, 0, 1):
                    nx = (animal1.x + dx) % config.GRID_SIZE
                    ny = (animal1.y + dy) % config.GRID_SIZE
                    
                    for animal2 in pos_map[(nx, ny)]:
                        if animal2.id <= animal1.id:  # Avoid duplicate pairs and self
                            continue
                        if animal2.id in mated_animals:
                            continue
                        if animal2.mating_cooldown > 0:
                            continue
                        if animal2.energy < config.MATING_ENERGY_COST:
                            continue
                        if type(animal1) != type(animal2):
                            continue
                        
                        mating_prob = (config.MATING_PROBABILITY_PREY 
                                     if isinstance(animal1, Prey) 
                                     else config.MATING_PROBABILITY_PREDATOR)
                        
                        if random.random() < mating_prob:
                            # Create offspring
                            child_x = (animal1.x + animal2.x) // 2
                            child_y = (animal1.y + animal2.y) // 2
                            
                            if isinstance(animal1, Prey):
                                child = Prey(child_x, child_y, animal1.name, animal1.color, {animal1.id, animal2.id})
                                stats['prey_births'] += 1
                            else:
                                child = Predator(child_x, child_y, animal1.name, animal1.color, {animal1.id, animal2.id})
                                stats['predator_births'] += 1
                            
                            child.energy = config.INITIAL_ENERGY
                            animals.append(child)
                            
                            # Apply cooldown and energy cost
                            animal1.mating_cooldown = config.MATING_COOLDOWN
                            animal2.mating_cooldown = config.MATING_COOLDOWN
                            animal1.energy -= config.MATING_ENERGY_COST
                            animal2.energy -= config.MATING_ENERGY_COST
                            
                            mated_animals.add(animal1.id)
                            mated_animals.add(animal2.id)
                            mated = True
                            break
        
        # Deposit pheromones
        for animal in animals:
            animal.deposit_pheromones(animals, pheromone_map, config, step_idx=step_idx)
        
        pheromone_map.update()
        
        # Check extinction
        if len(animals) == 0:
            print(f"  Episode ended at step {step_idx + 1}: All animals extinct")
            break
    
    # === FINALIZE REMAINING ACTIVE EVENTS ===
    print(f"Finalizing {len(active_prey_events)} active prey events...")
    for event_idx, tracking in active_prey_events.items():
        event = tracking['event']
        prey = next((a for a in animals if a.id == event.prey_id), None)
        event.alive_after_T = prey is not None
        event.escape_success = prey is not None
        
        # Fill in missing distance measurements if prey still alive
        if prey and event.distance_after_1 is None:
            pred = next((a for a in animals if a.id == tracking['nearest_pred_id']), None)
            if pred:
                event.distance_after_1 = compute_toroidal_distance(prey.x, prey.y, pred.x, pred.y, config.GRID_SIZE)
                event.distance_after_5 = event.distance_after_1
    
    print(f"Finalizing {len(active_pred_events)} active predator events...")
    for pred_id, tracking in active_pred_events.items():
        event = tracking['event']
        if not event.capture_success:
            event.capture_success = False
            event.steps_to_capture = None
    
    # === AGGREGATE METRICS ===
    final_prey_count = sum(1 for a in animals if isinstance(a, Prey))
    final_predator_count = sum(1 for a in animals if isinstance(a, Predator))
    
    # Prey escape metrics
    if prey_events:
        prey_escape_rate = sum(e.escape_success for e in prey_events) / len(prey_events)
        
        # Distance gains (only for events where we have data)
        dist_gains_1 = [e.distance_after_1 - e.initial_distance 
                       for e in prey_events if e.distance_after_1 is not None]
        dist_gains_5 = [e.distance_after_5 - e.initial_distance 
                       for e in prey_events if e.distance_after_5 is not None]
        
        prey_dist_gain_1_mean = np.mean(dist_gains_1) if dist_gains_1 else 0.0
        prey_dist_gain_5_mean = np.mean(dist_gains_5) if dist_gains_5 else 0.0
    else:
        prey_escape_rate = 0.0
        prey_dist_gain_1_mean = 0.0
        prey_dist_gain_5_mean = 0.0
    
    # Predator hunt metrics
    if predator_events:
        predator_capture_rate = sum(e.capture_success for e in predator_events) / len(predator_events)
        
        capture_times = [e.steps_to_capture for e in predator_events 
                        if e.capture_success and e.steps_to_capture is not None]
        predator_time_to_capture_median = float(np.median(capture_times)) if capture_times else 0.0
    else:
        predator_capture_rate = 0.0
        predator_time_to_capture_median = 0.0
    
    # Meals per alive predator
    alive_predators = sum(1 for a in animals if isinstance(a, Predator))
    predator_meals_per_alive = stats['predator_meals'] / max(alive_predators, 1)
    
    metrics = EvalMetrics(
        episode=0,  # Will be set by caller
        checkpoint_episode=0,  # Will be set by caller
        final_prey_count=final_prey_count,
        final_predator_count=final_predator_count,
        prey_deaths=stats['prey_deaths'],
        predator_deaths=stats['predator_deaths'],
        prey_births=stats['prey_births'],
        predator_births=stats['predator_births'],
        prey_detection_events=len(prey_events),
        prey_escape_rate=prey_escape_rate,
        prey_dist_gain_1_mean=prey_dist_gain_1_mean,
        prey_dist_gain_5_mean=prey_dist_gain_5_mean,
        predator_detection_events=len(predator_events),
        predator_capture_rate=predator_capture_rate,
        predator_time_to_capture_median=predator_time_to_capture_median,
        prey_starvation_deaths=stats['prey_starvation_deaths'],
        predator_starvation_deaths=stats['predator_starvation_deaths'],
        predator_meals_total=stats['predator_meals'],
        predator_meals_per_alive=predator_meals_per_alive
    )
    
    return metrics, prey_events, predator_events


def evaluate_checkpoint_pair(checkpoint_dir: str, 
                             episode_num: int,
                             config: SimulationConfig,
                             device: torch.device,
                             num_eval_episodes: int = 3,
                             steps_per_episode: int = 200,
                             base_seed: int = 42) -> Dict:
    """
    Evaluate a checkpoint pair (prey + predator) across multiple episodes
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        episode_num: Episode number to evaluate (e.g., 10 for ep10)
        config: Simulation config
        device: Torch device
        num_eval_episodes: Number of eval episodes to average over
        steps_per_episode: Steps per episode
        base_seed: Base random seed
    
    Returns:
        Dictionary with aggregated metrics and per-episode details
    """
    # Find checkpoints - support both old and new formats:
    # Old: model_A_ppo_ep*.pth, model_B_ppo_ep*.pth  
    # New: {prefix}_ep*_model_A.pth, {prefix}_ep*_model_B.pth
    prey_checkpoint = None
    pred_checkpoint = None
    
    # Try new format first (search for any prefix)
    for ckpt_file in os.listdir(checkpoint_dir):
        if ckpt_file.endswith(f'_ep{episode_num}_model_A.pth'):
            prey_checkpoint = os.path.join(checkpoint_dir, ckpt_file)
        elif ckpt_file.endswith(f'_ep{episode_num}_model_B.pth'):
            pred_checkpoint = os.path.join(checkpoint_dir, ckpt_file)
    
    # Fall back to old format
    if prey_checkpoint is None:
        prey_checkpoint = os.path.join(checkpoint_dir, f"model_A_ppo_ep{episode_num}.pth")
    if pred_checkpoint is None:
        pred_checkpoint = os.path.join(checkpoint_dir, f"model_B_ppo_ep{episode_num}.pth")
    
    print(f"\n{'='*80}")
    print(f"Evaluating checkpoint pair: ep{episode_num}")
    print(f"{'='*80}")
    
    prey_model = load_checkpoint(prey_checkpoint, config, device)
    predator_model = load_checkpoint(pred_checkpoint, config, device)
    
    # Run multiple eval episodes
    all_metrics = []
    all_prey_events = []
    all_predator_events = []
    
    for eval_ep in range(num_eval_episodes):
        seed = base_seed + eval_ep
        print(f"\nEval episode {eval_ep + 1}/{num_eval_episodes} (seed={seed})")
        
        metrics, prey_events, pred_events = run_eval_episode(
            prey_model, predator_model, config, device,
            steps=steps_per_episode, seed=seed, deterministic=True
        )
        
        metrics.episode = eval_ep
        metrics.checkpoint_episode = episode_num
        all_metrics.append(metrics)
        all_prey_events.extend(prey_events)
        all_predator_events.extend(pred_events)
    
    # Aggregate across episodes
    aggregated = {
        'checkpoint_episode': episode_num,
        'num_eval_episodes': num_eval_episodes,
        'steps_per_episode': steps_per_episode,
        
        # Averaged metrics
        'final_prey_count_mean': np.mean([m.final_prey_count for m in all_metrics]),
        'final_predator_count_mean': np.mean([m.final_predator_count for m in all_metrics]),
        'prey_escape_rate_mean': np.mean([m.prey_escape_rate for m in all_metrics]),
        'prey_dist_gain_1_mean': np.mean([m.prey_dist_gain_1_mean for m in all_metrics]),
        'prey_dist_gain_5_mean': np.mean([m.prey_dist_gain_5_mean for m in all_metrics]),
        'predator_capture_rate_mean': np.mean([m.predator_capture_rate for m in all_metrics]),
        'predator_time_to_capture_median': np.median([m.predator_time_to_capture_median for m in all_metrics]),
        'predator_meals_per_alive_mean': np.mean([m.predator_meals_per_alive for m in all_metrics]),
        
        # Totals
        'prey_detection_events_total': sum(m.prey_detection_events for m in all_metrics),
        'predator_detection_events_total': sum(m.predator_detection_events for m in all_metrics),
        'prey_deaths_total': sum(m.prey_deaths for m in all_metrics),
        'predator_deaths_total': sum(m.predator_deaths for m in all_metrics),
        'prey_starvation_deaths_total': sum(m.prey_starvation_deaths for m in all_metrics),
        'predator_starvation_deaths_total': sum(m.predator_starvation_deaths for m in all_metrics),
        
        # Per-episode details
        'episodes': [asdict(m) for m in all_metrics]
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY (checkpoint ep{episode_num}, {num_eval_episodes} eval episodes)")
    print(f"{'='*80}")
    print(f"Final population:       Prey={aggregated['final_prey_count_mean']:.1f}, Pred={aggregated['final_predator_count_mean']:.1f}")
    print(f"Prey escape rate:       {aggregated['prey_escape_rate_mean']:.1%}")
    print(f"Prey dist gain (1-5):   {aggregated['prey_dist_gain_1_mean']:.2f}, {aggregated['prey_dist_gain_5_mean']:.2f}")
    print(f"Predator capture rate:  {aggregated['predator_capture_rate_mean']:.1%}")
    print(f"Predator meals/alive:   {aggregated['predator_meals_per_alive_mean']:.2f}")
    print(f"Detection events:       Prey={aggregated['prey_detection_events_total']}, Pred={aggregated['predator_detection_events_total']}")
    
    return aggregated


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/checkpoints',
                       help='Directory containing checkpoints')
    parser.add_argument('--episodes', type=int, nargs='+', default=[1, 10, 50, 100],
                       help='Episode numbers to evaluate (e.g., 1 10 50 100)')
    parser.add_argument('--num-eval-episodes', type=int, default=3,
                       help='Number of eval episodes per checkpoint')
    parser.add_argument('--steps', type=int, default=200,
                       help='Steps per eval episode')
    parser.add_argument('--output-dir', type=str, default='outputs/eval_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clean up old results
    output_path = Path(args.output_dir)
    for old_file in output_path.glob('*.json'):
        old_file.unlink()
        print(f"Removed old result: {old_file.name}")
    
    # Load config
    config = SimulationConfig()
    device = torch.device(args.device)
    
    print(f"Evaluating checkpoints from: {args.checkpoint_dir}")
    print(f"Episodes to evaluate: {args.episodes}")
    print(f"Output directory: {args.output_dir}")
    
    # Evaluate each checkpoint
    all_results = []
    total_checkpoints = len(args.episodes)
    for idx, ep_num in enumerate(args.episodes, 1):
        print(f"\n{'='*60}", flush=True)
        print(f"Evaluating checkpoint {idx}/{total_checkpoints}: Episode {ep_num}", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            result = evaluate_checkpoint_pair(
                args.checkpoint_dir, ep_num, config, device,
                num_eval_episodes=args.num_eval_episodes,
                steps_per_episode=args.steps
            )
            all_results.append(result)
            
            # Save individual checkpoint results
            output_file = os.path.join(args.output_dir, f'eval_ep{ep_num}.json')
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"[OK] Saved results to: {output_file}", flush=True)
            print(f"Progress: {idx}/{total_checkpoints} checkpoints completed ({int(idx/total_checkpoints*100)}%)", flush=True)
            
        except FileNotFoundError as e:
            print(f"Skipping episode {ep_num}: {e}")
            continue
    
    # Save combined results
    combined_output = os.path.join(args.output_dir, 'eval_summary.json')
    with open(combined_output, 'w') as f:
        json.dump({
            'checkpoints_evaluated': len(all_results),
            'results': all_results
        }, f, indent=2)
    print(f"\nSaved combined results to: {combined_output}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
