"""
Policy Evaluation Script - Compare Stochastic vs Deterministic Behavior

Runs evaluation episodes with both sampling (stochastic) and argmax (deterministic)
to quantify behavioral differences and policy quality.
"""

import sys
import random
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SimulationConfig
from src.core.animal import Prey, Predator
from src.models.actor_critic_network import ActorCriticNetwork
from src.core.pheromone_system import PheromoneMap


class PolicyEvaluator:
    """Evaluate trained policy with different action selection modes"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.device = torch.device("cpu")
        
        # Load trained models
        self.model_prey = ActorCriticNetwork(config).to(self.device)
        self.model_predator = ActorCriticNetwork(config).to(self.device)
        
        try:
            self.model_prey.load_state_dict(
                torch.load("outputs/checkpoints/model_A_ppo.pth", map_location=self.device)
            )
            self.model_predator.load_state_dict(
                torch.load("outputs/checkpoints/model_B_ppo.pth", map_location=self.device)
            )
            print("✅ Loaded trained PPO models")
        except FileNotFoundError:
            print("❌ No trained models found")
            sys.exit(1)
        
        # Set to eval mode
        self.model_prey.eval()
        self.model_predator.eval()
    
    def create_population(self):
        """Create initial population"""
        animals = []
        
        # Create prey
        for _ in range(self.config.INITIAL_PREY_COUNT):
            x = random.randint(self.config.FIELD_MIN, self.config.FIELD_MAX)
            y = random.randint(self.config.FIELD_MIN, self.config.FIELD_MAX)
            animal = Prey(x, y, "A", "#00ff00")
            animal.energy = self.config.INITIAL_ENERGY
            animals.append(animal)
        
        # Create predators
        for _ in range(self.config.INITIAL_PREDATOR_COUNT):
            x = random.randint(self.config.FIELD_MIN, self.config.FIELD_MAX)
            y = random.randint(self.config.FIELD_MIN, self.config.FIELD_MAX)
            animal = Predator(x, y, "B", "#ff0000")
            animal.energy = self.config.INITIAL_ENERGY
            animals.append(animal)
        
        return animals
    
    def run_episode(self, seed: int, deterministic: bool = False):
        """
        Run one evaluation episode
        
        Args:
            seed: Random seed for reproducibility
            deterministic: If True, use argmax; if False, use sampling
            
        Returns:
            dict: Episode metrics
        """
        # Set seeds
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize environment
        animals = self.create_population()
        pheromone_map = PheromoneMap(
            self.config.GRID_SIZE,
            decay_rate=self.config.PHEROMONE_DECAY,
            diffusion_rate=self.config.PHEROMONE_DIFFUSION
        )
        
        # Episode tracking
        metrics = {
            'total_reward': 0.0,
            'prey_reward': 0.0,
            'predator_reward': 0.0,
            'meals': 0,
            'births': 0,
            'deaths': 0,
            'starvation_deaths': 0,
            'old_age_deaths': 0,
            'eaten_deaths': 0,
            'final_prey': 0,
            'final_predators': 0,
            'prey_survival_time': [],
            'predator_survival_time': [],
            'action_changes': 0,  # Track action consistency
            'last_actions': {}  # Track last action per animal
        }
        
        # Run episode
        for step in range(self.config.STEPS_PER_EPISODE):
            animals_to_remove = []
            
            # Cache counts
            prey_list = [a for a in animals if isinstance(a, Prey)]
            predator_list = [a for a in animals if isinstance(a, Predator)]
            self.config._prey_count = len(prey_list)
            self.config._pred_count = len(predator_list)
            
            # Age updates and old age deaths
            for animal in animals:
                animal.update_age()
                if animal.is_old(self.config):
                    animals_to_remove.append(animal)
                    metrics['deaths'] += 1
                    metrics['old_age_deaths'] += 1
                    if isinstance(animal, Prey):
                        metrics['prey_survival_time'].append(animal.survival_time)
                    else:
                        metrics['predator_survival_time'].append(animal.survival_time)
            
            # Movement phase
            active_animals = [a for a in animals if a not in animals_to_remove]
            for animal in active_animals:
                model = self.model_prey if isinstance(animal, Prey) else self.model_predator
                
                # Get observation
                animal_input = animal.get_enhanced_input(animals, self.config, pheromone_map)
                visible_animals = animal.communicate(animals, self.config)
                visible_animals_input = torch.tensor(visible_animals, dtype=torch.float32).unsqueeze(0)
                
                # Select action with specified mode
                with torch.no_grad():
                    if hasattr(model, 'get_action'):
                        turn_action, move_action, _, _, _ = model.get_action(
                            animal_input, visible_animals_input, deterministic=deterministic
                        )
                        turn_action = int(turn_action.item())
                        move_action = int(move_action.item())
                        
                        # Track action changes (for consistency metric)
                        action_key = (turn_action, move_action)
                        if animal.id in metrics['last_actions']:
                            if metrics['last_actions'][animal.id] != action_key:
                                metrics['action_changes'] += 1
                        metrics['last_actions'][animal.id] = action_key
                    else:
                        turn_action = 1  # fallback
                        move_action = 0
                
                # Apply turn
                animal.apply_turn_action(turn_action)
                
                # Apply movement
                pos_before = (animal.x, animal.y)
                new_x, new_y = animal._apply_action_logic(move_action, animals, self.config, is_training=False)
                
                if not animal._position_occupied(animals, new_x, new_y):
                    animal.x, animal.y = new_x, new_y
                
                moved = (animal.x, animal.y) != pos_before
                animal.update_energy(self.config, moved)
                
                if animal.is_exhausted():
                    animals_to_remove.append(animal)
                    metrics['deaths'] += 1
                    if isinstance(animal, Prey):
                        metrics['prey_survival_time'].append(animal.survival_time)
                    else:
                        metrics['predator_survival_time'].append(animal.survival_time)
            
            # Remove old/exhausted animals
            for animal in animals_to_remove:
                if animal in animals:
                    animals.remove(animal)
            animals_to_remove.clear()
            
            # Predators eat
            for predator in [a for a in animals if isinstance(a, Predator)]:
                ate, _, eaten = predator.perform_eat(animals, self.config)
                if ate:
                    metrics['meals'] += 1
                    if eaten and eaten in animals:
                        animals.remove(eaten)
                        metrics['deaths'] += 1
                        metrics['eaten_deaths'] += 1
                        metrics['prey_survival_time'].append(eaten.survival_time)
                else:
                    predator.steps_since_last_meal += 1
                    if (self.config.STARVATION_ENABLED and
                            predator.steps_since_last_meal >= self.config.STARVATION_THRESHOLD):
                        animals_to_remove.append(predator)
                        metrics['deaths'] += 1
                        metrics['starvation_deaths'] += 1
                        metrics['predator_survival_time'].append(predator.survival_time)
            
            for animal in animals_to_remove:
                if animal in animals:
                    animals.remove(animal)
            animals_to_remove.clear()
            
            # Mating phase (simplified - no new animals to keep consistent)
            for animal in animals:
                if animal.mating_cooldown > 0:
                    animal.mating_cooldown -= 1
                animal.survival_time += 1
                animal.deposit_pheromones(animals, pheromone_map, self.config)
            
            pheromone_map.update()
        
        # Final counts
        metrics['final_prey'] = sum(1 for a in animals if isinstance(a, Prey))
        metrics['final_predators'] = sum(1 for a in animals if isinstance(a, Predator))
        
        # Compute averages
        if metrics['prey_survival_time']:
            metrics['avg_prey_survival'] = np.mean(metrics['prey_survival_time'])
        else:
            metrics['avg_prey_survival'] = 0.0
            
        if metrics['predator_survival_time']:
            metrics['avg_predator_survival'] = np.mean(metrics['predator_survival_time'])
        else:
            metrics['avg_predator_survival'] = 0.0
        
        # Action consistency (lower is more consistent)
        total_actions = len(metrics['last_actions']) * self.config.STEPS_PER_EPISODE
        metrics['action_change_rate'] = metrics['action_changes'] / max(total_actions, 1)
        
        return metrics
    
    def evaluate(self, num_episodes: int = 10):
        """
        Run evaluation comparing stochastic vs deterministic
        
        Args:
            num_episodes: Number of episodes per mode
        """
        print(f"\n{'='*70}")
        print("POLICY EVALUATION: Stochastic vs Deterministic")
        print(f"{'='*70}\n")
        
        # Run stochastic episodes
        print("🎲 Running STOCHASTIC episodes (sampling)...")
        stochastic_results = []
        for i in range(num_episodes):
            seed = 1000 + i
            metrics = self.run_episode(seed, deterministic=False)
            stochastic_results.append(metrics)
            print(f"  Episode {i+1}/{num_episodes}: "
                  f"Meals={metrics['meals']}, "
                  f"Final Prey={metrics['final_prey']}, "
                  f"Final Pred={metrics['final_predators']}")
        
        print("\n🎯 Running DETERMINISTIC episodes (argmax)...")
        deterministic_results = []
        for i in range(num_episodes):
            seed = 1000 + i  # Same seeds for fair comparison
            metrics = self.run_episode(seed, deterministic=True)
            deterministic_results.append(metrics)
            print(f"  Episode {i+1}/{num_episodes}: "
                  f"Meals={metrics['meals']}, "
                  f"Final Prey={metrics['final_prey']}, "
                  f"Final Pred={metrics['final_predators']}")
        
        # Aggregate statistics
        print(f"\n{'='*70}")
        print("RESULTS COMPARISON")
        print(f"{'='*70}\n")
        
        def aggregate(results):
            agg = {}
            for key in ['meals', 'deaths', 'starvation_deaths', 'eaten_deaths', 
                       'final_prey', 'final_predators', 'avg_prey_survival', 
                       'avg_predator_survival', 'action_change_rate']:
                values = [r[key] for r in results]
                agg[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            return agg
        
        stoch_agg = aggregate(stochastic_results)
        det_agg = aggregate(deterministic_results)
        
        # Print comparison
        metrics_to_show = [
            ('Meals per Episode', 'meals'),
            ('Final Prey Count', 'final_prey'),
            ('Final Predator Count', 'final_predators'),
            ('Starvation Deaths', 'starvation_deaths'),
            ('Prey Eaten', 'eaten_deaths'),
            ('Avg Prey Survival Time', 'avg_prey_survival'),
            ('Avg Predator Survival Time', 'avg_predator_survival'),
            ('Action Change Rate', 'action_change_rate'),
        ]
        
        print(f"{'Metric':<30} {'Stochastic':<25} {'Deterministic':<25} {'Diff'}")
        print(f"{'-'*30} {'-'*25} {'-'*25} {'-'*10}")
        
        for name, key in metrics_to_show:
            s_mean = stoch_agg[key]['mean']
            s_std = stoch_agg[key]['std']
            d_mean = det_agg[key]['mean']
            d_std = det_agg[key]['std']
            diff = d_mean - s_mean
            
            print(f"{name:<30} {s_mean:>8.2f} ± {s_std:<7.2f}   {d_mean:>8.2f} ± {d_std:<7.2f}   {diff:+.2f}")
        
        print(f"\n{'='*70}")
        print("INTERPRETATION")
        print(f"{'='*70}\n")
        
        # Analysis
        action_improvement = (stoch_agg['action_change_rate']['mean'] - 
                             det_agg['action_change_rate']['mean']) / stoch_agg['action_change_rate']['mean'] * 100
        
        print(f"Action Consistency: {action_improvement:+.1f}% improvement with deterministic")
        print(f"  (Lower action change rate = more consistent behavior)")
        
        if abs(det_agg['meals']['mean'] - stoch_agg['meals']['mean']) < stoch_agg['meals']['std']:
            print(f"\n✅ Similar performance between modes → Policy is stable")
        else:
            print(f"\n⚠️  Significant performance difference → Policy may be unstable")
        
        if action_improvement > 20:
            print(f"\n💡 Large consistency improvement suggests the 'hectic' behavior")
            print(f"   was mainly due to sampling randomness, not policy instability")
        else:
            print(f"\n💡 Small consistency improvement suggests the policy itself")
            print(f"   may be unstable (uniform/uncertain action probabilities)")


def main():
    """Run evaluation"""
    config = SimulationConfig()
    evaluator = PolicyEvaluator(config)
    evaluator.evaluate(num_episodes=10)


if __name__ == "__main__":
    main()
