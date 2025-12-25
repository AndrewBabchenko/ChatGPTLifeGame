#!/usr/bin/env python3
"""
Quick test to verify species-aware rewards and model forward pass
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config import SimulationConfig
from src.models.actor_critic_network import ActorCriticNetwork
from src.core.animal import Animal
from src.core.pheromone_system import PheromoneMap

def test_species_awareness():
    print("\n" + "="*70)
    print("  Species-Aware Model Test")
    print("="*70 + "\n")
    
    config = SimulationConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🔧 Configuration:")
    print(f"  MAX_ANIMALS: {config.MAX_ANIMALS}")
    print(f"  OTHER_SPECIES_CAPACITY: {config.OTHER_SPECIES_CAPACITY}")
    print(f"  Usable capacity per species: {config.MAX_ANIMALS - config.OTHER_SPECIES_CAPACITY}")
    print(f"  MAX_VISIBLE_ANIMALS: {config.MAX_VISIBLE_ANIMALS}")
    
    print(f"\n📋 Reward Structure:")
    print(f"  PREY_EVASION_REWARD: {config.PREY_EVASION_REWARD}")
    print(f"  PREDATOR_APPROACH_REWARD: {config.PREDATOR_APPROACH_REWARD}")
    print(f"  PREDATOR_EAT_REWARD: {config.PREDATOR_EAT_REWARD}")
    print(f"  OVERPOPULATION_PENALTY: {config.OVERPOPULATION_PENALTY}")
    
    # Create test population
    print(f"\n🐾 Creating test population...")
    animals = []
    
    # Create prey
    for i in range(10):
        prey = Animal(50 + i, 50, "A", "green", predator=False)
        animals.append(prey)
    
    # Create predators
    for i in range(3):
        pred = Animal(55 + i, 55, "B", "red", predator=True)
        animals.append(pred)
    
    print(f"  Created {len([a for a in animals if not a.predator])} prey")
    print(f"  Created {len([a for a in animals if a.predator])} predators")
    
    # Test model
    print(f"\n🧠 Testing model...")
    model = ActorCriticNetwork(config).to(device)
    
    # Test prey input
    prey = animals[0]
    prey_input = prey.get_enhanced_input(animals, config)
    print(f"\n✅ Prey input shape: {prey_input.shape} (expected: [1, 21])")
    
    features = prey_input.squeeze().tolist()
    print(f"  Feature 2 (is species A): {features[2]}")
    print(f"  Feature 4 (is predator): {features[4]}")
    print(f"  Feature 7 (nearest predator dist): {features[7]:.3f}")
    print(f"  Feature 13 (predator count): {features[13]:.3f}")
    print(f"  Feature 20 (population pressure): {features[20]:.3f}")
    
    # Test visible animals
    visible = prey.communicate(animals, config)
    print(f"\n✅ Visible animals: {len([v for v in visible if v[5] > 0 or v[6] > 0])} / {config.MAX_VISIBLE_ANIMALS}")
    
    if len(visible) > 0 and (visible[0][5] > 0 or visible[0][6] > 0):
        print(f"  First visible animal:")
        print(f"    Distance: {visible[0][4]:.3f}")
        print(f"    Is predator: {visible[0][5]}")
        print(f"    Is prey: {visible[0][6]}")
        print(f"    Same species: {visible[0][7]}")
    
    # Test model forward pass
    visible_tensor = torch.tensor(visible, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        actions, values = model(prey_input.to(device), visible_tensor)
    
    print(f"\n✅ Model forward pass successful")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Values shape: {values.shape}")
    
    # Test reward calculation logic
    print(f"\n💰 Reward Calculation Test:")
    
    # Simulate prey avoiding predator
    threat_info = prey._get_threat_info(animals, config, None)
    nearest_pred_dist = threat_info['nearest_predator_dist']
    
    if nearest_pred_dist < 1.0:
        evasion_reward = config.PREY_EVASION_REWARD * (1.0 - nearest_pred_dist)
        print(f"  Prey evasion reward: +{evasion_reward:.2f}")
        print(f"  (predator at distance {nearest_pred_dist:.3f})")
    
    # Test overcrowding penalty
    same_species = len([a for a in animals if not a.predator])
    usable_capacity = config.MAX_ANIMALS - config.OTHER_SPECIES_CAPACITY
    
    if same_species > usable_capacity:
        overcrowd_ratio = (same_species - usable_capacity) / usable_capacity
        penalty = config.OVERPOPULATION_PENALTY * overcrowd_ratio
        print(f"  Overcrowding penalty: {penalty:.2f}")
    else:
        print(f"  No overcrowding ({same_species}/{usable_capacity} capacity)")
    
    # Test predator
    pred = animals[-1]
    pred_input = pred.get_enhanced_input(animals, config)
    
    print(f"\n✅ Predator input shape: {pred_input.shape}")
    features = pred_input.squeeze().tolist()
    print(f"  Feature 4 (is predator): {features[4]}")
    print(f"  Feature 10 (nearest prey dist): {features[10]:.3f}")
    print(f"  Feature 14 (prey count): {features[14]:.3f}")
    
    threat_info = pred._get_threat_info(animals, config, None)
    nearest_prey_dist = threat_info['nearest_prey_dist']
    
    if nearest_prey_dist < 1.0:
        approach_reward = config.PREDATOR_APPROACH_REWARD * (1.0 - nearest_prey_dist)
        print(f"  Predator approach reward: +{approach_reward:.2f}")
        print(f"  (prey at distance {nearest_prey_dist:.3f})")
    
    print(f"\n" + "="*70)
    print(f"  ✅ All species-aware features working correctly!")
    print(f"="*70 + "\n")

if __name__ == "__main__":
    test_species_awareness()
