"""
Life Game - Demo Mode (Inference Only)

Main entry point for running the simulation without training.
Loads pre-trained models and displays the learned behaviors on a game field.
"""

import torch
import matplotlib.pyplot as plt
from config import SimulationConfig
from neural_network import SimpleNN
from simulation import run_simulation, create_population


def main():
    """Main entry point for demo mode with restart capability"""
    
    # Header
    print("\n" + "=" * 70)
    print("  🎮 LIFE GAME - DEMO MODE (Inference Only)")
    print("  Predator-Prey Ecosystem with Neural Networks")
    print("=" * 70)
    
    # Initialize configuration
    config = SimulationConfig()
    
    # Create and load models
    model_prey = SimpleNN(config)
    model_predator = SimpleNN(config)
    
    try:
        model_prey.load_state_dict(torch.load("model_A_fixed.pth"))
        model_predator.load_state_dict(torch.load("model_B_fixed.pth"))
        print("\n✅ Loaded trained models successfully")
        models_loaded = True
    except FileNotFoundError:
        print("\n⚠️  WARNING: No trained models found!")
        print("   Using untrained models (random behavior)")
        print("   Run Life_Game_Fixed.py first to train models\n")
        models_loaded = False
    
    # Set to evaluation mode
    model_prey.eval()
    model_predator.eval()
    
    # Main game loop with restart capability
    while True:
        # Create initial population
        animals = create_population(config)
        
        print(f"\n✅ Created {len(animals)} initial animals")
        print(f"  • 🐰 Prey: {config.INITIAL_PREY_COUNT}")
        print(f"  • 🦊 Predators: {config.INITIAL_PREDATOR_COUNT}")
        
        print("\n" + "-" * 70)
        print("🎮 SIMULATION RUNNING")
        print("-" * 70)
        print("Close the plot window to exit | Click Restart button to restart")
        
        # Run simulation
        steps = 1000  # Max steps per simulation
        
        try:
            stats = run_simulation(animals, steps, model_prey, model_predator, config)
            
            # Check if restart was requested during simulation
            if stats.get('restart_requested'):
                print("\n🔄 Restarting simulation...")
                plt.close('all')
                continue
            
            # Display final statistics
            print("\n" + "=" * 70)
            print("🏁 SIMULATION COMPLETE")
            print("=" * 70)
            print(f"\n📊 Final Population:")
            print(f"  • 🐰 Prey: {stats['final_prey']}")
            print(f"  • 🦊 Predators: {stats['final_predators']}")
            print(f"\n📈 Total Events:")
            print(f"  • 👶 Births: {stats['total_births']}")
            print(f"  • 💀 Deaths: {stats['total_deaths']}")
            print(f"  • 🍽️  Predator Meals: {stats['total_meals']}")
            print(f"\n📉 Population Extremes:")
            print(f"  • 🔝 Peak Prey: {stats['peak_prey']}")
            print(f"  • 🔝 Peak Predators: {stats['peak_predators']}")
            print(f"  • ⬇️  Minimum Prey: {stats['min_prey']}")
            print(f"\n⏱️  Duration: {stats['duration']:.2f} seconds")
            
            # Show detailed statistics window
            print("\n📊 Opening detailed statistics window...")
            viz = stats.get('viz')
            if viz:
                viz.show_final_stats(stats)
            
            # Wait for user to close stats window
            plt.show(block=False)
            plt.pause(3)  # Show for 3 seconds
            
            # Ask for restart
            print("\n" + "=" * 70)
            response = input("\n🔄 Run again? (y/n): ").strip().lower()
            plt.close('all')  # Close all windows before restart
            if response != 'y':
                break
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Simulation interrupted by user")
            response = input("🔄 Restart simulation? (y/n): ").strip().lower()
            if response != 'y':
                break
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "=" * 70)
    print("  👋 Thank you for using Life Game Demo!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
