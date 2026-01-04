"""
Phase Training Runner Script

Easily run training for a specific phase with the correct config.
This script dynamically imports the correct phase config and runs training.

Usage:
    python scripts/run_phase.py --phase 1  # Run Phase 1: Pure Hunt/Evade
    python scripts/run_phase.py --phase 2  # Run Phase 2: Add Starvation
    python scripts/run_phase.py --phase 3  # Run Phase 3: Add Reproduction
    python scripts/run_phase.py --phase 4  # Run Phase 4: Full Ecosystem

To continue from a specific checkpoint (override config):
    python scripts/run_phase.py --phase 2 --checkpoint outputs/checkpoints/phase1_best
"""

import sys
import os
from pathlib import Path
import argparse
import importlib

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description='Run curriculum training for a specific phase')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4], default=4,
                        help='Phase number to run (1-4)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint prefix to load (e.g., outputs/checkpoints/phase1_best)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU mode')
    args = parser.parse_args()
    
    # Map phase number to config module name
    phase_configs = {
        1: 'src.config_phase1',
        2: 'src.config_phase2', 
        3: 'src.config_phase3',
        4: 'src.config',
    }
    
    phase_names = {
        1: "Pure Hunt/Evade",
        2: "Add Starvation",
        3: "Add Reproduction",
        4: "Full Ecosystem",
    }
    
    print(f"\n{'='*70}")
    print(f"  STARTING PHASE {args.phase}: {phase_names[args.phase]}")
    print(f"{'='*70}")
    
    # Dynamically import the correct config
    config_module_name = phase_configs[args.phase]
    print(f"Loading config: {config_module_name}")
    
    # Import the phase-specific config module
    config_module = importlib.import_module(config_module_name)
    
    # Make this module appear as 'src.config' so train.py imports it
    sys.modules['src.config'] = config_module
    
    # Override checkpoints if specified via command line
    if args.checkpoint:
        # If user provides prefix format, convert to full paths
        if not args.checkpoint.endswith('.pth'):
            config_module.LOAD_PREY_CHECKPOINT = f"{args.checkpoint}_model_A.pth"
            config_module.LOAD_PREDATOR_CHECKPOINT = f"{args.checkpoint}_model_B.pth"
        else:
            # User provided specific file, apply to both (unusual but supported)
            config_module.LOAD_PREY_CHECKPOINT = args.checkpoint
            config_module.LOAD_PREDATOR_CHECKPOINT = args.checkpoint
        print(f"Checkpoint override: prey={config_module.LOAD_PREY_CHECKPOINT}, pred={config_module.LOAD_PREDATOR_CHECKPOINT}")
    
    # Display phase settings
    print(f"Phase settings:")
    print(f"  PHASE_NUMBER: {getattr(config_module, 'PHASE_NUMBER', 'N/A')}")
    print(f"  PHASE_NAME: {getattr(config_module, 'PHASE_NAME', 'N/A')}")
    print(f"  LOAD_PREY_CHECKPOINT: {getattr(config_module, 'LOAD_PREY_CHECKPOINT', None)}")
    print(f"  LOAD_PREDATOR_CHECKPOINT: {getattr(config_module, 'LOAD_PREDATOR_CHECKPOINT', None)}")
    print(f"  SAVE_CHECKPOINT_PREFIX: {getattr(config_module, 'SAVE_CHECKPOINT_PREFIX', 'N/A')}")

    print(f"  NUM_EPISODES: {getattr(config_module.SimulationConfig, 'NUM_EPISODES', 'N/A')}")
    print(f"{'='*70}\n")
    
    # Pass CPU flag
    if args.cpu:
        os.environ['FORCE_CPU'] = '1'
    
    # Import and run train (it will use the modified src.config)
    from scripts.train import main as train_main
    train_main()


if __name__ == "__main__":
    main()
