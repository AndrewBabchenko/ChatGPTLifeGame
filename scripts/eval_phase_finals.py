"""
Evaluate final checkpoints from each training phase
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.eval_checkpoints import evaluate_checkpoint_pair, load_checkpoint
from src.config import SimulationConfig
import torch

def main():
    checkpoint_dir = Path("outputs/checkpoints")
    output_dir = Path("outputs/eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cpu")
    config = SimulationConfig()
    
    # Define the final checkpoint for each phase
    phase_finals = [
        ("phase1", 200),
        ("phase2", 50),
        ("phase3", 50),
        ("phase4", 150),
    ]
    
    results = []
    
    for prefix, ep_num in phase_finals:
        prey_ckpt = checkpoint_dir / f"{prefix}_ep{ep_num}_model_A.pth"
        pred_ckpt = checkpoint_dir / f"{prefix}_ep{ep_num}_model_B.pth"
        
        if not prey_ckpt.exists() or not pred_ckpt.exists():
            print(f"Skipping {prefix} ep{ep_num} - checkpoint not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"Evaluating {prefix} ep{ep_num}")
        print(f"{'='*70}")
        
        try:
            result = evaluate_checkpoint_pair(
                episode_num=ep_num,
                checkpoint_dir=str(checkpoint_dir),
                config=config,
                device=device,
                num_eval_episodes=5,
                steps_per_episode=300,
                base_seed=42,
                prefix=prefix  # Pass the prefix
            )
            result['phase'] = prefix
            results.append(result)
            
            # Save individual result
            out_file = output_dir / f"eval_{prefix}_ep{ep_num}.json"
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved: {out_file}")
            
        except Exception as e:
            print(f"Error evaluating {prefix} ep{ep_num}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY: Final Checkpoint Performance by Phase")
    print(f"{'='*70}")
    print(f"{'Phase':<10} {'Episode':<8} {'Prey':<8} {'Pred':<8} {'Escape%':<10} {'Capture%':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r.get('phase', 'N/A'):<10} "
              f"{r['checkpoint_episode']:<8} "
              f"{r['final_prey_count_mean']:<8.1f} "
              f"{r['final_predator_count_mean']:<8.1f} "
              f"{r['prey_escape_rate_mean']*100:<10.1f} "
              f"{r['predator_capture_rate_mean']*100:<10.1f}")
    
    # Save combined summary
    summary_file = output_dir / "eval_phase_finals_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary: {summary_file}")

if __name__ == "__main__":
    main()
