"""
Behaviors Tab - Analyze learned behaviors from model checkpoints
"""
import tkinter as tk
from tkinter import ttk
import sys
import re
import importlib
from pathlib import Path
from threading import Thread

import numpy as np
import torch

from .base_tab import BaseTab


class BehaviorsTab(BaseTab):
    """Behavior analysis tab"""
    
    def setup_ui(self):
        """Setup behavior analysis UI"""
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Model Behavior Analysis", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(control_frame, text="Run Behavior Analysis", 
                  command=self.run_behavior_analysis).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame, text="Copy CSV", 
                  command=self.copy_csv).pack(side=tk.LEFT, padx=(0, 10))
        
        self.behavior_status = ttk.Label(control_frame, text="Click button to analyze", 
                                        font=('Arial', 10), foreground='gray')
        self.behavior_status.pack(side=tk.LEFT)
        
        # Progress bar for analysis
        self.behavior_progress = ttk.Progressbar(control_frame, mode='determinate', length=200, maximum=100)
        self.behavior_progress.pack(side=tk.RIGHT, padx=(10, 0))
        self.behavior_progress.pack_forget()  # Hidden by default
        
        # Progress label showing percentage
        self.progress_label = ttk.Label(control_frame, text="", font=('Arial', 9))
        self.progress_label.pack(side=tk.RIGHT, padx=(5, 0))
        self.progress_label.pack_forget()  # Hidden by default
        
        # Legend
        legend_frame = ttk.Frame(main_frame)
        legend_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(legend_frame, text="Legend:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        for color, text in [('#d4edda', '>=80% Excellent'), ('#fff3cd', '60-79% Good'), ('#f8d7da', '<60% Poor')]:
            box = tk.Label(legend_frame, bg=color, width=3, height=1, relief=tk.RIDGE)
            box.pack(side=tk.LEFT, padx=(0, 5))
            ttk.Label(legend_frame, text=text, font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 15))
        
        # Scrollable table area
        table_container = ttk.Frame(main_frame)
        table_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbars
        canvas = tk.Canvas(table_container)
        h_scrollbar = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL, command=canvas.xview)
        v_scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=canvas.yview)
        
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Header frame (part of scrollable)
        self.behavior_header_frame = tk.Frame(self.scrollable_frame, bg='#e9ecef')
        self.behavior_header_frame.pack(fill=tk.X)
        
        # Table frame (part of scrollable)
        self.behavior_table_frame = tk.Frame(self.scrollable_frame, bg='white')
        self.behavior_table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Info text
        info_text = ("Behavior analysis evaluates learned behaviors by testing model responses "
                    "to synthetic scenarios. Higher percentages indicate better alignment with "
                    "expected behaviors.")
        ttk.Label(main_frame, text=info_text, font=('Arial', 9), 
                 foreground='gray', wraplength=800).pack(pady=(10, 0))
    
    def refresh(self):
        """Refresh tab - no auto-refresh for behavior analysis"""
        pass  # Behavior analysis is manual-only
    
    def _extract_episode_num(self, ckpt_path: Path) -> int:
        """Extract episode number from checkpoint filename"""
        # Handles both formats:
        # Old: model_A_ppo_ep10.pth -> 10
        # New: phase1_ep10_model_A.pth -> 10
        match = re.search(r'ep(\d+)', ckpt_path.stem)
        return int(match.group(1)) if match else 0
    
    def _get_config_for_checkpoint(self, ckpt_path: Path, cache: dict):
        """Pick the matching phase config module for a checkpoint (phase-aware)."""
        stem = ckpt_path.stem.lower()
        module_name = "src.config"
        if "phase1" in stem:
            module_name = "src.config_phase1"
        elif "phase2" in stem:
            module_name = "src.config_phase2"
        elif "phase3" in stem:
            module_name = "src.config_phase3"
        elif "phase4" in stem:
            module_name = "src.config"
        
        if module_name not in cache:
            cfg_module = importlib.import_module(module_name)
            cache[module_name] = cfg_module.SimulationConfig()
        return cache[module_name]
    
    def run_behavior_analysis(self):
        """Run behavior analysis on all checkpoints"""
        self.behavior_status.config(text="Running analysis...", foreground="orange")
        self.app.status_label.config(text="Running behavior analysis...", foreground="orange")
        
        # Show progress bar and label
        self.progress_label.config(text="0%")
        self.progress_label.pack(side=tk.RIGHT, padx=(5, 0))
        self.behavior_progress['value'] = 0
        self.behavior_progress.pack(side=tk.RIGHT, padx=(10, 0))
        self.root.update()
        
        # Run in thread to keep UI responsive
        Thread(target=self._run_analysis_thread, daemon=True).start()
    
    def _run_analysis_thread(self):
        """Background thread for behavior analysis"""
        try:
            # Add project root to Python path
            project_root = str(Path(__file__).parent.parent.parent.parent.absolute())
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from src.models.actor_critic_network import ActorCriticNetwork
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            config_cache = {}
            
            # Find all checkpoints
            ckpt_dir = Path(project_root) / "outputs" / "checkpoints"
            if not ckpt_dir.exists():
                self.root.after(0, lambda: self._analysis_complete(None, "No checkpoints found"))
                return
            
            # Find checkpoints - support both old and new formats:
            # Old: model_A_ppo_ep*.pth, model_B_ppo_ep*.pth
            # New: {prefix}_ep*_model_A.pth, {prefix}_ep*_model_B.pth (e.g., phase1_ep10_model_A.pth)
            pred_ckpts = []
            prey_ckpts = []
            
            # Old format
            pred_ckpts.extend(ckpt_dir.glob("model_B_ppo_ep*.pth"))
            prey_ckpts.extend(ckpt_dir.glob("model_A_ppo_ep*.pth"))
            
            # New format (any prefix)
            pred_ckpts.extend(ckpt_dir.glob("*_ep*_model_B.pth"))
            prey_ckpts.extend(ckpt_dir.glob("*_ep*_model_A.pth"))
            
            # Remove duplicates and sort
            pred_ckpts = sorted(set(pred_ckpts), key=lambda p: self._extract_episode_num(p))
            prey_ckpts = sorted(set(prey_ckpts), key=lambda p: self._extract_episode_num(p))
            
            # Skip best checkpoints to avoid fake episode 0 rows (phase best are loaded by phase-specific runs)
            pred_ckpts = [p for p in pred_ckpts if "_best_" not in p.stem]
            prey_ckpts = [p for p in prey_ckpts if "_best_" not in p.stem]
            
            if not pred_ckpts and not prey_ckpts:
                self.root.after(0, lambda: self._analysis_complete(None, "No checkpoints found"))
                return
            
            # Calculate total work for progress tracking
            total_checkpoints = len(pred_ckpts) + len(prey_ckpts)
            processed = 0
            
            # Organize by episode
            episode_data = {}
            
            # Process predator checkpoints (model_B)
            for ckpt_path in pred_ckpts:
                if "_best_" in ckpt_path.stem:
                    continue
                ep_match = re.search(r'ep(\d+)', ckpt_path.stem)
                if ep_match:
                    episode = int(ep_match.group(1))
                    if episode not in episode_data:
                        episode_data[episode] = {}
                    
                    cfg = self._get_config_for_checkpoint(ckpt_path, config_cache)
                    model = ActorCriticNetwork(cfg).to(device)
                    try:
                        state = torch.load(ckpt_path, map_location=device, weights_only=False)
                        model.load_state_dict(state, strict=True)
                    except Exception as e:
                        print(f"[Behavior Analysis] Skipping {ckpt_path.name}: {e}")
                        continue
                    
                    hunt = self.evaluate_hunting(model, cfg, device, samples=100)
                    hunger = self.evaluate_hunger_response(model, cfg, device, samples=100)
                    mate_pred = self.evaluate_mating_behavior(model, cfg, device, is_predator=True, samples=100)
                    selective = self.evaluate_selective_hunting(model, cfg, device, samples=100)
                    energy_mgmt = self.evaluate_energy_management(model, cfg, device, samples=100)
                    
                    episode_data[episode]['pred_hunting'] = self.format_behavior_score(hunt['rate'], hunt.get('success'), hunt.get('total'))
                    episode_data[episode]['pred_hunger'] = self.format_behavior_score(hunger['rate'], hunger.get('success'), hunger.get('total'))
                    episode_data[episode]['pred_mating'] = self.format_behavior_score(mate_pred['rate'], mate_pred.get('success'), mate_pred.get('total'))
                    episode_data[episode]['pred_selective'] = self.format_behavior_score(selective['rate'], selective.get('success'), selective.get('total'))
                    episode_data[episode]['pred_energy_mgmt'] = self.format_behavior_score(energy_mgmt['rate'], energy_mgmt.get('success'), energy_mgmt.get('total'))
                
                # Update progress
                processed += 1
                pct = int(processed / total_checkpoints * 100)
                self.root.after(0, lambda p=pct: self._update_progress(p))
            
            # Process prey checkpoints (model_A)
            for ckpt_path in prey_ckpts:
                if "_best_" in ckpt_path.stem:
                    continue
                ep_match = re.search(r'ep(\d+)', ckpt_path.stem)
                if ep_match:
                    episode = int(ep_match.group(1))
                    if episode not in episode_data:
                        episode_data[episode] = {}
                    
                    cfg = self._get_config_for_checkpoint(ckpt_path, config_cache)
                    model = ActorCriticNetwork(cfg).to(device)
                    try:
                        state = torch.load(ckpt_path, map_location=device, weights_only=False)
                        model.load_state_dict(state, strict=True)
                    except Exception as e:
                        print(f"[Behavior Analysis] Skipping {ckpt_path.name}: {e}")
                        continue
                    
                    evade = self.evaluate_evasion(model, cfg, device, samples=100)
                    mate_prey = self.evaluate_mating_behavior(model, cfg, device, is_predator=False, samples=100)
                    multi_target = self.evaluate_multi_target_handling(model, cfg, device, samples=100)
                    flocking = self.evaluate_flocking(model, cfg, device, samples=100)
                    threat_assess = self.evaluate_threat_assessment(model, cfg, device, samples=100)
                    grass_seek = self.evaluate_grass_seeking(model, cfg, device, samples=100)
                    
                    episode_data[episode]['prey_evasion'] = self.format_behavior_score(evade['rate'], evade.get('success'), evade.get('total'))
                    episode_data[episode]['prey_mating'] = self.format_behavior_score(mate_prey['rate'], mate_prey.get('success'), mate_prey.get('total'))
                    episode_data[episode]['prey_multi_target'] = self.format_behavior_score(multi_target['rate'], multi_target.get('success'), multi_target.get('total'))
                    episode_data[episode]['prey_flocking'] = self.format_behavior_score(flocking['rate'], flocking.get('success'), flocking.get('total'))
                    episode_data[episode]['prey_threat_assess'] = self.format_behavior_score(threat_assess['rate'], threat_assess.get('success'), threat_assess.get('total'))
                    episode_data[episode]['prey_grass'] = self.format_behavior_score(grass_seek['rate'], grass_seek.get('success'), grass_seek.get('total'))
                
                # Update progress
                processed += 1
                pct = int(processed / total_checkpoints * 100)
                self.root.after(0, lambda p=pct: self._update_progress(p))
            
            # Complete analysis on main thread
            self.root.after(0, lambda: self._analysis_complete(episode_data, None))
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"Behavior analysis error:\n{traceback.format_exc()}")
            self.root.after(0, lambda: self._analysis_complete(None, error_msg))
    
    def _update_progress(self, percent):
        """Update progress bar from main thread"""
        self.behavior_progress['value'] = percent
        self.progress_label.config(text=f"{percent}%")
    
    def _analysis_complete(self, episode_data, error_msg):
        """Called on main thread when analysis completes"""
        # Hide progress bar and label
        self.behavior_progress['value'] = 0
        self.behavior_progress.pack_forget()
        self.progress_label.pack_forget()
        
        if error_msg:
            for widget in self.behavior_table_frame.winfo_children():
                widget.destroy()
            error_label = tk.Label(self.behavior_table_frame, text=f"Error: {error_msg}", 
                                  font=('Arial', 11), fg='red')
            error_label.grid(row=0, column=0)
            self.behavior_status.config(text="Error - see status bar", foreground="red")
            self.app.status_label.config(text=f"Error: {error_msg}", foreground="red")
        else:
            # Populate table
            self._populate_table(episode_data)
            # Store episode_data for copy function
            self._last_episode_data = episode_data
            self.behavior_status.config(text=f"Analyzed {len(episode_data)} episodes", foreground="green")
            self.app.status_label.config(text="Behavior analysis complete", foreground="green")
    
    def copy_csv(self):
        """Copy behavior analysis to clipboard in CSV format"""
        import io
        csv = io.StringIO()
        
        # Check if we have data
        if not hasattr(self, '_last_episode_data') or not self._last_episode_data:
            self.app.status_label.config(text="No behavior data to copy. Run analysis first.", foreground="red")
            self.root.after(3000, lambda: self.app.status_label.config(text="Ready", foreground="gray"))
            return
        
        csv.write("# Behavior Analysis Results\n")
        csv.write("episode,pred_hunting,pred_hunger,pred_mating,pred_selective,pred_energy_mgmt,")
        csv.write("prey_evasion,prey_mating,prey_multi_target,prey_flocking,prey_threat_assess,prey_grass\n")
        
        for ep in sorted(self._last_episode_data.keys()):
            data = self._last_episode_data[ep]
            row = [str(ep)]
            for key in ['pred_hunting', 'pred_hunger', 'pred_mating', 'pred_selective', 'pred_energy_mgmt',
                       'prey_evasion', 'prey_mating', 'prey_multi_target', 'prey_flocking', 'prey_threat_assess', 'prey_grass']:
                val = data.get(key, '')
                # Remove % sign for pure CSV
                if isinstance(val, str) and '%' in val:
                    val = val.replace('%', '')
                row.append(str(val) if val else '')
            csv.write(','.join(row) + '\n')
        
        self.root.clipboard_clear()
        self.root.clipboard_append(csv.getvalue())
        self.app.status_label.config(text=f"Copied behavior analysis ({len(self._last_episode_data)} episodes)", foreground="green")
        self.root.after(3000, lambda: self.app.status_label.config(text="Ready", foreground="gray"))
    
    def _populate_table(self, episode_data):
        """Populate the behavior table with data"""
        # Clear existing widgets
        for widget in self.behavior_table_frame.winfo_children():
            widget.destroy()
        for widget in self.behavior_header_frame.winfo_children():
            widget.destroy()
        
        colors = {
            'excellent': '#d4edda',
            'good': '#fff3cd',
            'poor': '#f8d7da'
        }
        
        # Create header
        headers = ["Episode", "Pred: Hunting", "Pred: Hunger", "Pred: Mating",
                  "Pred: Selective", "Pred: Energy",
                  "Prey: Evasion", "Prey: Mating",
                  "Prey: Multi-Tgt", "Prey: Flocking", "Prey: Threat", "Prey: Grass"]
        
        for col, header_text in enumerate(headers):
            header_label = tk.Label(self.behavior_header_frame, text=header_text, 
                                   font=('Arial', 10, 'bold'), bg='#e9ecef', 
                                   relief=tk.RIDGE, borderwidth=1, width=14, padx=5, pady=5)
            header_label.grid(row=0, column=col, sticky='ew')
        
        # Data rows
        for row_idx, episode in enumerate(sorted(episode_data.keys(), reverse=True)):
            data = episode_data[episode]
            
            # Episode number
            tk.Label(self.behavior_table_frame, text=str(episode),
                    font=('Arial', 10, 'bold'), bg='white',
                    relief=tk.RIDGE, borderwidth=1, width=14, padx=5, pady=4).grid(row=row_idx, column=0, sticky='ew')
            
            # Behavior cells
            keys = ['pred_hunting', 'pred_hunger', 'pred_mating', 'pred_selective', 'pred_energy_mgmt',
                   'prey_evasion', 'prey_mating', 'prey_multi_target', 'prey_flocking', 
                   'prey_threat_assess', 'prey_grass']
            
            for col_idx, key in enumerate(keys, start=1):
                value_str = data.get(key, 'N/A')
                
                bg_color = 'white'
                match = re.search(r'(\d+)%', value_str)
                if match:
                    try:
                        score = int(match.group(1))
                        if score >= 80:
                            bg_color = colors['excellent']
                        elif score >= 60:
                            bg_color = colors['good']
                        else:
                            bg_color = colors['poor']
                    except ValueError:
                        pass
                
                tk.Label(self.behavior_table_frame, text=value_str,
                        font=('Arial', 10), bg=bg_color,
                        relief=tk.RIDGE, borderwidth=1, width=14, padx=5, pady=4).grid(row=row_idx, column=col_idx, sticky='ew')
    
    @staticmethod
    def format_behavior_score(rate, success=None, total=None):
        """Format behavior score with compact ASCII icons and optional counts."""
        if rate >= 80:
            emoji = '[+]'
        elif rate >= 60:
            emoji = '[~]'
        else:
            emoji = '[x]'

        suffix = f"{rate:.0f}%"
        if success is not None and total:
            suffix = f"{rate:.0f}% ({int(success)}/{int(total)})"
        return f"{emoji} {suffix}"
    # === Behavior Evaluation Methods ===
    
    @staticmethod
    def make_vis_batch(B, N, targets, device="cpu", subject_is_predator=False, vision_range=8):
        """Create visible animals batch with properly normalized features.
        
        Features (9 per animal):
        0: dx / vision_range (normalized direction x)
        1: dy / vision_range (normalized direction y)
        2: distance / vision_range (normalized distance)
        3: is_predator (binary)
        4: is_prey (binary)
        5: same_species (binary)
        6: same_type (binary) 
        7: 0.0 (unused)
        8: is_present (binary)
        """
        vis = torch.zeros(B, N, 9, device=device)
        for i, target in enumerate(targets):
            if i >= N:
                break
            dx = target['dx']
            dy = target['dy']
            distance = float((dx**2 + dy**2) ** 0.5)
            
            vis[:, i, 0] = dx / vision_range  # Normalized dx
            vis[:, i, 1] = dy / vision_range  # Normalized dy
            vis[:, i, 2] = distance / vision_range  # Normalized distance
            
            is_pred = target.get('is_predator', False)
            is_prey = target.get('is_prey', False)
            vis[:, i, 3] = 1.0 if is_pred else 0.0
            vis[:, i, 4] = 1.0 if is_prey else 0.0
            
            same_species = (subject_is_predator and is_pred) or ((not subject_is_predator) and is_prey)
            vis[:, i, 5] = 1.0 if same_species else 0.0
            vis[:, i, 6] = 1.0 if same_species else 0.0  # same_type = same_species for now
            vis[:, i, 7] = 0.0  # unused
            vis[:, i, 8] = 1.0  # is_present
        return vis
    
    @staticmethod
    def get_action_dirs():
        """Get normalized action direction vectors"""
        dirs = torch.tensor([
            [0.0, -1.0], [1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
            [0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0], [-1.0, -1.0]
        ], dtype=torch.float32)
        dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
        return dirs
    
    @staticmethod
    def get_intentional_threshold(config):
        """Calculate threshold for intentional movement based on entropy coefficient.
        
        Higher entropy = softer policies during training = need LOWER threshold
        to detect intentional behavior (more lenient).
        Lower entropy = sharper policies = can use HIGHER threshold (stricter).
        
        With 8 discrete directions and uniform random policy:
        - dot > 0: 50% random baseline
        - dot > 0.5: ~31% random baseline  
        - dot > 0.7: ~25% random baseline
        
        Formula: threshold = 0.65 - entropy_coef * 3.0
        - entropy=0.01 -> threshold=0.62 (low exploration, strict test)
        - entropy=0.04 -> threshold=0.53 (moderate)
        - entropy=0.10 -> threshold=0.35 (high exploration, lenient test)
        """
        entropy_coef = getattr(config, 'ENTROPY_COEF', 0.04)
        threshold = 0.65 - entropy_coef * 3.0
        return min(0.70, max(0.30, threshold))  # Clamp between 0.30 and 0.70
    
    @torch.no_grad()
    def evaluate_hunting(self, model, config, device, samples=50):
        """Test predator hunting: moving toward prey"""
        model.eval()
        threshold = self.get_intentional_threshold(config)
        obs = torch.zeros(1, config.SELF_FEATURE_DIM, device=device)
        obs[0, 3] = 1.0  # species B
        obs[0, 4] = 1.0  # is_predator
        obs[0, 5] = 0.2  # hunger_level (20 steps since last meal, not hungry)
        obs[0, 16] = 0.8  # energy (80% full)
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        approach_count = 0
        for _ in range(samples):
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(2, 10)
            dx, dy = dist * np.cos(angle), dist * np.sin(angle)
            
            targets = [{'dx': dx, 'dy': dy, 'is_prey': True}]
            vis = self.make_vis_batch(1, N, targets, device, subject_is_predator=True, vision_range=config.PREDATOR_VISION_RANGE)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            target_vec = torch.tensor([[dx, dy]], device=device, dtype=torch.float32)
            target_vec = target_vec / (target_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            
            # Dynamic threshold based on entropy coefficient
            if (action_vec @ target_vec.T).item() > threshold:
                approach_count += 1
        
        return {'rate': (approach_count / samples) * 100, 'success': approach_count, 'total': samples}
    
    @torch.no_grad()
    def evaluate_evasion(self, model, config, device, samples=50):
        """Test prey evasion: moving away from predators"""
        model.eval()
        threshold = self.get_intentional_threshold(config)
        obs = torch.zeros(1, config.SELF_FEATURE_DIM, device=device)
        obs[0, 2] = 1.0  # species A / is_prey
        obs[0, 16] = 0.8  # energy (80% full)
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        evade_count = 0
        for _ in range(samples):
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(2, 10)
            dx, dy = dist * np.cos(angle), dist * np.sin(angle)
            
            targets = [{'dx': dx, 'dy': dy, 'is_predator': True}]
            vis = self.make_vis_batch(1, N, targets, device, subject_is_predator=False, vision_range=config.PREY_VISION_RANGE)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            target_vec = torch.tensor([[dx, dy]], device=device, dtype=torch.float32)
            target_vec = target_vec / (target_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            
            # Dynamic threshold based on entropy coefficient (negative for fleeing)
            if (action_vec @ target_vec.T).item() < -threshold:
                evade_count += 1
        
        return {'rate': (evade_count / samples) * 100, 'success': evade_count, 'total': samples}
    
    @torch.no_grad()
    def evaluate_mating_behavior(self, model, config, device, is_predator=True, samples=50):
        """Test mating: approaching same-species when high energy"""
        model.eval()
        threshold = self.get_intentional_threshold(config)
        obs = torch.zeros(1, config.SELF_FEATURE_DIM, device=device)
        if is_predator:
            obs[0, 3] = 1.0  # species B
            obs[0, 4] = 1.0  # predator flag
        else:
            obs[0, 2] = 1.0  # species A
        obs[0, 5] = 0.1  # hunger_level (10 steps since last meal, not hungry)
        obs[0, 16] = 0.9  # energy (90% full, high)
        obs[0, 6] = 0.5  # mature age
        
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        approach_count = 0
        for _ in range(samples):
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(1, 5)
            dx, dy = dist * np.cos(angle), dist * np.sin(angle)
            
            targets = [{'dx': dx, 'dy': dy, 'is_predator': is_predator, 'is_prey': not is_predator}]
            vision_range = config.PREDATOR_VISION_RANGE if is_predator else config.PREY_VISION_RANGE
            vis = self.make_vis_batch(1, N, targets, device, subject_is_predator=is_predator, vision_range=vision_range)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            target_vec = torch.tensor([[dx, dy]], device=device, dtype=torch.float32)
            target_vec = target_vec / (target_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            
            # Dynamic threshold based on entropy coefficient
            if (action_vec @ target_vec.T).item() > threshold:
                approach_count += 1
        
        return {'rate': (approach_count / samples) * 100, 'success': approach_count, 'total': samples}
    
    @torch.no_grad()
    def evaluate_hunger_response(self, model, config, device, samples=50):
        """Test hunger response: predator approaching prey when low energy"""
        model.eval()
        threshold = self.get_intentional_threshold(config)
        obs = torch.zeros(1, config.SELF_FEATURE_DIM, device=device)
        obs[0, 3] = 1.0  # species B
        obs[0, 4] = 1.0  # is_predator
        obs[0, 5] = 0.85  # hunger_level (85 steps since last meal, hungry)
        obs[0, 16] = 0.2  # energy (20% full, LOW)
        
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        hunt_count = 0
        for _ in range(samples):
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(2, 10)
            dx, dy = dist * np.cos(angle), dist * np.sin(angle)
            
            targets = [{'dx': dx, 'dy': dy, 'is_prey': True}]
            vis = self.make_vis_batch(1, N, targets, device, subject_is_predator=True, vision_range=config.PREDATOR_VISION_RANGE)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            target_vec = torch.tensor([[dx, dy]], device=device, dtype=torch.float32)
            target_vec = target_vec / (target_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            
            # Dynamic threshold based on entropy coefficient
            if (action_vec @ target_vec.T).item() > threshold:
                hunt_count += 1
        
        return {'rate': (hunt_count / samples) * 100, 'success': hunt_count, 'total': samples}
    
    @torch.no_grad()
    def evaluate_selective_hunting(self, model, config, device, samples=50):
        """Test selective hunting: predators choosing closer prey"""
        model.eval()
        threshold = self.get_intentional_threshold(config)
        obs = torch.zeros(1, config.SELF_FEATURE_DIM, device=device)
        obs[0, 3] = 1.0  # species B
        obs[0, 4] = 1.0  # is_predator
        obs[0, 5] = 0.6  # hunger_level (60 steps since last meal)
        obs[0, 16] = 0.6  # energy (60% full)
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        chose_closer = 0
        for _ in range(samples):
            angle_close = np.random.uniform(0, 2*np.pi)
            angle_far = np.random.uniform(0, 2*np.pi)
            dist_close = np.random.uniform(2, 4)
            dist_far = np.random.uniform(7, 10)
            
            dx_close, dy_close = dist_close * np.cos(angle_close), dist_close * np.sin(angle_close)
            dx_far, dy_far = dist_far * np.cos(angle_far), dist_far * np.sin(angle_far)
            
            targets = [
                {'dx': dx_close, 'dy': dy_close, 'is_prey': True},
                {'dx': dx_far, 'dy': dy_far, 'is_prey': True}
            ]
            vis = self.make_vis_batch(1, N, targets, device, subject_is_predator=True, vision_range=config.PREDATOR_VISION_RANGE)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            close_vec = torch.tensor([[dx_close, dy_close]], device=device, dtype=torch.float32)
            far_vec = torch.tensor([[dx_far, dy_far]], device=device, dtype=torch.float32)
            close_vec = close_vec / (close_vec.norm() + 1e-8)
            far_vec = far_vec / (far_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            
            # Require clear preference with dynamic threshold
            dot_close = (action_vec @ close_vec.T).item()
            dot_far = (action_vec @ far_vec.T).item()
            if dot_close > threshold * 0.6 and dot_close > dot_far + 0.2:
                chose_closer += 1
        
        return {'rate': (chose_closer / samples) * 100, 'success': chose_closer, 'total': samples}
    
    @torch.no_grad()
    def evaluate_multi_target_handling(self, model, config, device, samples=50):
        """Test multi-target handling: prey fleeing from nearest predator"""
        model.eval()
        threshold = self.get_intentional_threshold(config)
        obs = torch.zeros(1, config.SELF_FEATURE_DIM, device=device)
        obs[0, 2] = 1.0  # species A / is_prey
        obs[0, 16] = 0.8  # energy (80% full)
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        fled_nearest = 0
        for _ in range(samples):
            angle_close = np.random.uniform(0, 2*np.pi)
            angle_far = np.random.uniform(0, 2*np.pi)
            dist_close = np.random.uniform(2, 4)
            dist_far = np.random.uniform(7, 10)
            
            dx_close, dy_close = dist_close * np.cos(angle_close), dist_close * np.sin(angle_close)
            dx_far, dy_far = dist_far * np.cos(angle_far), dist_far * np.sin(angle_far)
            
            targets = [
                {'dx': dx_close, 'dy': dy_close, 'is_predator': True},
                {'dx': dx_far, 'dy': dy_far, 'is_predator': True}
            ]
            vis = self.make_vis_batch(1, N, targets, device, subject_is_predator=False, vision_range=config.PREY_VISION_RANGE)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            close_vec = torch.tensor([[dx_close, dy_close]], device=device, dtype=torch.float32)
            close_vec = close_vec / (close_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            
            # Dynamic threshold (negative for fleeing)
            if (action_vec @ close_vec.T).item() < -threshold:
                fled_nearest += 1
        
        return {'rate': (fled_nearest / samples) * 100, 'success': fled_nearest, 'total': samples}
    
    @torch.no_grad()
    def evaluate_flocking(self, model, config, device, samples=50):
        """Test flocking: prey moving toward same-species when predator nearby"""
        model.eval()
        threshold = self.get_intentional_threshold(config)
        obs = torch.zeros(1, config.SELF_FEATURE_DIM, device=device)
        obs[0, 2] = 1.0  # species A / is_prey
        obs[0, 16] = 0.7  # energy (70% full)
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        flocked = 0
        for _ in range(samples):
            angle_pred = np.random.uniform(0, 2*np.pi)
            angle_prey = np.random.uniform(0, 2*np.pi)
            
            dx_pred = 9 * np.cos(angle_pred)
            dy_pred = 9 * np.sin(angle_pred)
            dx_prey = 3 * np.cos(angle_prey)
            dy_prey = 3 * np.sin(angle_prey)
            
            targets = [
                {'dx': dx_pred, 'dy': dy_pred, 'is_predator': True},
                {'dx': dx_prey, 'dy': dy_prey, 'is_prey': True}
            ]
            vis = self.make_vis_batch(1, N, targets, device, subject_is_predator=False, vision_range=config.PREY_VISION_RANGE)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            prey_vec = torch.tensor([[dx_prey, dy_prey]], device=device, dtype=torch.float32)
            prey_vec = prey_vec / (prey_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            
            # Dynamic threshold based on entropy coefficient
            if (action_vec @ prey_vec.T).item() > threshold:
                flocked += 1
        
        return {'rate': (flocked / samples) * 100, 'success': flocked, 'total': samples}
    
    @torch.no_grad()
    def evaluate_threat_assessment(self, model, config, device, samples=50):
        """Test threat assessment: prey reacts more to near vs far predators"""
        model.eval()
        threshold = self.get_intentional_threshold(config)
        obs = torch.zeros(1, config.SELF_FEATURE_DIM, device=device)
        obs[0, 2] = 1.0  # species A / is_prey
        obs[0, 16] = 0.8  # energy (80% full)
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        evade_near = 0
        evade_far = 0
        
        for _ in range(samples):
            angle = np.random.uniform(0, 2*np.pi)
            
            # Near predator test
            dx_near, dy_near = 2 * np.cos(angle), 2 * np.sin(angle)
            targets = [{'dx': dx_near, 'dy': dy_near, 'is_predator': True}]
            vis = self.make_vis_batch(1, N, targets, device, subject_is_predator=False, vision_range=config.PREY_VISION_RANGE)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            target_vec = torch.tensor([[dx_near, dy_near]], device=device, dtype=torch.float32)
            target_vec = target_vec / (target_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            
            # Dynamic threshold (negative for evade)
            if (action_vec @ target_vec.T).item() < -threshold:
                evade_near += 1
            
            # Far predator test
            dx_far, dy_far = 9 * np.cos(angle), 9 * np.sin(angle)
            targets = [{'dx': dx_far, 'dy': dy_far, 'is_predator': True}]
            vis = self.make_vis_batch(1, N, targets, device, subject_is_predator=False, vision_range=config.PREY_VISION_RANGE)
            
            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()
            
            target_vec = torch.tensor([[dx_far, dy_far]], device=device, dtype=torch.float32)
            target_vec = target_vec / (target_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            
            # Dynamic threshold (negative for evade)
            if (action_vec @ target_vec.T).item() < -threshold:
                evade_far += 1
        
        # Better threat assessment = stronger reaction to near threats
        diff = (evade_near - evade_far) / samples * 100
        return {'rate': max(0, min(100, 50 + diff))}
    
    @torch.no_grad()
    def evaluate_energy_management(self, model, config, device, samples=50):
        """Test energy management: hunting urgency increases as energy drops"""
        model.eval()
        threshold = self.get_intentional_threshold(config)
        obs_high = torch.zeros(1, config.SELF_FEATURE_DIM, device=device)
        obs_high[0, 3] = 1.0  # species B
        obs_high[0, 4] = 1.0  # is_predator
        obs_high[0, 5] = 0.2  # hunger_level (20 steps since meal, not hungry)
        obs_high[0, 16] = 0.8  # energy (80% full, HIGH)
        
        obs_low = torch.zeros(1, config.SELF_FEATURE_DIM, device=device)
        obs_low[0, 3] = 1.0  # species B
        obs_low[0, 4] = 1.0  # is_predator
        obs_low[0, 5] = 0.85  # hunger_level (85 steps since meal, hungry)
        obs_low[0, 16] = 0.15  # energy (15% full, LOW)
        
        N = config.MAX_VISIBLE_ANIMALS
        action_dirs = self.get_action_dirs().to(device)
        
        hunt_high = 0
        hunt_low = 0
        
        for _ in range(samples):
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(4, 8)
            dx, dy = dist * np.cos(angle), dist * np.sin(angle)
            
            targets = [{'dx': dx, 'dy': dy, 'is_prey': True}]
            vis = self.make_vis_batch(1, N, targets, device, subject_is_predator=True, vision_range=config.PREDATOR_VISION_RANGE)
            
            target_vec = torch.tensor([[dx, dy]], device=device, dtype=torch.float32)
            target_vec = target_vec / (target_vec.norm() + 1e-8)
            
            _, move_probs_high, _ = model.forward(obs_high, vis)
            action_high = torch.argmax(move_probs_high, dim=1).item()
            # Dynamic threshold
            if (action_dirs[action_high:action_high+1] @ target_vec.T).item() > threshold:
                hunt_high += 1
            
            _, move_probs_low, _ = model.forward(obs_low, vis)
            action_low = torch.argmax(move_probs_low, dim=1).item()
            # Dynamic threshold
            if (action_dirs[action_low:action_low+1] @ target_vec.T).item() > threshold:
                hunt_low += 1
        
        urgency_increase = (hunt_low - hunt_high) / samples * 100
        return {'rate': max(0, min(100, 50 + urgency_increase))}
    
    @torch.no_grad()
    def evaluate_grass_seeking(self, model, config, device, samples=50):
        """Test prey moving toward visible grass"""
        model.eval()
        threshold = self.get_intentional_threshold(config)
        obs = torch.zeros(1, config.SELF_FEATURE_DIM, device=device)
        obs[0, 2] = 1.0  # species A / is_prey
        obs[0, 16] = 0.6  # energy (60% full)
        obs[0, 20] = 0.0
        obs[0, 21] = -1.0

        patch_size = getattr(config, "GRASS_PATCH_SIZE", 0)
        radius = getattr(config, "PREY_VISION_RANGE", 0)
        diam = getattr(config, "GRASS_PATCH_DIAMETER", 1)
        action_dirs = self.get_action_dirs().to(device)

        success = 0
        vis = self.make_vis_batch(1, config.MAX_VISIBLE_ANIMALS, [], device, subject_is_predator=False, vision_range=config.PREY_VISION_RANGE)

        for _ in range(samples):
            if patch_size == 0:
                break
            dx = int(np.random.randint(-radius, radius + 1))
            dy = int(-np.random.randint(1, radius + 1))
            if dx == 0 and dy == 0:
                dy = -1
            idx = (dy + radius) * diam + (dx + radius)
            if idx < 0 or idx >= patch_size:
                continue

            obs[:, 34:] = 0.0
            obs[0, 34 + idx] = 1.0

            _, move_probs, _ = model.forward(obs, vis)
            action = torch.argmax(move_probs, dim=1).item()

            target_vec = torch.tensor([[float(dx), float(dy)]], device=device)
            if target_vec.norm() == 0:
                continue
            target_vec = target_vec / (target_vec.norm() + 1e-8)
            action_vec = action_dirs[action:action+1]
            # Dynamic threshold
            if (action_vec @ target_vec.T).item() > threshold:
                success += 1

        return {'rate': (success / max(1, samples)) * 100, 'success': success, 'total': samples}





