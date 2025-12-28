"""
Experience Replay Buffer for PPO Training
Stores and samples experiences for better learning efficiency
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from collections import deque
import random


class PPOMemory:
    """
    Memory buffer for PPO algorithm with hierarchical policy support
    
    Supports two modes:
    1. Simple mode (backward compatible): stores combined state/action/log_prob
    2. Hierarchical mode: stores separate turn and move observations/actions/log_probs
    """
    def __init__(self, batch_size: int = 64, hierarchical: bool = False):
        # Simple mode storage (backward compatible)
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.next_values = []  # TD(0) bootstrapping
        self.traj_ids = []     # Trajectory grouping
        self.rewards = []
        self.dones = []
        
        # Hierarchical mode storage
        self.hierarchical = hierarchical
        if hierarchical:
            self.obs_turn = []  # Pre-turn observations
            self.vis_turn = []  # Pre-turn visible animals
            self.turn_actions = []
            self.turn_log_probs_old = []
            
            self.obs_move = []  # Post-turn observations
            self.vis_move = []  # Post-turn visible animals
            self.move_actions = []
            self.move_log_probs_old = []
        
        self.batch_size = batch_size
        
    def add(self, state: torch.Tensor = None, action: torch.Tensor = None, log_prob: torch.Tensor = None,
            value: torch.Tensor = None, reward: float = None, done: bool = None,
            transition: Dict = None):
        """Add experience (simple mode) or hierarchical transition
        
        Simple mode (backward compatible):
            add(state, action, log_prob, value, reward, done)
            
        Hierarchical mode:
            add(transition={...}, reward=..., done=...)
            where transition contains obs_turn, vis_turn, turn_action, turn_logp_old,
                                    obs_move, vis_move, move_action, move_logp_old, value_old
        """
        # Enforce done flag default (never None)
        if done is None:
            done = False
        
        if transition is not None:
            # Hierarchical mode
            if not self.hierarchical:
                raise ValueError("Hierarchical transition provided but memory not in hierarchical mode")
            
            # Convert actions to tensors with consistent scalar shape
            turn_action = transition['turn_action']
            if isinstance(turn_action, torch.Tensor):
                turn_action = turn_action.view(-1)[0].long()  # Extract scalar
            else:
                turn_action = torch.tensor(turn_action, dtype=torch.long)
            
            move_action = transition['move_action']
            if isinstance(move_action, torch.Tensor):
                move_action = move_action.view(-1)[0].long()  # Extract scalar
            else:
                move_action = torch.tensor(move_action, dtype=torch.long)
            
            self.obs_turn.append(transition['obs_turn'])
            self.vis_turn.append(transition['vis_turn'])
            self.turn_actions.append(turn_action)
            self.turn_log_probs_old.append(transition['turn_logp_old'].detach().view(1))
            
            self.obs_move.append(transition['obs_move'])
            self.vis_move.append(transition['vis_move'])
            self.move_actions.append(move_action)
            self.move_log_probs_old.append(transition['move_logp_old'].detach().view(1))
            
            self.values.append(transition['value_old'].detach().view(1, 1))
            self.next_values.append(transition['value_next'].detach().view(1, 1))
            self.traj_ids.append(int(transition['traj_id']))
            self.rewards.append(reward)
            self.dones.append(done)
        else:
            # Simple mode (backward compatible)
            if self.hierarchical:
                raise ValueError("Simple experience provided but memory in hierarchical mode")
            
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.rewards.append(reward)
            self.dones.append(done)
    
    def compute_returns_and_advantages(self, gamma: float = 0.99):
        """
        Compute returns and advantages using TD(0)
        
        Args:
            gamma: Discount factor
        """
        if len(self.values) == 0:
            device = torch.device("cpu")
            empty = torch.tensor([], device=device)
            return empty, empty

        device = self.values[0].device

        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device).view(-1)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device).view(-1)

        values = torch.cat(self.values).to(device).view(-1)         # V(s_t)
        next_values = torch.cat(self.next_values).to(device).view(-1)  # V(s_{t+1})

        # TD(0) advantages: delta = r + gamma * V(s_{t+1}) * (1-done) - V(s_t)
        deltas = rewards + gamma * next_values * (1.0 - dones) - values
        advantages = deltas
        returns = advantages + values

        # Normalize advantages (with safety checks for edge cases)
        std = advantages.std(unbiased=False)
        if torch.isfinite(std) and std > 1e-6:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)
        else:
            # If std is too small or non-finite, just center
            advantages = advantages - advantages.mean()

        # Store for batching
        self.returns = returns
        self.advantages = advantages

        return returns, advantages
    
    def get_batches(self):
        """
        Generate random mini-batches for training
        """
        if self.hierarchical:
            n_samples = len(self.rewards)
        else:
            n_samples = len(self.states)
        
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            
            if self.hierarchical:
                yield {
                    'obs_turn': [self.obs_turn[i] for i in batch_indices],
                    'vis_turn': [self.vis_turn[i] for i in batch_indices],
                    'turn_actions': torch.stack([self.turn_actions[i] for i in batch_indices]),
                    'turn_log_probs_old': torch.stack([self.turn_log_probs_old[i] for i in batch_indices]),
                    
                    'obs_move': [self.obs_move[i] for i in batch_indices],
                    'vis_move': [self.vis_move[i] for i in batch_indices],
                    'move_actions': torch.stack([self.move_actions[i] for i in batch_indices]),
                    'move_log_probs_old': torch.stack([self.move_log_probs_old[i] for i in batch_indices]),
                    
                    # S3: Direct tensor slicing (much faster than list comprehension)
                    'returns': self.returns[batch_indices] if isinstance(self.returns, torch.Tensor) else torch.tensor([self.returns[i] for i in batch_indices], dtype=torch.float32),
                    'advantages': self.advantages[batch_indices] if isinstance(self.advantages, torch.Tensor) else torch.tensor([self.advantages[i] for i in batch_indices], dtype=torch.float32)
                }
            else:
                yield {
                    'states': [self.states[i] for i in batch_indices],
                    'actions': torch.stack([self.actions[i] for i in batch_indices]),
                    'old_log_probs': torch.stack([self.log_probs[i] for i in batch_indices]),
                    # S3: Direct tensor slicing
                    'returns': self.returns[batch_indices] if isinstance(self.returns, torch.Tensor) else torch.tensor([self.returns[i] for i in batch_indices], dtype=torch.float32),
                    'advantages': self.advantages[batch_indices] if isinstance(self.advantages, torch.Tensor) else torch.tensor([self.advantages[i] for i in batch_indices], dtype=torch.float32)
                }
    
    def clear(self):
        """Clear all stored experiences"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.next_values.clear()
        self.traj_ids.clear()
        
        # Reset computed tensors
        self.returns = None
        self.advantages = None
        
        if self.hierarchical:
            self.obs_turn.clear()
            self.vis_turn.clear()
            self.turn_actions.clear()
            self.turn_log_probs_old.clear()
            
            self.obs_move.clear()
            self.vis_move.clear()
            self.move_actions.clear()
            self.move_log_probs_old.clear()


class ExperienceReplayBuffer:
    """
    Large experience replay buffer for storing diverse experiences
    Improves sample efficiency by reusing past experiences
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, experience: Dict):
        """
        Add an experience tuple
        
        Args:
            experience: Dict containing {state, action, reward, next_state, done, ...}
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample random batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def sample_recent(self, n: int) -> List[Dict]:
        """Sample most recent n experiences"""
        return list(self.buffer)[-n:] if len(self.buffer) >= n else list(self.buffer)
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()


class PrioritizedExperienceReplay:
    """
    Prioritized Experience Replay
    Samples important experiences more frequently
    """
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.array([])
        self.position = 0
    
    def add(self, experience: Dict, priority: float = None):
        """
        Add experience with priority
        
        Args:
            experience: Experience dictionary
            priority: Priority value (higher = more important)
        """
        max_priority = self.priorities.max() if len(self.priorities) > 0 else 1.0
        
        if priority is None:
            priority = max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities = np.append(self.priorities, priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Sample batch based on priorities
        
        Returns:
            experiences: Batch of experiences
            indices: Indices of sampled experiences
            weights: Importance sampling weights
        """
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increase beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)
