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
    Memory buffer for PPO algorithm
    Stores trajectories and computes advantages
    """
    def __init__(self, batch_size: int = 64):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def add(self, state: torch.Tensor, action: torch.Tensor, log_prob: torch.Tensor,
            value: torch.Tensor, reward: float, done: bool):
        """Add a single experience"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, next_value: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation)
        
        Args:
            next_value: Value of next state (for bootstrapping)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        if len(self.values) == 0:
            device = next_value.device
            empty = torch.tensor([], device=device)
            return empty, empty

        device = self.values[0].device
        next_value = next_value.to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        values = torch.cat(self.values).to(device).view(-1)  # Flatten to 1D
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.tensor(0.0, device=device)
        last_value = next_value
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def get_batches(self):
        """
        Generate random mini-batches for training
        """
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)
        
        for start_idx in range(0, len(self.states), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            
            yield {
                'states': [self.states[i] for i in batch_indices],
                'actions': torch.stack([self.actions[i] for i in batch_indices]),
                'old_log_probs': torch.stack([self.log_probs[i] for i in batch_indices]),
                'returns': torch.tensor([self.returns[i] for i in batch_indices], dtype=torch.float32),
                'advantages': torch.tensor([self.advantages[i] for i in batch_indices], dtype=torch.float32)
            }
    
    def clear(self):
        """Clear all stored experiences"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()


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
