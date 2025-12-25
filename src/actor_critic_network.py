"""
Advanced Actor-Critic Neural Network with Multi-Head Attention
Implements PPO (Proximal Policy Optimization) architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention for focusing on different aspects of visible animals"""
    def __init__(self, embed_dim: int, num_heads: int = 3):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Separate attention for threats, prey, and mates
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out(attended)
        
        return output


class ActorCriticNetwork(nn.Module):
    """
    Advanced Actor-Critic network with multi-head attention
    Separate networks for policy (actor) and value estimation (critic)
    """
    def __init__(self, config):
        super(ActorCriticNetwork, self).__init__()
        
        # Enhanced input: 20 features (position, type, state, threat info, age, energy)
        self.self_embed = nn.Linear(20, 64)
        
        # Visible animals: 8 features per animal
        self.animal_embed = nn.Linear(8, 64)
        # DirectML-compatible transform (no Dropout, no RNN)
        self.animal_transform1 = nn.Linear(64, 64)
        self.animal_transform2 = nn.Linear(64, 64)
        
        # Multi-head attention for focusing on important animals
        self.attention = MultiHeadAttention(64, num_heads=4)
        
        # Pooling layer to aggregate attended information
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Actor (policy) network
        self.actor_shared = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.actor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # 8 action directions
        )
        
        # Critic (value) network - DirectML compatible (no Dropout)
        self.critic_shared = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single value estimate
        )
        
    def forward(self, animal_input: torch.Tensor, 
                visible_animals_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both policy and value
        
        Returns:
            action_probs: Probability distribution over actions
            state_value: Estimated value of current state
        """
        # Process self-state
        self_features = F.relu(self.self_embed(animal_input))
        
        # Process visible animals
        if visible_animals_input.size(1) > 0:
            # Embed each visible animal
            animal_embeds = self.animal_embed(visible_animals_input)
            
            # Transform with DirectML-compatible layers
            transformed = F.relu(self.animal_transform1(animal_embeds))
            transformed = self.animal_transform2(transformed)
            
            # Multi-head attention to focus on important animals
            attended = self.attention(transformed)
            
            # Pool to single vector
            context = self.pool(attended.transpose(1, 2)).squeeze(-1)
        else:
            context = torch.zeros(animal_input.size(0), 64, device=animal_input.device)
        
        # Combine self-state with context
        combined = torch.cat([self_features, context], dim=-1)
        
        # Actor network (policy)
        actor_features = self.actor_shared(combined)
        action_logits = self.actor_head(actor_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic network (value)
        critic_features = self.critic_shared(combined)
        state_value = self.critic_head(critic_features)
        
        return action_probs, state_value
    
    def get_action(self, animal_input: torch.Tensor, 
                   visible_animals_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and return log prob and value
        Used during training
        """
        action_probs, state_value = self.forward(animal_input, visible_animals_input)
        
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, state_value
    
    def evaluate_actions(self, animal_input: torch.Tensor,
                        visible_animals_input: torch.Tensor,
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions
        Used for PPO updates
        """
        action_probs, state_value = self.forward(animal_input, visible_animals_input)
        
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, state_value, entropy
