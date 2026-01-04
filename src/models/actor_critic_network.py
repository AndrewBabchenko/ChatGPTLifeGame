"""
Advanced Actor-Critic Neural Network with Cross-Attention and Dual Action Heads
Implements PPO (Proximal Policy Optimization) architecture with NN-controlled heading

OBS_VERSION = 5 (must match Animal.OBS_VERSION)
- Self-state: 34 base features + prey-only grass FOV map (flattened, length=GRASS_PATCH_SIZE), stacked by OBS_HISTORY_LEN
- Visible animals: 9 features per animal (no grass flag; index 7 is zero)
- Dual action heads: turn (3 actions) + move (8 actions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossAttention(nn.Module):
    """
    Cross-attention: self-state queries visible animals
    This is better than self-attention because what matters depends on MY state
    """
    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.head_dim = query_dim // num_heads
        
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        
        self.query = nn.Linear(query_dim, query_dim)
        self.key = nn.Linear(key_dim, query_dim)
        self.value = nn.Linear(key_dim, query_dim)
        self.out = nn.Linear(query_dim, query_dim)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, 
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, query_dim) - self-state embedding
            key_value: (batch, seq_len, key_dim) - visible animals embeddings
            mask: (batch, seq_len) - True means ignore this position (padding)
        Returns:
            output: (batch, query_dim) - context from visible animals
            attention_weights: (batch, num_heads, seq_len) - attention scores
        """
        batch_size = query.size(0)
        seq_len = key_value.size(1)
        
        # Expand query for multi-head
        Q = self.query(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D)
        K = self.key(key_value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D)
        V = self.value(key_value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, 1, S)
        
        # Apply mask if provided (padding = True → -1e9 for numerical stability)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            scores = scores.masked_fill(mask, -1e9)  # Use -1e9 instead of -inf
        
        attention_weights = F.softmax(scores, dim=-1)  # (B, H, 1, S)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (B, H, 1, D)
        
        # Concatenate heads and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, self.query_dim).squeeze(1)  # (B, query_dim)
        output = self.out(attended)
        
        return output, attention_weights.squeeze(2)  # (B, query_dim), (B, H, S)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network with dual action heads and cross-attention
    
        OBS_VERSION = 5:
        - Self-state: 34 base features + prey-only grass FOV map (flattened, length=GRASS_PATCH_SIZE), stacked by OBS_HISTORY_LEN
        - Visible animals: 9 features per animal
            [0-1]: relative dx/dy (signed, normalized)
            [2]: distance (normalized)
            [3-6]: is_predator, is_prey, same_species, same_type (binary)
            [7]: grass_present (always 0; grass is separate)
            [8]: is_present (1.0 = real, 0.0 = padding)
    
    Action space:
    - Turn: 3 actions (left=-1, straight=0, right=+1)
    - Move: 8 directions (N, NE, E, SE, S, SW, W, NW)
    """
    def __init__(self, config):
        super(ActorCriticNetwork, self).__init__()
        
        self.obs_version = 5  # Must match Animal.OBS_VERSION
        self.self_dim = getattr(config, "SELF_FEATURE_DIM", 34 + getattr(config, "GRASS_PATCH_SIZE", 0))
        
        # Self-state embedding: stacked base + grass map -> 256
        self.self_embed = nn.Linear(self.self_dim, 256)
        
        # Visible animals embedding: 9 features → 256
        self.animal_embed = nn.Linear(9, 256)
        self.animal_transform = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Cross-attention: self-state queries visible animals
        self.cross_attention = CrossAttention(query_dim=256, key_dim=256, num_heads=8)
        
        # Store attention weights for analysis
        self.last_attention_weights = None
        
        # Combined feature processing
        # Input: self(256) + context(256) = 512
        self.feature_fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Dual actor heads
        self.turn_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # LEFT, STRAIGHT, RIGHT
        )
        
        self.move_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # N, NE, E, SE, S, SW, W, NW
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Temperature for exploration (with safety clamp)
        self.temperature = max(0.1, getattr(config, 'ACTION_TEMPERATURE', 1.0))
        
        # Bias toward going straight (reduces jittery direction changes)
        # Applied as logit bonus to action index 1 (TURN_STRAIGHT)
        self.turn_straight_bias = getattr(config, 'TURN_STRAIGHT_BIAS', 0.0)
        
    def forward(self, animal_input: torch.Tensor, 
                visible_animals_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with cross-attention and dual action heads
        
        Args:
            animal_input: (batch, self_dim) - stacked self-state (current frame first)
            visible_animals_input: (batch, max_animals, 9) - visible animals
            
        Returns:
            turn_probs: (batch, 3) - turn action probabilities
            move_probs: (batch, 8) - move action probabilities
            state_value: (batch, 1) - state value estimate
        """
        batch_size = animal_input.size(0)
        
        # Contract checks (SHOULD DO for safety)
        assert animal_input.size(-1) == self.self_dim, f"Expected {self.self_dim} self features, got {animal_input.size(-1)}"
        assert visible_animals_input.size(-1) == 9, f"Expected 9 visible animal features, got {visible_animals_input.size(-1)}"
        
        # Process self-state
        self_features = F.relu(self.self_embed(animal_input))  # (B, 256)
        
        # Process visible animals with all-padding safety
        # Check if any real animals exist (is_present flag at index 8)
        has_any_real = (visible_animals_input[:, :, 8] > 0.5).any(dim=1)  # (B,)
        
        if visible_animals_input.size(1) > 0 and has_any_real.any():
            # Embed all visible animal slots
            animal_embeds = self.animal_embed(visible_animals_input)  # (B, N, 256)
            animal_embeds = self.animal_transform(animal_embeds)  # (B, N, 256)
            
            # Create padding mask from is_present flag (index 8)
            # is_present = 0.0 → padding → mask = True
            padding_mask = (visible_animals_input[:, :, 8] < 0.5)  # (B, N)
            
            # Initialize context
            context = torch.zeros(batch_size, 256, device=animal_input.device)
            
            # Vectorized processing: only run attention on valid rows
            valid_idx = has_any_real.nonzero(as_tuple=False).squeeze(-1)  # Indices of valid rows
            
            if valid_idx.numel() > 0:
                # Run attention only for valid rows (vectorized)
                valid_self_features = self_features[valid_idx]  # (V, 256)
                valid_animal_embeds = animal_embeds[valid_idx]  # (V, N, 256)
                valid_padding_mask = padding_mask[valid_idx]  # (V, N)
                
                valid_context, attention_weights = self.cross_attention(
                    valid_self_features, valid_animal_embeds, mask=valid_padding_mask
                )  # (V, 256), (V, H, N)
                
                # Scatter back to full batch
                context[valid_idx] = valid_context
                
                # Store attention weights (for debugging, only when batch_size==1)
                if batch_size == 1 and valid_idx.numel() == 1:
                    self.last_attention_weights = attention_weights
                else:
                    self.last_attention_weights = None
            else:
                self.last_attention_weights = None
        else:
            # No visible animals at all
            context = torch.zeros_like(self_features)
            self.last_attention_weights = None
        
        # Fuse self and context features
        combined = torch.cat([self_features, context], dim=-1)  # (B, 512)
        fused_features = self.feature_fusion(combined)  # (B, 512)
        
        # Dual actor heads with temperature
        turn_logits = self.turn_head(fused_features) / self.temperature  # (B, 3)
        move_logits = self.move_head(fused_features) / self.temperature  # (B, 8)
        
        # Apply straight bias to turn logits (index 1 = TURN_STRAIGHT)
        if self.turn_straight_bias > 0:
            turn_logits = turn_logits.clone()
            turn_logits[:, 1] = turn_logits[:, 1] + self.turn_straight_bias
        
        turn_probs = F.softmax(turn_logits, dim=-1)
        move_probs = F.softmax(move_logits, dim=-1)
        
        # Critic
        state_value = self.critic(fused_features)  # (B, 1)
        
        return turn_probs, move_probs, state_value
    
    def get_action(self, animal_input: torch.Tensor, 
                   visible_animals_input: torch.Tensor,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, 
                                                                   torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample both turn and move actions, return log probs and value
        Used during training and inference
        
        Args:
            animal_input: (batch, 34) - animal state
            visible_animals_input: (batch, max_animals, 9) - visible animals
            deterministic: If True, use argmax instead of sampling (for evaluation)
        
        Returns:
            turn_action: (batch,) - sampled/argmax turn action
            move_action: (batch,) - sampled/argmax move action
            turn_log_prob: (batch,) - log prob of turn action
            move_log_prob: (batch,) - log prob of move action
            state_value: (batch, 1) - value estimate
        """
        turn_probs, move_probs, state_value = self.forward(animal_input, visible_animals_input)
        
        # Create distributions for log_prob computation
        turn_dist = torch.distributions.Categorical(turn_probs)
        move_dist = torch.distributions.Categorical(move_probs)
        
        if deterministic:
            # Argmax instead of sampling (no randomness for evaluation)
            turn_action = torch.argmax(turn_probs, dim=-1)
            move_action = torch.argmax(move_probs, dim=-1)
        else:
            # Sample actions (training mode)
            turn_action = turn_dist.sample()
            move_action = move_dist.sample()
        
        # Compute log probs for the selected actions
        turn_log_prob = turn_dist.log_prob(turn_action)
        move_log_prob = move_dist.log_prob(move_action)
        
        return turn_action, move_action, turn_log_prob, move_log_prob, state_value
    
    def evaluate_actions(self, animal_input: torch.Tensor,
                        visible_animals_input: torch.Tensor,
                        turn_actions: torch.Tensor,
                        move_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions (for PPO updates)
        
        Returns:
            turn_log_probs: (batch,) - log probs of turn actions
            move_log_probs: (batch,) - log probs of move actions
            state_value: (batch, 1) - value estimates
            total_entropy: (batch,) - combined entropy
        """
        turn_probs, move_probs, state_value = self.forward(animal_input, visible_animals_input)
        
        turn_dist = torch.distributions.Categorical(turn_probs)
        move_dist = torch.distributions.Categorical(move_probs)
        
        turn_log_probs = turn_dist.log_prob(turn_actions)
        move_log_probs = move_dist.log_prob(move_actions)
        
        turn_entropy = turn_dist.entropy()
        move_entropy = move_dist.entropy()
        
        total_entropy = turn_entropy + move_entropy
        
        return turn_log_probs, move_log_probs, state_value, total_entropy
    
    def log_prob_turn(self, animal_input: torch.Tensor, 
                      visible_animals_input: torch.Tensor,
                      turn_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate turn actions only (for hierarchical PPO)
        
        Args:
            animal_input: (batch, 34) - pre-turn observation
            visible_animals_input: (batch, max_animals, 9)
            turn_actions: (batch,) - turn actions taken
            
        Returns:
            log_probs: (batch,) - log probabilities of turn actions
            entropy: (batch,) - entropy of turn distribution
        """
        turn_probs, _, _ = self.forward(animal_input, visible_animals_input)
        dist = torch.distributions.Categorical(turn_probs)
        return dist.log_prob(turn_actions), dist.entropy()
    
    def log_prob_move(self, animal_input: torch.Tensor,
                      visible_animals_input: torch.Tensor,
                      move_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate move actions only (for hierarchical PPO)
        
        Args:
            animal_input: (batch, 34) - post-turn observation
            visible_animals_input: (batch, max_animals, 9)
            move_actions: (batch,) - move actions taken
            
        Returns:
            log_probs: (batch,) - log probabilities of move actions
            entropy: (batch,) - entropy of move distribution
        """
        _, move_probs, _ = self.forward(animal_input, visible_animals_input)
        dist = torch.distributions.Categorical(move_probs)
        return dist.log_prob(move_actions), dist.entropy()
