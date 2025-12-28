# Hierarchical PPO Usage Guide

## Overview

The system now supports **correct hierarchical PPO** for the turn→move policy. Each micro-step produces two decisions conditioned on different observations:

1. **Turn decision**: Based on pre-turn observation (current heading/FOV)
2. **Move decision**: Based on post-turn observation (new heading/FOV after turning)

## API Changes

### 1. PPOMemory - Hierarchical Mode

```python
from src.models.replay_buffer import PPOMemory

# Create hierarchical memory buffer
memory = PPOMemory(batch_size=64, hierarchical=True)

# Add hierarchical transitions (returned by move_training)
transitions = animal.move_training(model, animals, config, pheromone_map)
for transition in transitions:
    memory.add(transition=transition, reward=reward, done=done)
```

### 2. Animal.move_training() - Returns Transitions

**Old API (DEPRECATED):**
```python
log_probs_list = []
values_list = []
animal.move_training(model, animals, log_probs_list, config, pheromone_map, values_list)
```

**New API (CORRECT):**
```python
# Returns list of transition dicts
transitions = animal.move_training(model, animals, config, pheromone_map)

# Each transition contains:
{
    'obs_turn': tensor,      # Pre-turn observation
    'vis_turn': tensor,      # Pre-turn visible animals
    'turn_action': int,      # Turn action taken
    'turn_logp_old': tensor, # Log prob of turn
    
    'obs_move': tensor,      # Post-turn observation
    'vis_move': tensor,      # Post-turn visible animals
    'move_action': tensor,   # Move action taken
    'move_logp_old': tensor, # Log prob of move
    
    'value_old': tensor      # Pre-turn state value
}
```

### 3. ActorCriticNetwork - Helper Methods

```python
# For PPO ratio computation
turn_log_prob, turn_entropy = model.log_prob_turn(
    obs_turn, vis_turn, turn_actions
)
move_log_prob, move_entropy = model.log_prob_move(
    obs_move, vis_move, move_actions
)

# Combined for hierarchical policy
new_log_prob_total = turn_log_prob + move_log_prob
old_log_prob_total = turn_logp_old + move_logp_old
ratio = torch.exp(new_log_prob_total - old_log_prob_total)
```

## Complete Training Loop Example

```python
from src.models.replay_buffer import PPOMemory
from src.models.actor_critic_network import ActorCriticNetwork

# Initialize
memory = PPOMemory(batch_size=64, hierarchical=True)
model = ActorCriticNetwork(config)

# Rollout phase
for step in range(num_steps):
    for animal in animals:
        # Get hierarchical transitions
        transitions = animal.move_training(model, animals, config, pheromone_map)
        
        # Store each transition with reward/done
        for transition in transitions:
            reward = compute_reward(animal)  # Your reward function
            done = check_done(animal)
            memory.add(transition=transition, reward=reward, done=done)

# Compute returns and advantages
next_value = get_next_value()  # Bootstrap value
returns, advantages = memory.compute_returns_and_advantages(next_value)

# PPO update phase
for epoch in range(ppo_epochs):
    for batch in memory.get_batches():
        # Extract hierarchical data
        obs_turn = torch.stack(batch['obs_turn'])
        vis_turn = torch.stack(batch['vis_turn'])
        turn_actions = batch['turn_actions']
        turn_logp_old = batch['turn_log_probs_old']
        
        obs_move = torch.stack(batch['obs_move'])
        vis_move = torch.stack(batch['vis_move'])
        move_actions = batch['move_actions']
        move_logp_old = batch['move_log_probs_old']
        
        returns = batch['returns']
        advantages = batch['advantages']
        
        # Recompute log probs on stored observations
        turn_logp_new, turn_entropy = model.log_prob_turn(
            obs_turn, vis_turn, turn_actions
        )
        move_logp_new, move_entropy = model.log_prob_move(
            obs_move, vis_move, move_actions
        )
        
        # Compute combined log probs
        old_logp_total = turn_logp_old + move_logp_old
        new_logp_total = turn_logp_new + move_logp_new
        
        # PPO ratio and clipping
        ratio = torch.exp(new_logp_total - old_logp_total)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus
        entropy = turn_entropy + move_entropy
        entropy_loss = -entropy_coef * entropy.mean()
        
        # Value loss (can use either obs_turn or recompute)
        _, _, values = model.forward(obs_turn, vis_turn)
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Total loss
        loss = policy_loss + value_loss_coef * value_loss + entropy_loss
        
        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Clear memory for next episode
memory.clear()
```

## Backward Compatibility

The system supports both modes:

**Simple Mode (old code still works):**
```python
memory = PPOMemory(batch_size=64, hierarchical=False)  # Default
memory.add(state, action, log_prob, value, reward, done)
```

**Hierarchical Mode (correct for turn→move policy):**
```python
memory = PPOMemory(batch_size=64, hierarchical=True)
memory.add(transition=transition_dict, reward=reward, done=done)
```

## Critical Fix Summary

### ✅ What Was Wrong

- Stored only `log_prob_total = turn_log_prob + move_log_prob`
- Couldn't recompute ratios because observations weren't stored separately
- PPO would use wrong ratios (turn and move conditioned on different states)

### ✅ What's Now Fixed

- Store both observations: `obs_turn` and `obs_move`
- Store both actions: `turn_action` and `move_action`
- Store both log probs: `turn_logp_old` and `move_logp_old`
- Use `model.log_prob_turn()` and `model.log_prob_move()` for correct recomputation
- Combine for ratio: `exp((new_turn + new_move) - (old_turn + old_move))`

## Migration Guide for train_advanced.py

The current `train_advanced.py` uses `model.get_action()` which samples both heads from the same observation. This needs updating:

**Current (INCORRECT):**
```python
actions, log_probs, values = model.get_action(inputs_batch, visible_batch)
memory.add(state, actions, log_probs, values, reward, done)
```

**Should Be (CORRECT):**
```python
# Use animal.move_training() which returns hierarchical transitions
transitions = animal.move_training(model, animals, config, pheromone_map)
for transition in transitions:
    memory.add(transition=transition, reward=reward, done=done)
```

Note: This requires refactoring the training loop to process animals individually rather than in batches for the hierarchical policy. The performance trade-off is worth it for correctness.
