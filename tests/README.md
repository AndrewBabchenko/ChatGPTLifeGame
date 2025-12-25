# Tests Directory

This directory contains test scripts for validating the training pipeline.

## Available Tests

### `test_training_minimal.py`
Minimal test script that validates the core training loop with a single animal and a few experiences. Useful for:
- Quick validation after code changes
- Debugging PPO update issues
- Verifying device handling (CPU/GPU)

**Usage:**
```bash
python tests/test_training_minimal.py
```

## Planned Unit Tests

Future unit tests to be added using pytest:

- **Animal behavior tests**: Verify animal movement, energy, aging
- **Neural network tests**: Validate model architecture and forward pass
- **PPO algorithm tests**: Ensure correct advantage computation, policy updates
- **Pheromone system tests**: Check decay, diffusion, and sensing

## Running Tests

```bash
# Install pytest (when unit tests are added)
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_animal.py
```
