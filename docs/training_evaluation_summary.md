# Training Evaluation Summary

**Date:** January 4, 2026  
**Evaluation Method:** 5 episodes × 300 steps per checkpoint, averaged results

## Final Checkpoint Performance by Phase

| Phase | Episode | Final Prey | Final Pred | Prey Escape% | Pred Capture% |
|-------|---------|------------|------------|--------------|---------------|
| Phase 1 | 200 | 322.8 | 44.8 | **87.0%** | 20.8% |
| Phase 2 | 50 | 341.4 | 42.0 | 84.5% | 22.5% |
| Phase 3 | 50 | 237.6 | 47.0 | 77.5% | **34.6%** |
| Phase 4 | 150 | **2.0** | 34.0 | 70.5% | 30.7% |

## Key Metrics Explained

- **Final Prey/Pred**: Average population at step 300 (started with 50 prey, 20 predators)
- **Prey Escape%**: Percentage of prey that escaped after being detected by a predator
- **Pred Capture%**: Percentage of hunts that resulted in successful captures

## Analysis

### Phase 1 (200 episodes) - Prey Dominance
- Prey learned effective escape behaviors early
- High escape rate (87%) prevents predator population control
- Prey population explodes due to reproduction outpacing predation

### Phase 2 (50 episodes) - Continued Prey Advantage
- Similar dynamics to Phase 1
- Slight improvement in predator capture rate (20.8% → 22.5%)
- Prey still dominate with 341 average final population

### Phase 3 (50 episodes) - Predator Adaptation
- **Significant shift**: Capture rate jumps to 34.6%
- Prey escape rate drops to 77.5%
- More balanced ecosystem, but prey still growing (237.6 avg)
- Predators learned more effective hunting strategies

### Phase 4 (150 episodes) - Predator Victory
- **Prey near extinction**: Only 2.0 average survivors
- Escape rate dropped to 70.5%
- Predators successfully hunt prey to near-extinction in most runs
- Arms race "won" by predators

## Adversarial Co-Evolution Pattern

```
Phase 1-2: Prey learn escape → Population explosion
     ↓
Phase 3:   Predators adapt → Improved hunting
     ↓
Phase 4:   Predators dominate → Prey extinction
```

## Entropy Observations (from training logs)

| Species | Early Training | Late Training (Phase 4) |
|---------|---------------|-------------------------|
| Prey | ~3.0 | ~1.6-1.7 |
| Predator | ~3.0 | ~2.5-2.6 |

### Entropy vs Performance Analysis

**Why predators have higher entropy despite winning:**

1. **Two-phase predator behavior (observed visually)**
   - **Hunt phase (prey visible):** Low entropy - learned efficient chase/intercept
   - **Search phase (no prey nearby):** High entropy - random wandering
   
2. **Entropy is averaged across all situations**
   - Predators spend significant time searching (no prey in vision)
   - Random search behavior inflates overall entropy metric
   - The "smart" hunting behavior is masked by "dumb" searching

3. **Why search behavior stays random**
   - No gradient signal when prey is not visible
   - Reward only comes from captures, not from finding prey
   - Network can't learn what it can't observe → defaults to random

4. **Prey have low entropy because they always have clear signals**
   - When predator visible: run away (learned)
   - When no predator: eat grass nearby (simple heuristic)
   - Both situations have clear objectives → low entropy

**Key Insight:** High predator entropy is a symptom of the **sparse reward problem** during search, not sophisticated flexibility. Consider adding exploration rewards or prey detection signals to improve search behavior.

## Recommendations for Future Training

1. **Balance predator rewards** - Current Phase 4 predators may be "too good"
2. **Consider prey curriculum** - Train against weaker predators first
3. **Population-based rewards** - Penalize species extinction
4. **Checkpoint selection** - Phase 3 models show best ecosystem balance
