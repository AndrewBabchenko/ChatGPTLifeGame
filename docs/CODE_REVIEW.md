# Code Review Report - ChatGPTLifeGame

## Scope
- Reviewed core runtime and training code under `src/` and `scripts/`.
- Checked tests under `tests/` and docs under `README.md` and `docs/`.
- Focus: correctness, behavior consistency, performance, and test gaps.
- Tests were not executed for this review.

## Findings (ordered by severity)

### High
1) Curriculum base reset breaks after first application
- Impact: `apply_curriculum_stage` can wipe the base snapshot because `_BASE` is included in itself, so later resets no longer restore original values and stage overrides can leak.
- Evidence: `src/config.py:12-13`, `src/config.py:256-258`, `src/config.py:306-307`.
- Recommendation: exclude `_BASE` (and other private keys) from the snapshot, for example `if k.isupper() and not k.startswith("_")`.

2) Ctrl+C interrupt checkpoint path points to a non-existent directory
- Impact: interrupt saves can fail, losing progress if the user stops training with Ctrl+C.
- Evidence: `scripts/train.py:1609-1612` saves to `models/...` while `scripts/train.py:1637` only creates `outputs/checkpoints`.
- Recommendation: save to `outputs/checkpoints` or create the `models/` directory before saving.

3) Hunger semantics diverge between training and demo
- Impact: demo behavior is not representative of training because hunger uses different units in different places, and `steps_since_last_meal` is never updated in the demo. Predators never enter hungry mode in the demo even though training uses hunger-based speed and penalties.
- Evidence: `src/core/animal.py:923-927` (hunger uses steps since last meal), `scripts/train.py:1309-1314` (steps increment), `scripts/simulation_demo.py:420-470` (no steps update), `scripts/simulation_demo.py:727, 749` (hunger based on energy).
- Recommendation: define a single hunger metric (energy or steps) and apply it consistently in both training and demo. If steps are intended, update the demo to increment `steps_since_last_meal` and use it for display and movement.

4) Inference path recomputes visibility twice per micro-step
- Impact: CPU-heavy inference and demo loops do redundant world scans, which reduces FPS and wastes time.
- Evidence: `src/core/animal.py:306-330` computes `get_enhanced_input()` (which calls `communicate`) and then calls `communicate()` again for the same step.
- Recommendation: call `communicate()` once and pass the result into `get_enhanced_input` via the `visible_animals` parameter.

### Medium
1) Visibility cache freshness guard never advances
- Impact: the cache validation in `deposit_pheromones` never invalidates across steps because `CURRENT_STEP` is never updated, so stale visibility can be reused if movement is skipped or ordering changes.
- Evidence: `src/config.py:15-16`, `src/core/animal.py:877-881`, `src/core/animal.py:1058-1061` with no updates in training or demo loops.
- Recommendation: increment `CURRENT_STEP` once per simulation step (training and demo), or remove the guard and always pass fresh `visible_animals`.

2) Prey reproduction threshold mismatch
- Impact: `PREY_MATING_ENERGY_THRESHOLD` is used for supervision but actual reproduction checks use `MATING_ENERGY_COST`, which can cause conflicting behavior vs reward shaping.
- Evidence: `src/core/animal.py:866-870` vs `scripts/train.py:145-157`.
- Recommendation: use `PREY_MATING_ENERGY_THRESHOLD` in `Prey.can_reproduce`, or remove that config setting if it is not intended to gate reproduction.

3) Demo chart rendering rebuilds the entire figure each step
- Impact: unnecessary re-layout and redraw overhead that causes UI lag for longer runs.
- Evidence: `scripts/simulation_demo.py:564-599`.
- Recommendation: keep `Line2D` objects and update their data (`set_data`) with `draw_idle()` instead of clearing and recreating axes each step.

4) Demo visibility is computed twice per step
- Impact: extra `communicate()` call per animal in the demo loop.
- Evidence: `scripts/simulation_demo.py:424-427`.
- Recommendation: reuse the `visible_animals` list when building the observation.

### Low
1) Docs reference a missing `scripts/demo.py` and outdated config values
- Impact: onboarding confusion and broken instructions.
- Evidence: `README.md:21-23`, `README.md:83`, `docs/SAFE_TRAINING_GUIDE.md:253-276`, `docs/ARCHITECTURE.md:130`.
- Recommendation: update docs to match current scripts (`scripts/simulation_demo.py`) or restore the missing file.

2) Safe training runner prints a fixed episode count
- Impact: misleading run output if config is changed.
- Evidence: `scripts/run_training_safe.ps1:27-28`, `src/config.py:151`.
- Recommendation: read values from config or remove the printed count.

3) Dead code: `update_post_action` is never used
- Impact: increases maintenance surface and obscures where hunger updates actually happen.
- Evidence: `src/core/animal.py:89-92` with no call sites.
- Recommendation: remove or call it from both training and demo loops to centralize per-step updates.

4) Actor-Critic input size is hard-coded
- Impact: any change to observation features requires manual edits in multiple places.
- Evidence: `src/models/actor_critic_network.py:93-99`.
- Recommendation: derive the input size from config (or assert that config.OBS_VERSION matches expected feature count at init).

## Opportunities and enhancements
- Consider sharing a single step function between training and demo to keep behavior consistent.
- Add optional GPU inference in the demo when available for smoother rendering.
- Add a requirements file to formalize dependencies for demos and dashboard.

## Test coverage and gaps
- Existing tests cover action mapping, directional supervision, GPU smoke, and cached visibility behavior.
- Missing tests for:
  - Curriculum base reset behavior.
  - `CURRENT_STEP` cache invalidation.
  - Hunger/steps-since-meal semantics and starvation logic.
  - Demo vs training environment parity.

## Open questions / assumptions
- Should the demo mirror the training environment exactly, or is it intentionally simplified?
- Is `HUNGER_THRESHOLD` intended to be energy-based or steps-since-meal?
- Is `PREY_MATING_ENERGY_THRESHOLD` meant to gate reproduction or only drive supervision?
