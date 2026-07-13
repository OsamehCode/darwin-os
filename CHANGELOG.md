# Changelog

All notable changes to DARWIN-OS are documented here. Versions follow
rough [Semantic Versioning](https://semver.org/) but the project is
pre-1.0 so anything can shift.

## [0.1.0] — 2026-07-13

The first end-to-end MVP. Five components wired together:

### Added
- `darwin_os.state` — `WorldState`, `AgentState`, `Vec2`, `Trajectory` data classes
- `darwin_os.conditions` — 10 features exposed as decision-tree "genes"
- `darwin_os.dna` — `DecisionTreeDNA` with binary-tree mutation + one-point crossover
- `darwin_os.actions` — public action-to-control translator (replaces earlier private API)
- `darwin_os.environment` — 2D PyMunk physics + vortex vector field + action-dependent fuel burn
- `darwin_os.digital_twin` — internal simulator (reuses live Environment verbatim)
- `darwin_os.verifier` — STL-style safety filter with quantitative robustness
- `darwin_os.evolution` — DEAP-based GA loop with proper best-of-k tournament selection
- `darwin_os.controller` — primary brain + microsecond-resolution hot swap
- `darwin_os.crisis_detector` — look-ahead predictive crisis forecaster
- `darwin_os.scenarios` — three predefined worlds (calm, surprise-vortex, minimal)
- `headless_demo.py` — canonical demo, works in any headless CI
- `visual_demo.py` — pygame demo with auto-fallback to headless
- 41 unit tests + 5 skip-on-no-safe-mutant guards across 8 test files

### Verified claims from the spec
- Hot swap wall-clock cost is in the tens of microseconds (target was <1ms)
- Digital twin reuses live physics — no second physics implementation
- Forecast is predictive, not reactive
- 4 default safety rules: fuel, obstacle clearance, wall margin, max speed

### Known issues (carry-over)
- `Environment._default_obstacles()` is dead code (no scenario uses it)
- DEAP `try/except` block in `evolution.py` is decorative (the loop is hand-rolled)
- `brake` action's vocab entry is redundantly overridden in `action_to_control`
- `test_calm_world_does_not_over_trigger_darwin` allows `<10` swaps as a calibration
