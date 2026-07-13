"""STL-style Safety Verifier.

We don't ship the full Signal Temporal Logic (that requires a dedicated
tool like RTAMT, which has heavy C++ dependencies and is awkward to install
on headless CI containers). Instead, we implement the **practical subset
of STL we actually need** in 50 lines of Python:

  * `always(predicate)`  — predicate holds at every step
  * `eventually(predicate, within=K)` — predicate becomes true within K steps

The verifier returns BOTH a binary verdict AND a continuous **robustness
score** (positive when satisfied, negative when violated). The sign and
magnitude of the robustness score is fed into the evolution fitness so the
GA can *feel* how close a candidate is to violating a rule — which is
exactly the kind of gradient signal vanilla GAs do not have.

The reward design follows Maler & Nickovic, "Monitoring Temporal Properties
of Continuous Signals", FTRTFT 2004.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable

from .state import Trajectory, WorldState


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    INVALID = "invalid"


@dataclass
class RuleViolation:
    rule_name: str
    step: int
    observed_value: float
    threshold: float
    robustness: float   # negative; 0 = at threshold


@dataclass
class SafetyReport:
    verdict: Verdict
    robustness_sum: float        # sum of per-rule robustness scores
    min_robustness: float        # worst rule (gate)
    violations: list[RuleViolation] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return self.verdict == Verdict.SAFE


# ---------------------------------------------------------------------------
# Rule abstraction
# ---------------------------------------------------------------------------

@dataclass
class SafetyRule:
    """A single STL-style property with closure semantics.

    A rule is a callable: `predicate(state, step_index) -> float`
    Negative result = violation. Magnitude = how badly violated.
    """
    name: str
    semantics: str                 # "always" or "eventually_within"
    predicate: Callable[[WorldState, int], float]
    horizon: int = 0               # only used by eventually_within

    def evaluate(self, trajectory: Trajectory) -> tuple[float, list[RuleViolation]]:
        violations: list[RuleViolation] = []
        if self.semantics == "always":
            rmin = float("inf")
            for i, step in enumerate(trajectory.steps):
                r = self.predicate(step.state, i)
                if r < 0:
                    violations.append(RuleViolation(self.name, i, r, 0.0, r))
                rmin = min(rmin, r)
            return rmin, violations

        if self.semantics == "eventually_within":
            K = self.horizon
            rmax = -float("inf")
            for i in range(len(trajectory.steps)):
                r = self.predicate(trajectory.steps[i].state, i)
                rmax = max(rmax, r)
                if r >= 0 and i <= K:
                    return rmax, []   # early exit happy
            return rmax, [RuleViolation(self.name, len(trajectory.steps) - 1,
                                       rmax, 0.0, rmax)]

        # Unknown semantics: be conservative.
        return -float("inf"), [RuleViolation(self.name, -1, 0.0, 0.0, -1e9)]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class SafetyVerifier:
    """Runs a list of `SafetyRule`s over a `Trajectory`.

    Robustness interpretation:
      * sum_robustness = Σ rule_robustness. Larger is better.
      * min_robustness = min rule_robustness. Gate; if any rule is
        even slightly violated, this is negative.
    """

    def __init__(self, rules: Iterable[SafetyRule]):
        self.rules = list(rules)

    def verify(self, trajectory: Trajectory) -> SafetyReport:
        if not trajectory.steps:
            return SafetyReport(verdict=Verdict.INVALID,
                                robustness_sum=-1e9,
                                min_robustness=-1e9)
        total = 0.0
        worst = float("inf")
        all_violations: list[RuleViolation] = []
        for rule in self.rules:
            r, viols = rule.evaluate(trajectory)
            total += r
            worst = min(worst, r)
            all_violations.extend(viols)
        verdict = Verdict.SAFE if worst >= 0 else Verdict.UNSAFE
        return SafetyReport(
            verdict=verdict,
            robustness_sum=total,
            min_robustness=worst,
            violations=all_violations,
        )


# ---------------------------------------------------------------------------
# Default rule set used by the demo
# ---------------------------------------------------------------------------

def default_rules() -> list[SafetyRule]:
    """Sensible default rules inspired by the project README.

    Rules are listed in evaluation order; the order does not affect the
    SAFETY verdict (worst-rule) but does affect which rule's robustness
    contributes to the safety report's *first violation* in step-order.
    """
    def fuel_above_ten(state: WorldState, _i: int) -> float:
        # (fuel_percent - 10). > 0 means safe.
        return state.fuel_fraction * 100.0 - 10.0

    def obstacle_margin(state: WorldState, _i: int) -> float:
        # (clearance - 1.5). > 0 means at least 1.5m of margin.
        return state.obstacle_clearance - 1.5

    def wall_margin(state: WorldState, _i: int) -> float:
        # Use the World's wall_margin property (PyMunk-correct) and add
        # the 1.5m reserved safety buffer on top of that.
        return state.wall_margin - 1.5

    def max_speed(state: WorldState, _i: int) -> float:
        # (max_speed * 1.4 - actual_speed). > 0 means under the limit.
        return state.agent.max_speed * 1.4 - state.speed

    return [
        SafetyRule(name="fuel_above_10pct", semantics="always", predicate=fuel_above_ten),
        SafetyRule(name="obstacle_margin", semantics="always", predicate=obstacle_margin),
        SafetyRule(name="wall_margin",     semantics="always", predicate=wall_margin),
        SafetyRule(name="max_speed",       semantics="always", predicate=max_speed),
    ]
