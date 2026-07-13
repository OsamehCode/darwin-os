"""Feature extractors used by decision-tree genes.

A "gene" in this project is just a comparison against a precomputed feature
of the WorldState. We expose them as Python callables (input: WorldState,
output: float). Each gene is given a stable name and a numeric index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .state import WorldState


# Each function takes a WorldState and returns a float. Comparing a float
# against a threshold is what a decision-tree node does.
Gene = Callable[[WorldState], float]


def _g_dist_to_goal(s: WorldState) -> float:
    return s.dist_to_goal


def _g_obstacle_clearance(s: WorldState) -> float:
    return s.obstacle_clearance


def _g_fuel(s: WorldState) -> float:
    return s.fuel_fraction * 100.0


def _g_speed(s: WorldState) -> float:
    return s.speed


def _g_vortex_dist(s: WorldState) -> float:
    return s.vortex_dist


def _g_vortex_strength(s: WorldState) -> float:
    return s.vortex_strength


def _g_inside_vortex(s: WorldState) -> float:
    return float(s.inside_vortex)


def _g_bearing_to_goal(s: WorldState) -> float:
    return s.bearing_to_goal


def _g_dx_to_goal(s: WorldState) -> float:
    return s.goal.x - s.agent.position.x


def _g_dy_to_goal(s: WorldState) -> float:
    return s.goal.y - s.agent.position.y


GENES: dict[str, Gene] = {
    "dist_to_goal":       _g_dist_to_goal,
    "obstacle_clearance": _g_obstacle_clearance,
    "fuel_percent":       _g_fuel,
    "speed":              _g_speed,
    "vortex_dist":        _g_vortex_dist,
    "vortex_strength":    _g_vortex_strength,
    "inside_vortex":      _g_inside_vortex,
    "bearing_to_goal":    _g_bearing_to_goal,
    "dx_to_goal":         _g_dx_to_goal,
    "dy_to_goal":         _g_dy_to_goal,
}

# Inverse map for use during mutation: stable numeric id -> gene name.
GENE_INDEX: dict[int, str] = {i: name for i, name in enumerate(GENES)}


@dataclass(frozen=True)
class GeneSpec:
    """Symbolic reference to a feature, suitable for a tree node."""
    name: str
    threshold: float

    def evaluate(self, state: WorldState) -> bool:
        return GENES[self.name](state) < self.threshold


def named_genes() -> list[str]:
    """Return a stable, ordered list of gene names."""
    return list(GENES.keys())
