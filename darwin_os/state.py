"""Sensor and world-state data structures.

Everything that flows between the live environment, the Digital Twin, the
crisis detector, the safety verifier and a DNA (decision tree) is captured
here. Keeping this dataclass very small and explicit makes it trivial to
log, replay, and unit-test.

Coordinate convention: PyMunk uses screen-down +y. We keep that
convention *everywhere* in this codebase. There is no y-flip in
Voronoi-rule computations, only inside PyMunk function calls themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class Vec2:
    """A 2D vector (x, y). PyMunk convention: +x right, +y down."""
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vec2":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vec2":
        return Vec2(self.x / scalar, self.y / scalar)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        n = self.length()
        if n < 1e-9:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / n, self.y / n)

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class AgentState:
    """Physical state of the probe."""
    position: Vec2
    velocity: Vec2
    heading: float            # radians, 0 = +x axis
    fuel: float               # 0..100, percent
    radius: float = 1.0
    max_speed: float = 6.0    # m/s
    max_steering: float = 4.0 # rad/s
    thrust_budget: float = 12.0  # N (abstract unit)


@dataclass
class WorldState:
    """Snapshot of the world the agent perceives.

    This is what a DNA reads through its decision tree. It is intentionally
    *pre-computed* (e.g. distance to goal, vortex strength) so the tree only
    needs trivial comparisons like "if dist < 5".

    PyMunk convention: +y is down. So `arena_min.y = 0` is the TOP wall and
    `arena_max.y` is the BOTTOM wall.
    """
    agent: AgentState
    goal: Vec2
    obstacles: list[tuple[Vec2, float]]   # (center, radius)
    vortex_center: Vec2 | None
    vortex_strength: float                # 0 means no vortex
    vortex_radius: float                  # influence radius
    arena_min: Vec2
    arena_max: Vec2

    # --- pre-computed features the DNA can read cheaply ---

    @property
    def dist_to_goal(self) -> float:
        return (self.goal - self.agent.position).length()

    @property
    def bearing_to_goal(self) -> float:
        """Counter-clockwise radians from current heading to goal.

        Note: PyMunk y-down, so positive rotation appears clockwise on
        screen. Sign is not as important as magnitude for the genes.
        """
        d = self.goal - self.agent.position
        return -math.atan2(d.y, d.x)

    @property
    def vortex_dist(self) -> float:
        if self.vortex_center is None:
            return float("inf")
        return (self.vortex_center - self.agent.position).length()

    @property
    def inside_vortex(self) -> bool:
        return (
            self.vortex_center is not None
            and self.vortex_strength > 0
            and self.vortex_dist <= self.vortex_radius
        )

    @property
    def obstacle_clearance(self) -> float:
        """Distance from agent surface to the nearest obstacle surface."""
        if not self.obstacles:
            return float("inf")
        center, r = min(self.obstacles, key=lambda o: (o[0] - self.agent.position).length())
        return (center - self.agent.position).length() - r - self.agent.radius

    @property
    def wall_margin(self) -> float:
        """Distance from agent surface to the nearest ARENA WALL in PyMunk coords.

        PyMunk y-down: `arena_min.y = 0` is the TOP wall, `arena_max.y` is the
        BOTTOM wall. So distance_to_top = position.y - arena_min.y, and
        distance_to_bottom = arena_max.y - position.y.
        """
        right = self.arena_max.x - self.agent.position.x
        top = self.agent.position.y - self.arena_min.y
        left = self.agent.position.x - self.arena_min.x
        bottom = self.arena_max.y - self.agent.position.y
        return min(right, top, left, bottom) - self.agent.radius

    @property
    def speed(self) -> float:
        return self.agent.velocity.length()

    @property
    def fuel_fraction(self) -> float:
        return max(0.0, min(1.0, self.agent.fuel / 100.0))

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self.agent)
        d["position"] = (self.agent.position.x, self.agent.position.y)
        d["velocity"] = (self.agent.velocity.x, self.agent.velocity.y)
        d["goal"] = (self.goal.x, self.goal.y)
        d["vortex_center"] = None if self.vortex_center is None else (
            self.vortex_center.x, self.vortex_center.y
        )
        d["arena_min"] = (self.arena_min.x, self.arena_min.y)
        d["arena_max"] = (self.arena_max.x, self.arena_max.y)
        d["pre"] = {
            "dist_to_goal": self.dist_to_goal,
            "bearing_to_goal": self.bearing_to_goal,
            "vortex_dist": self.vortex_dist,
            "inside_vortex": float(self.inside_vortex),
            "obstacle_clearance": self.obstacle_clearance,
            "speed": self.speed,
            "fuel_fraction": self.fuel_fraction,
        }
        return d


@dataclass
class TrajectoryStep:
    """One logged step of a rollout."""
    t: float
    state: WorldState
    action: tuple           # (thrust_x, thrust_y, yaw_rate)
    crashed: bool = False
    reached_goal: bool = False
    out_of_fuel: bool = False


@dataclass
class Trajectory:
    """Full log of a roll-out, used by SafetyVerifier and fitness functions."""
    steps: list[TrajectoryStep] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    @property
    def final_state(self) -> WorldState | None:
        return self.steps[-1].state if self.steps else None

    @property
    def final_distance_to_goal(self) -> float:
        if not self.steps:
            return float("inf")
        return self.steps[-1].state.dist_to_goal

    @property
    def fuel_used(self) -> float:
        if not self.steps:
            return 0.0
        return self.steps[0].state.agent.fuel - self.steps[-1].state.agent.fuel

    @property
    def total_path_length(self) -> float:
        s = 0.0
        for a, b in zip(self.steps, self.steps[1:]):
            s += (b.state.agent.position - a.state.agent.position).length()
        return s
