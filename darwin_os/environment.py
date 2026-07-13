"""2D physics environment using PyMunk.

Reality-gap note: PyMunk is a deterministic, side-effect-free 2D engine.
We therefore use it for BOTH the live world and the Digital Twin. This
keeps the twin honest: it is *not* a separate, faster-but-wrong sim.

Coordinate convention (single source of truth): PyMunk uses screen-
coords where +x is right and **+y is down**. We keep that convention
everywhere — gene features, observations, action interpretation — so
the project has no hidden y-flips.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import pymunk

from .state import AgentState, Vec2, WorldState


DT_SECONDS = 0.1   # physics step = 100 ms


@dataclass
class Obstacle:
    center: Vec2
    radius: float


@dataclass
class Vortex:
    """A spinning sink centred at `center`, active for `t_start` <= t < `t_end`."""
    center: Vec2
    strength: float          # peak tangential "wind" speed at the edge (m/s)
    radius: float            # influence radius (m)
    inward_bias: float = 0.25
    t_start: float = 0.0
    t_end: float = float("inf")

    def is_active(self, t: float) -> bool:
        return self.t_start <= t < self.t_end

    def force_at(self, position: Vec2, t: float) -> Vec2:
        if not self.is_active(t):
            return Vec2(0.0, 0.0)
        delta = position - self.center
        dist = delta.length()
        if dist < 1e-6 or dist > self.radius:
            return Vec2(0.0, 0.0)
        decay = (1.0 - dist / self.radius) ** 2
        tang = Vec2(-delta.y, delta.x).normalized() * self.strength * decay
        inward = (-delta).normalized() * (self.strength * decay * self.inward_bias)
        return Vec2(tang.x + inward.x, tang.y + inward.y)


class Environment:
    """A PyMunk-backed 2D world."""

    def __init__(self,
                 arena_size: float = 100.0,
                 start: Vec2 = Vec2(10.0, 10.0),
                 goal: Vec2 = Vec2(90.0, 90.0),
                 obstacles: Optional[list[Obstacle]] = None,
                 vortex: Optional[Vortex] = None,
                 seed: int = 0,
                 agent_radius: float = 1.0,
                 initial_fuel: float = 100.0,
                 fuel_idle_drain: float = 0.5,
                 fuel_thrust_cost: float = 1.0):
        self.arena_size = arena_size
        self.start = start
        self.goal = goal
        self._rng = random.Random(seed)
        self.obstacles = obstacles or self._default_obstacles()
        self.vortex = vortex
        self._t = 0.0
        self._initial_fuel = initial_fuel
        self._fuel = initial_fuel
        self._fuel_idle_drain = fuel_idle_drain   # fuel per second when idle
        self._fuel_thrust_cost = fuel_thrust_cost # fuel per (N*s) of thrust
        self._setup_space(agent_radius)

    # -- setup ----------------------------------------------------------

    def _default_obstacles(self) -> list[Obstacle]:
        return [
            Obstacle(Vec2(30.0, 25.0), 4.0),
            Obstacle(Vec2(30.0, 55.0), 4.0),
            Obstacle(Vec2(55.0, 40.0), 5.0),
            Obstacle(Vec2(70.0, 70.0), 4.0),
            Obstacle(Vec2(80.0, 30.0), 3.0),
        ]

    def _setup_space(self, agent_radius: float) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.02

        w = 1.0
        walls = [
            pymunk.Segment(self.space.static_body, (0, 0), (self.arena_size, 0), w),
            pymunk.Segment(self.space.static_body, (self.arena_size, 0),
                          (self.arena_size, self.arena_size), w),
            pymunk.Segment(self.space.static_body, (self.arena_size, self.arena_size),
                          (0, self.arena_size), w),
            pymunk.Segment(self.space.static_body, (0, self.arena_size), (0, 0), w),
        ]
        for wall in walls:
            wall.elasticity = 0.0
            wall.friction = 0.5
            self.space.add(wall)

        self._obstacle_bodies = []
        for ob in self.obstacles:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (ob.center.x, ob.center.y)
            shape = pymunk.Circle(body, ob.radius)
            shape.elasticity = 0.1
            shape.friction = 0.6
            self.space.add(body, shape)
            self._obstacle_bodies.append((body, ob.radius))

        mass = 1.0
        moment = pymunk.moment_for_circle(mass, 0, agent_radius)
        self.agent_body = pymunk.Body(mass, moment)
        self.agent_body.position = (self.start.x, self.start.y)
        self.agent_body.velocity = (0.0, 0.0)
        self.agent_body.angular_velocity = 0.0
        self.agent_body.angle = 0.0
        self.agent_shape = pymunk.Circle(self.agent_body, agent_radius)
        self.agent_shape.elasticity = 0.0
        self.agent_shape.friction = 0.5
        self.space.add(self.agent_body, self.agent_shape)

    # -- reset / state injection ---------------------------------------

    def reset(self) -> WorldState:
        self.agent_body.position = (self.start.x, self.start.y)
        self.agent_body.velocity = (0.0, 0.0)
        self.agent_body.angular_velocity = 0.0
        self.agent_body.angle = 0.0
        self._t = 0.0
        self._fuel = self._initial_fuel
        return self.observe()

    def set_state_from_world(self, state: WorldState) -> None:
        """Inject a snapshot as the current state — used by the crisis detector
        to start a forecast from a real, observed state.

        Note: this resets _t to 0 so the forecast horizon is well-defined.
        """
        self.agent_body.position = (state.agent.position.x, state.agent.position.y)
        self.agent_body.velocity = (state.agent.velocity.x, state.agent.velocity.y)
        self.agent_body.angular_velocity = 0.0
        self.agent_body.angle = state.agent.heading
        self._t = 0.0
        self._fuel = state.agent.fuel

    # -- observation ----------------------------------------------------

    def observe(self) -> WorldState:
        pos = self.agent_body.position
        vel = self.agent_body.velocity
        return WorldState(
            agent=AgentState(
                position=Vec2(pos.x, pos.y),
                velocity=Vec2(vel.x, vel.y),
                heading=self.agent_body.angle,
                fuel=self._fuel,
                radius=self.agent_shape.radius,
            ),
            goal=self.goal,
            obstacles=[(Vec2(body.position.x, body.position.y), r)
                       for (body, r) in self._obstacle_bodies],
            vortex_center=(self.vortex.center if (self.vortex and self.vortex.is_active(self._t)) else None),
            vortex_strength=(self.vortex.strength if (self.vortex and self.vortex.is_active(self._t)) else 0.0),
            vortex_radius=(self.vortex.radius if self.vortex else 0.0),
            arena_min=Vec2(0.0, 0.0),
            arena_max=Vec2(self.arena_size, self.arena_size),
        )

    # -- step -----------------------------------------------------------

    def step(self, thrust: Vec2, yaw_rate: float) -> WorldState:
        max_thrust = 6.0
        mag = thrust.length()
        applied = thrust if mag <= max_thrust else thrust * (max_thrust / mag)
        self.agent_body.apply_force_at_world_point(
            (applied.x, applied.y),
            (self.agent_body.position.x, self.agent_body.position.y),
        )
        self.agent_body.torque = yaw_rate * 50.0

        if self.vortex and self.vortex.is_active(self._t):
            f = self.vortex.force_at(Vec2(self.agent_body.position.x, self.agent_body.position.y), self._t)
            self.agent_body.apply_force_at_world_point(
                (f.x, f.y),
                (self.agent_body.position.x, self.agent_body.position.y),
            )

        # Action-dependent fuel burn: idle is cheap, thrust costs more.
        # This is what makes the verifier's "fuel > 10%" rule meaningful
        # — a DNA that loiters in the vortex is punished relative to one
        # that escapes quickly.
        drain = (self._fuel_idle_drain + self._fuel_thrust_cost * mag) * DT_SECONDS
        self._fuel = max(0.0, self._fuel - drain)

        v = self.agent_body.velocity
        if not (math.isfinite(v.x) and math.isfinite(v.y)):
            self.agent_body.velocity = (0.0, 0.0)

        self.space.step(DT_SECONDS)
        self._t += DT_SECONDS

        return self.observe()

    # -- status ---------------------------------------------------------

    @property
    def time(self) -> float:
        return self._t

    def reached_goal(self, state: WorldState) -> bool:
        return state.dist_to_goal <= 3.0

    def crashed(self, state: WorldState) -> bool:
        for (center, r) in state.obstacles:
            if (center - state.agent.position).length() < r + state.agent.radius - 0.1:
                return True
        margin = 0.5
        if (state.agent.position.x < margin
            or state.agent.position.y < margin
            or state.agent.position.x > self.arena_size - margin
            or state.agent.position.y > self.arena_size - margin):
            return True
        return False

    def out_of_fuel(self, state: WorldState) -> bool:
        return state.agent.fuel <= 0.0
