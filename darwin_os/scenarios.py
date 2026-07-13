"""Predefined scenarios for demos and tests.

Each scenario is just an `Environment` factory: args in, world out. Keeping
them in one place makes the demo deterministic and testable.
"""

from __future__ import annotations

from .environment import Environment, Obstacle, Vortex, Vec2
from .dna import DecisionTreeDNA, seed_dna_zero


def calm_world(seed: int = 42) -> Environment:
    """No vortex, just static obstacles. The probe can succeed or fail by its own merit."""
    return Environment(
        arena_size=100.0,
        start=Vec2(10.0, 10.0),
        goal=Vec2(90.0, 90.0),
        obstacles=[
            Obstacle(Vec2(30.0, 25.0), 4.0),
            Obstacle(Vec2(30.0, 55.0), 4.0),
            Obstacle(Vec2(55.0, 40.0), 5.0),
            Obstacle(Vec2(70.0, 70.0), 4.0),
            Obstacle(Vec2(80.0, 30.0), 3.0),
        ],
        vortex=None,
        seed=seed,
    )


def surprise_vortex_world(seed: int = 42,
                          vortex_t_start: float = 6.0,
                          vortex_t_end: float = 30.0,
                          vortex_center: Vec2 = Vec2(50.0, 50.0),
                          vortex_strength: float = 9.0,
                          vortex_radius: float = 25.0) -> Environment:
    """Vortex appears at t=6s when the probe is already in transit.

    This is the README's "the controller has never seen this" scenario.
    The seed_dna_zero (just avoid obstacles, head to goal) will be sucked
    in and either crash on an obstacle or run out of fuel.
    """
    return Environment(
        arena_size=100.0,
        start=Vec2(10.0, 10.0),
        goal=Vec2(95.0, 95.0),
        obstacles=[
            Obstacle(Vec2(30.0, 25.0), 4.0),
            Obstacle(Vec2(40.0, 70.0), 4.0),
        ],
        vortex=Vortex(
            center=vortex_center,
            strength=vortex_strength,
            radius=vortex_radius,
            t_start=vortex_t_start,
            t_end=vortex_t_end,
            inward_bias=0.35,
        ),
        seed=seed,
    )


def minimal_vortex_world(seed: int = 42) -> Environment:
    """Simplest possible crisis: agent starts INSIDE the vortex influence."""
    return Environment(
        arena_size=60.0,
        start=Vec2(28.0, 32.0),       # ~7m from vortex center, inside r=20
        goal=Vec2(55.0, 30.0),
        obstacles=[],
        vortex=Vortex(
            center=Vec2(35.0, 30.0),
            strength=8.0,
            radius=20.0,
            t_start=0.0,
            t_end=15.0,
        ),
        seed=seed,
    )
