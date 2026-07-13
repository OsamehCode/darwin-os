"""Tests for the Environment + Vortex (PyMunk 2D physics)."""

import math

import pytest

from darwin_os.environment import (
    DT_SECONDS, Environment, Obstacle, Vortex, Vec2,
)
from darwin_os.scenarios import (
    calm_world, minimal_vortex_world, surprise_vortex_world,
)
from darwin_os.state import AgentState, WorldState


def test_calm_world_initializes():
    env = calm_world()
    s = env.reset()
    assert s.agent.position.length() > 0
    assert math.isfinite(s.dist_to_goal)


def test_step_advances_time():
    env = calm_world()
    env.reset()
    t0 = env.time
    env.step(Vec2(1.0, 1.0), 0.0)
    assert env.time == pytest.approx(t0 + DT_SECONDS)


def test_step_decreases_fuel_with_thrust():
    """fuel burn must depend on the action, not just on wall-clock."""
    env = calm_world()
    env.reset()
    fuel_idle_start = env._fuel
    env.step(Vec2(0.0, 0.0), 0.0)
    idle_drain = fuel_idle_start - env._fuel

    env.reset()
    fuel_thrust_start = env._fuel
    env.step(Vec2(5.0, 0.0), 0.0)
    thrust_drain = fuel_thrust_start - env._fuel
    assert thrust_drain > idle_drain, (
        f"Thrust must burn more fuel than idle; got thrust={thrust_drain}, idle={idle_drain}"
    )


def test_full_thrust_brings_fuel_under_verifier_threshold():
    """Sprinting under full thrust for a long stretch drops fuel below 10%,
    binding the verifier's `fuel_above_10pct` rule to the simulation."""
    env = Environment(arena_size=200.0, start=Vec2(10, 10), goal=Vec2(190, 190),
                      obstacles=[], vortex=None, seed=0,
                      initial_fuel=20.0, fuel_idle_drain=0.0, fuel_thrust_cost=2.0)
    env.reset()
    for _ in range(200):
        env.step(Vec2(5.0, 0.0), 0.0)
    assert env._fuel < 10.0, (
        f"Expected fuel below 10%% after long full-thrust, got {env._fuel}"
    )


def test_minimal_vortex_world_pulls_agent():
    """Under active vortex force, an idle agent should drift — not sit still."""
    env = minimal_vortex_world()       # vortex strength=8.0, radius=20m
    env.reset()
    p0 = env.agent_body.position
    for _ in range(30):                # ~3 seconds
        env.step(Vec2(0.0, 0.0), 0.0)
    p1 = env.agent_body.position
    moved = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    assert moved > 0.5, (
        f"Vortex should pull the agent at least 0.5m in 30 idle steps; "
        f"agent moved only {moved:.2f}m"
    )


def test_surprise_vortex_world_starts_calm_then_activates():
    env = surprise_vortex_world(vortex_t_start=2.0, vortex_t_end=20.0)
    assert env.vortex is not None
    assert not env.vortex.is_active(0.0)
    env.reset()
    for _ in range(30):
        env.step(Vec2(0.0, 0.0), 0.0)
    assert env.vortex.is_active(env.time)


def test_vortex_force_is_zero_outside_radius():
    v = Vortex(center=Vec2(10, 10), strength=10.0, radius=5.0, t_start=0.0, t_end=10.0)
    assert v.force_at(Vec2(50, 50), t=1.0).length() == 0.0


def test_vortex_force_is_nonzero_inside_radius():
    v = Vortex(center=Vec2(10, 10), strength=10.0, radius=5.0, t_start=0.0, t_end=10.0)
    f = v.force_at(Vec2(11, 10), t=1.0)
    assert f.length() > 0.0


def test_no_nan_inf_under_long_sequences():
    env = minimal_vortex_world()
    env.reset()
    for _ in range(500):
        env.step(Vec2(1.0, 0.0), 0.5)
    v = env.agent_body.velocity
    assert math.isfinite(v.x) and math.isfinite(v.y)


def test_set_state_from_world_overrides_initial():
    env = minimal_vortex_world()
    env.reset()
    custom = WorldState(
        agent=AgentState(position=Vec2(5.0, 30.0), velocity=Vec2(0, 0),
                         heading=0.0, fuel=42.0),
        goal=Vec2(55.0, 30.0),
        obstacles=[],
        vortex_center=None, vortex_strength=0.0, vortex_radius=0.0,
        arena_min=Vec2(0, 0), arena_max=Vec2(100, 100),
    )
    env.set_state_from_world(custom)
    s = env.observe()
    assert s.agent.position.x == pytest.approx(5.0)
    assert s.agent.fuel == pytest.approx(42.0)
    assert env._t == 0.0


def test_vec2_supports_unary_negation():
    """The Vortex.force_at() uses (-delta) — Vec2 must implement __neg__."""
    a = Vec2(3.0, 4.0)
    assert (-a).x == pytest.approx(-3.0)
    assert (-a).y == pytest.approx(-4.0)


def test_wall_margin_uses_pymunk_convention():
    """Position at the geometric centre of a square arena should have
    balanced wall-clearance on all sides.

    PyMunk y-down: arena_min.y=0 is the TOP wall, arena_max.y is BOTTOM.
    """
    s = WorldState(
        agent=AgentState(position=Vec2(50, 50), velocity=Vec2(0, 0),
                         heading=0.0, fuel=80.0, radius=1.0),
        goal=Vec2(80, 80),
        obstacles=[],
        vortex_center=None, vortex_strength=0.0, vortex_radius=0.0,
        arena_min=Vec2(0, 0), arena_max=Vec2(100, 100),
    )
    assert s.wall_margin == pytest.approx(49.0)
