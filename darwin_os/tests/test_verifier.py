"""Tests for the SafetyVerifier."""

import pytest

from darwin_os.state import AgentState, Trajectory, TrajectoryStep, Vec2, WorldState
from darwin_os.verifier import SafetyRule, SafetyVerifier, Verdict, default_rules


def _make_state(**overrides) -> WorldState:
    base = dict(
        agent=AgentState(position=Vec2(50, 50), velocity=Vec2(0, 0),
                         heading=0.0, fuel=80.0),
        goal=Vec2(90, 90),
        obstacles=[],
        vortex_center=None, vortex_strength=0.0, vortex_radius=0.0,
        arena_min=Vec2(0, 0), arena_max=Vec2(100, 100),
    )
    base.update(overrides)
    base["agent"] = overrides.get("agent", base["agent"])
    return WorldState(**base)


def test_safe_trajectory_passes_default_rules():
    # Build states manually so we can be sure of the structure.
    states = []
    for i in range(30):
        s = WorldState(
            agent=AgentState(
                position=Vec2(50.0 + i, 50.0),
                velocity=Vec2(2.0, 0.0),
                heading=0.0,
                fuel=80.0 - i * 0.5,
                radius=1.0,
            ),
            goal=Vec2(90, 90),
            obstacles=[],
            vortex_center=None, vortex_strength=0.0, vortex_radius=0.0,
            arena_min=Vec2(0, 0), arena_max=Vec2(100, 100),
        )
        states.append(s)
    traj = Trajectory(steps=[
        TrajectoryStep(t=i * 0.1, state=s, action=(0, 0, 0))
        for i, s in enumerate(states)
    ])
    v = SafetyVerifier(default_rules()).verify(traj)
    assert v.is_safe, f"Expected safe trajectory; got {v}"
    assert v.min_robustness > 0


def test_unsafe_fuel_violation_detected():
    states = []
    for i in range(30):
        # Fuel steadily drops below 10%.
        states.append(_make_state(agent=AgentState(
            position=Vec2(50 + i, 50), velocity=Vec2(2, 0),
            heading=0.0, fuel=max(-1.0, 80.0 - i * 5.0))))
    traj = Trajectory(steps=[
        TrajectoryStep(t=i * 0.1, state=s, action=(0, 0, 0))
        for i, s in enumerate(states)
    ])
    v = SafetyVerifier(default_rules()).verify(traj)
    assert v.verdict == Verdict.UNSAFE
    assert any("fuel" in v.rule_name for v in v.violations) or \
           any(vio.rule_name == "fuel_above_10pct" for vio in v.violations)


def test_robustness_is_continuous_sign():
    """Higher fuel → higher robustness on the fuel rule."""
    def manual_robustness(fuel_pct: float) -> float:
        s = WorldState(
            agent=AgentState(
                position=Vec2(50.0, 50.0), velocity=Vec2(0.0, 0.0),
                heading=0.0, fuel=fuel_pct, radius=1.0,
            ),
            goal=Vec2(90, 90),
            obstacles=[],
            vortex_center=None, vortex_strength=0.0, vortex_radius=0.0,
            arena_min=Vec2(0, 0), arena_max=Vec2(100, 100),
        )
        traj = Trajectory(steps=[TrajectoryStep(t=0, state=s, action=(0, 0, 0))])
        return SafetyVerifier(default_rules()).verify(traj).min_robustness
    r_high = manual_robustness(50.0)
    r_low  = manual_robustness(5.0)
    assert r_high > r_low, f"r_high={r_high} should be > r_low={r_low}"
    assert r_low < 0


def test_empty_trajectory_is_invalid():
    v = SafetyVerifier(default_rules()).verify(Trajectory())
    assert v.verdict == Verdict.INVALID
