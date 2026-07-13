"""Tests for the public actions module (action_to_control).

PyMunk convention: +x right, +y DOWN. So "up-left" on screen is (-x, -y).
"""

from darwin_os.actions import action_to_control
from darwin_os.state import AgentState, Vec2, WorldState


def _make_world(agent_x=20.0, agent_y=30.0,
                goal_x=80, goal_y=30,
                vortex_center=None, vortex_strength=0.0):
    return WorldState(
        agent=AgentState(position=Vec2(agent_x, agent_y),
                         velocity=Vec2(0.0, 0.0),
                         heading=0.0, fuel=80.0),
        goal=Vec2(goal_x, goal_y),
        obstacles=[],
        vortex_center=(Vec2(vortex_center[0], vortex_center[1])
                       if vortex_center is not None else None),
        vortex_strength=vortex_strength,
        vortex_radius=20.0,
        arena_min=Vec2(0, 0), arena_max=Vec2(100, 100),
    )


def test_cruise_to_goal_to_the_right():
    """Goal right of agent (same y) -> thrust in +x direction."""
    state = _make_world(agent_x=20, agent_y=30, goal_x=80, goal_y=30)
    thrust, _ = action_to_control("cruise_to_goal", state)
    assert thrust.x > 0.5, f"expected thrust.x > 0.5 rightward, got {thrust}"
    assert abs(thrust.y) < 0.1


def test_cruise_to_goal_south_east():
    """Goal south-east of agent (positive x and positive y in PyMunk)
    -> thrust has positive x AND positive y."""
    state = _make_world(agent_x=20, agent_y=10, goal_x=80, goal_y=80)
    thrust, _ = action_to_control("cruise_to_goal", state)
    assert thrust.x > 0.0
    assert thrust.y > 0.0


def test_escape_vortex_away_from_vortex_in_pymunk():
    """Vortex at (10, 10), agent at (50, 50). Outward vector = (+40, +40)
    in PyMunk convention (= down-right on screen). Thrust should be
    positive in both x and y."""
    state = _make_world(agent_x=50, agent_y=50, vortex_center=(10, 10),
                        vortex_strength=10.0)
    thrust, _ = action_to_control("escape_vortex", state)
    assert thrust.x > 0.0, f"expected positive x (push right), got {thrust.x}"
    assert thrust.y > 0.0, f"expected positive y (push down), got {thrust.y}"
    # magnitude should match the base magnitude sqrt(1.4^2 + 1.0^2)
    import math
    expected = math.hypot(1.4, 1.0)
    assert abs(thrust.length() - expected) < 1e-6


def test_escape_vortex_when_vortex_is_to_the_left():
    """Vortex at (60, 50), agent at (20, 50). Outward = (-40, 0) i.e.
    push LEFT (negative x in PyMunk)."""
    state = _make_world(agent_x=20, agent_y=50, vortex_center=(60, 50),
                        vortex_strength=10.0)
    thrust, _ = action_to_control("escape_vortex", state)
    assert thrust.x < 0.0, f"expected negative x (push left), got {thrust.x}"
    assert abs(thrust.y) < 1e-6


def test_brake_action_pushes_against_velocity():
    """Brake should thrust opposite to current velocity."""
    state = _make_world(agent_x=50, agent_y=50)
    state.agent.velocity = Vec2(3.0, 4.0)
    thrust, _ = action_to_control("brake", state)
    import math
    # Brake base magnitude (sqrt((-0.6)^2 + 0^2) = 0.6), so thrust magnitude
    # should be ~0.6.
    expected_mag = 0.6
    assert abs(thrust.length() - expected_mag) < 1e-6
    # Direction is opposite to (3, 4) -> thrust should be in (-3, -4) direction
    norm_thrust = thrust.normalized()
    expected = Vec2(-3.0, -4.0).normalized()
    assert abs(norm_thrust.x - expected.x) < 1e-6
    assert abs(norm_thrust.y - expected.y) < 1e-6


def test_unknown_action_raises_key_error():
    """Pinned: unknown action currently raises KeyError."""
    state = _make_world()
    try:
        action_to_control("nonexistent_action", state)
    except KeyError:
        return
    raise AssertionError("Expected KeyError for unknown action name")
