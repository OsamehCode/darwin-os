"""Public translation layer: DNA action name -> (thrust, yaw_rate).

Coordinate convention (chosen for consistency throughout the project):

  * WorldState uses PyMunk conventions: x is right, y is down.
  * `atan2(dy, dx)` therefore points "south-east" if dy > 0.

Template actions (`cruise_to_goal`, `boost_to_goal`, `escape_vortex`, etc.)
compute a *desired direction* and scale the action's base magnitude by it,
rather than rotating the base thrust by an absolute angle.
"""

from __future__ import annotations

import math
from typing import Tuple

from .dna import ACTION_BY_NAME
from .state import Vec2, WorldState


def action_to_control(action_name: str,
                      state: WorldState) -> Tuple[Vec2, float]:
    """Translate an action name into a (thrust_vector, yaw_rate) pair.

    The thrust vector is in the same convention as the rest of the
    codebase: PyMunk's screen-down +y. Direction-templated actions are
    computed by:
      1. Determine the desired DIRECTION (unit vector toward goal,
         outward-from-vortex, etc.).
      2. Scale it by the action's base magnitude so the thrust has the
         right energy.
    """
    spec = ACTION_BY_NAME[action_name]
    base_thrust = spec["thrust"]
    yaw = spec["yaw"]
    base_mag = math.hypot(base_thrust[0], base_thrust[1])

    # Default: the base vector verbatim (no directional template).
    tx, ty = base_thrust[0], base_thrust[1]

    name = spec["name"]
    is_directional = name in ("cruise_to_goal", "boost_to_goal",
                              "circle_left", "circle_right",
                              "tangent_left", "tangent_right")

    if is_directional:
        # Aim the base thrust at the goal.
        dx = state.goal.x - state.agent.position.x
        dy = state.goal.y - state.agent.position.y
        if not (dx or dy):
            tx, ty = base_thrust[0], base_thrust[1]
        else:
            m = math.hypot(dx, dy)
            tx = (dx / m) * base_mag
            ty = (dy / m) * base_mag

    if name == "escape_vortex" and state.vortex_center is not None:
        # Aim AWAY from vortex centre.
        vdx = state.agent.position.x - state.vortex_center.x
        vdy = state.agent.position.y - state.vortex_center.y
        if vdx or vdy:
            m = math.hypot(vdx, vdy)
            tx = (vdx / m) * base_mag
            ty = (vdy / m) * base_mag
    elif name == "brake":
        # Brake applies opposite to current velocity (capped magnitude).
        v = state.agent.velocity
        vmag = v.length()
        if vmag > 1e-3:
            tx = -v.x / vmag * base_mag
            ty = -v.y / vmag * base_mag
    return Vec2(tx, ty), yaw
