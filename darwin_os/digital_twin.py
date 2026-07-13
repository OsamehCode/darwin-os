"""Digital Twin: replay a DNA in the same physics, with deterministic seeding.

This is the *sandbox* where mutant DNAs are evaluated. It reuses
`Environment` verbatim so there is no second physics implementation to
disagree with the live world.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .actions import action_to_control
from .dna import DecisionTreeDNA
from .environment import Environment, Vec2
from .state import Trajectory, TrajectoryStep, WorldState


@dataclass
class TwinConfig:
    max_steps: int = 300            # ~30 seconds of simulated time
    crash_on_contact: bool = True


@dataclass
class TwinResult:
    trajectory: Trajectory
    crashed: bool
    reached_goal: bool
    out_of_fuel: bool
    final_distance: float
    fuel_used: float


class DigitalTwin:
    """A thin, deterministic wrapper around an `Environment` for replaying DNAs."""

    def __init__(self, base_env: Environment, config: TwinConfig | None = None):
        self._env_template = copy.deepcopy(base_env)
        self.config = config or TwinConfig()

    def reset_env(self) -> Environment:
        return copy.deepcopy(self._env_template)

    def evaluate(self, dna: DecisionTreeDNA) -> TwinResult:
        env = self.reset_env()
        traj = Trajectory()
        s = env.reset()
        traj.steps.append(TrajectoryStep(
            t=env.time, state=s, action=(0.0, 0.0, 0.0)))

        crashed = reached = oof = False

        for _ in range(self.config.max_steps):
            action_name = dna.to_action(s)
            thrust_vec, yaw = action_to_control(action_name, s)
            s = env.step(thrust_vec, yaw)

            crashed_now = env.crashed(s)
            reached_now = env.reached_goal(s)
            oof_now = env.out_of_fuel(s)

            traj.steps.append(TrajectoryStep(
                t=env.time,
                state=s,
                action=(thrust_vec.x, thrust_vec.y, yaw),
                crashed=crashed_now,
                reached_goal=reached_now,
                out_of_fuel=oof_now,
            ))

            if crashed_now:
                crashed = True
                break
            if reached_now:
                reached = True
                break
            if oof_now:
                # keep rolling — out of fuel by itself isn't fatal if the
                # agent has already proven safe behaviour. We mark a flag.
                pass

        return TwinResult(
            trajectory=traj,
            crashed=crashed,
            reached_goal=reached,
            out_of_fuel=any(st.out_of_fuel for st in traj.steps),
            final_distance=traj.final_distance_to_goal,
            fuel_used=traj.fuel_used,
        )
