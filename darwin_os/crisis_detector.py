"""Predictive crisis detector.

We don't wait for the probe to crash. We project its behaviour forward in
the Digital Twin and ask the Safety Verifier for a verdict on the
forecasted trajectory. If any rule is going to be violated within the
lookahead horizon, we treat that as a CRISIS and hand control to the
Darwin Engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from .actions import action_to_control
from .controller import Controller
from .digital_twin import DigitalTwin
from .state import Trajectory, TrajectoryStep, WorldState
from .verifier import SafetyVerifier


@dataclass
class CrisisAssessment:
    triggered: bool
    reason: str
    horizon: int                    # earliest violation step (0..lookahead-1)
    best_robustness: float
    report_min_robustness: float


class CrisisDetector:
    """Look-ahead crisis detector using the Digital Twin's twin."""

    def __init__(self,
                 twin: DigitalTwin,
                 verifier: SafetyVerifier,
                 lookahead_steps: int = 30):  # ~3 s at dt=0.1
        self.twin = twin
        self.verifier = verifier
        self.lookahead_steps = lookahead_steps

    def assess(self, controller: Controller, state: WorldState) -> CrisisAssessment:
        """Replicate `state` in a fresh twin, run controller.act() loop, check safety.

        The detector gets a fresh environment from the twin (one deep copy),
        forces it into the live `state` via `set_state_from_world`, then
        rolls N lookahead steps under the active DNA. We break early on
        crash or goal-reached to avoid spurious post-terminal steps.
        """
        env = self.twin.reset_env()
        env.set_state_from_world(state)

        traj = Trajectory()
        traj.steps.append(TrajectoryStep(t=env.time, state=env.observe(),
                                         action=(0.0, 0.0, 0.0)))

        s = env.observe()
        for _ in range(self.lookahead_steps):
            action_name = controller.active_dna.to_action(s)
            thrust_vec, yaw = action_to_control(action_name, s)
            s = env.step(thrust_vec, yaw)
            traj.steps.append(TrajectoryStep(t=env.time, state=s,
                                             action=(thrust_vec.x, thrust_vec.y, yaw),
                                             crashed=env.crashed(s),
                                             reached_goal=env.reached_goal(s),
                                             out_of_fuel=env.out_of_fuel(s)))
            if env.crashed(s) or env.reached_goal(s):
                break

        report = self.verifier.verify(traj)
        if report.is_safe:
            return CrisisAssessment(
                triggered=False, reason="safe",
                horizon=self.lookahead_steps,
                best_robustness=report.min_robustness,
                report_min_robustness=report.min_robustness,
            )

        earliest_v = min(report.violations, key=lambda v: v.step)
        return CrisisAssessment(
            triggered=True, reason=earliest_v.rule_name,
            horizon=earliest_v.step,
            best_robustness=report.min_robustness,
            report_min_robustness=report.min_robustness,
        )
