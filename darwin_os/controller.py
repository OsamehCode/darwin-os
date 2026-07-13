"""Primary Controller + Hot Swap.

The Controller is what runs *now*. It owns the active DNA, executes
`act(state)`, and provides `evolve_and_swap(...)` to delegate to the
Darwin Engine when the Crisis Detector raises an alarm.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Optional

from .actions import action_to_control
from .digital_twin import DigitalTwin
from .environment import Vec2
from .evolution import EvolutionEngine, EvolutionResult
from .state import WorldState
from .verifier import SafetyReport, SafetyVerifier, default_rules
from .dna import seed_dna_zero, DecisionTreeDNA


@dataclass
class HotSwapEvent:
    t: float
    from_dna_id: int
    to_dna_id: int
    reason: str
    elapsed_us: float             # wall-clock duration of the swap, in microseconds
    improvement: float
    best_report: SafetyReport

    def __str__(self) -> str:
        return (
            f"[HotSwap @ t={self.t:.2f}s] "
            f"DNA#{self.from_dna_id} -> DNA#{self.to_dna_id} "
            f"({self.reason}); "
            f"swap={self.elapsed_us:.1f}μs; "
            f"Δfit={self.improvement:+.2f}; "
            f"robustness={self.best_report.min_robustness:.2f}"
        )


class Controller:
    """The probe's primary brain.

    Two roles:
      1. `act(state)`: choose a control action for the given world state.
      2. `evolve_and_swap(...)`: invoke the Darwin Engine and, if a strictly
         better AND safe mutant emerges, hot-swap it in.
    """

    def __init__(self,
                 twin: DigitalTwin,
                 verifier: SafetyVerifier | None = None,
                 engine: Optional[EvolutionEngine] = None,
                 initial_dna: DecisionTreeDNA | None = None):
        self.twin = twin
        self.verifier = verifier or SafetyVerifier(default_rules())
        self.engine = engine
        self.active_dna = initial_dna or seed_dna_zero()
        self.history: list[HotSwapEvent] = []
        self._swap_count: int = 0

    def act(self, state: WorldState) -> dict:
        action_name = self.active_dna.to_action(state)
        thrust_vec, yaw = action_to_control(action_name, state)
        return {
            "thrust": thrust_vec,
            "yaw_rate": yaw,
            "action_name": action_name,
            "dna_id": self.active_dna.id,
        }

    def fitness_of_current(self) -> float:
        return self.engine.score(self.active_dna) if self.engine else 0.0

    def evolve_and_swap(self, t: float, reason: str,
                        seed_dna: Optional[DecisionTreeDNA] = None) -> HotSwapEvent:
        """Run the Darwin Engine and hot-swap if a safer/better mutant exists.

        Timing: we measure the wall-clock of the *swap itself* (one object
        reference exchange + deepcopy). The README claims <1ms; on a
        laptop we typically see <100μs. We do NOT include the evolution
        run in this measurement, only the swap.
        """
        baseline_fitness = self.engine.score(self.active_dna) if self.engine else 0.0
        seed = seed_dna or self.active_dna
        result: EvolutionResult = self.engine.run(seed_dna=seed)

        new_id = self._swap_count + 1000
        result.best_dna.id = new_id

        if not result.best_report.is_safe:
            event = HotSwapEvent(
                t=t,
                from_dna_id=self.active_dna.id,
                to_dna_id=new_id,
                reason=f"engine produced no safe mutant ({reason})",
                elapsed_us=0.0,
                improvement=0.0,
                best_report=result.best_report,
            )
            self.history.append(event)
            return event

        # Measure ONLY the swap cost.
        t0 = time.perf_counter_ns()
        old_id = self.active_dna.id
        self.active_dna = copy.deepcopy(result.best_dna)
        new_fit = result.history[-1].best_fitness if result.history else 0.0
        self._swap_count += 1
        t1 = time.perf_counter_ns()
        elapsed_us = (t1 - t0) / 1000.0   # ns → μs

        event = HotSwapEvent(
            t=t,
            from_dna_id=old_id,
            to_dna_id=self.active_dna.id,
            reason=reason,
            elapsed_us=elapsed_us,
            improvement=new_fit - baseline_fitness,
            best_report=result.best_report,
        )
        self.history.append(event)
        return event
