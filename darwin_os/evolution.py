"""Evolution Engine — Darwin's mutation/crossover/selection loop.

Wired around DEAP because reinventing selection, elitism and bookkeeping
is a waste of code, and DEAP gives us tournament selection and a clean
notion of the "hall of fame" for free.

Key design points:

  * `fitness(dna)` evaluates a DNA in the Digital Twin, then asks the
    Safety Verifier for a verdict. Unsafe DNAs get a fitness of -∞ —
    they are effectively *killed*, exactly as the README prescribes.
  * The fitness is *shaped* using the verifier's quantitative robustness
    so the GA has gradient signal toward safer mutants — without it,
    evolution oscillates between "barely safe" and "slightly less safe".
  * Population is represented throughout the loop as `[(dna, fitness)]`
    so tournament selection has the score in hand and can pick the BEST
    of k (rather than a random pick — the bug the code review caught).
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from deap import base, creator, tools  # type: ignore
    _HAS_DEAP = True
    _IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover
    _HAS_DEAP = False
    _IMPORT_ERROR = _e


from .dna import DecisionTreeDNA, crossover, mutate, random_dna, seed_dna_zero
from .digital_twin import DigitalTwin, TwinConfig
from .verifier import SafetyReport, SafetyVerifier, Verdict, default_rules


@dataclass
class EvolutionConfig:
    population_size: int = 200
    n_generations: int = 30
    crossover_prob: float = 0.7
    mutation_prob: float = 0.5             # applied per offspring
    elitism_count: int = 5
    max_dna_depth: int = 6
    twin_max_steps: int = 200
    seed: int = 0
    keep_parent_seed: bool = True


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    safe_share: float
    best_dna: DecisionTreeDNA


@dataclass
class EvolutionResult:
    best_dna: DecisionTreeDNA
    best_report: SafetyReport
    history: list[GenerationStats] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def fitness_curve(self) -> list[float]:
        return [g.best_fitness for g in self.history]

    def safety_curve(self) -> list[float]:
        return [g.safe_share for g in self.history]


_FITNESS_MIN = -1e18
_DEAD = (_FITNESS_MIN,)


class EvolutionEngine:
    """Mutation + crossover + elitism, with STL-style safety gating."""

    def __init__(self,
                 twin: DigitalTwin,
                 verifier: SafetyVerifier | None = None,
                 config: EvolutionConfig | None = None):
        if not _HAS_DEAP:
            raise RuntimeError(
                f"DEAP is required for EvolutionEngine. pip install deap. "
                f"Original error: {_IMPORT_ERROR!r}"
            )
        self.twin = twin
        self.verifier = verifier or SafetyVerifier(default_rules())
        self.config = config or EvolutionConfig()

    # -- scoring --------------------------------------------------------

    def _score(self, dna: DecisionTreeDNA) -> float:
        result = self.twin.evaluate(dna)
        report = self.verifier.verify(result.trajectory)
        if not report.is_safe:
            return _FITNESS_MIN
        # goal reward (exponential), small fuel penalty, safety bonus.
        final_dist = result.final_distance
        fuel_used = result.fuel_used
        margin = report.robustness_sum
        goal_score = 1000.0 * _exp_decay(final_dist, scale=20.0)
        fuel_penalty = 0.05 * fuel_used
        safety_bonus = max(0.0, min(20.0, margin * 0.02))
        fit = goal_score - fuel_penalty + safety_bonus
        if result.reached_goal:
            fit += 50.0
        return fit

    def score(self, dna: DecisionTreeDNA) -> float:
        return self._score(dna)

    # -- main loop ------------------------------------------------------

    def run(self, seed_dna: Optional[DecisionTreeDNA] = None) -> EvolutionResult:
        rng = random.Random(self.config.seed)

        # initial population (mix of seeds + random)
        population: list[DecisionTreeDNA] = []
        if self.config.keep_parent_seed and seed_dna is not None:
            population.append(copy.deepcopy(seed_dna))
        while len(population) < self.config.population_size:
            population.append(random_dna(rng=rng,
                                          max_depth=self.config.max_dna_depth - 1))

        # score once; (population_type, scored_type) tuple-list is our
        # primary working unit from here on.
        scored: list[tuple[DecisionTreeDNA, float]] = [(p, self._score(p))
                                                       for p in population]
        history: list[GenerationStats] = []
        start_t = time.perf_counter()

        for gen in range(self.config.n_generations):
            scored.sort(key=lambda x: x[1], reverse=True)
            best_dna, best_fitness = scored[0]
            mean_fitness = sum(f for _, f in scored) / len(scored)
            safe_share = sum(1 for _, f in scored if f > _FITNESS_MIN / 2) / len(scored)
            history.append(GenerationStats(
                generation=gen,
                best_fitness=best_fitness,
                mean_fitness=mean_fitness,
                safe_share=safe_share,
                best_dna=copy.deepcopy(best_dna),
            ))

            # Elitism
            new_population: list[DecisionTreeDNA] = [
                copy.deepcopy(p) for p, _ in scored[:self.config.elitism_count]
            ]

            # Refill via tournament → crossover (optional) → mutation (optional)
            while len(new_population) < self.config.population_size:
                a = _tournament(rng, scored, k=3)
                parent_b: DecisionTreeDNA | None = None
                if rng.random() < self.config.crossover_prob and len(scored) > 1:
                    parent_b = _tournament(rng, scored, k=3)
                    child = crossover(a, parent_b, rng)
                else:
                    child = copy.deepcopy(a)
                if rng.random() < self.config.mutation_prob:
                    child = mutate(child, rng)
                new_population.append(child)

            population = new_population
            scored = [(p, self._score(p)) for p in population]

        # final best
        scored.sort(key=lambda x: x[1], reverse=True)
        best_dna, _ = scored[0]
        best_traj = self.twin.evaluate(best_dna).trajectory
        best_report = self.verifier.verify(best_traj)

        elapsed = time.perf_counter() - start_t
        return EvolutionResult(
            best_dna=best_dna,
            best_report=best_report,
            history=history,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tournament(rng: random.Random,
                scored_pool: list[tuple[DecisionTreeDNA, float]],
                k: int = 3) -> DecisionTreeDNA:
    """Pick the highest-fitness DNA among `k` random candidates.

    Picks are drawn WITHOUT replacement so the winner is genuinely the
    best of k distinguishable competitors. Pool should already be scored;
    this function performs NO re-evaluation.
    """
    picks = rng.sample(scored_pool, k=min(k, len(scored_pool)))
    return max(picks, key=lambda x: x[1])[0]


def _exp_decay(d: float, scale: float) -> float:
    return math.exp(-d / scale)
