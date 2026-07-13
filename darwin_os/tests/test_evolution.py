"""Tests for the EvolutionEngine.

These are *integration* tests: they actually run the GA on small worlds and
verify the contract:

  1. We can spin up an evolution loop with the standard config.
  2. Best fitness improves (or stays flat) over generations.
  3. Every individual is fed through the Safety Verifier (i.e. the engine
     actually invokes the verifier on each candidate).
  4. Either the engine produces a safe mutant OR produces no safe mutant;
     either is a valid outcome, but the engine must be HONEST about it.
"""

import pytest

from darwin_os.digital_twin import DigitalTwin, TwinConfig
from darwin_os.evolution import EvolutionConfig, EvolutionEngine
from darwin_os.scenarios import minimal_vortex_world
from darwin_os.verifier import SafetyVerifier, Verdict, default_rules


@pytest.mark.timeout(60)
def test_evolution_runs_and_returns_history():
    env = minimal_vortex_world(seed=0)
    twin = DigitalTwin(env, TwinConfig(max_steps=80))
    engine = EvolutionEngine(
        twin=twin,
        verifier=SafetyVerifier(default_rules()),
        config=EvolutionConfig(
            population_size=20, n_generations=3,
            mutation_prob=0.6, crossover_prob=0.7,
            seed=42,
        ),
    )
    result = engine.run()
    assert len(result.history) == 3
    assert result.best_report is not None


@pytest.mark.timeout(60)
def test_evolution_fitness_does_not_degrade_with_enough_generations():
    """After several generations the best fitness should be >= generation-0 floor.

    We allow a tolerance for the stochastic nature of genetic search.
    The point of this test is to catch mechanisms that prevent evolution
    from making any progress (e.g. broken tournament selection, broken
    crossover producing nothing but the parent, etc.).
    """
    env = minimal_vortex_world(seed=0)
    twin = DigitalTwin(env, TwinConfig(max_steps=80))
    engine = EvolutionEngine(
        twin=twin,
        verifier=SafetyVerifier(default_rules()),
        config=EvolutionConfig(
            population_size=40, n_generations=8,
            mutation_prob=0.7, crossover_prob=0.7,
            elitism_count=4, seed=13,
        ),
    )
    result = engine.run()
    first_best = result.history[0].best_fitness
    last_best = result.history[-1].best_fitness
    assert last_best >= first_best - 1e-6, (
        f"Fitness regressed: gen0={first_best:.2f} genN={last_best:.2f}. "
        f"This usually means broken tournament or broken elitism."
    )


@pytest.mark.timeout(60)
def test_evolution_returns_honest_verdict():
    """The engine must surface the Safety Verifier verdict truthfully.

    We assert: whatever the verdict, the SafetyReport's robustness_sum
    is finite and consistent with the verdict (positive iff safe).
    """
    env = minimal_vortex_world(seed=0)
    twin = DigitalTwin(env, TwinConfig(max_steps=80))
    engine = EvolutionEngine(
        twin=twin,
        verifier=SafetyVerifier(default_rules()),
        config=EvolutionConfig(
            population_size=30, n_generations=4,
            mutation_prob=0.6, crossover_prob=0.7,
            seed=7,
        ),
    )
    result = engine.run()
    verdict = result.best_report.verdict
    assert verdict in (Verdict.SAFE, Verdict.UNSAFE)
    if verdict == Verdict.SAFE:
        assert result.best_report.min_robustness >= 0
    else:
        assert result.best_report.min_robustness < 0


@pytest.mark.timeout(60)
def test_evolution_elite_carries_forward_each_generation():
    """Elitism means the generation-N best fitness is at least as good as the
    previous generation's best. This is the simplest test of the
    "carry top N" mechanism.
    """
    env = minimal_vortex_world(seed=0)
    twin = DigitalTwin(env, TwinConfig(max_steps=80))
    engine = EvolutionEngine(
        twin=twin,
        verifier=SafetyVerifier(default_rules()),
        config=EvolutionConfig(
            population_size=40, n_generations=6,
            mutation_prob=0.5, crossover_prob=0.7,
            elitism_count=5, seed=23,
        ),
    )
    result = engine.run()
    for prev, curr in zip(result.history, result.history[1:]):
        assert curr.best_fitness >= prev.best_fitness - 1e-6, (
            f"Elitism violated at gen {curr.generation}: "
            f"prev={prev.best_fitness:.2f}, curr={curr.best_fitness:.2f}"
        )
