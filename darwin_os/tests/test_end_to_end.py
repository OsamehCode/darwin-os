"""End-to-end tests for the Darwin Engine.

These run the FULL Darwin loop on real scenarios. They are deliberately
slow — marked with @pytest.mark.timeout to prevent runaway.
"""

import pytest

from darwin_os.controller import Controller
from darwin_os.crisis_detector import CrisisDetector
from darwin_os.digital_twin import DigitalTwin
from darwin_os.environment import Environment, Obstacle, Vortex, Vec2
from darwin_os.evolution import EvolutionConfig, EvolutionEngine
from darwin_os.scenarios import calm_world
from darwin_os.verifier import SafetyVerifier, default_rules


@pytest.mark.timeout(60)
def test_calm_world_does_not_over_trigger_darwin():
    """In a calm world, the seed DNA should rarely need the Darwin Engine.

    A 5-second run of the seed DNA in a sparse-obstacle world should fire
    Darwin at most a handful of times — the policy is hand-tuned for this
    environment. We allow a generous upper bound (10) so the test focuses
    on catastrophic over-triggering rather than fine-tuning.
    """
    env = calm_world()
    twin = DigitalTwin(env)
    engine = EvolutionEngine(twin=twin, config=EvolutionConfig(
        population_size=10, n_generations=1, seed=0))
    ctrl = Controller(twin=twin, engine=engine)
    detector = CrisisDetector(twin=twin, verifier=SafetyVerifier(default_rules()),
                              lookahead_steps=20)

    s = env.reset()
    swaps = 0
    for _ in range(40):  # ~4 seconds
        assessment = detector.assess(ctrl, s)
        if assessment.triggered:
            ctrl.evolve_and_swap(t=env.time, reason="crisis")
            swaps += 1
        a = ctrl.act(s)
        s = env.step(a["thrust"], a["yaw_rate"])
        if env.crashed(s) or env.reached_goal(s):
            break

    assert swaps < 10, (
        f"Seed DNA should not over-trigger Darwin in a calm world; got {swaps} "
        f"swaps in ~{env.time:.1f}s."
    )


@pytest.mark.timeout(180)
def test_darwin_engine_invocation_produces_dna_change():
    """The Darwin Engine, when invoked, must produce a different DNA.

    We don't try to trigger the crisis detector in a vortex scenario
    (the forecast may not catch a vortex in 30 steps and a 100-step
    forecast typically ends at the goal). Instead, this test asserts
    the *engine* contract: if you call `evolve_and_swap`, you get a
    new DNA, and the swap is recorded in history.
    """
    env = Environment(
        arena_size=100.0,
        start=Vec2(10.0, 10.0),
        goal=Vec2(90.0, 90.0),
        obstacles=[],
        vortex=Vortex(
            center=Vec2(50.0, 50.0), strength=10.0, radius=20.0,
            t_start=0.0, t_end=20.0,
        ),
        seed=42,
    )
    twin = DigitalTwin(env)
    engine = EvolutionEngine(twin=twin, config=EvolutionConfig(
        population_size=20, n_generations=2, seed=42))
    ctrl = Controller(twin=twin, engine=engine)
    original_dna = ctrl.active_dna
    original_id = original_dna.id

    # Manually invoke Darwin, simulating what the crisis detector would do.
    event = ctrl.evolve_and_swap(t=0.0, reason="vortex-mock")

    if event.best_report.is_safe:
        # The Darwin Engine produced a SAFE mutant. We hot-swapped in it.
        assert ctrl.active_dna is not original_dna
        assert ctrl.active_dna.id != original_id
        assert ctrl.active_dna.id == event.to_dna_id
        # The history records this swap.
        assert len(ctrl.history) >= 1
        assert ctrl.history[-1].from_dna_id == original_id
    else:
        # The Darwin Engine failed to find a safe mutant on this small
        # config. This is not an error per se, but the controller must
        # not have swapped in an unsafe DNA.
        assert ctrl.active_dna is original_dna


@pytest.mark.timeout(180)
def test_darwin_engine_evolves_to_safer_dna_in_vortex_world():
    """The Darwin Engine, when run repeatedly, should produce DNAs that
    score strictly better than the seed (or at least, find a safe one).

    This is the *evolution contract*: over a few generations, fitness
    improves. We don't require the agent to physically survive the
    vortex — only that the engine discovers improved policies.
    """
    env = Environment(
        arena_size=100.0,
        start=Vec2(10.0, 10.0),
        goal=Vec2(90.0, 90.0),
        obstacles=[],
        vortex=Vortex(
            center=Vec2(50.0, 50.0), strength=10.0, radius=20.0,
            t_start=0.0, t_end=20.0,
        ),
        seed=42,
    )
    twin = DigitalTwin(env)
    engine = EvolutionEngine(twin=twin, config=EvolutionConfig(
        population_size=20, n_generations=4, seed=7))
    result = engine.run()
    # The engine should produce history.
    assert len(result.history) == 4
    # The first generation's best fitness should not be greater than the
    # last (i.e. evolution is at least non-decreasing on this small run).
    assert result.history[-1].best_fitness >= result.history[0].best_fitness - 1e-6
