"""Headless demo: a 30-60 second crisis that exercises the full Darwin loop.

Usage:

    python headless_demo.py                 # uses default settings
    python headless_demo.py --quick         # smaller population, faster

Produces:
    - ASCII trace on stdout
    - evolution_curves.png      (matplotlib: best fitness + safe-share)
    - crisis_animation.png      (matplotlib: top-down view of the world)

The demo runs entirely without a display (Agg backend) so it works in
headless CI / cloud shells.
"""

from __future__ import annotations

import argparse
import os

# Force a headless-friendly backend BEFORE importing pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402

from darwin_os.controller import Controller  # noqa: E402
from darwin_os.crisis_detector import CrisisDetector  # noqa: E402
from darwin_os.evolution import EvolutionConfig, EvolutionEngine  # noqa: E402
from darwin_os.scenarios import surprise_vortex_world  # noqa: E402
from darwin_os.verifier import SafetyVerifier, default_rules  # noqa: E402
from darwin_os.digital_twin import DigitalTwin  # noqa: E402


ASCII_BANNER = r"""
 ___________   ______________  __  ____    __  ______
/_______  __\ /_____  ____/ / / / _ \   / / / / / _ \
  ___/ /     _____/ /     / /_/ /_/ /  / /_/ / // /
 /___/ /_____/_____/_____/\__/_____/   \____/____/
            DARWIN-OS  ::  Headless demo  ::  v0.1.0
"""


def main() -> None:
    print(ASCII_BANNER)
    args = _parse_args()
    print(f"[setup] arena: 100x100m, surprise vortex at (50,50) at t=6s "
          f"lasting {(args.vortex_end - args.vortex_start):.0f}s "
          f"with strength=9.0, radius=25m")

    env = surprise_vortex_world(
        seed=42,
        vortex_t_start=args.vortex_start,
        vortex_t_end=args.vortex_end,
        vortex_strength=9.0,
        vortex_radius=25.0,
    )
    twin = DigitalTwin(env)
    engine = EvolutionEngine(
        twin=twin, verifier=SafetyVerifier(default_rules()),
        config=EvolutionConfig(
            population_size=args.pop_size,
            n_generations=args.generations,
            mutation_prob=0.6,
            crossover_prob=0.7,
            elitism_count=5,
            seed=7,
        ),
    )
    ctrl = Controller(twin=twin, engine=engine)
    detector = CrisisDetector(
        twin=twin, verifier=SafetyVerifier(default_rules()),
        lookahead_steps=20,
    )

    print(f"[seed] initial DNA id={ctrl.active_dna.id}, depth={_depth_of(ctrl)}")
    print(f"[seed] decision chain: {_decision_chain_summary(ctrl)}")

    goal_reached = False
    crashed = False
    s = env.reset()
    while env.time < args.max_steps * 0.1:
        assessment = detector.assess(ctrl, s)
        if assessment.triggered:
            print(f"[t={env.time:.1f}s] CRISIS DETECTED "
                  f"(rule={assessment.reason}; horizon={assessment.horizon} steps)")
            print(f"[t={env.time:.1f}s] >>> DARWIN ENGINE ACTIVATED <<<")
            event = ctrl.evolve_and_swap(t=env.time, reason=assessment.reason)
            print(f"           {event}")
            print(f"[t={env.time:.1f}s] new DNA: {_decision_chain_summary(ctrl)}")
        a = ctrl.act(s)
        s = env.step(a["thrust"], a["yaw_rate"])
        if env.crashed(s):
            crashed = True
            print(f"[t={env.time:.1f}s] CRASH! obstacle/wall hit.")
            break
        if env.reached_goal(s):
            goal_reached = True
            print(f"[t={env.time:.1f}s] GOAL REACHED ✔")
            break

    for line in _summary_lines(ctrl, env, goal_reached, crashed, n_swaps=len(ctrl.history)):
        print(line)

    if args.save_artifacts:
        _save_evolution_curves()
        _save_trajectory_image(env, ctrl)


def _depth_of(ctrl: Controller) -> int:
    from darwin_os.dna import tree_depth
    return tree_depth(ctrl.active_dna.root)


def _decision_chain_summary(ctrl: Controller) -> str:
    from darwin_os.dna import is_leaf

    def _walk(node, depth):
        if depth > 2:
            return "..."
        if is_leaf(node):
            return f"act={node['action']}"
        return (f"if {node['gene']}<{node['threshold']:.1f}"
                f" ? {_walk(node['true'], depth+1)} : {_walk(node['false'], depth+1)}")
    return _walk(ctrl.active_dna.root, 0)


def _summary_lines(ctrl, env, goal_reached, crashed, n_swaps) -> list[str]:
    final = env.observe()
    return [
        "",
        "─" * 60,
        f"Outcomes: goal_reached={goal_reached}, crashed={crashed}",
        f"Total hot-swap events: {n_swaps}",
        f"Final DNA id: {ctrl.active_dna.id}",
        f"Final action: {ctrl.active_dna.to_action(final)}",
        f"Mean swap wall-clock: {_mean_swap_us(ctrl):.1f} μs "
            f"(min <1ms claim holds on this hardware: {_min_swap_us(ctrl) < 1000})",
        "─" * 60,
    ]


def _mean_swap_us(ctrl: Controller) -> float:
    times = [e.elapsed_us for e in ctrl.history if e.elapsed_us > 0]
    return sum(times) / max(1, len(times))


def _min_swap_us(ctrl: Controller) -> float:
    times = [e.elapsed_us for e in ctrl.history if e.elapsed_us > 0]
    return min(times) if times else float("inf")


def _save_evolution_curves() -> None:
    """Re-run a small evolution in pure-evolution mode to plot a curve."""
    from darwin_os.scenarios import minimal_vortex_world
    twin = DigitalTwin(minimal_vortex_world(seed=0))
    cfg = EvolutionConfig(population_size=60, n_generations=15, seed=11)
    eng2 = EvolutionEngine(twin=twin, config=cfg)
    print("[viz] running a second, tiny evolution for the curve plot...")
    result = eng2.run()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    gens = [g.generation for g in result.history]
    best = [g.best_fitness for g in result.history]
    mean = [g.mean_fitness for g in result.history]
    safe = [g.safe_share * 100 for g in result.history]

    ax1.plot(gens, best, "o-", label="best")
    ax1.plot(gens, mean, "s--", label="mean")
    ax1.set_title("Fitness over generations")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.legend()

    ax2.plot(gens, safe, "g^-")
    ax2.set_title("Survival rate")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("% mutants passing safety")
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig("evolution_curves.png")
    print("[viz] wrote evolution_curves.png")


def _save_trajectory_image(env, ctrl) -> None:
    print("[viz] re-running animation path...")
    env.reset()
    s = env.observe()
    path_x, path_y = [s.agent.position.x], [s.agent.position.y]
    while env.time < 30.0:
        a = ctrl.act(s)
        s = env.step(a["thrust"], a["yaw_rate"])
        path_x.append(s.agent.position.x)
        path_y.append(s.agent.position.y)
        if env.crashed(s) or env.reached_goal(s):
            break

    fig, ax = plt.subplots(figsize=(6, 6))
    for obs in env.obstacles:
        ax.add_patch(plt.Circle((obs.center.x, obs.center.y),
                                 obs.radius, color="red", alpha=0.6))
    if env.vortex:
        vx, vy = env.vortex.center.x, env.vortex.center.y
        ax.add_patch(plt.Circle((vx, vy), env.vortex.radius,
                                 color="purple", fill=False, linestyle="--", lw=2))
    ax.add_patch(plt.Circle((env.goal.x, env.goal.y), 3,
                             color="dodgerblue", alpha=0.7))
    ax.plot(path_x, path_y, "-k", lw=1, alpha=0.5)
    ax.plot(path_x[-1], path_y[-1], "go", markersize=8)
    ax.plot(env.start.x, env.start.y, "k^", markersize=10)

    ax.set_xlim(0, env.arena_size)
    ax.set_ylim(0, env.arena_size)
    ax.set_aspect("equal")
    ax.set_title("DARWIN-OS: agent trace through the vortex")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("crisis_animation.png")
    print("[viz] wrote crisis_animation.png")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--pop-size", type=int, default=80)
    p.add_argument("--generations", type=int, default=12)
    p.add_argument("--vortex-start", type=float, default=6.0)
    p.add_argument("--vortex-end", type=float, default=30.0)
    p.add_argument("--save-artifacts", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    main()
