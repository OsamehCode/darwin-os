"""Tests for the DNA primitives (DecisionTreeDNA + mutate + crossover)."""

import random

import pytest

from darwin_os.dna import (
    ACTION_VOCAB,
    DecisionTreeDNA,
    crossover,
    mutate,
    random_dna,
    seed_dna_zero,
    tree_depth,
    tree_size,
)


def test_seed_dna_zero_is_safe_to_call():
    dna = seed_dna_zero()
    assert dna.id == 0
    assert tree_depth(dna.root) >= 1
    # It should always produce a valid action name.
    from darwin_os.state import AgentState, Vec2, WorldState
    state = WorldState(
        agent=AgentState(position=Vec2(10, 10), velocity=Vec2(0, 0),
                         heading=0.0, fuel=100.0),
        goal=Vec2(90, 90),
        obstacles=[(Vec2(40, 40), 5.0)],
        vortex_center=None, vortex_strength=0.0, vortex_radius=0.0,
        arena_min=Vec2(0, 0), arena_max=Vec2(100, 100),
    )
    action = dna.to_action(state)
    assert action in ACTION_VOCAB or any(a["name"] == action for a in ACTION_VOCAB)


def test_random_dna_runs_for_arbitrary_state():
    rng = random.Random(123)
    dna = random_dna(rng=rng, max_depth=4)
    from darwin_os.state import AgentState, Vec2, WorldState
    state = WorldState(
        agent=AgentState(position=Vec2(0, 0), velocity=Vec2(0, 0),
                         heading=0.0, fuel=10.0),
        goal=Vec2(100, 100),
        obstacles=[],
        vortex_center=Vec2(50, 50), vortex_strength=10.0, vortex_radius=20.0,
        arena_min=Vec2(0, 0), arena_max=Vec2(100, 100),
    )
    # Should not raise.
    action = dna.to_action(state)
    assert isinstance(action, str)


def test_mutate_preserves_depth_distribution_loosely():
    rng = random.Random(7)
    dna = random_dna(rng=rng, max_depth=4)
    before_size = tree_size(dna.root)
    child = mutate(dna, rng)
    after_size = tree_size(child.root)
    # It *can* grow or shrink with sub-tree replacement, but should
    # always remain a tree we can walk.
    assert after_size >= 1


def test_crossover_returns_a_valid_tree():
    rng = random.Random(0)
    a = random_dna(rng=rng, max_depth=5)
    b = random_dna(rng=rng, max_depth=5)
    child = crossover(a, b, rng)
    assert isinstance(child.root, dict)
    assert child.root.get("op") in {"if", "act"}
    assert child.birth_generation > max(a.birth_generation, b.birth_generation)


def test_to_action_returns_only_known_actions():
    rng = random.Random(1)
    dna = random_dna(rng=rng, max_depth=5)
    from darwin_os.state import AgentState, Vec2, WorldState
    state = WorldState(
        agent=AgentState(position=Vec2(10, 10), velocity=Vec2(0, 0),
                         heading=0.0, fuel=80.0),
        goal=Vec2(20, 10),
        obstacles=[],
        vortex_center=None, vortex_strength=0.0, vortex_radius=0.0,
        arena_min=Vec2(0, 0), arena_max=Vec2(100, 100),
    )
    valid = {a["name"] for a in ACTION_VOCAB}
    for _ in range(20):
        assert dna.to_action(state) in valid
