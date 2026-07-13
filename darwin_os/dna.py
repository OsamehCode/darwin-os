"""DecisionTreeDNA — the "genome" of a control policy.

A DNA is a binary decision tree. Internal nodes are *GeneSpec* comparisons
(< threshold). Leaves are discrete control *actions* drawn from a small
vocabulary of (thrust_vector, yaw_rate) pairs.

The representation is deliberately simple and explicit:

    Node = {"op": "if",  "gene": "fuel_percent", "threshold": 12.0,
            "true":  <Node>,  "false": <Node>}
    Node = {"op": "act", "action": "boost_toward_goal"}

We use dicts instead of a heavyweight AST class so that the structure is:

  * JSON-serialisable for snapshotting after a successful hot-swap
  * easy to deep-copy with `copy.deepcopy`
  * easy to randomise, mutate, and visualise

The action vocabulary is intentionally tiny: 8 hand-tuned behaviours are
already more than enough for an MVP. The Darwin Engine discovers *which*
action to fire *when*; it does not invent new actions.
"""

from __future__ import annotations

import copy
import random
from typing import Any, TypedDict, Union, Optional

from .conditions import GENES, GENE_INDEX, named_genes


# ---------------------------------------------------------------------------
# Action vocabulary
# ---------------------------------------------------------------------------
# An "action" is a dict with keys: thrust (Vec2-like floats), yaw (rad/s).
# The vocabulary enumerates a small but expressive set of probe manoeuvres.

ACTION_VOCAB: list[dict[str, tuple[float, float]]] = [
    # thrust_x, thrust_y, yaw_rate
    {"name": "cruise_to_goal",  "thrust": (1.0, 0.0),  "yaw":  0.0},
    {"name": "boost_to_goal",   "thrust": (1.6, 0.0),  "yaw":  0.0},
    {"name": "circle_left",     "thrust": (0.4, 0.7),  "yaw":  1.6},
    {"name": "circle_right",    "thrust": (0.4, -0.7), "yaw": -1.6},
    {"name": "brake",           "thrust": (-0.6, 0.0), "yaw":  0.0},
    {"name": "escape_vortex",   "thrust": (1.4, 1.0),  "yaw":  2.5},
    {"name": "tangent_left",    "thrust": (0.7, 0.6),  "yaw":  1.0},
    {"name": "tangent_right",   "thrust": (0.7, -0.6), "yaw": -1.0},
]

ACTION_BY_NAME: dict[str, dict[str, tuple[float, float]]] = {
    a["name"]: a for a in ACTION_VOCAB
}


# ---------------------------------------------------------------------------
# Tree node type
# ---------------------------------------------------------------------------

class IfNode(TypedDict):
    op: str              # "if"
    gene: str
    threshold: float
    true: "Node"
    false: "Node"


class LeafNode(TypedDict):
    op: str              # "act"
    action: str


Node = Union[IfNode, LeafNode]


def is_leaf(node: Node) -> bool:
    return node["op"] == "act"


# ---------------------------------------------------------------------------
# Random tree construction
# ---------------------------------------------------------------------------

def random_tree(rng: random.Random,
                max_depth: int = 5,
                leaf_prob_at_max: float = 0.7) -> Node:
    """Build a random decision tree of bounded depth.

    Depth 0 means a single leaf. At each non-leaf level we draw a gene and
    a threshold from a sensible range, then recurse.
    """
    if max_depth == 0 or rng.random() < leaf_prob_at_max * (1.0 / max(1, max_depth)):
        return _random_leaf(rng)
    gene_name = rng.choice(named_genes())
    threshold = _sample_threshold(rng, gene_name)
    return IfNode(
        op="if",
        gene=gene_name,
        threshold=threshold,
        true=random_tree(rng, max_depth - 1, leaf_prob_at_max),
        false=random_tree(rng, max_depth - 1, leaf_prob_at_max),
    )


def _random_leaf(rng: random.Random) -> LeafNode:
    return LeafNode(op="act", action=rng.choice(ACTION_VOCAB)["name"])


def _sample_threshold(rng: random.Random, gene_name: str) -> float:
    """Sample a sensible threshold for the given gene."""
    rngs: dict[str, tuple[float, float]] = {
        "dist_to_goal":       (0.0,  120.0),
        "obstacle_clearance": (0.0,   20.0),
        "fuel_percent":       (0.0,  100.0),
        "speed":              (0.0,    8.0),
        "vortex_dist":        (0.0,  100.0),
        "vortex_strength":    (0.0,   20.0),
        "inside_vortex":      (0.0,    1.0),
        "bearing_to_goal":    (-3.2,  3.2),
        "dx_to_goal":         (-120.0, 120.0),
        "dy_to_goal":         (-120.0, 120.0),
    }
    lo, hi = rngs.get(gene_name, (-1.0, 1.0))
    return rng.uniform(lo, hi)


# ---------------------------------------------------------------------------
# DNA class
# ---------------------------------------------------------------------------

class DecisionTreeDNA:
    """Wrapper around the tree-of-dicts representation.

    Mostly a thin wrapper that records the generation it was born in and
    carries convenience methods for mutation and crossover.
    """

    __slots__ = ("root", "birth_generation", "parent_ids", "id")

    def __init__(self,
                 root: Optional[Node] = None,
                 birth_generation: int = 0,
                 parent_ids: tuple[int, int] = (-1, -1),
                 dna_id: int = -1):
        self.root: Node = root if root is not None else LeafNode(op="act", action="cruise_to_goal")
        self.birth_generation = birth_generation
        self.parent_ids = parent_ids
        self.id = dna_id

    # convenience -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "root": copy.deepcopy(self.root),
            "birth_generation": self.birth_generation,
            "parent_ids": list(self.parent_ids),
            "id": self.id,
        }

    def __repr__(self) -> str:
        return f"DNA(id={self.id} gen={self.birth_generation} depth={tree_depth(self.root)})"

    # tree traversal -----------------------------------------------------
    def to_action(self, state) -> str:
        """Walk the tree and return the leaf action name."""
        node = self.root
        # safety: bound walk depth to avoid accidental infinite recursion
        for _ in range(64):
            if is_leaf(node):
                return node["action"]
            spec = _gene_spec(node)
            if spec.evaluate(state):
                node = node["true"]
            else:
                node = node["false"]
        return "cruise_to_goal"  # safe fallback


def _gene_spec(node: IfNode):
    from .conditions import GeneSpec
    return GeneSpec(name=node["gene"], threshold=node["threshold"])


def tree_depth(node: Node) -> int:
    if is_leaf(node):
        return 1
    return 1 + max(tree_depth(node["true"]), tree_depth(node["false"]))


def tree_size(node: Node) -> int:
    if is_leaf(node):
        return 1
    return 1 + tree_size(node["true"]) + tree_size(node["false"])


# ---------------------------------------------------------------------------
# Mutation & crossover
# ---------------------------------------------------------------------------

def mutate(dna: DecisionTreeDNA,
           rng: random.Random,
           p_subtree: float = 0.10,
           p_threshold: float = 0.25,
           p_gene_swap: float = 0.15,
           p_leaf_swap: float = 0.30) -> DecisionTreeDNA:
    """Mutate a DNA in place by deep-copying it first.

    Several mutation operators are applied randomly at each non-leaf node:
      * replace a sub-tree with a fresh random one (rare, big jumps)
      * nudge a threshold (small numeric shift)
      * swap to a different gene in the same family (medium)
      * swap a leaf action (small)
    """
    new = copy.deepcopy(dna.root)
    _mutate_in_place(new, rng, p_subtree, p_threshold, p_gene_swap, p_leaf_swap)
    return DecisionTreeDNA(
        root=new,
        birth_generation=dna.birth_generation,
        parent_ids=(dna.id, -1),
        dna_id=-1,
    )


def _mutate_in_place(node: Node,
                     rng: random.Random,
                     p_subtree: float,
                     p_threshold: float,
                     p_gene_swap: float,
                     p_leaf_swap: float) -> None:
    if is_leaf(node):
        if rng.random() < p_leaf_swap:
            node["action"] = rng.choice(ACTION_VOCAB)["name"]
        return

    # mutate this internal node
    r = rng.random()
    if r < p_subtree:
        # replace this whole sub-tree
        replacement = random_tree(rng, max_depth=max(1, tree_depth(node) - 1))
        node.clear()
        node.update(copy.deepcopy(replacement))
        return
    elif r < p_subtree + p_threshold:
        node["threshold"] *= rng.uniform(0.6, 1.4)
    elif r < p_subtree + p_threshold + p_gene_swap:
        node["gene"] = rng.choice(named_genes())
        node["threshold"] = _sample_threshold(rng, node["gene"])

    _mutate_in_place(node["true"],  rng, p_subtree, p_threshold, p_gene_swap, p_leaf_swap)
    _mutate_in_place(node["false"], rng, p_subtree, p_threshold, p_gene_swap, p_leaf_swap)


def crossover(parent_a: DecisionTreeDNA,
              parent_b: DecisionTreeDNA,
              rng: random.Random) -> DecisionTreeDNA:
    """One-point sub-tree crossover: pick a random node in each parent, swap."""
    a = copy.deepcopy(parent_a.root)
    b = copy.deepcopy(parent_b.root)

    path_a = _random_node_path(a, rng)
    path_b = _random_node_path(b, rng)

    if path_a and path_b:
        node_a = _follow_path(a, path_a)
        node_b = _follow_path(b, path_b)
        # swap by exchanging the dict
        _replace_at_path(a, path_a, copy.deepcopy(node_b))
        _replace_at_path(b, path_b, copy.deepcopy(node_a))

    # Default: keep the modified `a` as the child
    return DecisionTreeDNA(
        root=a,
        birth_generation=max(parent_a.birth_generation, parent_b.birth_generation) + 1,
        parent_ids=(parent_a.id, parent_b.id),
        dna_id=-1,
    )


def _random_node_path(root: Node, rng: random.Random) -> list[int]:
    """Return a random path from root to some descendant, encoded as a
    list of 0s (true) and 1s (false). Empty list = the root itself."""
    path: list[int] = []
    node = root
    for _ in range(20):  # hard cap
        if is_leaf(node):
            break
        bit = rng.randint(0, 1)
        path.append(bit)
        node = node["true"] if bit == 0 else node["false"]
    return path


def _follow_path(root: Node, path: list[int]) -> Node:
    node = root
    for bit in path:
        if is_leaf(node):
            return node
        node = node["true"] if bit == 0 else node["false"]
    return node


def _replace_at_path(root: Node, path: list[int], new_sub: Node) -> None:
    parent = root
    for bit in path[:-1]:
        parent = parent["true"] if bit == 0 else parent["false"]
    if not path:
        # replace the root itself
        root.clear()
        root.update(new_sub)
        return
    if path:
        last = path[-1]
        parent["true" if last == 0 else "false"] = new_sub


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def random_dna(rng: random.Random | None = None,
               max_depth: int = 4,
               birth_generation: int = 0,
               dna_id: int = -1) -> DecisionTreeDNA:
    rng = rng or random.Random()
    return DecisionTreeDNA(
        root=random_tree(rng, max_depth=max_depth),
        birth_generation=birth_generation,
        parent_ids=(-1, -1),
        dna_id=dna_id,
    )


def seed_dna_zero() -> DecisionTreeDNA:
    """The "father" DNA — a sane, hand-crafted baseline policy.

    "If there is an obstacle near me, brake; otherwise cruise toward the goal."
    We later show that this policy fails in a vortex that it has never seen.
    """
    root: Node = IfNode(
        op="if",
        gene="obstacle_clearance",
        threshold=4.0,
        true=LeafNode(op="act", action="brake"),
        false=IfNode(
            op="if",
            gene="dist_to_goal",
            threshold=4.0,
            true=LeafNode(op="act", action="boost_to_goal"),
            false=LeafNode(op="act", action="cruise_to_goal"),
        ),
    )
    return DecisionTreeDNA(root=root, birth_generation=0, dna_id=0)
