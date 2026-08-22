"""Explicability and recovery metrics for extracted circuit graphs.

Two families:

* ``explicability_metrics`` -- graph-theoretic properties of an extracted
  circuit graph that are observable *without* a ground truth:
  sparsity, modularity (community structure), effective depth, width
  bottleneck, and role diversity.
* ``recovery_*`` -- how well the extracted graph recovers a *known*
  ground-truth circuit's wiring. For boolean tasks this is a functional
  wiring overlap: do the model's hidden units co-wire pairs of inputs the
  same way the ground-truth gates do?

``mes_score`` (Mechanistic Explicability Score) combines recovery,
sparsity, modularity and depth fidelity into a single [0, 1] number.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Graph-theoretic explicability metrics (no ground truth needed)
# ---------------------------------------------------------------------------

def _to_nx(graph: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for nd in graph["nodes"]:
        G.add_node(nd["id"], layer=nd["layer"], role=nd["role"])
    for e in graph["edges"]:
        if len(e) == 3:
            G.add_edge(e[0], e[1], weight=e[2])
        else:
            G.add_edge(e[0], e[1], weight=1.0)
    return G


def explicability_metrics(graph: dict) -> dict:
    """Graph properties of an extracted circuit graph, each in [0, 1] where
    higher is *more* explicable."""
    G = _to_nx(graph)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # -- sparsity: fraction of possible layered edges that are absent ------
    unit_dims = graph.get("unit_dims")
    if unit_dims and len(unit_dims) >= 2:
        possible = sum(d * unit_dims[k + 1] for k, d in enumerate(unit_dims[:-1]))
    else:
        possible = n_nodes * (n_nodes - 1)
    density = n_edges / possible if possible else 0.0
    sparsity = max(0.0, 1.0 - density)

    # -- modularity: community structure (greedy Louvain on undirected) ----
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        U = G.to_undirected()
        if U.number_of_edges() > 0:
            comms = list(greedy_modularity_communities(U, weight="weight"))
            mod = nx.algorithms.community.modularity(U, comms, weight="weight")
        else:
            mod = 0.0
    except Exception:
        mod = 0.0
    modularity = float(np.clip(mod, 0.0, 1.0))

    # -- effective depth: longest path (longest chain of computation) ------
    try:
        depth = nx.dag_longest_path_length(G, weight=None) if n_edges else 0
    except Exception:
        depth = 0

    # -- bottleneck: minimum width of any cut between input and output -----
    topo = list(nx.topological_sort(G)) if n_edges else []
    width = n_nodes
    if topo:
        layers = {}
        for nd in G.nodes:
            layers[nd] = G.nodes[nd]["layer"]
        for k in set(layers.values()):
            w = sum(1 for v in layers.values() if v == k)
            width = min(width, w)
    bottleneck = 1.0 - (width / n_nodes) if n_nodes else 0.0
    # higher bottleneck = narrower middle = more structured

    # -- role diversity: distinct structural roles via degree signature -----
    sigs = set()
    for nd in G.nodes:
        sig = (G.in_degree(nd), G.out_degree(nd))
        sigs.add(sig)
    role_diversity = (len(sigs) / n_nodes) if n_nodes else 0.0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "sparsity": sparsity,
        "modularity": modularity,
        "effective_depth": depth,
        "bottleneck": bottleneck,
        "role_diversity": role_diversity,
    }


# ---------------------------------------------------------------------------
# Recovery metrics (extracted graph vs. known ground-truth circuit)
# ---------------------------------------------------------------------------

def _input_supports(graph: dict) -> list[set[int]]:
    """For each non-input node, the set of input-node indices in its
    support (reachable from layer-0 nodes via directed paths)."""
    G = _to_nx(graph)
    inputs = [nd["id"] for nd in graph["nodes"] if nd["role"] == "input"]
    idx = {i: k for k, i in enumerate(inputs)}
    supports: list[set[int]] = []
    for nd in graph["nodes"]:
        if nd["role"] == "input":
            continue
        reach = nx.ancestors(G, nd["id"])
        s = {idx[r] for r in reach if r in idx}
        supports.append(s)
    return [s for s in supports if len(s) >= 2]


def _pair_set(supports: list[set[int]]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for s in supports:
        for a, b in itertools.combinations(sorted(s), 2):
            pairs.add((a, b))
    return pairs


def recovery_wiring_overlap(extracted: dict, ground_truth: dict) -> dict:
    """Functional wiring overlap between extracted and ground-truth graphs.

    Ground truth must be a boolean DAG whose input nodes are the model's
    inputs. For each pair of inputs, does some gate (GT) / some hidden
    unit (extracted) co-wire them? Precision/recall/F1 over all input
    pairs measures how faithfully the model reproduced the circuit's
    input-co-dependence structure.
    """
    gt_supports = _input_supports(ground_truth)
    ex_supports = _input_supports(extracted)
    gt_pairs = _pair_set(gt_supports)
    ex_pairs = _pair_set(ex_supports)
    if not gt_pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_gt_pairs": 0,
                "n_ex_pairs": len(ex_pairs)}
    inter = gt_pairs & ex_pairs
    precision = len(inter) / len(ex_pairs) if ex_pairs else 0.0
    recall = len(inter) / len(gt_pairs)
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(f1), "n_gt_pairs": len(gt_pairs),
            "n_ex_pairs": len(ex_pairs)}


def depth_fidelity(extracted_depth: int, gt_depth: int) -> float:
    """How close the extracted effective depth is to the ground truth."""
    if gt_depth <= 0:
        return 0.0
    return float(min(1.0, extracted_depth / gt_depth))


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def mes_score(expl: dict, recovery: dict | None = None,
              gt_depth: int | None = None) -> dict:
    """Mechanistic Explicability Score, a weighted combination in [0, 1].

    Weights: 0.4 recovery (when a ground truth exists), 0.2 sparsity,
    0.2 modularity, 0.2 depth fidelity. When no ground truth is available
    the recovery weight is redistributed to the other three components.
    """
    if recovery is not None:
        rec = recovery.get("f1", 0.0)
        sp, mo, df = expl["sparsity"], expl["modularity"], expl["effective_depth"]
        if gt_depth is not None:
            df = depth_fidelity(df, gt_depth)
        else:
            df = 1.0
        score = 0.4 * rec + 0.2 * sp + 0.2 * mo + 0.2 * df
        return {"mes": float(np.clip(score, 0.0, 1.0)), "recovery_f1": float(rec),
                "sparsity": float(sp), "modularity": float(mo),
                "depth_fidelity": float(df)}
    sp, mo = expl["sparsity"], expl["modularity"]
    score = (sp + mo) / 2.0
    return {"mes": float(np.clip(score, 0.0, 1.0)), "recovery_f1": None,
            "sparsity": float(sp), "modularity": float(mo),
            "depth_fidelity": None}
