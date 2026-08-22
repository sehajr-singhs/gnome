"""Ground-truth circuit synthesis.

Every task in GNOmE carries a *known* circuit graph, so circuit recovery
is measurable instead of asserted. Two families:

* ModularTask: (a op b) mod p. The ground-truth circuit is the canonical
  Fourier/trigonometric algorithm (Nanda et al. 2023): frequency features
  per residue, trig products that implement the addition formula, a sum,
  and an argmax readout. Structure is known by construction.
* RandomBooleanCircuit: a random DAG of {AND, OR, XOR} gates over n_in
  inputs. The ground-truth graph is literally the DAG we generated.

Input representation to the model is concatenated one-hot vectors, so a
task's input dimension is n_onehot. Output is a class index in [0, n_out).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CircuitTask:
    name: str
    n_input: int
    n_output: int
    family: str  # "modular" | "boolean"

    def generate(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) with X one-hot (n, n_input), y classes (n,)."""
        raise NotImplementedError

    def ground_truth_graph(self) -> dict:
        """Known circuit graph: nodes with roles/layers, and edges."""
        raise NotImplementedError


def _onehot(vals: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((vals.shape[0], width), dtype=np.float32)
    out[np.arange(vals.shape[0]), vals.astype(int)] = 1.0
    return out


# ---------------------------------------------------------------------------
# Modular arithmetic with the known trigonometric circuit
# ---------------------------------------------------------------------------

class ModularTask(CircuitTask):
    """(a op b) mod p. Input = onehot(a) ++ onehot(b), output = class."""

    def __init__(self, p: int, op: str = "add", name: str | None = None):
        self.p = p
        self.op = op
        super().__init__(
            name=name or f"mod_{op}_p{p}",
            n_input=2 * p,
            n_output=p,
            family="modular",
        )

    def _apply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.op == "add":
            return (a + b) % self.p
        if self.op == "mul":
            return (a * b) % self.p
        if self.op == "sub":
            return (a - b) % self.p
        raise ValueError(self.op)

    def generate(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        a = rng.integers(0, self.p, size=n)
        b = rng.integers(0, self.p, size=n)
        X = np.concatenate([_onehot(a, self.p), _onehot(b, self.p)], axis=1)
        return X, self._apply(a, b)

    def ground_truth_graph(self) -> dict:
        """The known Fourier/trig circuit (Nanda et al. 2023), as a graph."""
        p = self.p
        K = (p - 1) // 2
        nodes: list[dict] = [{"id": "a", "role": "input", "layer": 0},
                             {"id": "b", "role": "input", "layer": 0}]
        edges: list[tuple[str, str]] = []
        for k in range(1, K + 1):
            ca, sa = f"cos_a_{k}", f"sin_a_{k}"
            cb, sb = f"cos_b_{k}", f"sin_b_{k}"
            pc, ps = f"prod_cos_{k}", f"prod_sin_{k}"
            nodes += [
                {"id": ca, "role": "freq_feature", "layer": 1},
                {"id": sa, "role": "freq_feature", "layer": 1},
                {"id": cb, "role": "freq_feature", "layer": 1},
                {"id": sb, "role": "freq_feature", "layer": 1},
                {"id": pc, "role": "product", "layer": 2},
                {"id": ps, "role": "product", "layer": 2},
            ]
            edges += [(ca, pc), (cb, pc), (sa, ps), (sb, ps)]
        nodes += [{"id": "sum_cos", "role": "sum", "layer": 3},
                  {"id": "sum_sin", "role": "sum", "layer": 3},
                  {"id": "out", "role": "output", "layer": 4}]
        for k in range(1, K + 1):
            edges += [(f"prod_cos_{k}", "sum_cos"), (f"prod_sin_{k}", "sum_sin")]
        edges += [("sum_cos", "out"), ("sum_sin", "out")]
        return {"nodes": nodes, "edges": edges, "family": "modular",
                "note": f"known Fourier/trig circuit for ({self.op}) mod {p}"}


# ---------------------------------------------------------------------------
# Random Boolean circuits with a literally-known DAG
# ---------------------------------------------------------------------------

_GATES = {"AND", "OR", "XOR"}


def _gate_eval(gate: str, x: int, y: int) -> int:
    if gate == "AND":
        return x & y
    if gate == "OR":
        return x | y
    return x ^ y


class RandomBooleanCircuit(CircuitTask):
    """Random layered DAG of {AND, OR, XOR} gates. Graph is known exactly."""

    def __init__(self, n_in: int = 7, n_gates: int = 12, seed: int = 0,
                 name: str | None = None):
        self.n_in = n_in
        self.n_gates = n_gates
        self.seed = seed
        self._dag = self._build_dag()
        super().__init__(
            name=name or f"bool_n{n_in}_g{n_gates}_s{seed}",
            n_input=n_in,
            n_output=2,  # binary output
            family="boolean",
        )

    def _build_dag(self) -> dict:
        rng = random.Random(self.seed)
        # nodes: list of dicts {id, gate, parents:[ids], layer}
        # parents are indices into the node list (inputs first, then gates).
        nodes: list[dict] = [{"id": f"x{i}", "gate": "INPUT", "parents": [],
                              "layer": 0} for i in range(self.n_in)]
        for g in range(self.n_gates):
            # parents from any earlier node (inputs or earlier gates)
            avail = list(range(len(nodes)))
            p1, p2 = rng.sample(avail, 2)
            layer = max(nodes[p1]["layer"], nodes[p2]["layer"]) + 1
            gate = rng.choice(sorted(_GATES))
            nodes.append({"id": f"g{g}", "gate": gate, "parents": [p1, p2],
                          "layer": layer})
        return {"nodes": nodes, "output": len(nodes) - 1}

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        """X (n, n_in) bits -> y (n,) output bit of the DAG."""
        n = X.shape[0]
        vals = np.zeros((n, len(self._dag["nodes"])), dtype=np.int64)
        vals[:, : self.n_in] = X.astype(np.int64)
        nodes = self._dag["nodes"]
        for idx in range(self.n_in, len(nodes)):
            nd = nodes[idx]
            v = _gate_eval(nd["gate"], vals[:, nd["parents"][0]],
                           vals[:, nd["parents"][1]])
            vals[:, idx] = v
        return vals[:, self._dag["output"]]

    def generate(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        # Truth table over all 2^n_in inputs; sample n rows (seeded).
        grid = np.array(list(itertools.product([0, 1], repeat=self.n_in)),
                        dtype=np.float32)
        y_all = self._evaluate(grid)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(grid), size=min(n, len(grid)), replace=False)
        return grid[idx].copy(), y_all[idx]

    def ground_truth_graph(self) -> dict:
        nodes = [{"id": nd["id"], "role": "input" if nd["gate"] == "INPUT"
                  else nd["gate"].lower(), "layer": nd["layer"]}
                 for nd in self._dag["nodes"]]
        edges = []
        for idx, nd in enumerate(self._dag["nodes"]):
            for p in nd["parents"]:
                edges.append((nodes[p]["id"], nd["id"]))
        return {"nodes": nodes, "edges": edges, "family": "boolean",
                "note": "the literal generated DAG"}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def make_task(name: str, seed: int = 0) -> CircuitTask:
    if name.startswith("mod_add_p"):
        return ModularTask(int(name.split("p")[1]), "add")
    if name.startswith("mod_mul_p"):
        return ModularTask(int(name.split("p")[1]), "mul")
    if name.startswith("mod_sub_p"):
        return ModularTask(int(name.split("p")[1]), "sub")
    if name.startswith("bool_"):
        parts = name.split("_")
        n_in = int(parts[1][1:])  # n<N>
        n_g = int(parts[2][1:])   # g<G>
        s = int(parts[3][1:]) if len(parts) > 3 and parts[3].startswith("s") else seed
        return RandomBooleanCircuit(n_in=n_in, n_gates=n_g, seed=s)
    raise ValueError(f"unknown task {name}")
