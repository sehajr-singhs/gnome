"""GNN explicability experiment (honest protocol, v3).

Claim tested: a graph network that *reads the circuit graph* extracts more
explicability signal than a graph-agnostic model given the same node
features. Both tasks are designed so the answer is NOT readable from a
one-hot positional feature:

Experiment 1 -- few-shot node-depth inference (no positional features).
  Node features carry only degree and edge-weight statistics --- no layer
  index. The task is to infer a node's depth bin (input / early / mid /
  late / output) from graph structure alone. Depth is a multi-hop
  quantity: it requires propagating "distance from input" along paths,
  which message passing does and a feature-only MLP cannot. We train on K
  graphs and test on held-out graphs (different seeds), reporting
  macro-F1.

Experiment 2 -- transfer of structural reading: size generalization and
  cross-architecture control.
  The readout GNN is trained on circuit graphs extracted from MLPs on
  bool_n7_g12 (7 inputs, seeds 0-5) and tested on three held-out sets:
  (a) same-size MLP graphs (bool_n7_g12 seeds 6-7) -- a sanity check;
  (b) larger-size MLP graphs (bool_n9_g12 seeds 0-3, an input count
  NEVER seen during training) -- tests whether the reading is
  *structural* (size-invariant) rather than a lookup over node counts;
  (c) transformer graphs on the same task (bool_n7_g12 seeds 0-3) -- the
  control: different architecture, wildly different wiring statistics
  (MLP ~137 nodes, sparsity ~0.71; transformer ~250+ nodes, sparsity
  ~0.4). If a graph network reads structure, (b) must transfer while (c)
  is expected to fail -- depth-bin semantics and degree statistics are
  architecture-specific. Reported per readout: macro-F1 on each set.

Run:  python benchmarks/run_gnn_experiment.py [--exp 1|2|both] [--fast]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnome.circuits import make_task  # noqa: E402
from gnome.extraction import extract_circuit_graph  # noqa: E402
from gnome.graphnets import CircuitGNN, graph_to_batch  # noqa: E402
from gnome.metrics import recovery_wiring_overlap  # noqa: E402
from gnome.models import build_model  # noqa: E402
from gnome.training import train_classifier  # noqa: E402

DEPTH_BINS = ["input", "early", "mid", "late", "output"]


def depth_bin(layer: int, n_layers: int) -> int:
    """Map unit layer index to a depth bin (input/early/mid/late/output)."""
    if layer == 0:
        return 0
    if layer == n_layers - 1:
        return 4
    frac = layer / (n_layers - 1)
    if frac < 0.4:
        return 1
    if frac < 0.7:
        return 2
    return 3


def build_and_extract(task, seed, epochs, h, rel_thresh=0.5, train=True,
                      kind="mlp"):
    X, y = task.generate(n=128, seed=seed)
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.long)
    if kind == "mlp":
        model = build_model("mlp", task, h=h)
    else:
        model = build_model("transformer", task, d_model=32, d_ff=64, n_heads=2)
    if train:
        train_classifier(model, Xt, yt, epochs=epochs, seed=seed)
    g = extract_circuit_graph(model, Xt, rel_thresh=rel_thresh)
    meta = {"task": task.name, "seed": seed, "h": h,
            "trained": train, "kind": kind}
    if task.family == "boolean":
        rec = recovery_wiring_overlap(g, task.ground_truth_graph())
        meta["recovery_f1"] = float(rec["f1"])
    g["_meta"] = meta
    return g


def node_depth_labels(graph) -> torch.Tensor:
    n_layers = max(nd["layer"] for nd in graph["nodes"]) + 1
    return torch.tensor([depth_bin(nd["layer"], n_layers)
                         for nd in graph["nodes"]], dtype=torch.long)


def macro_f1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(1)
    f1s = []
    for c in range(logits.shape[1]):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return float(np.mean(f1s))


class FeatureOnlyBaseline(nn.Module):
    """MLP on node features: same information, no edges."""

    def __init__(self, in_f: int, hidden: int = 64, n_roles: int = 5,
                 n_graph_targets: int = 0):
        super().__init__()
        self.n_graph_targets = n_graph_targets
        self.net = nn.Sequential(nn.Linear(in_f, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.role_head = nn.Linear(hidden, n_roles)
        if n_graph_targets > 0:
            self.pool = nn.Linear(hidden, hidden)
            self.graph_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_graph_targets))

    def forward(self, data: dict):
        x = self.net(data["x"])
        role_logits = self.role_head(x)
        out = {"roles": role_logits}
        if self.n_graph_targets > 0:
            pooled = torch.zeros(data["n_graphs"], x.shape[1], device=x.device)
            pooled = pooled.index_add(0, data["batch"], self.pool(x))
            counts = torch.bincount(data["batch"], minlength=data["n_graphs"])
            pooled = pooled / counts.unsqueeze(1).clamp(min=1.0)  # mean pool
            out["graph"] = self.graph_head(pooled)
        return out


def fit(model, data, roles, gt, epochs, seed=0, w_graph=0.5):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data)
        loss = nn.functional.cross_entropy(out["roles"], roles)
        if gt is not None:
            loss = loss + w_graph * nn.functional.mse_loss(out["graph"], gt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def make_models(in_f, n_graph_targets):
    return {
        "gcn": CircuitGNN(n_layers=1, hidden=64, n_roles=5,
                          n_graph_targets=n_graph_targets, kind="gcn",
                          depth=3, in_f=in_f),
        "gat": CircuitGNN(n_layers=1, hidden=64, n_roles=5,
                          n_graph_targets=n_graph_targets, kind="gat",
                          depth=3, in_f=in_f),
        "baseline_mlp": FeatureOnlyBaseline(in_f=in_f, n_roles=5,
                                            n_graph_targets=n_graph_targets),
    }


def exp1_depth_fewshot(graphs, train_seeds, test_seeds, epochs):
    """Few-shot depth-bin inference from structure alone."""
    train_g = [g for g in graphs if g["_meta"]["seed"] in train_seeds]
    test_g = [g for g in graphs if g["_meta"]["seed"] in test_seeds]
    tr = graph_to_batch(train_g, positional=False)
    te = graph_to_batch(test_g, positional=False)
    in_f = tr["x"].shape[1]
    tr_roles = torch.cat([node_depth_labels(g) for g in train_g])
    te_roles = torch.cat([node_depth_labels(g) for g in test_g])
    out = {"n_train_graphs": len(train_g), "n_test_graphs": len(test_g)}
    for name, model in make_models(in_f, 0).items():
        fit(model, tr, tr_roles, None, epochs)
        with torch.no_grad():
            pred = model(te)["roles"]
        out[name] = {"test_macro_f1": macro_f1(pred, te_roles),
                     "test_acc": float((pred.argmax(1) == te_roles).float().mean())}
    return out


def exp2_transfer(graphs_small, graphs_large, graphs_tf, epochs):
    """Transfer of depth-role reading (train MLP bool_n7, seeds 0-5).

    Test sets: (a) held-out same-size MLP graphs, (b) larger-size MLP
    graphs with an input count never seen in training, (c) transformer
    graphs as the architecture control. Node features are positional=False
    throughout, so the reading must come from structure.
    """
    train_g = graphs_small[:6]
    holdout = graphs_small[6:]
    tr = graph_to_batch(train_g, positional=False)
    te_same = graph_to_batch(holdout, positional=False)
    te_large = graph_to_batch(graphs_large, positional=False)
    te_tf = graph_to_batch(graphs_tf, positional=False)
    in_f = tr["x"].shape[1]
    tr_roles = torch.cat([node_depth_labels(g) for g in train_g])
    te_same_roles = torch.cat([node_depth_labels(g) for g in holdout])
    te_large_roles = torch.cat([node_depth_labels(g) for g in graphs_large])
    te_tf_roles = torch.cat([node_depth_labels(g) for g in graphs_tf])
    out = {"n_train_graphs": len(train_g),
           "n_test_same_size": len(holdout),
           "n_test_larger_size": len(graphs_large),
           "n_test_transformer": len(graphs_tf),
           "train_arch": "mlp_bool_n7_g12",
           "larger_size_arch": "mlp_bool_n9_g12",
           "transformer_arch": "tf_bool_n7_g12"}
    for name, model in make_models(in_f, 0).items():
        fit(model, tr, tr_roles, None, epochs)
        with torch.no_grad():
            p_same = model(te_same)["roles"]
            p_large = model(te_large)["roles"]
            p_tf = model(te_tf)["roles"]
        out[name] = {"test_same_size_macro_f1": macro_f1(p_same, te_same_roles),
                     "test_larger_size_macro_f1": macro_f1(p_large, te_large_roles),
                     "test_tf_macro_f1": macro_f1(p_tf, te_tf_roles)}
    return out


def save(results: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    torch.set_num_threads(8)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gnn_experiment.json")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--gnn-epochs", type=int, default=350)
    ap.add_argument("--exp", choices=["1", "2", "both"], default="both")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    # merge into any existing results so --exp 1 and --exp 2 can be run
    # separately without overwriting each other
    results = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    results["config"] = vars(args)

    if args.exp in ("1", "both"):
        print("exp1: collecting same-architecture graphs (8 seeds)...", flush=True)
        seeds = [0, 1, 2, 3, 4, 5, 6, 7] if not args.fast else [0, 1, 2]
        graphs1 = [build_and_extract(make_task(f"bool_n7_g12_s{s}"), s,
                                     args.epochs, h=32) for s in seeds]
        results["exp1"] = {}
        for K in ([2, 4, 6] if not args.fast else [2]):
            results["exp1"][f"K{K}"] = exp1_depth_fewshot(
                graphs1, seeds[:K], seeds[K:], args.gnn_epochs)
            print(f"  K={K}: " + ", ".join(
                f"{k} F1={v['test_macro_f1']:.3f}" for k, v in
                results["exp1"][f"K{K}"].items() if isinstance(v, dict)),
                flush=True)
        save(results, args.out)
        print("  (partial saved)", flush=True)

    if args.exp in ("2", "both"):
        print("exp2: collecting MLP graphs (n7 seeds 0-7, n9 seeds 0-3) + "
              "transformer graphs (n7 seeds 0-3)...", flush=True)
        seeds7 = [0, 1, 2, 3, 4, 5, 6, 7] if not args.fast else [0, 1, 2]
        seeds9 = [0, 1, 2, 3] if not args.fast else [0]
        tf_seeds = [0, 1, 2, 3] if not args.fast else [0]
        graphs_small = [build_and_extract(make_task(f"bool_n7_g12_s{s}"), s,
                                          args.epochs, h=32) for s in seeds7]
        graphs_large = [build_and_extract(make_task(f"bool_n9_g12_s{s}"), s,
                                          args.epochs, h=32) for s in seeds9]
        graphs_tf = [build_and_extract(make_task(f"bool_n7_g12_s{s}"), s,
                                       args.epochs, h=32, kind="transformer")
                     for s in tf_seeds]
        results["exp2"] = exp2_transfer(graphs_small, graphs_large, graphs_tf,
                                         args.gnn_epochs)
        print("  exp2: " + ", ".join(
            f"{k} same={v['test_same_size_macro_f1']:.3f} "
            f"larger={v['test_larger_size_macro_f1']:.3f} "
            f"tf={v['test_tf_macro_f1']:.3f}"
            for k, v in results["exp2"].items() if isinstance(v, dict)),
            flush=True)
        save(results, args.out)
        print("  (partial saved)", flush=True)

    results["total_time_s"] = round(time.time() - t0, 1)
    save(results, args.out)
    print(f"wrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
