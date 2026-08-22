"""Recovery-prediction control (the boundary condition in the GNOmE paper).

Question: can a graph readout predict recovery F1 -- how much of an unseen
target circuit a model learned -- from the extracted circuit graph alone,
with the ground-truth circuit NEVER shown?

Protocol (clean held-out size):
  * graphs from boolean circuits: input counts {5, 7} (train) and {9} (test,
    an input count never seen during readout training), gate counts {8, 16},
    widths {16, 32}, trained and untrained models. Recovery F1 spreads from
    ~0.4 (untrained / large inputs) to ~0.95 (trained, small inputs).
  * readouts: GCN, GAT, and a feature-only MLP baseline on the same node
    features (degree + edge-weight statistics, no positional features).
  * target: graph-level recovery F1; metric: test MAE (in F1 units) and R^2.

Expected and reported: a null result for every readout. The feature-only
baseline failing identically shows the barrier is informational (recovery
is a statement about a target the extractor never sees), not architectural.

Run:  python benchmarks/run_recovery_control.py --out results/recovery_control.json
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


def build_and_extract(task, seed, epochs, h, rel_thresh=0.5, train=True):
    X, y = task.generate(n=128, seed=seed)
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.long)
    model = build_model("mlp", task, h=h)
    if train:
        train_classifier(model, Xt, yt, epochs=epochs, seed=seed)
    g = extract_circuit_graph(model, Xt, rel_thresh=rel_thresh)
    rec = recovery_wiring_overlap(g, task.ground_truth_graph())
    g["_meta"] = {"task": task.name, "h": h, "trained": train,
                  "recovery_f1": float(rec["f1"])}
    return g


def node_depth_labels(graph) -> torch.Tensor:
    n_layers = max(nd["layer"] for nd in graph["nodes"]) + 1
    return torch.tensor([depth_bin(nd["layer"], n_layers)
                         for nd in graph["nodes"]], dtype=torch.long)


class FeatureOnlyBaseline(nn.Module):
    def __init__(self, in_f: int, hidden: int = 64, n_roles: int = 5,
                 n_graph_targets: int = 1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_f, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.role_head = nn.Linear(hidden, n_roles)
        self.pool = nn.Linear(hidden, hidden)
        self.graph_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_graph_targets))

    def forward(self, data: dict):
        x = self.net(data["x"])
        role_logits = self.role_head(x)
        pooled = torch.zeros(data["n_graphs"], x.shape[1], device=x.device)
        pooled = pooled.index_add(0, data["batch"], self.pool(x))
        counts = torch.bincount(data["batch"], minlength=data["n_graphs"])
        pooled = pooled / counts.unsqueeze(1).clamp(min=1.0)
        return {"roles": role_logits, "graph": self.graph_head(pooled)}


def make_models(in_f):
    return {
        "gcn": CircuitGNN(n_layers=1, hidden=64, n_roles=5,
                          n_graph_targets=1, kind="gcn", depth=3, in_f=in_f),
        "gat": CircuitGNN(n_layers=1, hidden=64, n_roles=5,
                          n_graph_targets=1, kind="gat", depth=3, in_f=in_f),
        "baseline_mlp": FeatureOnlyBaseline(in_f=in_f),
    }


def fit(model, data, roles, gt, epochs, seed=0, w_graph=0.5):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data)
        loss = nn.functional.cross_entropy(out["roles"], roles) \
            + w_graph * nn.functional.mse_loss(out["graph"], gt)
        loss.backward()
        opt.step()
    model.eval()
    return model


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


def main() -> None:
    torch.set_num_threads(8)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/recovery_control.json")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--gnn-epochs", type=int, default=350)
    args = ap.parse_args()
    t0 = time.time()

    cfgs = []
    for n_in in (5, 7):                       # train input counts
        for n_g in (8, 16):
            for h in (16, 32):
                for tr in (True, False):
                    cfgs.append((n_in, n_g, h, tr))
    for n_g in (8, 16):                       # test input count (unseen)
        for h in (16, 32):
            for tr in (True, False):
                cfgs.append((9, n_g, h, tr))
    print(f"collecting {len(cfgs)} boolean graphs "
          f"(train n_in in {{5,7}}, test n_in=9)...", flush=True)
    graphs = [build_and_extract(make_task(f"bool_n{n_in}_g{n_g}_s0"), 0,
                                args.epochs, h=h, train=tr)
              for (n_in, n_g, h, tr) in cfgs]
    f1s = [g["_meta"]["recovery_f1"] for g in graphs]
    print(f"  recovery F1 range: {min(f1s):.3f}..{max(f1s):.3f}", flush=True)

    n_tr = 16  # n_in in {5, 7}: 2 sizes x 2 gates x 2 widths x 2 trained
    train_g, test_g = graphs[:n_tr], graphs[n_tr:]
    tr = graph_to_batch(train_g, positional=False)
    te = graph_to_batch(test_g, positional=False)
    in_f = tr["x"].shape[1]
    tr_roles = torch.cat([node_depth_labels(g) for g in train_g])
    te_roles = torch.cat([node_depth_labels(g) for g in test_g])
    tr_gt = torch.tensor([g["_meta"]["recovery_f1"] for g in train_g],
                         dtype=torch.float32).unsqueeze(1)
    te_gt = torch.tensor([g["_meta"]["recovery_f1"] for g in test_g],
                         dtype=torch.float32).unsqueeze(1)

    out = {"n_train_graphs": len(train_g), "n_test_graphs": len(test_g),
           "target": "recovery_f1",
           "test_recovery_f1_min": float(te_gt.min().item()),
           "test_recovery_f1_max": float(te_gt.max().item()),
           "train_recovery_f1_min": float(tr_gt.min().item()),
           "train_recovery_f1_max": float(tr_gt.max().item())}
    for name, model in make_models(in_f).items():
        fit(model, tr, tr_roles, tr_gt, args.gnn_epochs)
        with torch.no_grad():
            pred = model(te)["graph"]
        mae = float((pred - te_gt).abs().mean().item())
        ss_res = float(((pred - te_gt) ** 2).sum().item())
        ss_tot = float(((te_gt - te_gt.mean(0)) ** 2).sum().item())
        r2 = 1.0 - ss_res / (ss_tot + 1e-9)
        out[name] = {"test_graph_mae": mae, "test_graph_r2": r2}
        print(f"  {name}: MAE={mae:.3f} R2={r2:.3f}", flush=True)
    out["total_time_s"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
