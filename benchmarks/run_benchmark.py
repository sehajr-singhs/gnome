"""Benchmark runner: train base models, extract circuit graphs, score.

For every task we:
  1. train a base model (MLP and/or transformer),
  2. extract its circuit graph via blockwise Jacobian attribution,
  3. compute explicability metrics and (for boolean tasks) recovery
     against the known ground-truth circuit,
  4. write one JSON row.

Run:  python benchmarks/run_benchmark.py [--cpu] [--out results/benchmark_cpu.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnome.circuits import make_task  # noqa: E402
from gnome.extraction import extract_circuit_graph  # noqa: E402
from gnome.metrics import (  # noqa: E402
    explicability_metrics,
    mes_score,
    recovery_wiring_overlap,
)
from gnome.models import build_model  # noqa: E402
from gnome.training import train_classifier  # noqa: E402


def run_boolean(task, seed: int, epochs: int, h: int, rel_thresh: float) -> dict:
    X, y = task.generate(n=128, seed=seed)
    Xt = torch.tensor(X)
    yt = torch.tensor(y, dtype=torch.long)
    model = build_model("mlp", task, h=h)
    t0 = time.time()
    train_stats = train_classifier(model, Xt, yt, epochs=epochs, seed=seed)
    train_t = time.time() - t0

    t0 = time.time()
    graph = extract_circuit_graph(model, Xt, rel_thresh=rel_thresh)
    extract_t = time.time() - t0

    expl = explicability_metrics(graph)
    gt = task.ground_truth_graph()
    rec = recovery_wiring_overlap(graph, gt)
    gt_depth = max(nd["layer"] for nd in gt["nodes"])
    mes = mes_score(expl, rec, gt_depth)
    return {
        "task": task.name, "family": "boolean", "model": "mlp",
        "seed": seed, "h": h, "rel_thresh": rel_thresh,
        "train_accuracy": train_stats["accuracy"],
        "train_time_s": round(train_t, 3), "extract_time_s": round(extract_t, 3),
        "explicability": expl, "recovery": rec, **mes,
    }


def run_modular(task, seed: int, epochs: int, kind: str, rel_thresh: float) -> dict:
    n = task.p * task.p * 8
    X, y = task.generate(n=n, seed=seed)
    Xt = torch.tensor(X)
    yt = torch.tensor(y, dtype=torch.long)
    model = build_model(kind, task, h=64) if kind == "mlp" else build_model(
        kind, task, d_model=32, d_ff=64, n_heads=2)
    t0 = time.time()
    train_stats = train_classifier(model, Xt, yt, epochs=epochs, seed=seed)
    train_t = time.time() - t0

    t0 = time.time()
    graph = extract_circuit_graph(model, Xt, rel_thresh=rel_thresh)
    extract_t = time.time() - t0

    expl = explicability_metrics(graph)
    gt = task.ground_truth_graph()
    gt_depth = max(nd["layer"] for nd in gt["nodes"])
    mes = mes_score(expl, None, gt_depth)
    return {
        "task": task.name, "family": "modular", "model": kind,
        "seed": seed, "rel_thresh": rel_thresh,
        "train_accuracy": train_stats["accuracy"],
        "train_time_s": round(train_t, 3), "extract_time_s": round(extract_t, 3),
        "explicability": expl, "recovery": None, **mes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/benchmark_cpu.json")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--rel-thresh", type=float, default=0.5)
    ap.add_argument("--fast", action="store_true", help="tiny smoke run")
    args = ap.parse_args()

    results = []
    t_all = time.time()

    # Boolean: 3 gate counts x 3 seeds
    if args.fast:
        bool_cfgs = [(7, 8, 0)]
    else:
        bool_cfgs = [(7, g, s) for g in (8, 12, 16) for s in (0, 1, 2)]
    for (n_in, n_gates, seed) in bool_cfgs:
        task = make_task(f"bool_n{n_in}_g{n_gates}_s{seed}")
        results.append(run_boolean(task, seed, args.epochs, h=32,
                                   rel_thresh=args.rel_thresh))
        print(f"  done {task.name}")

    # Modular: p=5 and p=7, MLP + transformer
    if args.fast:
        mod_cfgs = [(5, "mlp")]
    else:
        mod_cfgs = [(p, k) for p in (5, 7) for k in ("mlp", "transformer")]
    for (p, kind) in mod_cfgs:
        task = make_task(f"mod_add_p{p}")
        results.append(run_modular(task, 0, args.epochs, kind,
                                   rel_thresh=args.rel_thresh))
        print(f"  done {task.name} [{kind}]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": results, "total_time_s": round(time.time() - t_all, 1)},
                  f, indent=2)
    print(f"\nwrote {len(results)} rows -> {args.out} "
          f"({time.time() - t_all:.1f}s total)")


if __name__ == "__main__":
    main()
