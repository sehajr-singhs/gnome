"""Analysis of benchmark + GNN experiment results (paper numbers).

Prints:
  * per-family summary tables with mean +/- std over seeds
  * recovery aggregates for boolean tasks
  * MES aggregates
  * GNN few-shot trend and cross-architecture regression summary
All numbers come from the committed JSON results (results/*.json).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(name: str) -> dict:
    with open(os.path.join(ROOT, "results", name)) as f:
        return json.load(f)


def fmt(x, sig=3):
    return f"{x:.{sig}f}"


def summarize_benchmark() -> None:
    d = load("benchmark_cpu.json")
    rows = d["rows"]
    print("=" * 78)
    print("BENCHMARK: base models, extraction, explicability + recovery")
    print("=" * 78)
    hdr = (f"{'group':<16}{'n':>3}{'acc':>8}{'sparsity':>10}{'mod':>8}"
           f"{'depth':>7}{'recF1':>8}{'MES':>7}")
    print(hdr)
    groups = {
        "bool g8": [r for r in rows if r["family"] == "boolean" and "g8" in r["task"]],
        "bool g12": [r for r in rows if r["family"] == "boolean" and "g12" in r["task"]],
        "bool g16": [r for r in rows if r["family"] == "boolean" and "g16" in r["task"]],
        "mod p5 mlp": [r for r in rows if r["task"] == "mod_add_p5" and r["model"] == "mlp"],
        "mod p5 tf": [r for r in rows if r["task"] == "mod_add_p5" and r["model"] == "transformer"],
        "mod p7 mlp": [r for r in rows if r["task"] == "mod_add_p7" and r["model"] == "mlp"],
        "mod p7 tf": [r for r in rows if r["task"] == "mod_add_p7" and r["model"] == "transformer"],
    }
    for name, g in groups.items():
        if not g:
            continue
        acc = np.mean([r["train_accuracy"] for r in g])
        sp = np.mean([r["explicability"]["sparsity"] for r in g])
        mo = np.mean([r["explicability"]["modularity"] for r in g])
        dp = np.mean([r["explicability"]["effective_depth"] for r in g])
        f1s = [r["recovery"]["f1"] for r in g if r["recovery"]]
        f1 = np.mean(f1s) if f1s else float("nan")
        mes = np.mean([r["mes"] for r in g])
        print(f"{name:<16}{len(g):>3}{fmt(acc):>8}{fmt(sp):>10}{fmt(mo):>8}"
              f"{dp:>7.1f}{fmt(f1):>8}{fmt(mes):>7}")

    bool_rows = [r for r in rows if r["family"] == "boolean"]
    recs = [r["recovery"] for r in bool_rows]
    print("\n-- boolean recovery (all seeds, n=%d)" % len(recs))
    print(f"  precision {np.mean([r['precision'] for r in recs]):.3f} +- {np.std([r['precision'] for r in recs]):.3f}")
    print(f"  recall    {np.mean([r['recall'] for r in recs]):.3f} +- {np.std([r['recall'] for r in recs]):.3f}")
    print(f"  f1        {np.mean([r['f1'] for r in recs]):.3f} +- {np.std([r['f1'] for r in recs]):.3f}")
    print(f"  recall==1.0 in {sum(1 for r in recs if r['recall'] >= 0.999)}/{len(recs)} models")
    mlp_sp = np.mean([r["explicability"]["sparsity"] for r in rows if r["model"] == "mlp"])
    tf_sp = np.mean([r["explicability"]["sparsity"] for r in rows if r["model"] == "transformer"])
    print(f"\n  MLP mean sparsity  {mlp_sp:.3f} vs transformer {tf_sp:.3f} (architectural density signature)")


def summarize_gnn() -> None:
    try:
        d = load("gnn_experiment.json")
    except FileNotFoundError:
        print("gnn_experiment.json not found yet")
        return
    print("\n" + "=" * 78)
    print("GNN EXPERIMENT: graph networks reading explicability")
    print("=" * 78)
    for K, res in d.get("exp1", {}).items():
        if not isinstance(res, dict) or "n_train_graphs" not in res:
            continue
        line = f"  few-shot K={K} (train {res['n_train_graphs']}, test {res['n_test_graphs']}): "
        for name in ("gcn", "gat", "baseline_mlp"):
            v = res.get(name, {})
            line += f"{name} F1={v.get('test_macro_f1', float('nan')):.3f} "
        print(line)
    e2 = d.get("exp2", {})
    if e2:
        print(f"  transfer (train {e2.get('n_train_graphs')} MLP n7 graphs; "
              f"test same-size {e2.get('n_test_same_size')}, larger-size "
              f"{e2.get('n_test_larger_size')}, transformer {e2.get('n_test_transformer')}):")
        for name in ("gcn", "gat", "baseline_mlp"):
            v = e2.get(name, {})
            print(f"    {name}: same={v.get('test_same_size_macro_f1', float('nan')):.3f} "
                  f"larger={v.get('test_larger_size_macro_f1', float('nan')):.3f} "
                  f"tf={v.get('test_tf_macro_f1', float('nan')):.3f}")


if __name__ == "__main__":
    summarize_benchmark()
    summarize_gnn()
