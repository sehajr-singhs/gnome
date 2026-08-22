"""Paper figures for GNOmE, generated from committed result JSONs.

Figures:
  fig1_pipeline.png    GNOmE pipeline schematic (drawn, not measured)
  fig2_recovery.png    recovery precision/recall/F1 per boolean task
  fig3_density.png     MLP vs transformer graph density (modular tasks)
  fig4_gnn_fewshot.png few-shot role macro-F1: GCN/GAT vs baseline
  fig5_gnn_transfer.png    size generalization + cross-architecture control

Run:  python benchmarks/make_figures.py   (writes into figs/)
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)

INK = "#1a1a1a"
MUTED = "#8c8e90"
BLUE = "#226999"
RED = "#b03a2e"
GREEN = "#2e6b4f"


def load(name):
    with open(os.path.join(ROOT, "results", name)) as f:
        return json.load(f)


def fig1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    boxes = [
        (0.03, 0.42, 0.16, 0.3, "trained model\n(MLP / transformer)", INK),
        (0.25, 0.42, 0.17, 0.3, "blockwise\nJacobians W_k", BLUE),
        (0.48, 0.42, 0.17, 0.3, "circuit graph\n(thresholded DAG)", INK),
        (0.71, 0.42, 0.17, 0.3, "explicability\nmetrics + MES", GREEN),
        (0.13, 0.02, 0.22, 0.22, "ground-truth circuit\n(known wiring)", MUTED),
        (0.55, 0.02, 0.22, 0.22, "graph network\n(reads roles, metrics)", BLUE),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=color,
                                   lw=1.4, zorder=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8.5, color=color, zorder=3)
    for (x1, y1, x2, y2) in [(0.19, 0.57, 0.25, 0.57), (0.42, 0.57, 0.48, 0.57),
                             (0.65, 0.57, 0.71, 0.57),
                             (0.24, 0.42, 0.24, 0.24), (0.66, 0.42, 0.66, 0.24)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.text(0.5, 0.97, "GNOmE: Graph Networks for Mechanistic Explicability",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(0.24, 0.21, "recovery F1 (wiring overlap)", ha="center", fontsize=7.5,
            color=MUTED)
    ax.text(0.66, 0.21, "roles, sparsity, modularity, depth", ha="center",
            fontsize=7.5, color=MUTED)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig1_pipeline.png"), dpi=200)
    plt.close()


def fig2_recovery() -> None:
    d = load("benchmark_cpu.json")
    rows = [r for r in d["rows"] if r["family"] == "boolean"]
    tasks = [r["task"] for r in rows]
    P = [r["recovery"]["precision"] for r in rows]
    R = [r["recovery"]["recall"] for r in rows]
    F = [r["recovery"]["f1"] for r in rows]
    x = np.arange(len(tasks))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.bar(x - w, P, w, label="precision", color=BLUE)
    ax.bar(x, R, w, label="recall", color=GREEN)
    ax.bar(x + w, F, w, label="F1", color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("bool_n7_", "").replace("_s", "\ns") for t in tasks],
                       fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recovery score")
    ax.set_title("Circuit recovery: wiring overlap between extracted graph\nand known ground-truth circuit (boolean tasks)")
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig2_recovery.png"), dpi=200)
    plt.close()


def fig3_density() -> None:
    d = load("benchmark_cpu.json")
    mods = [r for r in d["rows"] if r["family"] == "modular"]
    names = sorted({r["model"] for r in mods})
    tasks = ["mod_add_p5", "mod_add_p7"]
    x = np.arange(len(tasks))
    w = 0.34
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = {"mlp": BLUE, "transformer": RED}
    for i, m in enumerate(names):
        vals = [next((r["explicability"]["sparsity"] for r in mods
                      if r["task"] == t and r["model"] == m), np.nan) for t in tasks]
        ax.bar(x + (i - 0.5) * w, vals, w, label=m, color=colors[m])
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("graph sparsity (1 - density)")
    ax.set_title("Architectural density signature: transformers wire\ndenser circuit graphs than MLPs on the same task")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig3_density.png"), dpi=200)
    plt.close()


def fig4_gnn_fewshot() -> None:
    d = load("gnn_experiment.json")
    exp1 = d.get("exp1", {})
    ks = sorted(int(k[1:]) for k in exp1 if k.startswith("K"))
    if not ks:
        print("no exp1 data yet")
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for name, color, mk in (("gcn", BLUE, "o"), ("gat", RED, "s"),
                            ("baseline_mlp", MUTED, "D")):
        vals = [exp1[f"K{k}"][name]["test_macro_f1"] for k in ks]
        ax.plot(ks, vals, marker=mk, color=color, label=name, lw=1.6)
    ax.set_xlabel("training graphs (few-shot K)")
    ax.set_ylabel("test macro-F1 (roles)")
    ax.set_xticks(ks)
    ax.set_title("Graph networks read structure from few graphs")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig4_gnn_fewshot.png"), dpi=200)
    plt.close()


def fig5_gnn_transfer() -> None:
    d = load("gnn_experiment.json")
    e2 = d.get("exp2", {})
    if not e2:
        print("no exp2 data yet")
        return
    names = [n for n in ("gcn", "gat", "baseline_mlp") if n in e2]
    sets = [("same-size", "test_same_size_macro_f1", BLUE),
            ("larger size\n(unseen n_in)", "test_larger_size_macro_f1", GREEN),
            ("transformer\n(control)", "test_tf_macro_f1", MUTED)]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, (label, key, color) in enumerate(sets):
        vals = [e2[n][key] for n in names]
        ax.bar(x + (i - 1) * w, vals, w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("test macro-F1 (depth roles)")
    ax.set_title("Reading transfers across input sizes (structure is "
                 "size-invariant)\nbut not across architectures (control)")
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(0.33, color=INK, lw=0.7, ls=":")
    ax.text(2.55, 0.35, "chance", fontsize=7, color=MUTED)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig5_gnn_transfer.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    fig1_pipeline()
    fig2_recovery()
    fig3_density()
    print("fig1_pipeline, fig2_recovery, fig3_density written")
    if os.path.exists(os.path.join(ROOT, "results", "gnn_experiment.json")):
        fig4_gnn_fewshot()
        fig5_gnn_transfer()
        print("fig4_gnn_fewshot, fig5_gnn_transfer written")
    print(f"-> {FIGS}")
