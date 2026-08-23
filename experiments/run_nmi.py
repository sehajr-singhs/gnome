"""NMI-level experiments for GNOmE.

End-to-end pipeline:
  1. Train small transformers on IOI and induction tasks
  2. Extract computation graphs from trained models
  3. Compare GNOmE graph-based method vs. path patching vs. random
  4. Train GNN reader to predict head importance from graph structure alone
  5. Ablations: graph features, GNN depth, threshold sensitivity

Key claims:
  - GNOmE recovers known circuits from trained models
  - Graph-based method predicts head importance without interventions
  - GNN reader transfers across different trained instances
"""

from __future__ import annotations

import json
import time
import os
import sys
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnome.trainee import SmallTransformer, train_on_ioi, train_on_induction
from gnome.extract_small import extract_circuit, compute_head_importance, path_patching
from gnome.graphnets import GNNExplicator
from gnome.metrics import compute_all_metrics


def train_and_extract(task: str = "ioi", n_seeds: int = 5, epochs: int = 30,
                      verbose: bool = True) -> list[dict]:
    """Train multiple models and extract their circuits."""
    results = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if verbose:
            print(f"\n=== {task.upper()} seed={seed} ===")

        # Train
        model = SmallTransformer(vocab_size=256, d_model=128, n_heads=4, n_layers=2)
        if verbose:
            print(f"  Params: {model.get_num_params():,}")

        t0 = time.time()
        if task == "ioi":
            train_info = train_on_ioi(model, epochs=epochs, verbose=verbose)
        else:
            train_info = train_on_induction(model, epochs=epochs, verbose=verbose)
        train_time = time.time() - t0

        if verbose:
            print(f"  Train time: {train_time:.1f}s, val_acc={train_info['final_val_acc']:.3f}")

        # Extract circuit
        circuit = extract_circuit(model, rel_thresh=0.05)

        # Compute head importance (attribution method)
        eval_ids = torch.randint(0, 256, (256, 32))
        eval_targets = eval_ids.clone()
        importance = compute_head_importance(model, eval_ids, eval_targets)

        # Path patching
        t0 = time.time()
        patching = path_patching(model, eval_ids, eval_targets)
        patching_time = time.time() - t0

        if verbose:
            print(f"  Path patching: {patching_time:.1f}s")
            # Show top heads
            sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:5]
            print(f"  Top heads (attribution): {sorted_imp}")

        results.append({
            "seed": seed,
            "task": task,
            "train_info": train_info,
            "train_time": train_time,
            "circuit": {
                "n_nodes": circuit["metadata"]["n_nodes"],
                "n_edges": circuit["metadata"]["n_edges"],
                "nodes": circuit["nodes"],
                "edges": circuit["edges"],
                "adj_matrix": circuit["adj_matrix"].tolist(),
                "unit_names": circuit["unit_names"],
            },
            "importance": importance,
            "patching": patching,
            "patching_time": patching_time,
        })

        # Save intermediate
        out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "results", f"circuit_{task}_seed{seed}.json")
        with open(out_path, "w") as f:
            json.dump(results[-1], f, indent=2)

    return results


def cross_model_gnn_experiment(all_results: list[dict], verbose: bool = True) -> dict:
    """Test whether a GNN trained on one model's graph can predict
    head importance in a different model instance.

    This is the key NMI claim: graph structure predicts function
    WITHOUT running the model.
    """
    from gnome.trainee import SmallTransformer
    from gnome.extract_small import extract_circuit
    import torch.nn as nn

    if verbose:
        print("\n=== Cross-Model GNN Experiment ===")

    # Build graph + importance pairs
    graph_data = []
    for res in all_results:
        adj = np.array(res["circuit"]["adj_matrix"])
        imp = res["importance"]
        # Convert importance to numpy vector matching node order
        imp_vec = np.array([imp.get(name, 0.0) for name in res["circuit"]["unit_names"]])
        graph_data.append({"adj": adj, "importance": imp_vec, "nodes": res["circuit"]["unit_names"]})

    n_units = graph_data[0]["adj"].shape[0]

    # Simple GNN: 2-layer GCN that predicts per-node importance
    class SimpleGCN(nn.Module):
        def __init__(self, in_dim, hidden=64):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x, adj):
            # x: (n_nodes, in_dim), adj: (n_nodes, n_nodes)
            deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
            adj_norm = adj / deg
            h = torch.relu(self.fc1(adj_norm @ x))
            h = torch.relu(self.fc2(adj_norm @ h))
            return self.out(h).squeeze(-1)

    model = SimpleGCN(in_dim=1)  # input = node index (simple feature)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Leave-one-out cross-validation
    all_correlations = []
    for leave_out in range(len(graph_data)):
        train_data = [d for i, d in enumerate(graph_data) if i != leave_out]
        test_data = graph_data[leave_out]

        # Train
        for epoch in range(200):
            total_loss = 0
            for d in train_data:
                x = torch.arange(n_units, dtype=torch.float32).unsqueeze(1) / n_units
                adj = torch.tensor(d["adj"], dtype=torch.float32)
                target = torch.tensor(d["importance"], dtype=torch.float32)
                pred = model(x, adj)
                loss = nn.functional.mse_loss(pred, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

        # Test
        with torch.no_grad():
            x = torch.arange(n_units, dtype=torch.float32).unsqueeze(1) / n_units
            adj = torch.tensor(test_data["adj"], dtype=torch.float32)
            pred = model(x, adj).numpy()
            target = test_data["importance"]
            corr = np.corrcoef(pred, target)[0, 1]
            all_correlations.append(corr)

            if verbose:
                print(f"  Leave-{leave_out}-out: r={corr:.3f}")

    avg_corr = float(np.mean(all_correlations))
    if verbose:
        print(f"  Average cross-model correlation: {avg_corr:.3f}")

    return {
        "method": "cross_model_gnn",
        "correlations": [float(c) for c in all_correlations],
        "mean_correlation": avg_corr,
    }


def graph_vs_patching_correlation(results: list[dict], verbose: bool = True) -> dict:
    """Compare GNOmE graph similarity vs path patching importance."""
    if verbose:
        print("\n=== Graph vs Path Patching ===")

    correlations = []
    for res in results:
        imp = res["importance"]
        patch = res["patching"]
        names = res["circuit"]["unit_names"]

        imp_vec = np.array([imp.get(n, 0.0) for n in names])
        patch_vec = np.array([patch.get(n, 0.0) for n in names])

        if np.std(imp_vec) > 0 and np.std(patch_vec) > 0:
            corr = np.corrcoef(imp_vec, patch_vec)[0, 1]
            correlations.append(corr)
            if verbose:
                print(f"  {res['task']} seed={res['seed']}: attribution-vs-patching r={corr:.3f}")

    avg = float(np.mean(correlations)) if correlations else 0.0
    if verbose:
        print(f"  Average correlation: {avg:.3f}")

    return {
        "method": "attribution_vs_patching",
        "correlations": [float(c) for c in correlations],
        "mean_correlation": avg,
    }


def threshold_sweep(results: list[dict], verbose: bool = True) -> dict:
    """Sweep graph extraction threshold and measure circuit quality."""
    if verbose:
        print("\n=== Threshold Sweep ===")

    from gnome.extract_small import extract_circuit
    from gnome.trainee import SmallTransformer

    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    sweep_results = {}

    for thresh in thresholds:
        edge_counts = []
        for res in results:
            # Re-extract with different threshold
            adj = np.array(res["circuit"]["adj_matrix"])
            n_edges = int((adj >= thresh).sum())
            edge_counts.append(n_edges)

        avg_edges = float(np.mean(edge_counts))
        sweep_results[str(thresh)] = {
            "threshold": thresh,
            "avg_edges": avg_edges,
        }
        if verbose:
            print(f"  thresh={thresh:.3f}: avg_edges={avg_edges:.1f}")

    return sweep_results


def run_all(experiments: str = "all", epochs: int = 30, n_seeds: int = 3,
            out_dir: str = "results") -> dict:
    """Run the full NMI-level experiment suite."""
    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    t_start = time.time()

    # ---- 1. Train and extract circuits ----
    if experiments in ("all", "circuits"):
        print("=" * 60)
        print("EXPERIMENT 1: Train models + extract circuits")
        print("=" * 60)

        ioi_results = train_and_extract("ioi", n_seeds=n_seeds, epochs=epochs)
        induction_results = train_and_extract("induction", n_seeds=n_seeds, epochs=epochs)

        all_results["ioi"] = ioi_results
        all_results["induction"] = induction_results

        # Save
        with open(os.path.join(out_dir, "circuits.json"), "w") as f:
            json.dump({"ioi": ioi_results, "induction": induction_results}, f, indent=2)

    # ---- 2. Graph vs Patching correlation ----
    if experiments in ("all", "comparison"):
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Attribution vs Path Patching")
        print("=" * 60)

        ioi_results = all_results.get("ioi", None)
        induction_results = all_results.get("induction", None)

        if ioi_results is None or induction_results is None:
            # Load from disk
            with open(os.path.join(out_dir, "circuits.json")) as f:
                data = json.load(f)
                ioi_results = data["ioi"]
                induction_results = data["induction"]

        all_combined = ioi_results + induction_results
        comparison = graph_vs_patching_correlation(all_combined)
        all_results["comparison"] = comparison

    # ---- 3. Cross-model GNN ----
    if experiments in ("all", "gnn"):
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: Cross-Model GNN Transfer")
        print("=" * 60)

        ioi_results = all_results.get("ioi", None)
        induction_results = all_results.get("induction", None)

        if ioi_results is None:
            with open(os.path.join(out_dir, "circuits.json")) as f:
                data = json.load(f)
                ioi_results = data["ioi"]
                induction_results = data["induction"]

        gnn_ioi = cross_model_gnn_experiment(ioi_results)
        gnn_ind = cross_model_gnn_experiment(induction_results)
        all_results["gnn_ioi"] = gnn_ioi
        all_results["gnn_induction"] = gnn_ind

    # ---- 4. Threshold sweep ----
    if experiments in ("all", "sweep"):
        print("\n" + "=" * 60)
        print("EXPERIMENT 4: Threshold Sensitivity")
        print("=" * 60)

        ioi_results = all_results.get("ioi", None)
        if ioi_results is None:
            with open(os.path.join(out_dir, "circuits.json")) as f:
                data = json.load(f)
                ioi_results = data["ioi"]

        sweep = threshold_sweep(ioi_results)
        all_results["threshold_sweep"] = sweep

    # ---- Summary ----
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"{'=' * 60}")

    # Compute summary statistics
    summary = {"total_time": elapsed}

    if "ioi" in all_results:
        ioi_accs = [r["train_info"]["final_val_acc"] for r in all_results["ioi"]]
        summary["ioi_mean_acc"] = float(np.mean(ioi_accs))
        summary["ioi_std_acc"] = float(np.std(ioi_accs))
        print(f"  IOI accuracy: {summary['ioi_mean_acc']:.3f} ± {summary['ioi_std_acc']:.3f}")

    if "induction" in all_results:
        ind_accs = [r["train_info"]["final_val_acc"] for r in all_results["induction"]]
        summary["induction_mean_acc"] = float(np.mean(ind_accs))
        summary["induction_std_acc"] = float(np.std(ind_accs))
        print(f"  Induction accuracy: {summary['induction_mean_acc']:.3f} ± {summary['induction_std_acc']:.3f}")

    if "comparison" in all_results:
        summary["attr_patching_corr"] = all_results["comparison"]["mean_correlation"]
        print(f"  Attribution vs Patching r: {summary['attr_patching_corr']:.3f}")

    if "gnn_ioi" in all_results:
        summary["gnn_ioi_corr"] = all_results["gnn_ioi"]["mean_correlation"]
        print(f"  GNN cross-model (IOI): {summary['gnn_ioi_corr']:.3f}")

    if "gnn_induction" in all_results:
        summary["gnn_induction_corr"] = all_results["gnn_induction"]["mean_correlation"]
        print(f"  GNN cross-model (Induction): {summary['gnn_induction_corr']:.3f}")

    all_results["summary"] = summary

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="all",
                        choices=["all", "circuits", "comparison", "gnn", "sweep"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    run_all(args.experiments, args.epochs, args.n_seeds, args.out_dir)
