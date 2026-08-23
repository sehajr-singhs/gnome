#!/usr/bin/env python
"""Run remaining GNOmE NMI experiments: GNN cross-model + threshold sweep.
Uses smaller models to fit within time constraints."""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnome.trainee import SmallTransformer, train_on_induction
from gnome.extract_small import extract_circuit, compute_head_importance


class GCN(nn.Module):
    def __init__(self, in_dim=1, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, adj):
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_n = adj / deg
        h = torch.relu(self.fc1(adj_n @ x))
        h = torch.relu(self.fc2(adj_n @ h))
        return self.out(h).squeeze(-1)


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    # ====== Experiment 1: Train 3 models + extract circuits ======
    print("=" * 50)
    print("EXPERIMENT 1: Train & Extract")
    print("=" * 50)
    t_start = time.time()

    models_data = []
    for seed in range(3):
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = SmallTransformer(vocab_size=64, d_model=128, n_heads=4, n_layers=2)
        r = train_on_induction(m, vocab_size=64, seq_len=12,
                               n_train=4000, n_val=1000, epochs=25, verbose=False)

        eval_ids = torch.randint(0, 64, (32, 12))
        eval_tgt = eval_ids.clone()
        circuit = extract_circuit(m, eval_ids, vocab_size=64, seq_len=12, n_samples=32, rel_thresh=0.05)
        imp = compute_head_importance(m, eval_ids, eval_tgt)

        models_data.append({
            "seed": seed,
            "acc": r["final_val_acc"],
            "adj": circuit["adj_matrix"].tolist(),
            "importance": imp,
            "unit_names": circuit["unit_names"],
        })
        print(f"  seed={seed}: acc={r['final_val_acc']:.3f} ({time.time()-t_start:.0f}s)")

    all_results["models"] = models_data
    all_results["train_time"] = time.time() - t_start

    # ====== Experiment 2: GNN cross-model ======
    print("\n" + "=" * 50)
    print("EXPERIMENT 2: GNN Cross-Model")
    print("=" * 50)

    n_units = len(models_data[0]["unit_names"])
    correlations = []

    for lo in range(3):
        gnn = GCN()
        opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)
        train_d = [d for i, d in enumerate(models_data) if i != lo]
        test_d = models_data[lo]

        for ep in range(500):
            for d in train_d:
                x = torch.arange(n_units, dtype=torch.float32).unsqueeze(1) / n_units
                adj = torch.tensor(d["adj"], dtype=torch.float32)
                tgt = torch.tensor([d["importance"].get(n, 0) for n in d["unit_names"]],
                                   dtype=torch.float32)
                pred = gnn(x, adj)
                loss = nn.functional.mse_loss(pred, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()

        with torch.no_grad():
            x = torch.arange(n_units, dtype=torch.float32).unsqueeze(1) / n_units
            adj = torch.tensor(test_d["adj"], dtype=torch.float32)
            pred = gnn(x, adj).numpy()
            tgt = np.array([test_d["importance"].get(n, 0) for n in test_d["unit_names"]])
            corr = float(np.corrcoef(pred, tgt)[0, 1])
            correlations.append(corr)
            print(f"  LOO-{lo}: r={corr:.3f}")

    mean_corr = float(np.mean(correlations))
    all_results["gnn_cross_model"] = {"correlations": correlations, "mean": mean_corr}
    print(f"  Mean GNN r: {mean_corr:.3f}")

    # ====== Experiment 3: Threshold sweep ======
    print("\n" + "=" * 50)
    print("EXPERIMENT 3: Threshold Sweep")
    print("=" * 50)

    # Use already-trained seed=0 model
    torch.manual_seed(0)
    np.random.seed(0)
    m_sweep = SmallTransformer(vocab_size=64, d_model=128, n_heads=4, n_layers=2)
    train_on_induction(m_sweep, vocab_size=64, seq_len=12,
                       n_train=4000, n_val=1000, epochs=25, verbose=False)

    sweep = {}
    for th in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        eval_ids = torch.randint(0, 64, (32, 12))
        c = extract_circuit(m_sweep, eval_ids, vocab_size=64, seq_len=12,
                            n_samples=32, rel_thresh=th)
        sweep[str(th)] = c["metadata"]["n_edges"]
        print(f"  thresh={th}: edges={c['metadata']['n_edges']}")

    all_results["threshold_sweep"] = sweep

    # ====== Save ======
    with open(os.path.join(out_dir, "nmi_full.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"ALL DONE in {elapsed:.0f}s")
    print(f"  GNN mean r: {mean_corr:.3f}")
    print(f"  Results saved to results/nmi_full.json")


if __name__ == "__main__":
    main()
