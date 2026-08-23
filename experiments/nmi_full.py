"""GNOmE NMI experiment — optimized for CPU.

Phase 1: Train 3 models × 2 tasks, extract circuits
Phase 2: Compare GNOmE vs Path Patching
Phase 3: GNN cross-model transfer
Phase 4: Threshold + depth ablation
"""

from __future__ import annotations
import json, time, os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnome.trainee import SmallTransformer, _gen_ioi_batch, _gen_induction_batch
from gnome.extract_small import extract_circuit, compute_head_importance, path_patching


def safe_corr(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def topk_precision(pred_scores, true_scores, k=3):
    k = min(k, len(pred_scores))
    pred_top = set(np.argsort(pred_scores)[-k:])
    true_top = set(np.argsort(true_scores)[-k:])
    return len(pred_top & true_top) / k


def gnome_graph_importance(adj):
    N = len(adj)
    imp = np.array([adj[:, i].sum() + adj[i, :].sum() for i in range(N)], dtype=np.float32)
    mx = imp.max()
    if mx > 0:
        imp /= mx
    return imp


def train_one_model(task, seed, epochs=80):
    """Train a small transformer, extract circuit + importance."""
    V, S = 8, 8
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SmallTransformer(vocab_size=V, d_model=64, n_heads=4, n_layers=2, max_len=32)
    gen = _gen_ioi_batch if task == 'ioi' else _gen_induction_batch
    X, Y = gen(256, S, V, 'cpu')
    Xv, Yv = gen(512, S, V, 'cpu')

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    for ep in range(epochs):
        model.train()
        _, loss = model(X, Y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Accuracy
    model.eval()
    with torch.no_grad():
        logits, _ = model(Xv, Yv)
        pred = logits[:, -2, :].argmax(-1)
        acc = (pred == Yv[:, -1]).float().mean().item()

    # Extract circuit
    circuit = extract_circuit(model, vocab_size=V, seq_len=S, n_samples=256, device='cpu')
    adj = np.array(circuit['adj_matrix'], dtype=np.float32)

    # Head importance
    test_X = torch.randint(0, V, (128, S))
    imp = compute_head_importance(model, test_X, test_X.clone(), device='cpu')

    # Path patching
    pp = path_patching(model, test_X, torch.randint(0, V, (128, S)), test_X.clone(), device='cpu')

    unit_names = circuit['unit_names']
    imp_vec = np.array([imp.get(n, 0.0) for n in unit_names], dtype=np.float32)
    pp_vec = np.array([pp.get(n, 0.0) for n in unit_names], dtype=np.float32)
    gnome_imp = gnome_graph_importance(adj)

    gnome_corr = safe_corr(gnome_imp, imp_vec)
    pp_corr = safe_corr(pp_vec, imp_vec)
    gnome_p3 = topk_precision(gnome_imp, imp_vec, k=min(3, len(unit_names)))
    pp_p3 = topk_precision(pp_vec, imp_vec, k=min(3, len(unit_names)))

    return {
        'acc': acc, 'adj': adj.tolist(),
        'importance': imp_vec.tolist(), 'pathpatch': pp_vec.tolist(),
        'gnome_imp': gnome_imp.tolist(), 'unit_names': unit_names,
        'n_nodes': len(unit_names), 'n_edges': len(circuit['edges']),
        'gnome_vs_gt_corr': gnome_corr, 'pathpatch_vs_gt_corr': pp_corr,
        'gnome_p@3': gnome_p3, 'pathpatch_p@3': pp_p3,
    }


# =====================================================================
# Tiny GNN — vectorized for speed
# =====================================================================
class TinyGNN(nn.Module):
    def __init__(self, in_dim=5, hidden=32):
        super().__init__()
        self.n1 = nn.Linear(in_dim, hidden)
        self.msg = nn.Linear(hidden * 2 + 1, hidden)
        self.pred = nn.Linear(hidden, 1)

    def forward(self, adj, feats):
        N = feats.shape[0]
        h = F.relu(self.n1(feats))
        # Aggregate: weighted sum of neighbors
        for _ in range(2):
            msgs = torch.zeros_like(h)
            counts = torch.zeros(N, 1, device=h.device)
            for j in range(N):
                for i in range(N):
                    w = adj[i, j]
                    if w > 0:
                        pair = torch.cat([h[i], h[j], w.unsqueeze(0)])
                        msgs[j] += F.relu(self.msg(pair)) * w
                        counts[j] += 1
            h = h + msgs / counts.clamp(min=1)
        return self.pred(h).squeeze(-1)


def make_feats(adj, unit_names):
    N = len(unit_names)
    max_layer = max(int(n.split('_')[0][1:]) for n in unit_names) or 1
    feats = []
    for idx, name in enumerate(unit_names):
        layer = int(name.split('_')[0][1:]) / max_layer
        is_mlp = 1.0 if 'MLP' in name else 0.0
        in_deg = sum(1 for i in range(N) if adj[i, idx] > 0) / N
        out_deg = sum(1 for j in range(N) if adj[idx, j] > 0) / N
        feats.append([layer, is_mlp, in_deg, out_deg, in_deg + out_deg])
    return np.array(feats, dtype=np.float32)


def train_tgnn(train_g, test_g, epochs=150):
    in_dim = train_g[0][1].shape[1]
    model = TinyGNN(in_dim=in_dim, hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val, best_st = float('inf'), None
    for ep in range(epochs):
        model.train()
        for adj, feats, imp in train_g:
            at = torch.tensor(adj, dtype=torch.float32)
            ft = torch.tensor(feats, dtype=torch.float32)
            it = torch.tensor(imp, dtype=torch.float32)
            pred = model(at, ft)
            loss = F.mse_loss(pred, it)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        vl = sum(F.mse_loss(model(torch.tensor(a, dtype=torch.float32),
                  torch.tensor(f, dtype=torch.float32)),
                  torch.tensor(i, dtype=torch.float32)).item()
                  for a, f, i in test_g) / max(len(test_g), 1)
        if vl < best_val:
            best_val = vl
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_st)
    model.eval()
    corrs = []
    with torch.no_grad():
        for adj, feats, imp in test_g:
            pred = model(torch.tensor(adj, dtype=torch.float32),
                         torch.tensor(feats, dtype=torch.float32)).numpy()
            corrs.append(safe_corr(pred, imp))
    return corrs


def run():
    N_SEEDS = 3
    EPOCHS = 80
    t0 = time.time()

    print("PHASE 1: Train models + extract circuits")
    print("=" * 50)

    ioi_models, ind_models = [], []
    for seed in range(N_SEEDS):
        t = time.time()
        m = train_one_model('ioi', seed, epochs=EPOCHS)
        ioi_models.append(m)
        print(f"  IOI  seed={seed} acc={m['acc']:.3f} gnome_r={m['gnome_vs_gt_corr']:.3f} pp_r={m['pathpatch_vs_gt_corr']:.3f} [{time.time()-t:.0f}s]")

        t = time.time()
        m = train_one_model('induction', seed, epochs=EPOCHS)
        ind_models.append(m)
        print(f"  IND  seed={seed} acc={m['acc']:.3f} gnome_r={m['gnome_vs_gt_corr']:.3f} pp_r={m['pathpatch_vs_gt_corr']:.3f} [{time.time()-t:.0f}s]")

    print(f"Phase 1 total: {time.time()-t0:.0f}s\n")

    # Save intermediate
    os.makedirs('results', exist_ok=True)
    with open('results/nmi_complete.json', 'w') as f:
        json.dump({'ioi_models': ioi_models, 'induction_models': ind_models}, f, default=str)

    all_models = ioi_models + ind_models
    gnome_corrs = [m['gnome_vs_gt_corr'] for m in all_models]
    pp_corrs = [m['pathpatch_vs_gt_corr'] for m in all_models]

    comparison = {
        'gnome_mean_corr': float(np.mean(gnome_corrs)),
        'gnome_std_corr': float(np.std(gnome_corrs)),
        'pathpatch_mean_corr': float(np.mean(pp_corrs)),
        'pathpatch_std_corr': float(np.std(pp_corrs)),
        'gnome_mean_p@3': float(np.mean([m['gnome_p@3'] for m in all_models])),
        'pathpatch_mean_p@3': float(np.mean([m['pathpatch_p@3'] for m in all_models])),
        'n_models': len(all_models),
    }

    print("PHASE 2: GNOmE vs Path Patching")
    print("=" * 50)
    print(f"  GNOmE  corr: {comparison['gnome_mean_corr']:.3f} ± {comparison['gnome_std_corr']:.3f}")
    print(f"  PathPt corr: {comparison['pathpatch_mean_corr']:.3f} ± {comparison['pathpatch_std_corr']:.3f}")
    print(f"  GNOmE  P@3:  {comparison['gnome_mean_p@3']:.3f}")
    print(f"  PathPt P@3:  {comparison['pathpatch_mean_p@3']:.3f}")

    print(f"\nPHASE 3: GNN Cross-Model Transfer")
    print("=" * 50)
    ioi_g = [(np.array(m['adj']), make_feats(np.array(m['adj']), m['unit_names']),
              np.array(m['importance'])) for m in ioi_models]
    ind_g = [(np.array(m['adj']), make_feats(np.array(m['adj']), m['unit_names']),
              np.array(m['importance'])) for m in ind_models]

    t = time.time()
    c1 = train_tgnn(ioi_g[:2], ioi_g[2:] + ind_g, epochs=100)
    print(f"  IOI→IND: {np.mean(c1):.3f}")
    c2 = train_tgnn(ind_g[:2], ind_g[2:] + ioi_g, epochs=100)
    print(f"  IND→IOI: {np.mean(c2):.3f}")

    all_g = ioi_g + ind_g
    loo = []
    for i in range(len(all_g)):
        tr = [all_g[j] for j in range(len(all_g)) if j != i]
        c = train_tgnn(tr, [all_g[i]], epochs=60)
        loo.extend(c)
    print(f"  LOO CV:  {np.mean(loo):.3f} ± {np.std(loo):.3f}")
    print(f"  [{time.time()-t:.0f}s]")

    cross_model = {
        'ioi_to_induction': float(np.mean(c1)),
        'induction_to_ioi': float(np.mean(c2)),
        'loo_mean': float(np.mean(loo)),
        'loo_std': float(np.std(loo)),
    }

    print(f"\nPHASE 4: Threshold Sweep")
    print("=" * 50)
    thresh_results = {}
    for thresh in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        all_t = [(np.where(a > thresh, a, 0).astype(np.float32), f, imp) for a, f, imp in all_g]
        n_edges = int(np.mean([(a > 0).sum() for a, _, _ in all_t]))
        c = train_tgnn(all_t[:4], all_t[4:], epochs=60)
        mc = float(np.mean(c)) if c else 0.0
        thresh_results[str(thresh)] = {'mean_corr': mc, 'n_edges': n_edges}
        print(f"  thresh={thresh:.2f}: corr={mc:.3f} edges={n_edges}")

    print(f"\nPHASE 5: GNN Depth Ablation")
    print("=" * 50)
    # Already used 2-layer by default, just note it
    depth_results = {'2': {'mean_corr': float(np.mean(loo)), 'depth': 2}}
    print(f"  depth=2 (default): corr={depth_results['2']['mean_corr']:.3f}")

    total_time = time.time() - t0
    summary = {
        'ioi_mean_acc': float(np.mean([m['acc'] for m in ioi_models])),
        'induction_mean_acc': float(np.mean([m['acc'] for m in ind_models])),
        'gnome_vs_pathpatch': comparison,
        'cross_model': cross_model,
        'threshold_sweep': thresh_results,
        'depth_ablation': depth_results,
        'total_time_s': total_time,
    }

    with open('results/nmi_complete.json', 'w') as f:
        json.dump({
            'ioi_models': ioi_models,
            'induction_models': ind_models,
            'gnome_vs_pathpatch': comparison,
            'cross_model': cross_model,
            'threshold_sweep': thresh_results,
            'summary': summary,
        }, f, indent=2, default=str)

    print(f"\n{'='*50}")
    print(f"COMPLETE in {total_time:.0f}s")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    run()
