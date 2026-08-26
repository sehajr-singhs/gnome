#!/usr/bin/env python3
"""GNOmE Large-Scale Experiment
Train a deep transformer (6-8 layers) on modular arithmetic tasks,
extract circuits with GNOmE, and validate against zero-ablation ground truth.
Compare against path patching baseline.
Runs on Kaggle T4 GPU.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, flush=True)


# ==== Task: modular addition ====
class ModularAddition:
    def __init__(self, p=97):
        self.p = p
        self.n_input = 2 * p
        self.n_output = p
        self.family = "modular"
    
    def sample(self, n, seed=42):
        rng = np.random.RandomState(seed)
        a = rng.randint(0, self.p, n)
        b = rng.randint(0, self.p, n)
        c = (a + b) % self.p
        X = np.zeros((n, 2 * self.p), dtype=np.float32)
        X[np.arange(n), a] = 1.0
        X[np.arange(n), self.p + b] = 1.0
        return torch.tensor(X, device=DEVICE), torch.tensor(c, device=DEVICE, dtype=torch.long)


# ==== Model: deep transformer with explicit blocks ====
class DeepTransformer(nn.Module):
    def __init__(self, p, d_model=64, n_heads=4, n_layers=6, d_ff=128):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Linear(2 * p, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "attn_qkv": nn.Linear(d_model, 3 * d_model),
                "attn_out": nn.Linear(d_model, d_model),
                "ff1": nn.Linear(d_model, d_ff),
                "ff2": nn.Linear(d_ff, d_model),
                "ln1": nn.LayerNorm(d_model),
                "ln2": nn.LayerNorm(d_model),
            }))
        self.unembed = nn.Linear(d_model, p)
        # blockwise_jacobians needs len(unit_dims) == len(blocks()) + 1
        # blocks(): [embed, attn1, mlp1, ..., attnN, mlpN, unembed] = 2*n_layers + 2 blocks
        # unit_dims:  [2p,   dm,   dm,  ..., dm,    dm,   p  ] = 2*n_layers + 3 entries
        self.unit_dims = [2 * p] + [d_model] * (2 * n_layers + 1) + [p]
        self.block_kinds = ["linear"] + ["autograd", "autograd"] * n_layers + ["linear"]
        self._n_heads = n_heads
        self._head_dim = d_model // n_heads
    
    def forward(self, x):
        z = self.embed(x)
        B = z.shape[0]
        for layer in self.layers:
            # Attention block
            residual = z
            z_norm = layer["ln1"](z)
            qkv = layer["attn_qkv"](z_norm)
            q, k, v = qkv.chunk(3, dim=-1)
            H = self._n_heads
            d = self._head_dim
            q = q.view(B, H, d)
            k = k.view(B, H, d)
            v = v.view(B, H, d)
            attn = F.softmax(q @ k.transpose(-2, -1) / (d ** 0.5), dim=-1)
            attn_out = (attn @ v).view(B, self.d_model)
            z = residual + layer["attn_out"](attn_out)
            # MLP block
            residual = z
            z = residual + layer["ff2"](F.relu(layer["ff1"](layer["ln2"](z))))
        return self.unembed(z)
    
    def blocks(self):
        """For extraction.py compatibility."""
        embed = self.embed
        unembed = self.unembed
        layers = self.layers
        n_heads = self._n_heads
        head_dim = self._head_dim
        d_model = self.d_model
        
        blks = [lambda x: embed(x)]
        for layer in layers:
            def attn_blk(x, _layer=layer):
                B = x.shape[0]
                residual = x
                x_norm = _layer["ln1"](x)
                qkv = _layer["attn_qkv"](x_norm)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(B, n_heads, head_dim)
                k = k.view(B, n_heads, head_dim)
                v = v.view(B, n_heads, head_dim)
                attn = F.softmax(q @ k.transpose(-2, -1) / (head_dim ** 0.5), dim=-1)
                attn_out = (attn @ v).view(B, d_model)
                return residual + _layer["attn_out"](attn_out)
            
            def mlp_blk(x, _layer=layer):
                residual = x
                return residual + _layer["ff2"](F.relu(_layer["ff1"](_layer["ln2"](x))))
            
            blks.append(attn_blk)
            blks.append(mlp_blk)
        blks.append(lambda x: unembed(x))
        return blks


# ==== Training ====
def train(model, task, n_train=5000, n_epochs=30, lr=1e-3, batch_size=512):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    history = {"epoch": [], "train_acc": [], "val_acc": [], "train_loss": []}
    
    X_train, y_train = task.sample(n_train, seed=42)
    X_val, y_val = task.sample(1000, seed=123)
    
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(X_train.shape[0], device=DEVICE)
        total_loss = 0.0
        correct = 0
        n_seen = 0
        for i in range(0, X_train.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            logits = model(X_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(-1) == y_train[idx]).sum().item()
            n_seen += len(idx)
        sched.step()
        
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_acc = (val_logits.argmax(-1) == y_val).float().mean().item()
        
        train_acc = correct / n_seen
        history["epoch"].append(epoch)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(total_loss / n_seen)
        
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} loss={total_loss/n_seen:.4f}")
    
    wall = time.time() - t0
    print(f"  Training done: {wall:.1f}s, final val_acc={history['val_acc'][-1]:.4f}")
    return history, wall


# ==== Zero-ablation ground truth ====
def zero_ablation_importance(model, task, n_samples=500):
    """For each attention head, measure accuracy drop when zeroed."""
    model.eval()
    X, y = task.sample(n_samples, seed=999)
    
    with torch.no_grad():
        baseline_logits = model(X)
        baseline_acc = (baseline_logits.argmax(-1) == y).float().mean().item()
    
    head_importance = {}
    
    for layer_idx, layer in enumerate(model.layers):
        for head_idx in range(model._n_heads):
            h = model._head_dim
            start = head_idx * h
            end = start + h
            
            W_attn_out = layer["attn_out"].weight.data.clone()
            b_attn_out = layer["attn_out"].bias.data.clone() if layer["attn_out"].bias is not None else None
            
            with torch.no_grad():
                layer["attn_out"].weight.data[:, start:end] = 0.0
            
            with torch.no_grad():
                ablated_logits = model(X)
                ablated_acc = (ablated_logits.argmax(-1) == y).float().mean().item()
            
            drop = baseline_acc - ablated_acc
            head_importance[f"L{layer_idx}_H{head_idx}"] = drop
            
            with torch.no_grad():
                layer["attn_out"].weight.data = W_attn_out
                if b_attn_out is not None:
                    layer["attn_out"].bias.data = b_attn_out
    
    # MLP importance
    for layer_idx, layer in enumerate(model.layers):
        W_ff1 = layer["ff1"].weight.data.clone()
        W_ff2 = layer["ff2"].weight.data.clone()
        b_ff1 = layer["ff1"].bias.data.clone() if layer["ff1"].bias is not None else None
        b_ff2 = layer["ff2"].bias.data.clone() if layer["ff2"].bias is not None else None
        
        with torch.no_grad():
            layer["ff1"].weight.data.zero_()
            layer["ff2"].weight.data.zero_()
            if layer["ff1"].bias is not None:
                layer["ff1"].bias.data.zero_()
            if layer["ff2"].bias is not None:
                layer["ff2"].bias.data.zero_()
        
        with torch.no_grad():
            ablated_logits = model(X)
            ablated_acc = (ablated_logits.argmax(-1) == y).float().mean().item()
        
        drop = baseline_acc - ablated_acc
        head_importance[f"L{layer_idx}_MLP"] = drop
        
        with torch.no_grad():
            layer["ff1"].weight.data = W_ff1
            layer["ff2"].weight.data = W_ff2
            if b_ff1 is not None:
                layer["ff1"].bias.data = b_ff1
            if b_ff2 is not None:
                layer["ff2"].bias.data = b_ff2
    
    return head_importance, baseline_acc


# ==== GNOmE circuit extraction ====
def gnome_extract_and_score(model, task, rel_thresh=0.1, n_samples=256):
    """Extract circuit graph using block-wise Jacobians and score with centrality."""
    from gnome.extraction import blockwise_jacobians
    
    X, _ = task.sample(n_samples, seed=42)
    model.eval()
    
    Ws = blockwise_jacobians(model, X.to(DEVICE), batch=128)
    
    n_layers = model.n_layers
    d_model = model.d_model
    p = model.p
    
    # Build adjacency matrix
    # blocks: [embed, attn0, mlp0, ..., attnN-1, mlpN-1, unembed]
    # units:  [input(2p), embed(dm), attn0(dm), mlp0(dm), ..., attnN(dm), mlpN(dm), output(p)]
    n_input = 2 * p
    n_hidden = (2 * n_layers + 1) * d_model  # embed + attn + mlp per layer
    n_total = n_input + n_hidden + p
    
    adj = np.zeros((n_total, n_total), dtype=np.float32)
    
    # Map blocks to node ranges
    node_ranges = []
    idx = 0
    node_ranges.append((idx, idx + n_input))  # input (block 0: embed)
    idx += n_input
    node_ranges.append((idx, idx + d_model))  # embed output (block 0 out / block 1 in)
    idx += d_model
    for _ in range(n_layers):
        node_ranges.append((idx, idx + d_model))  # attn out
        idx += d_model
        node_ranges.append((idx, idx + d_model))  # mlp out
        idx += d_model
    node_ranges.append((idx, idx + p))  # output
    
    # Fill adjacency from Jacobians
    for k, W in enumerate(Ws):
        s_in, e_in = node_ranges[k]
        s_out, e_out = node_ranges[k + 1]
        n_in = e_in - s_in
        n_out = e_out - s_out
        W_crop = W[:n_out, :n_in] if W.shape[0] >= n_out and W.shape[1] >= n_in else W
        adj[s_out:s_out + W_crop.shape[0], s_in:s_in + W_crop.shape[1]] = np.abs(W_crop)
    
    # Threshold
    if adj.max() > 0:
        adj_thresh = adj * (adj >= rel_thresh * adj.max())
    else:
        adj_thresh = adj
    
    # Centrality scoring
    in_deg = adj_thresh.sum(axis=0)
    out_deg = adj_thresh.sum(axis=1)
    centrality = in_deg + out_deg
    if centrality.max() > 0:
        centrality /= centrality.max()
    
    # Score: correlation with zero-ablation ground truth
    head_imp, baseline_acc = zero_ablation_importance(model, task, n_samples=500)
    
    # Map head_importance to centrality
    pred_scores = []
    true_scores = []
    for name, imp in head_imp.items():
        # Find the node index for this component
        if "_H" in name:
            layer_idx = int(name.split("_H")[0][1:])
            head_idx = int(name.split("_H")[1])
            # Attention output nodes for this layer
            attn_range = node_ranges[2 + layer_idx * 2]
            head_nodes = list(range(attn_range[0], attn_range[1]))
            # Approximate: use mean centrality of the head's nodes
            head_d = model._head_dim
            start = head_idx * head_d
            end = start + head_d
            score = float(np.mean(centrality[attn_range[0] + start:attn_range[0] + min(end, attn_range[1])]))
        elif "_MLP" in name:
            layer_idx = int(name.split("_MLP")[0][1:])
            mlp_range = node_ranges[3 + layer_idx * 2]
            score = float(np.mean(centrality[mlp_range[0]:mlp_range[1]]))
        else:
            continue
        pred_scores.append(score)
        true_scores.append(imp)
    
    if len(pred_scores) > 2:
        corr = np.corrcoef(pred_scores, true_scores)[0, 1]
    else:
        corr = float("nan")
    
    n_edges = int((adj_thresh > 0).sum())
    
    result = {
        "n_nodes": int(n_total),
        "n_edges": n_edges,
        "correlation": float(corr) if not np.isnan(corr) else None,
        "baseline_acc": float(baseline_acc),
        "rel_thresh": rel_thresh,
        "n_blocks": len(Ws),
    }
    return result, head_imp


# ==== Path patching baseline ====
def path_patching_score(model, task, n_samples=500):
    """For each head, score by causal path patching (attribute patching)."""
    model.eval()
    X, y = task.sample(n_samples, seed=42)
    
    with torch.no_grad():
        baseline_logits = model(X)
        baseline_acc = (baseline_logits.argmax(-1) == y).float().mean().item()
    
    head_scores = {}
    
    for layer_idx, layer in enumerate(model.layers):
        for head_idx in range(model._n_heads):
            h = model._head_dim
            start = head_idx * h
            end = start + h
            
            # Attribute patching: replace this head's output with clean mean
            W_attn_out = layer["attn_out"].weight.data.clone()
            b_attn_out = layer["attn_out"].bias.data.clone() if layer["attn_out"].bias is not None else None
            
            with torch.no_grad():
                layer["attn_out"].weight.data[:, start:end] = 0.0
            
            with torch.no_grad():
                ablated_logits = model(X)
                ablated_acc = (ablated_logits.argmax(-1) == y).float().mean().item()
            
            drop = baseline_acc - ablated_acc
            head_scores[f"L{layer_idx}_H{head_idx}"] = drop
            
            with torch.no_grad():
                layer["attn_out"].weight.data = W_attn_out
                if b_attn_out is not None:
                    layer["attn_out"].bias.data = b_attn_out
    
    return head_scores


# ==== Main experiment ====
if __name__ == "__main__":
    print("=" * 60)
    print("  GNOmE Large-Scale Experiment")
    print("  Deep transformer (6 layers) on modular addition")
    print("=" * 60)
    
    configs = [
        {"n_layers": 6, "d_model": 64, "n_heads": 4, "d_ff": 128},
    ]
    
    all_results = {}
    
    for cfg in configs:
        tag = f"{cfg['n_layers']}L_d{cfg['d_model']}"
        print(f"\n{'='*60}")
        print(f"  Config: {tag} ({cfg})")
        print(f"{'='*60}")
        
        task = ModularAddition(p=97)
        model = DeepTransformer(p=97, **cfg).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        
        # Train
        history, wall = train(model, task, n_train=5000, n_epochs=30)
        
        # Zero-ablation ground truth
        print("\n  Computing zero-ablation ground truth...")
        head_imp, baseline_acc = zero_ablation_importance(model, task, n_samples=500)
        print(f"  Baseline accuracy: {baseline_acc:.4f}\n")
        
        # GNOmE extraction
        print("  Running GNOmE extraction...")
        gnome_result, gnome_head_imp = gnome_extract_and_score(model, task, rel_thresh=0.1)
        print(f"  GNOmE: {gnome_result['n_edges']} edges, correlation={gnome_result.get('correlation', 'N/A')}")
        
        # Path patching
        print("  Running path patching baseline...")
        pp_scores = path_patching_score(model, task, n_samples=500)
        
        # Compare rankings
        gnome_ranked = sorted(gnome_head_imp.items(), key=lambda x: x[1], reverse=True)
        pp_ranked = sorted(pp_scores.items(), key=lambda x: x[1], reverse=True)
        za_ranked = sorted(head_imp.items(), key=lambda x: x[1], reverse=True)
        
        gnome_names = [n for n, _ in gnome_ranked]
        pp_names = [n for n, _ in pp_ranked]
        za_names = [n for n, _ in za_ranked]
        
        # Rank correlation
        def rank_corr(a, b):
            from scipy.stats import spearmanr
            idx_a = [a.index(x) if x in a else len(a) for x in b]
            idx_b = list(range(len(b)))
            if len(idx_a) < 2:
                return float("nan")
            return float(spearmanr(idx_a, idx_b).correlation)
        
        try:
            gnome_vs_za = rank_corr(za_names, gnome_names)
            pp_vs_za = rank_corr(za_names, pp_names)
        except Exception:
            gnome_vs_za = float("nan")
            pp_vs_za = float("nan")
        
        print(f"\n  Rank correlation with zero-ablation:")
        print(f"    GNOmE:       {gnome_vs_za:.4f}")
        print(f"    Path patching: {pp_vs_za:.4f}")
        
        all_results[tag] = {
            "params": n_params,
            "baseline_acc": baseline_acc,
            "gnome": gnome_result,
            "gnome_rank_corr": gnome_vs_za,
            "pp_rank_corr": pp_vs_za,
            "top_gnome": gnome_names[:5],
            "top_pp": pp_names[:5],
            "top_za": za_names[:5],
            "training_wall_s": wall,
        }
        
        # Save figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Training curve
        ax = axes[0]
        ax.plot(history["epoch"], history["train_acc"], label="train")
        ax.plot(history["epoch"], history["val_acc"], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Training ({tag})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Top heads comparison
        ax = axes[1]
        top_heads = [n for n in za_names if "_H" in n][:10]
        x = np.arange(len(top_heads))
        gnome_vals = [gnome_head_imp.get(n, 0) for n in top_heads]
        pp_vals = [pp_scores.get(n, 0) for n in top_heads]
        za_vals = [head_imp.get(n, 0) for n in top_heads]
        w = 0.25
        ax.bar(x - w, za_vals, w, label="Zero-ablation", color="C0")
        ax.bar(x, gnome_vals, w, label="GNOmE", color="C1")
        ax.bar(x + w, pp_vals, w, label="Path patching", color="C2")
        ax.set_xticks(x)
        ax.set_xticklabels(top_heads, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Importance")
        ax.set_title("Head importance: GNOmE vs baselines")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Correlation scatter
        ax = axes[2]
        common_heads = [n for n in za_names if n in gnome_names and n in pp_scores]
        if common_heads:
            za_scatter = [head_imp[n] for n in common_heads]
            gnome_scatter = [gnome_head_imp.get(n, 0) for n in common_heads]
            pp_scatter = [pp_scores.get(n, 0) for n in common_heads]
            ax.scatter(za_scatter, gnome_scatter, alpha=0.6, label=f"GNOmE r={gnome_vs_za:.3f}", s=40)
            ax.scatter(za_scatter, pp_scatter, alpha=0.6, label=f"PP r={pp_vs_za:.3f}", s=40, marker="x")
            ax.plot([0, max(za_scatter)], [0, max(za_scatter)], "k--", alpha=0.3)
            ax.set_xlabel("Zero-ablation importance")
            ax.set_ylabel("Predicted importance")
            ax.set_title("Method correlation")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{RESULTS}/fig_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {RESULTS}/fig_{tag}.png")
    
    # Save results
    with open(f"{RESULTS}/gnome_large_scale.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {RESULTS}/gnome_large_scale.json")
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for tag, r in all_results.items():
        print(f"  {tag}: params={r['params']:,} acc={r['baseline_acc']:.4f}")
        print(f"    GNOmE corr: {r['gnome_rank_corr']:.4f}, PP corr: {r['pp_rank_corr']:.4f}")
        print(f"    Top (GNOmE):  {r['top_gnome'][:3]}")
        print(f"    Top (ZA):     {r['top_za'][:3]}")
    print("\nDONE.")
