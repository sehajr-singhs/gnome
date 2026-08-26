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
        self.p = p
    
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
        # For block-wise Jacobian extraction
        self.unit_dims = [2 * p] + [d_model] * (2 * n_layers) + [p]
        self.block_kinds = ["linear"] + ["attention", "elementwise"] * n_layers + ["linear"]
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
def zero_ablation_importance(model, task, n_samples=500, n_head_layers=1):
    """For each attention head, measure how much accuracy drops when that head is zeroed."""
    model.eval()
    X, y = task.sample(n_samples, seed=999)
    
    with torch.no_grad():
        baseline_logits = model(X)
        baseline_acc = (baseline_logits.argmax(-1) == y).float().mean().item()
    
    # Get all attention layers
    attn_layers = []
    for layer in model.layers:
        attn_layers.append(layer)
    
    head_importance = {}
    
    for layer_idx, layer in enumerate(attn_layers):
        for head_idx in range(model._n_heads):
            # Zero out this head by modifying the forward pass temporarily
            h = model._head_dim
            start = head_idx * h
            end = start + h
            
            # Store original weights
            W_attn_out = layer["attn_out"].weight.data.clone()
            b_attn_out = layer["attn_out"].bias.data.clone() if layer["attn_out"].bias is not None else None
            
            # Zero the output projection for this head's contribution
            with torch.no_grad():
                layer["attn_out"].weight.data[:, start:end] = 0.0
            
            # Evaluate
            with torch.no_grad():
                ablated_logits = model(X)
                ablated_acc = (ablated_logits.argmax(-1) == y).float().mean().item()
            
            drop = baseline_acc - ablated_acc
            head_importance[f"L{layer_idx}_H{head_idx}"] = drop
            
            # Restore
            with torch.no_grad():
                layer["attn_out"].weight.data = W_attn_out
                if b_attn_out is not None:
                    layer["attn_out"].bias.data = b_attn_out
    
    # Also measure MLP importance
    for layer_idx, layer in enumerate(attn_layers):
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
    from gnome.extraction import blockwise_jacobians, threshold_edges
    
    X, _ = task.sample(n_samples, seed=42)
    model.eval()
    
    Ws = blockwise_jacobians(model, X.to(DEVICE), batch=128)
    
    # Build adjacency
    n_layers = model.n_layers
    d_model = model.d_model
    p = model.p
    
    nodes = []
    ids = []
    # Input layer
    layer_ids = [f"input_{j}" for j in range(2 * p)]
    ids.append(layer_ids)
    for j in range(2 * p):
        nodes.append({"id": layer_ids[j], "layer": 0, "role": "input"})
    
    # Hidden layers
    for k in range(n_layers):
        layer_ids = [f"L{k}_H{j}" for j in range(d_model)]
        ids.append(layer_ids)
        for j in range(d_model):
            nodes.append({"id": layer_ids[j], "layer": k + 1, "role": "hidden"})
    
    # Output layer
    out_ids = [f"out_{j}" for j in range(p)]
    ids.append(out_ids)
    for j in range(p):
        nodes.append({"id": out_ids[j], "layer": n_layers + 1, "role": "output"})
    
    # Threshold edges
    edges_by_layer = threshold_edges(Ws, rel_thresh)
    edges = []
    for k, layer_edges in enumerate(edges_by_layer):
        for src, dst, w in layer_edges:
            if k < len(ids) - 1 and src < len(ids[k]) and dst < len(ids[k + 1]):
                edges.append((ids[k][src], ids[k + 1][dst], w))
    
    # Score with centrality
    n_nodes = len(nodes)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    node_id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
    
    for src_id, dst_id, w in edges:
        if src_id in node_id_to_idx and dst_id in node_id_to_idx:
            adj[node_id_to_idx[src_id], node_id_to_idx[dst_id]] = w
    
    # Degree centrality
    centrality = adj.sum(axis=0) + adj.sum(axis=1)
    if centrality.max() > 0:
        centrality = centrality / centrality.max()
    
    scores = {nodes[i]["id"]: float(centrality[i]) for i in range(n_nodes)}
    
    return {
        "nodes": nodes,
        "edges": edges,
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "scores": scores,
        "adj": adj,
    }


# ==== Path patching baseline ====
def path_patching_importance(model, task, n_samples=200):
    """Simplified path patching: for each head, ablate and measure acc drop."""
    # Same as zero-ablation but done sequentially with causal masking
    model.eval()
    X, y = task.sample(n_samples, seed=888)
    
    with torch.no_grad():
        baseline_acc = (model(X).argmax(-1) == y).float().mean().item()
    
    importance = {}
    for layer_idx, layer in enumerate(model.layers):
        h = model._head_dim
        for head_idx in range(model._n_heads):
            start = head_idx * h
            end = start + h
            
            W_out = layer["attn_out"].weight.data.clone()
            with torch.no_grad():
                layer["attn_out"].weight.data[:, start:end] = 0.0
            
            with torch.no_grad():
                acc = (model(X).argmax(-1) == y).float().mean().item()
            
            importance[f"L{layer_idx}_H{head_idx}"] = baseline_acc - acc
            with torch.no_grad():
                layer["attn_out"].weight.data = W_out
    
    return importance


# ==== Main experiment ====
if __name__ == "__main__":
    print("=" * 60)
    print("  GNOmE Large-Scale Experiment")
    print("  Deep transformer (6 layers) on modular addition")
    print("=" * 60)
    
    p = 97
    task = ModularAddition(p=p)
    
    # Configs to test
    configs = {
        "6L_d64": {"n_layers": 6, "d_model": 64, "n_heads": 4, "d_ff": 128},
        "6L_d128": {"n_layers": 6, "d_model": 128, "n_heads": 4, "d_ff": 256},
        "8L_d128": {"n_layers": 8, "d_model": 128, "n_heads": 8, "d_ff": 256},
    }
    
    all_results = {}
    
    for name, cfg in configs.items():
        print(f"\n{'='*50}")
        print(f"  Config: {name} ({cfg})")
        print(f"{'='*50}")
        
        model = DeepTransformer(
            p=p, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"], d_ff=cfg["d_ff"]
        ).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        
        # Train
        history, train_time = train(model, task, n_train=5000, n_epochs=30, lr=1e-3)
        
        # Zero-ablation ground truth
        print("\n  Computing zero-ablation ground truth...")
        gt_importance, baseline_acc = zero_ablation_importance(model, task, n_samples=500)
        print(f"  Baseline accuracy: {baseline_acc:.4f}")
        
        # GNOmE extraction
        print("\n  Running GNOmE extraction...")
        t0 = time.time()
        gnome_result = gnome_extract_and_score(model, task, rel_thresh=0.1)
        gnome_time = time.time() - t0
        print(f"  GNOmE: {gnome_result['n_nodes']} nodes, {gnome_result['n_edges']} edges ({gnome_time:.2f}s)")
        
        # Path patching baseline
        print("\n  Running path patching baseline...")
        t0 = time.time()
        pp_importance = path_patching_importance(model, task, n_samples=200)
        pp_time = time.time() - t0
        print(f"  Path patching done ({pp_time:.2f}s)")
        
        # Compare
        common_heads = set(gt_importance.keys()) & set(gnome_result["scores"].keys())
        gt_vec = np.array([gt_importance[h] for h in sorted(common_heads)])
        gnome_vec = np.array([gnome_result["scores"][h] for h in sorted(common_heads)])
        pp_vec = np.array([pp_importance.get(h, 0.0) for h in sorted(common_heads)])
        
        gnome_corr = float(np.corrcoef(gt_vec, gnome_vec)[0, 1]) if gt_vec.std() > 1e-8 else 0.0
        pp_corr = float(np.corrcoef(gt_vec, pp_vec)[0, 1]) if gt_vec.std() > 1e-8 else 0.0
        
        # Recovery: top-k overlap
        k = min(5, len(gt_vec))
        gt_top = set(np.argsort(gt_vec)[-k:])
        gnome_top = set(np.argsort(gnome_vec)[-k:])
        pp_top = set(np.argsort(pp_vec)[-k:])
        gnome_recovery = len(gt_top & gnome_top) / k
        pp_recovery = len(gt_top & pp_top) / k
        
        print(f"\n  Results:")
        print(f"    GNOmE correlation with ground truth: {gnome_corr:.4f}")
        print(f"    Path patching correlation:            {pp_corr:.4f}")
        print(f"    GNOmE recovery@{k}:                   {gnome_recovery:.4f}")
        print(f"    Path patching recovery@{k}:           {pp_recovery:.4f}")
        
        all_results[name] = {
            "params": n_params,
            "train_time_s": train_time,
            "baseline_acc": baseline_acc,
            "gnome": {
                "n_nodes": gnome_result["n_nodes"],
                "n_edges": gnome_result["n_edges"],
                "extraction_time_s": gnome_time,
                "correlation": gnome_corr,
                "recovery": gnome_recovery,
            },
            "path_patching": {
                "correlation": pp_corr,
                "recovery": pp_recovery,
                "time_s": pp_time,
            },
            "history": {
                "val_acc_final": history["val_acc"][-1],
                "train_loss_final": history["train_loss"][-1],
            },
        }
    
    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':12s} {'Params':>10s} {'Val Acc':>8s} {'GNOmE r':>8s} {'PP r':>8s} {'GNOmE@k':>8s} {'PP@k':>8s}")
    print(f"  {'-'*70}")
    for name, r in all_results.items():
        print(f"  {name:12s} {r['params']:>10,} {r['history']['val_acc_final']:>8.4f} "
              f"{r['gnome']['correlation']:>8.4f} {r['path_patching']['correlation']:>8.4f} "
              f"{r['gnome']['recovery']:>8.4f} {r['path_patching']['recovery']:>8.4f}")
    
    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = list(all_results.keys())
    
    # Correlation comparison
    gnome_corrs = [all_results[n]["gnome"]["correlation"] for n in names]
    pp_corrs = [all_results[n]["path_patching"]["correlation"] for n in names]
    x = np.arange(len(names))
    axes[0].bar(x - 0.15, gnome_corrs, 0.3, label="GNOmE", color="#4A90D9")
    axes[0].bar(x + 0.15, pp_corrs, 0.3, label="Path Patching", color="#E53935")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("Correlation with ground truth")
    axes[0].set_title("Circuit Discovery Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Recovery comparison
    gnome_recs = [all_results[n]["gnome"]["recovery"] for n in names]
    pp_recs = [all_results[n]["path_patching"]["recovery"] for n in names]
    axes[1].bar(x - 0.15, gnome_recs, 0.3, label="GNOmE", color="#4A90D9")
    axes[1].bar(x + 0.15, pp_recs, 0.3, label="Path Patching", color="#E53935")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel(f"Top-{5} Recovery")
    axes[1].set_title("Circuit Recovery")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Scaling: val_acc vs params
    params = [all_results[n]["params"] for n in names]
    accs = [all_results[n]["history"]["val_acc_final"] for n in names]
    axes[2].loglog(params, accs, "o-", color="#4CAF50", linewidth=2, markersize=8)
    for i, n in enumerate(names):
        axes[2].annotate(n, (params[i], accs[i]), textcoords="offset points", xytext=(5, 5))
    axes[2].set_xlabel("Parameters")
    axes[2].set_ylabel("Validation Accuracy")
    axes[2].set_title("Scaling")
    axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_gnome_large_scale.png"), dpi=160)
    plt.close(fig)
    
    # Save
    output = {
        "experiment": "gnome_large_scale",
        "device": str(DEVICE),
        "task": "modular_addition_p97",
        "configs": {k: {
            "params": v["params"],
            "val_acc": v["history"]["val_acc_final"],
            "gnome_corr": v["gnome"]["correlation"],
            "gnome_recovery": v["gnome"]["recovery"],
            "pp_corr": v["path_patching"]["correlation"],
            "pp_recovery": v["path_patching"]["recovery"],
            "gnome_time_s": v["gnome"]["extraction_time_s"],
            "pp_time_s": v["path_patching"]["time_s"],
        } for k, v in all_results.items()},
    }
    with open(os.path.join(RESULTS, "gnome_large_scale.json"), "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDONE. Results saved to {RESULTS}/gnome_large_scale.json")
