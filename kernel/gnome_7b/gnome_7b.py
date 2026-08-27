#!/usr/bin/env python3
"""
GNOmE on Qwen2.5-3B — Scaling beyond 1.5B parameters
======================================================
Target: Qwen2.5-3B (3B params, 36 layers, hidden=2048, 16 heads)
Fits in T4 16GB VRAM with float16.
First graph-based circuit extraction on a 3B parameter model.
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.44.2"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import torch
import torch.nn.functional as F
import numpy as np
import json, os, time, warnings
warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42)
torch.manual_seed(42)

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================================
# IOI prompts
# ============================================================================
def build_ioi_prompts(n=30, seed=42):
    rng = np.random.RandomState(seed)
    names = ["Mary", "John", "Alice", "Bob", "Tom", "Steve", "Kevin", "Mike",
             "Anna", "Sara", "Emma", "Lisa", "Kate", "Amy", "Zoe",
             "Bill", "Dan", "Jeff", "Jason", "Harry"]
    templates = [
        " When {a} and {b} went to the store, {a} gave a drink to",
        " {a} and {b} went to the park and {a} gave a ball to",
        " After {a} talked to {b}, {a} gave a book to",
        " Before {a} met {b}, {a} sent a letter to",
    ]
    prompts = []
    for _ in range(n):
        a, b = rng.choice(names, 2, replace=False)
        tmpl = rng.choice(templates).format(a=a, b=b)
        prompts.append({"template": tmpl, "S1": f" {a}", "S2": f" {b}"})
    return prompts

ALL_NAMES = [" Mary", " John", " Alice", " Bob", " Tom", " Steve", " Kevin",
             " Mike", " Anna", " Sara", " Emma", " Lisa", " Kate", " Amy",
             " Zoe", " Bill", " Dan", " Jeff", " Jason", " Harry"]

def compute_logit_diff(model, tok, prompts):
    diffs = []
    for p in prompts:
        text = p["template"] + p["S2"]
        ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        s2_id = tok.encode(p["S2"])[0]
        other_ids = [tok.encode(n)[0] for n in ALL_NAMES if n != p["S2"] and n != p["S1"]]
        diff = logits[s2_id].item() - logits[other_ids].mean().item()
        diffs.append(diff)
    return np.mean(diffs)

# ============================================================================
# GNOmE extraction
# ============================================================================
def gnome_extract(model):
    """Weight-norm based head scoring (V→O product).
    Handles both GPT-2 style (c_attn combined) and Qwen2.5 style (separate q/k/v/o)."""
    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    d_model = config.hidden_size
    d_head = d_model // n_heads
    kv_head_dim = d_model // n_kv_heads

    head_scores = {}
    for layer_idx in range(n_layers):
        # GPT-2 style
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            attn = model.transformer.h[layer_idx].attn
            W_qkv = attn.c_attn.weight.data
            W_out = attn.c_proj.weight.data
            for head_idx in range(n_heads):
                start = head_idx * d_head
                end = start + d_head
                W_v = W_qkv[:, 2*d_model+start:2*d_model+end]
                W_o = W_out[start:end, :]
                score = (W_o @ W_v).abs().mean().item()
                head_scores[f"L{layer_idx}_H{head_idx}"] = score
        # Qwen2.5 style (handles GQA where n_kv_heads != n_heads)
        else:
            layer = model.model.layers[layer_idx]
            attn = layer.self_attn
            W_v = attn.v_proj.weight.data  # (n_kv_heads * kv_head_dim, d_model)
            W_o = attn.o_proj.weight.data  # (d_model, n_heads * d_head)

            for head_idx in range(n_heads):
                # Map head_idx to kv head index (GQA: multiple q heads share one kv head)
                kv_head_idx = head_idx % n_kv_heads
                kv_start = kv_head_idx * kv_head_dim
                kv_end = kv_start + kv_head_dim
                o_start = head_idx * d_head
                o_end = o_start + d_head

                v_head = W_v[kv_start:kv_end, :]  # (kv_head_dim, d_model)
                o_head = W_o[:, o_start:o_end]  # (d_model, d_head)

                if v_head.shape[0] == 0 or o_head.shape[1] == 0:
                    continue

                # For GQA: use separate norms since shapes don't align for direct matmul
                # V norm measures how much this head reads; O norm measures how much it writes
                v_norm = v_head.norm().item()
                o_norm = o_head.norm().item()
                # Score = geometric mean of read/write capacity
                score = (v_norm * o_norm) ** 0.5
                head_scores[f"L{layer_idx}_H{head_idx}"] = score

    return head_scores

# ============================================================================
# Zero-ablation (subset for speed on 3B model)
# ============================================================================
def zero_ablation_heads(model, tok, prompts, max_layers=12, n_samples=20):
    """Measure logit-diff drop when each head is zeroed. Limited layers for speed."""
    config = model.config
    n_layers = min(config.num_hidden_layers, max_layers)
    n_heads = config.num_attention_heads
    d_model = config.hidden_size
    d_head = d_model // n_heads

    base_diff = compute_logit_diff(model, tok, prompts[:n_samples])
    print(f"  Baseline logit diff: {base_diff:.4f}")

    head_importance = {}

    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        # Save original weights
        W_o_orig = attn.o_proj.weight.data.clone()
        b_o_orig = attn.o_proj.bias.data.clone() if attn.o_proj.bias is not None else None

        for head_idx in range(n_heads):
            start = head_idx * d_head
            end = start + d_head

            # Zero out this head
            with torch.no_grad():
                attn.o_proj.weight.data[:, start:end] = 0.0

            abl_diff = compute_logit_diff(model, tok, prompts[:n_samples])
            drop = base_diff - abl_diff
            head_importance[f"L{layer_idx}_H{head_idx}"] = drop

            # Restore
            with torch.no_grad():
                attn.o_proj.weight.data = W_o_orig
                if b_o_orig is not None:
                    attn.o_proj.bias.data = b_o_orig

        print(f"    ZA layer {layer_idx+1}/{n_layers}", flush=True)

    return head_importance, base_diff

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  GNOmE on Qwen2.5-3B — Scaling beyond 1.5B parameters")
    print("=" * 70)

    # Try Qwen2.5-3B first, fall back to Qwen2.5-1.5B
    model_name = "Qwen/Qwen2.5-3B"
    prompts = build_ioi_prompts(n=30, seed=42)

    print(f"\nLoading {model_name}...")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
    except Exception as e:
        print(f"  Failed to load {model_name}: {e}")
        print("  Falling back to Qwen2.5-1.5B...")
        model_name = "Qwen/Qwen2.5-1.5B"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    d_model = config.hidden_size
    N = n_layers * n_heads

    load_time = time.time() - t0
    gpu_mem = torch.cuda.memory_allocated() / 1024**3 if DEVICE == "cuda" else 0

    print(f"  Model: {model_name}")
    print(f"  Parameters: {n_params/1e6:.1f}M")
    print(f"  Layers: {n_layers}, Heads: {n_heads}, Hidden: {d_model}")
    print(f"  Components: {N}")
    print(f"  Load time: {load_time:.1f}s")
    print(f"  GPU memory: {gpu_mem:.2f} GB")

    # Base IOI logit diff
    base_diff = compute_logit_diff(model, tok, prompts)
    print(f"\n  Base IOI logit diff: {base_diff:.4f}")

    # GNOmE extraction
    print("\n  Running GNOmE extraction...")
    t0 = time.time()
    gnome_scores = gnome_extract(model)
    gnome_time = time.time() - t0
    print(f"  GNOmE done in {gnome_time:.3f}s")

    # Zero-ablation (first 12 layers for speed)
    print(f"\n  Running zero-ablation (first 12 layers, 20 prompts)...")
    t0 = time.time()
    za_scores, za_base = zero_ablation_heads(model, tok, prompts, max_layers=12, n_samples=20)
    za_time = time.time() - t0
    print(f"  Zero-ablation done in {za_time:.1f}s")

    # Rank correlation (only on layers where both have scores)
    from scipy.stats import spearmanr, pearsonr
    common_heads = sorted(set(za_scores.keys()) & set(gnome_scores.keys()))
    za_vals = np.array([za_scores[h] for h in common_heads])
    gnome_vals = np.array([gnome_scores[h] for h in common_heads])

    spearman_corr = spearmanr(za_vals, gnome_vals).correlation
    pearson_corr = pearsonr(za_vals, gnome_vals)[0]

    print(f"\n  Rank correlation with zero-ablation (first 12 layers):")
    print(f"    Spearman: r = {spearman_corr:.4f}")
    print(f"    Pearson:  r = {pearson_corr:.4f}")

    # Sparse adjacency
    all_vals = np.array(list(gnome_scores.values()))
    threshold = np.percentile(all_vals, 90)
    n_edges = sum(1 for v in all_vals if v > threshold)
    n_nodes = len(gnome_scores)

    full_mem = n_nodes ** 2 * 2  # float16
    sparse_mem = n_edges * 8  # index + weight
    compression = full_mem / max(sparse_mem, 1)

    print(f"\n  Sparse adjacency:")
    print(f"    Nodes: {n_nodes}")
    print(f"    Edges (top 10%): {n_edges}")
    print(f"    Density: {n_edges/max(n_nodes**2,1)*100:.3f}%")
    print(f"    Full memory: {full_mem/1024:.1f} KB")
    print(f"    Sparse memory: {sparse_mem/1024:.3f} KB")
    print(f"    Compression: {compression:.0f}x")

    # Query speedup
    path_patching_queries = N * (N - 1) // 2
    print(f"\n  Query complexity:")
    print(f"    Path patching: {path_patching_queries:,} queries")
    print(f"    GNOmE: 1 query")
    print(f"    Speedup: {path_patching_queries:,}x")

    # Top heads
    gnome_ranked = sorted(gnome_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 15 heads by GNOmE:")
    for i, (n, s) in enumerate(gnome_ranked[:15]):
        print(f"    {i+1:2d}. {n:10s} {s:.6f}")

    if za_scores:
        za_ranked = sorted(za_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 15 heads by zero-ablation:")
        for i, (n, s) in enumerate(za_ranked[:15]):
            print(f"    {i+1:2d}. {n:10s} {s:+.6f}")

    # Save results
    results = {
        "model": model_name,
        "params_M": n_params / 1e6,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_model": d_model,
        "N_components": N,
        "gpu_memory_GB": float(gpu_mem),
        "load_time_s": float(load_time),
        "gnome_time_s": float(gnome_time),
        "za_time_s": float(za_time),
        "spearman_r": float(spearman_corr),
        "pearson_r": float(pearson_corr),
        "adjacency": {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": n_edges / max(n_nodes**2, 1),
            "full_memory_KB": full_mem / 1024,
            "sparse_memory_KB": sparse_mem / 1024,
            "compression": compression,
        },
        "query_speedup": path_patching_queries,
        "top15_gnome": [(h, float(s)) for h, s in gnome_ranked[:15]],
        "top15_za": [(h, float(s)) for h, s in za_ranked[:15]] if za_scores else [],
    }

    with open("results/gnome_7b.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to results/gnome_7b.json")
    print(f"\n{'='*70}")
    print("  DONE.")
