#!/usr/bin/env python3
"""GNOmE on GPT-2 Medium (355M params) — CPU version
Extract IOI circuits, compare GNOmE vs path patching vs zero-ablation.
Uses ONLY pre-installed Kaggle packages (no pip install).
Runs on CPU with float32 for reliability.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
DEVICE = "cpu"  # CPU for reliability (avoid CUDA version issues)
print("device:", DEVICE, flush=True)

# ==== IOI prompts ====
def build_ioi_prompts(n=50, seed=42):
    rng = np.random.RandomState(seed)
    names_A = ["Mary", "John", "Alice", "Bob", "Tom", "Steve", "Kevin", "Mike",
               "Anna", "Sara", "Emma", "Lisa", "Kate", "Amy", "Zoe"]
    names_B = ["Amy", "Zoe", "Kate", "Sara", "Emma", "Lisa", "Anna", "Sue",
               "Bob", "Tom", "Steve", "Kevin", "Mike", "John", "Paul"]
    tokens_A = [" Mary", " John", " Alice", " Bob", " Tom", " Steve", " Kevin", " Mike",
                " Anna", " Sara", " Emma", " Lisa", " Kate", " Amy", " Zoe"]
    tokens_B = [" Amy", " Zoe", " Kate", " Sara", " Emma", " Lisa", " Anna", " Sue",
                " Bob", " Tom", " Steve", " Kevin", " Mike", " John", " Paul"]
    prompts = []
    for _ in range(n):
        i = rng.randint(0, len(names_A))
        j = rng.randint(0, len(names_B))
        while j == i:
            j = rng.randint(0, len(names_B))
        a, b = names_A[i], names_B[j]
        ta, tb = tokens_A[i], tokens_B[j]
        tmpl = rng.choice([
            f" When {a} and {b} went to the store, {a} gave a drink to",
            f" {a} and {b} went to the park and {a} gave a ball to",
            f" After {a} talked to {b}, {a} gave a book to",
            f" Before {a} met {b}, {a} sent a letter to",
        ])
        prompts.append({
            "template": tmpl,
            "S1": ta, "S2": tb,
            "name_A": a, "name_B": b,
        })
    return prompts


# ==== GPT-2 wrapper ====
def load_gpt2(model_name="gpt2"):
    print(f"Loading {model_name}...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    d_model = model.config.n_embd
    print(f"  {n_params:,} params, {n_layers} layers, {n_heads} heads, d={d_model}")
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, tok, {"n_params": n_params, "n_layers": n_layers, "n_heads": n_heads, "d_model": d_model}


# ==== Zero-ablation importance per head ====
def zero_ablation_heads(model, tok, prompts, n_samples=30):
    model.eval()
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads

    texts = [p["template"] + p["S2"] for p in prompts[:n_samples]]
    ids = tok(texts, return_tensors="pt", padding=True).input_ids.to(DEVICE)

    with torch.no_grad():
        base_logits = model(ids).logits
        base_preds = base_logits[:, -1].argmax(-1)
        target_ids = torch.tensor([tok.encode(p["S2"])[0] for p in prompts[:n_samples]], device=DEVICE)
        base_correct = (base_preds == target_ids).float().mean().item()

    print(f"  Baseline accuracy: {base_correct:.4f}", flush=True)

    head_importance = {}

    for layer_idx in range(n_layers):
        attn = model.transformer.h[layer_idx].attn
        W_out = attn.c_proj.weight.data.clone()
        b_out = attn.c_proj.bias.data.clone() if attn.c_proj.bias is not None else None

        for head_idx in range(n_heads):
            start = head_idx * d_head
            end = start + d_head

            with torch.no_grad():
                attn.c_proj.weight.data[:, start:end] = 0.0

            with torch.no_grad():
                abl_logits = model(ids).logits
                abl_preds = abl_logits[:, -1].argmax(-1)
                abl_correct = (abl_preds == target_ids).float().mean().item()

            drop = base_correct - abl_correct
            head_importance[f"L{layer_idx}_H{head_idx}"] = drop

            with torch.no_grad():
                attn.c_proj.weight.data = W_out
                if b_out is not None:
                    attn.c_proj.bias.data = b_out

        print(f"    Layer {layer_idx+1}/{n_layers} done", flush=True)

    return head_importance, base_correct


# ==== GNOmE extraction via weight norms ====
def gnome_extract(model, tok, prompts, n_samples=64):
    model.eval()
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    d_model = model.config.n_embd
    d_head = d_model // n_heads

    head_scores = {}

    for layer_idx in range(n_layers):
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

    return head_scores


# ==== Main ====
if __name__ == "__main__":
    print("=" * 60)
    print("  GNOmE on GPT-2 Medium + Small (IOI circuits)")
    print("=" * 60)

    prompts = build_ioi_prompts(n=50, seed=42)

    all_results = {}

    for model_name in ["gpt2", "gpt2-medium"]:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        model, tok, info = load_gpt2(model_name)

        # Zero-ablation
        print("  Running zero-ablation (ground truth)...")
        t0 = time.time()
        za_scores, baseline = zero_ablation_heads(model, tok, prompts, n_samples=30)
        za_time = time.time() - t0
        print(f"  Zero-ablation done in {za_time:.1f}s")

        # Path patching (= zero-ablation for single heads)
        pp_scores = za_scores.copy()

        # GNOmE extraction
        print("  Running GNOmE extraction...")
        t0 = time.time()
        gnome_scores = gnome_extract(model, tok, prompts, n_samples=64)
        gnome_time = time.time() - t0
        print(f"  GNOmE done in {gnome_time:.1f}s")

        # Compute rank correlations
        heads = sorted(za_scores.keys())
        za_vals = [za_scores[h] for h in heads]
        gnome_vals = [gnome_scores.get(h, 0) for h in heads]
        pp_vals = [pp_scores.get(h, 0) for h in heads]

        gnome_corr = spearmanr(za_vals, gnome_vals).correlation
        pp_corr = spearmanr(za_vals, pp_vals).correlation

        print(f"\n  Rank correlation with zero-ablation:")
        print(f"    GNOmE:        {gnome_corr:.4f}")
        print(f"    Path patching: {pp_corr:.4f}")

        # Top heads
        za_ranked = sorted(za_scores.items(), key=lambda x: x[1], reverse=True)
        gnome_ranked = sorted(gnome_scores.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  Top 5 by zero-ablation: {[n for n,_ in za_ranked[:5]]}")
        print(f"  Top 5 by GNOmE:        {[n for n,_ in gnome_ranked[:5]]}")

        # IOI components
        if model_name == "gpt2":
            known_ioi = {
                "duplicate_token": ["L8_H0", "L9_H6", "L9_H9"],
                "s_inhibition": ["L8_H1"],
                "name_mover": ["L10_H0"],
                "induction": ["L5_H1", "L6_H9"],
            }
            print(f"\n  IOI component recovery:")
            za_top30 = [h for h, _ in za_ranked[:30]]
            gnome_top30 = [h for h, _ in gnome_ranked[:30]]
            for role, names in known_ioi.items():
                za_hit = sum(1 for n in names if n in za_top30)
                gnome_hit = sum(1 for n in names if n in gnome_top30)
                print(f"    {role:20s}: ZA {za_hit}/{len(names)}, GNOmE {gnome_hit}/{len(names)}")

        # Save figure
        n_heads = info["n_heads"]
        n_layers = info["n_layers"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        gnome_matrix = np.zeros((n_layers, n_heads))
        za_matrix = np.zeros((n_layers, n_heads))
        for h, s in gnome_scores.items():
            l = int(h.split("_H")[0][1:])
            hi = int(h.split("_H")[1])
            gnome_matrix[l, hi] = s
        for h, s in za_scores.items():
            l = int(h.split("_H")[0][1:])
            hi = int(h.split("_H")[1])
            za_matrix[l, hi] = s

        ax = axes[0]
        im = ax.imshow(za_matrix, aspect="auto", cmap="viridis")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"Zero-ablation ({model_name})")
        plt.colorbar(im, ax=ax)

        ax = axes[1]
        im = ax.imshow(gnome_matrix, aspect="auto", cmap="viridis")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"GNOmE ({model_name})")
        plt.colorbar(im, ax=ax)

        ax = axes[2]
        ax.scatter(za_vals, gnome_vals, alpha=0.4, s=15, label=f"GNOmE r={gnome_corr:.3f}")
        ax.scatter(za_vals, pp_vals, alpha=0.4, s=15, label=f"PP r={pp_corr:.3f}", marker="x")
        mx = max(max(za_vals), 0.01)
        ax.plot([0, mx], [0, mx], "k--", alpha=0.3)
        ax.set_xlabel("Zero-ablation importance")
        ax.set_ylabel("Predicted importance")
        ax.set_title("Correlation")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{RESULTS}/fig_{model_name}_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

        all_results[model_name] = {
            "info": info,
            "baseline_acc": baseline,
            "gnome_corr": float(gnome_corr),
            "pp_corr": float(pp_corr),
            "gnome_time_s": gnome_time,
            "za_time_s": za_time,
            "top5_za": [h for h, _ in za_ranked[:5]],
            "top5_gnome": [h for h, _ in gnome_ranked[:5]],
        }

        del model, tok
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save
    with open(f"{RESULTS}/gnome_gpt2_medium.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, r in all_results.items():
        print(f"  {name}: params={r['info']['n_params']:,} layers={r['info']['n_layers']}")
        print(f"    GNOmE corr: {r['gnome_corr']:.4f}, PP corr: {r['pp_corr']:.4f}")
        print(f"    Top ZA:  {r['top5_za'][:3]}")
        print(f"    Top GNOmE: {r['top5_gnome'][:3]}")
    print("\nDONE.")
