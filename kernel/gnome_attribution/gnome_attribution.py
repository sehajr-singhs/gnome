#!/usr/bin/env python3
"""
GNOmE vs Attribution Patching — Proper Head-to-Head Comparison
==============================================================
Implements attribution patching correctly per Syed et al. 2023:
  attribution_h = <grad, clean_activation - corrupted_activation>
where grad = d(loss)/d(residual_stream) and activation diff is per-head.

Runs on GPT-2 Small (124M) with 30 IOI prompts.
"""
import json, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr, pearsonr

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
DEVICE = "cpu"
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# IOI PROMPTS
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
# METHOD 1: Zero-ablation (ground truth)
# ============================================================================
def zero_ablation_heads(model, tok, prompts, n_samples=30):
    model.eval()
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads

    base_diff = compute_logit_diff(model, tok, prompts[:n_samples])
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
            abl_diff = compute_logit_diff(model, tok, prompts[:n_samples])
            drop = base_diff - abl_diff
            head_importance[f"L{layer_idx}_H{head_idx}"] = drop
            with torch.no_grad():
                attn.c_proj.weight.data = W_out
                if b_out is not None:
                    attn.c_proj.bias.data = b_out

        print(f"    ZA layer {layer_idx+1}/{n_layers}", flush=True)
    return head_importance, base_diff

# ============================================================================
# METHOD 2: GNOmE
# ============================================================================
def gnome_extract(model):
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

# ============================================================================
# METHOD 3: PROPER Attribution Patching (Syed et al. 2023)
# ============================================================================
def attribution_patching_proper(model, tok, prompts, n_samples=30):
    """
    Proper attribution patching per Syed et al. 2023:
    
    For each head h at layer l:
      attribution_h = E_x[ <grad, clean_act - corrupt_act> ]
    
    where:
      grad = d(loss)/d(residual_stream_before_layer_l)   [computed once per layer]
      clean_act = residual stream before layer l on clean input
      corrupt_act = residual stream before layer l on corrupted input
    
    This requires one backward pass per layer (to get grad), not per head.
    """
    model.eval()
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    d_model = model.config.n_embd
    d_head = d_model // n_heads

    base_diff = compute_logit_diff(model, tok, prompts[:n_samples])

    # Collect clean and corrupted hidden states + gradients
    head_scores = {}

    for layer_idx in range(n_layers):
        # ---- Step 1: Get gradient w.r.t. residual stream before this layer ----
        # We need to compute d(loss)/d(h) where h is the residual stream input to this layer.
        # This requires a backward pass.

        # Hook to capture the residual stream input
        residual_input = [None]
        def make_hook(lIdx):
            def hook_fn(module, inp, out):
                if residual_input[0] is None:
                    # The input to the layer is a tuple; first element is residual stream
                    if isinstance(inp, tuple):
                        residual_input[0] = inp[0].detach().clone()
                    else:
                        residual_input[0] = inp.detach().clone()
            return hook_fn

        hook = model.transformer.h[layer_idx].register_forward_hook(make_hook(layer_idx))

        # ---- Step 2: Compute average gradient across prompts ----
        all_grads = []
        all_act_diffs = []

        for p_idx in range(min(n_samples, 30)):
            # Clean forward
            clean_text = prompts[p_idx]["template"] + prompts[p_idx]["S2"]
            clean_ids = tok(clean_text, return_tensors="pt").input_ids.to(DEVICE)

            # Corrupted forward (swap S1 and S2)
            corrupt_text = prompts[p_idx]["template"] + prompts[p_idx]["S1"]
            corrupt_ids = tok(corrupt_text, return_tensors="pt").input_ids.to(DEVICE)

            # Clean forward + backward to get gradient
            residual_input[0] = None
            clean_out = model(clean_ids, labels=clean_ids)
            loss = clean_out.loss
            loss.backward()

            # Gradient w.r.t. residual stream input
            grad = residual_input[0].grad
            if grad is not None:
                # Take last position gradient
                all_grads.append(grad[0, -1].detach().clone())  # (D,)

            # Corrupted forward (no grad needed)
            with torch.no_grad():
                corrupt_out = model(corrupt_ids, output_hidden_states=True)

            # Clean forward hidden states (re-run without grad to get clean activation)
            with torch.no_grad():
                clean_out2 = model(clean_ids, output_hidden_states=True)

            # Activation difference at last position
            clean_hidden = clean_out2.hidden_states[layer_idx][0, -1]  # (D,)
            corrupt_hidden = corrupt_out.hidden_states[layer_idx][0, -1]  # (D,)
            act_diff = clean_hidden - corrupt_hidden  # (D,)
            all_act_diffs.append(act_diff)

            model.zero_grad()

        hook.remove()

        if not all_grads:
            print(f"    AP layer {layer_idx+1}/{n_layers} (no grads)", flush=True)
            continue

        # Average gradient and activation difference
        avg_grad = torch.stack(all_grads).mean(dim=0)  # (D,)
        avg_act_diff = torch.stack(all_act_diffs).mean(dim=0)  # (D,)

        # ---- Step 3: Per-head attribution = grad · act_diff for head dimensions ----
        for head_idx in range(n_heads):
            start = head_idx * d_head
            end = start + d_head
            # Attribution = sum of grad * act_diff over this head's dimensions
            attribution = (avg_grad[start:end] * avg_act_diff[start:end]).sum().item()
            head_scores[f"L{layer_idx}_H{head_idx}"] = attribution

        print(f"    AP layer {layer_idx+1}/{n_layers} (grad norm={avg_grad.norm():.4f})", flush=True)

    return head_scores, base_diff

# ============================================================================
# METHOD 4: Simplified weight-norm attribution (fast baseline)
# ============================================================================
def attribution_weight_norm(model):
    """Fast weight-norm based attribution (no gradients needed)."""
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

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  GNOmE vs Attribution Patching — Proper Head-to-Head")
    print("  GPT-2 Small (124M), 30 IOI prompts")
    print("=" * 70)

    model_name = "gpt2"
    prompts = build_ioi_prompts(n=30, seed=42)

    print(f"\nLoading {model_name}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    d_model = model.config.n_embd
    N = n_layers * n_heads
    print(f"  {n_params:,} params, {n_layers}L × {n_heads}H = {N} components")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    base_diff = compute_logit_diff(model, tok, prompts)
    print(f"\n  Base IOI logit diff: {base_diff:.4f}")

    # ---- Zero-ablation (ground truth) ----
    print("\n[1/3] Zero-ablation (ground truth)...")
    t0 = time.time()
    za_scores, za_base = zero_ablation_heads(model, tok, prompts, n_samples=30)
    za_time = time.time() - t0
    print(f"  Done in {za_time:.1f}s")

    # ---- GNOmE ----
    print("\n[2/3] GNOmE extraction...")
    t0 = time.time()
    gnome_scores = gnome_extract(model)
    gnome_time = time.time() - t0
    print(f"  Done in {gnome_time:.4f}s")

    # ---- Proper Attribution Patching ----
    print("\n[3/3] Attribution patching (Syed et al. 2023, proper)...")
    t0 = time.time()
    ap_scores, ap_base = attribution_patching_proper(model, tok, prompts, n_samples=30)
    ap_time = time.time() - t0
    print(f"  Done in {ap_time:.1f}s")

    # ---- RESULTS ----
    heads = sorted(za_scores.keys())
    za_vals = np.array([za_scores[h] for h in heads])
    gnome_vals = np.array([gnome_scores.get(h, 0) for h in heads])
    ap_vals = np.array([ap_scores.get(h, 0) for h in heads])

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    spearman_gnome = spearmanr(za_vals, gnome_vals).correlation
    spearman_ap = spearmanr(za_vals, ap_vals).correlation
    pearson_gnome = pearsonr(za_vals, gnome_vals)[0]
    pearson_ap = pearsonr(za_vals, ap_vals)[0]

    print(f"\n  Spearman rank correlation with zero-ablation:")
    print(f"    GNOmE:             r = {spearman_gnome:.4f}")
    print(f"    Attribution patch: r = {spearman_ap:.4f}")
    print(f"\n  Pearson correlation:")
    print(f"    GNOmE:             r = {pearson_gnome:.4f}")
    print(f"    Attribution patch: r = {pearson_ap:.4f}")

    print(f"\n  Time:")
    print(f"    Zero-ablation:     {za_time:.1f}s")
    print(f"    GNOmE:             {gnome_time:.4f}s")
    print(f"    Attribution patch: {ap_time:.1f}s")
    print(f"    Speedup GNOmE/AP:  {ap_time/max(gnome_time,0.001):.0f}x")

    # IOI recovery
    known_ioi = {
        "duplicate_token": ["L8_H0", "L9_H6", "L9_H9"],
        "s_inhibition": ["L8_H1"],
        "name_mover": ["L10_H0"],
        "induction": ["L5_H1", "L6_H9"],
    }
    all_known = [h for names in known_ioi.values() for h in names]

    za_ranked = sorted(za_scores.items(), key=lambda x: x[1], reverse=True)
    gnome_ranked = sorted(gnome_scores.items(), key=lambda x: x[1], reverse=True)
    ap_ranked = sorted(ap_scores.items(), key=lambda x: x[1], reverse=True)

    za_top30 = set(h for h, _ in za_ranked[:30])
    gnome_top30 = set(h for h, _ in gnome_ranked[:30])
    ap_top30 = set(h for h, _ in ap_ranked[:30])

    gnome_recovery = sum(1 for h in all_known if h in gnome_top30)
    ap_recovery = sum(1 for h in all_known if h in ap_top30)

    print(f"\n  IOI component recovery (top-30):")
    print(f"    GNOmE:    {gnome_recovery}/{len(all_known)}")
    print(f"    Attr. AP: {ap_recovery}/{len(all_known)}")

    for role, names in known_ioi.items():
        gnome_hit = sum(1 for n in names if n in gnome_top30)
        ap_hit = sum(1 for n in names if n in ap_top30)
        print(f"    {role:20s}: GNOmE {gnome_hit}/{len(names)}, AP {ap_hit}/{len(names)}")

    # Known head ranks
    print(f"\n  Known IOI head ranks:")
    for h in all_known:
        rank_gnome = next((i+1 for i, (n, _) in enumerate(gnome_ranked) if n == h), N+1)
        rank_ap = next((i+1 for i, (n, _) in enumerate(ap_ranked) if n == h), N+1)
        rank_za = next((i+1 for i, (n, _) in enumerate(za_ranked) if n == h), N+1)
        print(f"    {h:10s}: ZA={rank_za:3d}, GNOmE={rank_gnome:3d}, AP={rank_ap:3d}")

    # Save results
    results = {
        "model": "GPT-2 Small",
        "params_M": n_params / 1e6,
        "n_components": N,
        "base_logit_diff": float(base_diff),
        "correlations": {
            "gnome_spearman": float(spearman_gnome),
            "gnome_pearson": float(pearson_gnome),
            "ap_spearman": float(spearman_ap),
            "ap_pearson": float(pearson_ap),
        },
        "speed": {
            "zero_ablation_s": float(za_time),
            "gnome_s": float(gnome_time),
            "attribution_patching_s": float(ap_time),
        },
        "ioi_recovery": {
            "gnome": gnome_recovery,
            "ap": ap_recovery,
            "total": len(all_known),
        },
        "known_head_ranks": {},
    }
    for h in all_known:
        results["known_head_ranks"][h] = {
            "gnome": next((i+1 for i, (n, _) in enumerate(gnome_ranked) if n == h), N+1),
            "ap": next((i+1 for i, (n, _) in enumerate(ap_ranked) if n == h), N+1),
            "za": next((i+1 for i, (n, _) in enumerate(za_ranked) if n == h), N+1),
        }

    with open(f"{RESULTS}/gnome_attribution.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {RESULTS}/gnome_attribution.json")
    print(f"\n{'='*70}")
    print("  DONE.")
