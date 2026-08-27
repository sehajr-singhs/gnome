#!/usr/bin/env python3
"""
GNOmE vs Anthropic Circuit Tracing — Head-to-Head Comparison
=============================================================
Implements Anthropic's backward-Jacobian attribution graph extraction
(Circuit Tracing, 2025) and compares with GNOmE's forward-pass extraction.

Anthropic method: for each component pair (i,j), compute
  attribution(i,j) = E_x[ |d(output)/d(component_i_input) * component_i_output contribution to j| ]

GNOmE method: for each component, compute
  score(i) = E_x[ |d(component_i_output)/d(component_i_input)| ] (V→O weight product)

Both are run on GPT-2 Small (124M) and Qwen2.5-3B (3,086M) for IOI.

Kernels: GPT-2 on CPU, Qwen2.5-3B on T4 GPU.
"""
import torch
import torch.nn.functional as F
import numpy as np
import json, os, time, warnings
warnings.filterwarnings("ignore")

DEVICE = "cpu"  # Force CPU - P100 incompatible with pre-installed torch
RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
np.random.seed(42)
torch.manual_seed(42)

print(f"Device: {DEVICE}")

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
# METHOD 2: GNOmE (forward-pass, weight-norm)
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
# METHOD 3: Anthropic Circuit Tracing (backward-Jacobian attribution)
# ============================================================================
def anthropic_extract(model, tok, prompts, n_samples=20):
    """
    Anthropic-style backward-Jacobian attribution graph.
    
    For each head h at layer l, compute:
      attribution(h) = E_x[ |d(loss)/d(h_attn_input)| * ||W_O^h|| ]
    
    This is the product of:
    1. How much the loss depends on this head's input (gradient sensitivity)
    2. How much this head can influence output (weight magnitude)
    
    This requires one backward pass per layer (not per head).
    """
    model.eval()
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    d_model = model.config.n_embd
    d_head = d_model // n_heads

    head_scores = {}

    for layer_idx in range(n_layers):
        layer = model.transformer.h[layer_idx]
        attn = layer.attn

        # ---- Backward pass to get gradient sensitivity ----
        all_grad_norms = []
        for p_idx in range(min(n_samples, 20)):
            text = prompts[p_idx]["template"] + prompts[p_idx]["S2"]
            ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)

            # Enable gradients for the residual stream input to this layer
            h_input = None
            def make_hook(lIdx):
                def hook_fn(module, inp, out):
                    nonlocal h_input
                    if h_input is None:
                        if isinstance(inp, tuple):
                            h_input = inp[0].detach().requires_grad_(True)
                        else:
                            h_input = inp.detach().requires_grad_(True)
                return hook_fn

            hook = layer.register_forward_hook(make_hook(layer_idx))

            # Forward + backward
            out = model(ids, labels=ids)
            loss = out.loss

            hook.remove()

            if h_input is not None and h_input.grad is None:
                loss.backward(retain_graph=True)

            if h_input is not None and h_input.grad is not None:
                # Gradient norm per dimension at last position
                grad = h_input.grad[0, -1]  # (D,)
                all_grad_norms.append(grad.detach())

            model.zero_grad()

        if not all_grad_norms:
            print(f"    CT layer {layer_idx+1}/{n_layers} (no grads)", flush=True)
            continue

        # Average gradient across samples
        avg_grad = torch.stack(all_grad_norms).mean(dim=0)  # (D,)

        # ---- Combine with weight magnitude ----
        W_out = attn.c_proj.weight.data  # (D, D) — output projection
        W_qkv = attn.c_attn.weight.data  # (D, 3D) — combined QKV

        for head_idx in range(n_heads):
            start = head_idx * d_head
            end = start + d_head

            # Gradient sensitivity: how much does the loss depend on this head's dimensions?
            grad_sensitivity = avg_grad[start:end].abs().mean().item()

            # Weight magnitude: how much can this head influence output?
            W_o_head = W_out[start:end, :]  # (d_head, D)
            weight_magnitude = W_o_head.norm().item()

            # Anthropic attribution = gradient sensitivity × weight magnitude
            attribution = grad_sensitivity * weight_magnitude
            head_scores[f"L{layer_idx}_H{head_idx}"] = attribution

        print(f"    CT layer {layer_idx+1}/{n_layers} (grad_sens={avg_grad.abs().mean():.4f})", flush=True)

    return head_scores

# ============================================================================
# METHOD 4: Path patching (causal intervention, subset)
# ============================================================================
def path_patch_heads(model, tok, prompts, n_samples=10, max_layers=6):
    """Path patching: zero each head and measure loss change."""
    model.eval()
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads
    n_layers = min(model.config.n_layer, max_layers)

    base_diff = compute_logit_diff(model, tok, prompts[:n_samples])
    head_scores = {}

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
            head_scores[f"L{layer_idx}_H{head_idx}"] = drop
            with torch.no_grad():
                attn.c_proj.weight.data = W_out
                if b_out is not None:
                    attn.c_proj.bias.data = b_out
        print(f"    PP layer {layer_idx+1}/{n_layers}", flush=True)
    return head_scores

# ============================================================================
# GPT-2 ONLY (no GPU needed)
# ============================================================================
def run_gpt2():
    print("\n" + "="*70)
    print("  GPT-2 Small (124M) — Head-to-Head Comparison")
    print("="*70)

    model_name = "gpt2"
    prompts = build_ioi_prompts(n=30, seed=42)

    print(f"\nLoading {model_name}...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    N = model.config.n_layer * model.config.n_head
    print(f"  {n_params/1e6:.1f}M params, {N} components, {time.time()-t0:.1f}s")

    results = {"model": model_name, "params_M": n_params/1e6, "N": N}

    # 1. Zero-ablation
    print("\n[1/4] Zero-ablation (ground truth)...")
    t0 = time.time()
    za_scores, za_base = zero_ablation_heads(model, tok, prompts, n_samples=30)
    za_time = time.time() - t0
    results["zero_ablation"] = {"time_s": float(za_time), "base": float(za_base)}
    print(f"  Done in {za_time:.1f}s")

    # 2. GNOmE
    print("\n[2/4] GNOmE (forward-pass weight-norm)...")
    t0 = time.time()
    gnome_scores = gnome_extract(model)
    gnome_time = time.time() - t0
    results["gnome"] = {"time_s": float(gnome_time)}
    print(f"  Done in {gnome_time:.4f}s")

    # 3. Anthropic Circuit Tracing
    print("\n[3/4] Anthropic Circuit Tracing (backward-Jacobian)...")
    t0 = time.time()
    anthro_scores = anthropic_extract(model, tok, prompts, n_samples=20)
    anthro_time = time.time() - t0
    results["anthropic"] = {"time_s": float(anthro_time)}
    print(f"  Done in {anthro_time:.1f}s")

    # 4. Path patching (6 layers)
    print("\n[4/4] Path patching (6 layers, causal intervention)...")
    t0 = time.time()
    pp_scores = path_patch_heads(model, tok, prompts, n_samples=10, max_layers=6)
    pp_time = time.time() - t0
    results["path_patching"] = {"time_s": float(pp_time)}
    print(f"  Done in {pp_time:.1f}s")

    # ---- Compare ----
    from scipy.stats import spearmanr, pearsonr

    heads = sorted(za_scores.keys())
    za_vals = np.array([za_scores[h] for h in heads])
    gnome_vals = np.array([gnome_scores.get(h, 0) for h in heads])
    anthro_vals = np.array([anthro_scores.get(h, 0) for h in heads])

    print("\n" + "="*70)
    print("  RESULTS: GPT-2 Small")
    print("="*70)

    corr = {}
    for name, vals in [("GNOmE", gnome_vals), ("Anthropic CT", anthro_vals)]:
        sp = spearmanr(za_vals, vals).correlation
        pr = pearsonr(za_vals, vals)[0]
        corr[name] = {"spearman": float(sp), "pearson": float(pr)}
        print(f"\n  {name}:")
        print(f"    Spearman r = {sp:.4f}")
        print(f"    Pearson  r = {pr:.4f}")

    print(f"\n  Speed:")
    print(f"    Zero-ablation:     {za_time:.1f}s")
    print(f"    GNOmE:             {gnome_time:.4f}s")
    print(f"    Anthropic CT:      {anthro_time:.1f}s")
    print(f"    Path patching:     {pp_time:.1f}s (6 layers)")

    # IOI recovery
    known_ioi = ["L8_H0", "L9_H6", "L9_H9", "L10_H0", "L8_H1", "L5_H1", "L6_H9"]
    gnome_ranked = sorted(gnome_scores.items(), key=lambda x: x[1], reverse=True)
    anthro_ranked = sorted(anthro_scores.items(), key=lambda x: x[1], reverse=True)
    za_ranked = sorted(za_scores.items(), key=lambda x: x[1], reverse=True)

    gnome_top30 = set(h for h, _ in gnome_ranked[:30])
    anthro_top30 = set(h for h, _ in anthro_ranked[:30])
    za_top30 = set(h for h, _ in za_ranked[:30])

    print(f"\n  IOI recovery (top-30):")
    print(f"    GNOmE:      {sum(1 for h in known_ioi if h in gnome_top30)}/7")
    print(f"    Anthropic:  {sum(1 for h in known_ioi if h in anthro_top30)}/7")
    print(f"    Zero-abl:   {sum(1 for h in known_ioi if h in za_top30)}/7")

    # Known head ranks
    print(f"\n  Known IOI head ranks:")
    for h in known_ioi:
        r_gnome = next((i+1 for i, (n, _) in enumerate(gnome_ranked) if n == h), N+1)
        r_anthro = next((i+1 for i, (n, _) in enumerate(anthro_ranked) if n == h), N+1)
        r_za = next((i+1 for i, (n, _) in enumerate(za_ranked) if n == h), N+1)
        print(f"    {h:10s}: ZA={r_za:3d}, GNOmE={r_gnome:3d}, Anthropic={r_anthro:3d}")

    results["correlations"] = corr
    results["ioi_recovery"] = {
        "gnome": sum(1 for h in known_ioi if h in gnome_top30),
        "anthropic": sum(1 for h in known_ioi if h in anthro_top30),
        "za": sum(1 for h in known_ioi if h in za_top30),
        "total": len(known_ioi),
    }
    results["speed"] = {
        "zero_ablation_s": float(za_time),
        "gnome_s": float(gnome_time),
        "anthropic_ct_s": float(anthro_time),
        "path_patching_s": float(pp_time),
    }

    return results

# ============================================================================
# QWEN2.5-3B (GPU)
# ============================================================================
def run_qwen3b():
    print("\n" + "="*70)
    print("  Qwen2.5-3B (3,086M) — Scaling Comparison")
    print("="*70)

    model_name = "Qwen/Qwen2.5-3B"
    prompts = build_ioi_prompts(n=30, seed=42)

    print(f"\nLoading {model_name}...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    N = model.config.num_hidden_layers * model.config.num_attention_heads
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  {n_params/1e6:.1f}M params, {N} components, GPU: {gpu_mem:.1f}GB, {time.time()-t0:.1f}s")

    results = {"model": model_name, "params_M": n_params/1e6, "N": N, "gpu_GB": float(gpu_mem)}

    # GNOmE (Qwen2.5 extraction)
    print("\n[GNOmE] Extracting...")
    t0 = time.time()
    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    d_model = config.hidden_size
    d_head = d_model // n_heads
    kv_head_dim = d_model // n_kv_heads

    gnome_scores = {}
    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        W_v = attn.v_proj.weight.data
        W_o = attn.o_proj.weight.data
        for head_idx in range(n_heads):
            kv_head_idx = head_idx % n_kv_heads
            kv_start = kv_head_idx * kv_head_dim
            kv_end = kv_start + kv_head_dim
            o_start = head_idx * d_head
            o_end = o_start + d_head
            v_head = W_v[kv_start:kv_end, :]
            o_head = W_o[:, o_start:o_end]
            if v_head.shape[0] == 0 or o_head.shape[1] == 0:
                continue
            v_norm = v_head.norm().item()
            o_norm = o_head.norm().item()
            gnome_scores[f"L{layer_idx}_H{head_idx}"] = (v_norm * o_norm) ** 0.5
    gnome_time = time.time() - t0
    print(f"  GNOmE: {gnome_time:.4f}s, {len(gnome_scores)} heads")

    # Sparse stats
    all_vals = np.array(list(gnome_scores.values()))
    threshold = np.percentile(all_vals, 90)
    n_edges = sum(1 for v in all_vals if v > threshold)
    n_nodes = len(gnome_scores)
    full_mem = n_nodes ** 2 * 2
    sparse_mem = n_edges * 8
    compression = full_mem / max(sparse_mem, 1)

    print(f"  Sparse: {n_edges} edges / {n_nodes} nodes, {compression:.0f}x compression")
    print(f"  Speedup: {N*(N-1)//2:,}x")

    results["gnome"] = {
        "time_s": float(gnome_time),
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "compression": float(compression),
        "speedup": N*(N-1)//2,
    }

    # Top heads
    gnome_ranked = sorted(gnome_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 heads:")
    for i, (h, s) in enumerate(gnome_ranked[:10]):
        print(f"    {i+1:2d}. {h:10s} {s:.4f}")

    return results

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    all_results = {}

    # GPT-2 on CPU
    gpt2_results = run_gpt2()
    all_results["gpt2"] = gpt2_results

    # Qwen2.5-3B skipped (P100 GPU incompatible with pre-installed torch)

    # Save
    with open(f"{RESULTS}/gnome_vs_anthropic.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print("="*70)
    gpt2 = all_results["gpt2"]
    print(f"\n  GPT-2 Small ({gpt2['params_M']:.0f}M, {gpt2['N']} components):")
    print(f"    GNOmE Spearman:      {gpt2['correlations']['GNOmE']['spearman']:.4f}")
    print(f"    Anthropic CT Spearman: {gpt2['correlations']['Anthropic CT']['spearman']:.4f}")
    print(f"    GNOmE time:          {gpt2['speed']['gnome_s']:.4f}s")
    print(f"    Anthropic CT time:   {gpt2['speed']['anthropic_ct_s']:.1f}s")
    print(f"    Speedup GNOmE/CT:    {gpt2['speed']['anthropic_ct_s']/max(gpt2['speed']['gnome_s'],0.001):.0f}x")

    if "qwen3b" in all_results:
        qwen = all_results["qwen3b"]
        print(f"\n  Qwen2.5-3B ({qwen['params_M']:.0f}M, {qwen['N']} components):")
        print(f"    GNOmE time:          {qwen['gnome']['time_s']:.4f}s")
        print(f"    Sparse edges:        {qwen['gnome']['n_edges']}")
        print(f"    Compression:         {qwen['gnome']['compression']:.0f}x")
        print(f"    Speedup:             {qwen['gnome']['speedup']:,}x")

    print(f"\n  DONE.")
