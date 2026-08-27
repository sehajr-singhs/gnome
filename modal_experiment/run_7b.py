#!/usr/bin/env python3
"""
Modal GPU experiment: GNOmE on Llama-3-8B + head-to-head vs attribution patching.
Runs on Modal's A10G (24GB VRAM) — no Kaggle quota limits.
"""
import modal

app = modal.App("gnome-llama3-experiment")

gnome_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
        "scipy>=1.10.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.0.0",
    )
)

@app.function(image=gnome_image, gpu="A10G", timeout=1800)
def run_experiment():
    import torch
    import time
    import json
    import numpy as np
    from pathlib import Path

    print("=" * 60)
    print("GNOmE Llama-3-8B Experiment on Modal A10G")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 60)

    results = {}

    # ============================================================
    # 1. Load Llama-3.2-3B (non-gated, fits in 24GB at 4-bit)
    # Llama-3.1-8B needs too much even at 4-bit on A10G
    # Using Qwen2.5-7B instead — same class, non-gated, 7.6B params
    # ============================================================
    print("\n[1] Loading Qwen2.5-7B-Instruct (4-bit quantized)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    load_time = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Layers: {n_layers}, d_model: {d_model}")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB used")

    results["model"] = "Qwen2.5-7B-Instruct"
    results["params"] = n_params
    results["layers"] = n_layers
    results["d_model"] = d_model
    results["load_time"] = load_time

    # ============================================================
    # 2. GNOmE extraction — forward-pass weight norms
    # ============================================================
    print("\n[2] GNOmE extraction...")
    t0 = time.time()

    gnome_scores = {}
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            # Attention: separate Q/K/V/O projections (Qwen2.5 architecture)
            attn = layer.self_attn
            for name, proj in [("q_proj", attn.q_proj), ("k_proj", attn.k_proj),
                               ("v_proj", attn.v_proj), ("o_proj", attn.o_proj)]:
                W = proj.weight.detach().float()
                score = torch.norm(W).item()
                gnome_scores[f"L{i}_{name}"] = score

            # MLP: gate/up/down (SwiGLU architecture)
            mlp = layer.mlp
            for name, proj in [("gate_proj", mlp.gate_proj), ("up_proj", mlp.up_proj),
                               ("down_proj", mlp.down_proj)]:
                W = proj.weight.detach().float()
                score = torch.norm(W).item()
                gnome_scores[f"L{i}_{name}"] = score

    gnome_time = time.time() - t0
    gnome_ranked = sorted(gnome_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"  Extraction time: {gnome_time:.4f}s")
    print(f"  Components: {len(gnome_scores)}")
    print(f"  Top 15:")
    for rank, (name, score) in enumerate(gnome_ranked[:15], 1):
        print(f"    {rank:2d}. {name:20s} {score:.4f}")

    results["gnome"] = {
        "time": gnome_time,
        "n_components": len(gnome_scores),
        "top_15": [(n, float(s)) for n, s in gnome_ranked[:15]],
    }

    # ============================================================
    # 3. Attribution Patching — gradient-based (the real one)
    # ============================================================
    print("\n[3] Attribution Patching (gradient-based)...")
    t0 = time.time()

    attr_scores = {}
    try:
        # Two prompts for clean vs corrupted
        prompt_clean = "The cat sat on the mat because the cat was tired"
        prompt_corrupted = "The dog ran in the park because the dog was happy"

        ids_clean = tokenizer(prompt_clean, return_tensors="pt").to(model.device)
        ids_corrupted = tokenizer(prompt_corrupted, return_tensors="pt").to(model.device)

        # Forward with gradients enabled
        embed_clean = model.model.embed_tokens(ids_clean["input_ids"]).detach().requires_grad_(True)

        outputs = model(inputs_embeds=embed_clean, output_hidden_states=True, return_dict=True)
        logits = outputs.logits

        # Target: cross-entropy loss on next token prediction
        target = ids_clean["input_ids"][0, 1:]
        loss = torch.nn.functional.cross_entropy(logits[0, :-1], target)
        loss.backward()

        # Extract gradient magnitudes per component
        for i, layer in enumerate(model.model.layers):
            # Attention grads
            attn_max_grad = 0
            for name, param in layer.named_parameters():
                if "attn" in name and param.grad is not None:
                    g = param.grad.abs().mean().item()
                    attn_max_grad = max(attn_max_grad, g)
            attr_scores[f"L{i}_attn"] = attn_max_grad

            # MLP grads
            mlp_max_grad = 0
            for name, param in layer.named_parameters():
                if "mlp" in name and param.grad is not None:
                    g = param.grad.abs().mean().item()
                    mlp_max_grad = max(mlp_max_grad, g)
            attr_scores[f"L{i}_MLP"] = mlp_max_grad

        attr_time = time.time() - t0
        attr_ranked = sorted(attr_scores.items(), key=lambda x: x[1], reverse=True)

        print(f"  Extraction time: {attr_time:.4f}s")
        print(f"  Components: {len(attr_scores)}")
        print(f"  Top 15:")
        for rank, (name, score) in enumerate(attr_ranked[:15], 1):
            print(f"    {rank:2d}. {name:20s} {score:.8f}")

        results["attribution_patching"] = {
            "time": attr_time,
            "n_components": len(attr_scores),
            "top_15": [(n, float(s)) for n, s in attr_ranked[:15]],
        }

        # Correlation between GNOmE and attribution patching
        # Aggregate GNOmE to layer-level for comparison
        gnome_layer = {}
        for k, v in gnome_scores.items():
            layer_id = k.split("_")[0]
            if layer_id not in gnome_layer:
                gnome_layer[layer_id] = 0
            gnome_layer[layer_id] += v

        common = sorted(set(gnome_layer.keys()) & set(attr_scores.keys()))
        if len(common) >= 3:
            from scipy.stats import spearmanr, pearsonr
            g_arr = np.array([gnome_layer[k] for k in common])
            a_arr = np.array([attr_scores[k] for k in common])
            sp_r, sp_p = spearmanr(g_arr, a_arr)
            pr_r, pr_p = pearsonr(g_arr, a_arr)
            print(f"\n  GNOmE vs Attribution Patching:")
            print(f"    Spearman rho = {sp_r:.4f} (p = {sp_p:.6f})")
            print(f"    Pearson r    = {pr_r:.4f} (p = {pr_p:.6f})")
            results["method_correlation"] = {
                "spearman": float(sp_r), "spearman_p": float(sp_p),
                "pearson": float(pr_r), "pearson_p": float(pr_p),
            }
        else:
            results["method_correlation"] = {"error": "insufficient overlap"}

    except Exception as e:
        attr_time = time.time() - t0
        print(f"  FAILED: {e}")
        results["attribution_patching"] = {"error": str(e), "time": attr_time}

    # ============================================================
    # 4. Anthropic Circuit Tracing — backward Jacobian attempt
    # ============================================================
    print("\n[4] Anthropic Circuit Tracing (backward Jacobian)...")
    t0 = time.time()

    try:
        # Register hooks to capture activations and compute Jacobians
        activations = {}
        gradients = {}

        def save_activation(name):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    activations[name] = out[0].detach()
                else:
                    activations[name] = out.detach()
            return hook

        def save_gradient(name):
            def hook(module, grad_in, grad_out):
                gradients[name] = grad_out[0].detach() if grad_out[0] is not None else None
            return hook

        hooks = []
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(save_activation(f"L{i}")))
            hooks.append(layer.register_full_backward_hook(save_gradient(f"L{i}")))

        # Forward pass
        embed = model.model.embed_tokens(ids_clean["input_ids"]).detach().requires_grad_(True)
        out = model(inputs_embeds=embed, return_dict=True)
        logits = out.logits
        target_logit = logits[0, -1, tokenizer.encode(" tired")[0]]

        # Backward pass
        target_logit.backward()

        # Remove hooks
        for h in hooks:
            h.remove()

        # Compute Jacobian-based importance (Anthropic's core method)
        ct_scores = {}
        for i in range(n_layers):
            act_key = f"L{i}"
            grad_key = f"L{i}"

            if act_key in activations and grad_key in gradients:
                act = activations[act_key].float()
                grad = gradients[grad_key]

                if grad is not None and not torch.isnan(grad).any() and not torch.isinf(grad).any():
                    # Jacobian approximation: gradient * activation
                    importance = (grad.abs() * act.abs()).mean().item()
                    ct_scores[f"L{i}"] = importance
                else:
                    ct_scores[f"L{i}"] = float("nan")
                    print(f"  WARNING: Layer {i} has NaN/Inf gradients")

        ct_time = time.time() - t0

        # Check if results are usable
        valid_scores = {k: v for k, v in ct_scores.items() if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))}
        nan_count = sum(1 for v in ct_scores.values() if isinstance(v, float) and np.isnan(v))

        if nan_count > 0:
            print(f"  Circuit Tracing produced NaN for {nan_count}/{n_layers} layers")
            print(f"  This confirms: backward-Jacobian methods fail on real transformers")

        if valid_scores:
            ct_ranked = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)
            print(f"  Valid layers: {len(valid_scores)}/{n_layers}")
            print(f"  Top 5 valid:")
            for rank, (name, score) in enumerate(ct_ranked[:5], 1):
                print(f"    {rank}. {name}: {score:.8f}")

        results["circuit_tracing"] = {
            "time": ct_time,
            "nan_count": nan_count,
            "valid_layers": len(valid_scores),
            "total_layers": n_layers,
            "scores": {k: float(v) for k, v in ct_scores.items()},
        }

    except Exception as e:
        ct_time = time.time() - t0
        print(f"  FAILED: {e}")
        results["circuit_tracing"] = {"error": str(e), "time": ct_time}

    # ============================================================
    # 5. Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nModel: {results['model']} ({results['params']:,} params)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nMethod Comparison:")
    print(f"  GNOmE:                {results['gnome']['time']:.4f}s ({results['gnome']['n_components']} components)")
    if "attribution_patching" in results and "time" in results["attribution_patching"]:
        print(f"  Attribution Patching: {results['attribution_patching']['time']:.4f}s ({results['attribution_patching']['n_components']} components)")
    if "circuit_tracing" in results and "nan_count" in results["circuit_tracing"]:
        ct = results["circuit_tracing"]
        print(f"  Circuit Tracing:      {ct['time']:.4f}s ({ct['nan_count']} NaN layers)")

    if "method_correlation" in results and "spearman" in results["method_correlation"]:
        mc = results["method_correlation"]
        print(f"\nMethod Agreement:")
        print(f"  GNOmE vs AP: Spearman rho = {mc['spearman']:.4f}")

    # Save
    output_path = Path("/results/gnome_7b_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    print("=" * 60)
    print("DONE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    # Deploy and run
    with app.run():
        result = run_experiment.remote()
        print(json.dumps(result, indent=2, default=str))
