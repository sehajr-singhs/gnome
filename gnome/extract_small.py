"""Extract computation graphs from trained small transformers.

Core idea: the model's forward pass IS a graph. Each attention head and
MLP layer is a node. Edges represent how much each node contributes to
the next layer's residual stream.

This module works with SmallTransformer (trainee.py) at head-level granularity.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_circuit(
    model: nn.Module,
    input_ids: torch.Tensor | None = None,
    vocab_size: int = 64,
    seq_len: int = 12,
    n_samples: int = 256,
    rel_thresh: float = 0.05,
    device: str = "cpu",
) -> dict:
    """Extract circuit graph from a SmallTransformer.

    Nodes:
      - For each layer: L{k}_H{h} for attention heads
      - For each layer: L{k}_MLP for the feed-forward layer

    Edges: cosine similarity between each unit's contribution vector
    to the residual stream, across consecutive layers.
    """
    model = model.to(device).eval()

    if input_ids is None:
        input_ids = torch.randint(0, vocab_size, (n_samples, seq_len), device=device)

    n_heads = model.n_heads
    n_layers = model.n_layers
    d_model = model.d_model
    d_head = d_model // n_heads

    # ---- Hook to capture per-head outputs and MLP outputs ----
    captured = {}  # key -> tensor

    def make_attn_hook(layer_idx):
        def hook_fn(mod, inp, out):
            # inp[0]: input to attn module (B, S, D)
            # out: output of attn module (B, S, D) - after output proj
            x = inp[0]
            B, S, _ = x.shape

            # Recompute Q, K, V, attention, per-head output
            qkv = mod.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, S, n_heads, d_head).transpose(1, 2)
            k = k.view(B, S, n_heads, d_head).transpose(1, 2)
            v = v.view(B, S, n_heads, d_head).transpose(1, 2)

            scale = d_head ** 0.5
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / scale
            attn_probs = F.softmax(attn_scores, dim=-1)
            head_outs = torch.matmul(attn_probs, v)  # (B, H, S, d_head)

            # Per-head contribution = output_proj applied to this head's slice
            w_out = mod.out.weight  # (D, D)
            for h in range(n_heads):
                wh = w_out[:, h * d_head:(h + 1) * d_head]  # (D, d_head)
                # (B, S, d_head) @ (d_head, D) -> (B, S, D)
                contrib = torch.matmul(head_outs[:, h], wh.t())
                captured[(layer_idx, f'L{layer_idx}_H{h}')] = contrib.detach()
        return hook_fn

    def make_ff_hook(layer_idx):
        def hook_fn(mod, inp, out):
            # inp[0]: input to FF module
            # We want the FF's contribution (not the residual)
            # Recompute: ff(ln2(x)) where x is the input
            # But we need the residual input... simpler: capture raw output
            # Actually the FF module's forward returns w2(gelu(w1(x)))
            # which IS the contribution (the residual is added in the block)
            if isinstance(out, tuple):
                out = out[0]
            captured[(layer_idx, f'L{layer_idx}_MLP')] = out.detach()
        return hook_fn

    hooks = []
    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_attn_hook(i)))
        hooks.append(block.ff.register_forward_hook(make_ff_hook(i)))

    with torch.no_grad():
        _ = model(input_ids)

    for h in hooks:
        h.remove()

    # ---- Compute per-unit vectors (mean activation over batch and positions) ----
    unit_names = []
    unit_vectors = []

    for layer_idx in range(n_layers):
        for h in range(n_heads):
            key = (layer_idx, f'L{layer_idx}_H{h}')
            if key in captured:
                vec = captured[key].mean(dim=(0, 1)).cpu().numpy()  # (D,)
                unit_names.append(f'L{layer_idx}_H{h}')
                unit_vectors.append(vec)

        key = (layer_idx, f'L{layer_idx}_MLP')
        if key in captured:
            vec = captured[key].mean(dim=(0, 1)).cpu().numpy()
            unit_names.append(f'L{layer_idx}_MLP')
            unit_vectors.append(vec)

    unit_vectors = np.stack(unit_vectors, axis=0)  # (n_units, D)
    n_units = len(unit_names)

    # ---- Build nodes ----
    nodes = []
    for name in unit_names:
        layer_idx = int(name.split('_')[0][1:])
        role = 'mlp_layer' if 'MLP' in name else 'attention_head'
        nodes.append({"id": name, "layer": layer_idx, "role": role})

    # ---- Build edges between consecutive layers ----
    edges = []
    adj_matrix = np.zeros((n_units, n_units), dtype=np.float32)

    # Group units by layer
    layer_groups = {}
    for idx, name in enumerate(unit_names):
        layer_idx = int(name.split('_')[0][1:])
        if layer_idx not in layer_groups:
            layer_groups[layer_idx] = []
        layer_groups[layer_idx].append(idx)

    sorted_layers = sorted(layer_groups.keys())

    for li in range(len(sorted_layers) - 1):
        src_indices = layer_groups[sorted_layers[li]]
        dst_indices = layer_groups[sorted_layers[li + 1]]

        vecs_src = unit_vectors[src_indices]
        vecs_dst = unit_vectors[dst_indices]

        norms_src = np.linalg.norm(vecs_src, axis=1, keepdims=True).clip(min=1e-8)
        norms_dst = np.linalg.norm(vecs_dst, axis=1, keepdims=True).clip(min=1e-8)

        S = np.abs((vecs_src / norms_src) @ (vecs_dst / norms_dst).T)

        for ii, si in enumerate(src_indices):
            for jj, di in enumerate(dst_indices):
                w = float(S[ii, jj])
                if w >= rel_thresh:
                    edges.append((unit_names[si], unit_names[di], w))
                    adj_matrix[si, di] = w

    print(f"  Circuit: {n_units} nodes, {len(edges)} edges (thresh={rel_thresh})")

    return {
        "nodes": nodes,
        "edges": edges,
        "adj_matrix": adj_matrix,
        "unit_names": unit_names,
        "unit_vectors": unit_vectors,
        "metadata": {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_units": n_units,
            "n_edges": len(edges),
            "thresh": rel_thresh,
        },
    }


def compute_head_importance(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    device: str = "cpu",
) -> dict[str, float]:
    """Compute head importance via zero-ablation.

    For each head, zero out its output contribution and measure
    the increase in cross-entropy loss. Positive = head was helpful.
    """
    model = model.to(device).eval()
    n_heads = model.n_heads
    n_layers = model.n_layers
    d_head = model.d_model // n_heads

    # Baseline loss
    with torch.no_grad():
        logits_base, _ = model(input_ids, targets)
        base_loss = F.cross_entropy(
            logits_base[:, :-1].reshape(-1, logits_base.size(-1)),
            targets[:, 1:].reshape(-1),
        ).item()

    importances = {}

    for layer_idx in range(n_layers):
        attn = model.blocks[layer_idx].attn

        # Save original weights
        w_out_orig = attn.out.weight.data.clone()
        b_out_orig = attn.out.bias.data.clone() if attn.out.bias is not None else None

        for h in range(n_heads):
            with torch.no_grad():
                attn.out.weight.data[:, h * d_head:(h + 1) * d_head] = 0.0
                logits_h, _ = model(input_ids, targets)
                h_loss = F.cross_entropy(
                    logits_h[:, :-1].reshape(-1, logits_h.size(-1)),
                    targets[:, 1:].reshape(-1),
                ).item()

            name = f'L{layer_idx}_H{h}'
            importances[name] = h_loss - base_loss

            # Restore
            with torch.no_grad():
                attn.out.weight.data = w_out_orig.clone()
                if b_out_orig is not None:
                    attn.out.bias.data = b_out_orig.clone()

        # MLP importance
        ff = model.blocks[layer_idx].ff
        w1_orig = ff.w1.weight.data.clone()
        w2_orig = ff.w2.weight.data.clone()

        with torch.no_grad():
            ff.w1.weight.data.zero_()
            ff.w2.weight.data.zero_()
            logits_mlp, _ = model(input_ids, targets)
            mlp_loss = F.cross_entropy(
                logits_mlp[:, :-1].reshape(-1, logits_mlp.size(-1)),
                targets[:, 1:].reshape(-1),
            ).item()

        name = f'L{layer_idx}_MLP'
        importances[name] = mlp_loss - base_loss

        with torch.no_grad():
            ff.w1.weight.data = w1_orig.clone()
            ff.w2.weight.data = w2_orig.clone()

    return importances


def path_patching(
    model: nn.Module,
    input_ids_clean: torch.Tensor,
    input_ids_corrupt: torch.Tensor,
    targets: torch.Tensor,
    device: str = "cpu",
) -> dict[str, float]:
    """Path patching: measure how each head's output contributes
    when the model processes corrupted input.

    For each head, we zero it out during the corrupted run and
    measure the change in loss. This tells us which heads are
    doing useful work on the corrupted input.
    """
    model = model.to(device).eval()
    n_heads = model.n_heads
    n_layers = model.n_layers
    d_head = model.d_model // n_heads

    # Baseline on corrupted input
    with torch.no_grad():
        logits_corrupt, _ = model(input_ids_corrupt, targets)
        base_loss = F.cross_entropy(
            logits_corrupt[:, :-1].reshape(-1, logits_corrupt.size(-1)),
            targets[:, 1:].reshape(-1),
        ).item()

    results = {}

    for layer_idx in range(n_layers):
        attn = model.blocks[layer_idx].attn
        w_out_orig = attn.out.weight.data.clone()

        for h in range(n_heads):
            with torch.no_grad():
                attn.out.weight.data[:, h * d_head:(h + 1) * d_head] = 0.0
                logits_h, _ = model(input_ids_corrupt, targets)
                h_loss = F.cross_entropy(
                    logits_h[:, :-1].reshape(-1, logits_h.size(-1)),
                    targets[:, 1:].reshape(-1),
                ).item()

            name = f'L{layer_idx}_H{h}'
            results[name] = h_loss - base_loss

            with torch.no_grad():
                attn.out.weight.data = w_out_orig.clone()

    return results
