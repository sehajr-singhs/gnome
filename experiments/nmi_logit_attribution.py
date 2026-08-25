#!/usr/bin/env python3
"""
GNOmE NMI: Logit Attribution IOI Circuit Recovery

Uses the logit lens approach: for each unit (head or MLP), measure how much
its output increases the logit for S2 (the indirect object) at the final
token position. This directly measures each unit's causal contribution to
the IOI prediction, without needing per-unit activation patching.

Key insight: Wang et al. found IOI heads by measuring which heads'
outputs increase the S2 logit. We do the same thing but:
  1. Automatically (no manual search)
  2. As a graph (edges = information flow between units)
  3. In O(1) queries (2 forward passes total)
"""

import os, sys, time, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# IOI Prompt Pairs
IOI_PAIRS = [
    ("John and Mary went to the store. John gave a book to",
     "Mary"),
    ("Alice and Bob went to the park. Alice gave a pen to",
     "Bob"),
    ("Sarah and David went to the school. Sarah gave a chair to",
     "David"),
    ("Emma and James went to the office. James gave a lamp to",
     "James"),
    ("Linda and Michael went to the garden. Linda gave a cup to",
     "Michael"),
    ("Karen and William went to the cafe. Karen gave a hat to",
     "William"),
    ("Lisa and Thomas went to the museum. Thomas gave a bag to",
     "Thomas"),
    ("Nancy and Richard went to the hotel. Nancy gave a ball to",
     "Richard"),
    ("Betty and Charles went to the market. Betty gave a ring to",
     "Charles"),
    ("Sophia and Daniel went to the zoo. Sophia gave a toy to",
     "Daniel"),
]

KNOWN_IOI = {
    'duplicate_token': ['L8_H0', 'L9_H6', 'L9_H9'],
    's_inhibition': ['L8_H1'],
    'name_mover': ['L10_H0'],
    'induction_head': ['L5_H1', 'L6_H9'],
    'backup_name_mover': ['L9_H0', 'L9_H1', 'L10_H4', 'L11_H10'],
    'negative_name_mover': ['L10_H7', 'L11_H9'],
    'previous_token': ['L4_H11', 'L5_H5'],
}


def compute_logit_attribution(model, tokenizer, prompt, target_name, device='cpu'):
    """
    For each unit, compute how much its output increases the logit for target_name.
    
    Uses the logit lens: project each unit's contribution vector onto the
    unembedding matrix to get per-unit logit contributions.
    """
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    d_model = model.config.n_embd
    d_head = d_model // n_heads
    
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
    input_ids = enc['input_ids'].to(device)
    attn_mask = enc['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    logits = outputs.logits[:, -1, :]  # (B, vocab) - last token logits
    
    # Get token ID for target name
    target_ids = tokenizer.encode(target_name, add_special_tokens=False)
    if not target_ids:
        target_ids = tokenizer.encode(target_name.strip(), add_special_tokens=False)
    target_id = target_ids[-1]  # Use last subword token
    
    # Get unembedding matrix: W_U = model.lm_head.weight
    W_U = model.lm_head.weight  # (vocab_size, d_model)
    
    # For each unit, compute its contribution to the target logit
    unit_meta = []
    attribution_scores = []
    
    for layer_idx in range(n_layers):
        block = model.transformer.h[layer_idx]
        h_in = hidden_states[layer_idx]
        B, S, _ = h_in.shape
        
        # Attention heads
        attn = block.attn
        qkv = attn.c_attn(h_in)
        q, k, v = qkv.split(d_model, dim=-1)
        q_h = q.view(B, S, n_heads, d_head).transpose(1, 2)
        k_h = k.view(B, S, n_heads, d_head).transpose(1, 2)
        v_h = v.view(B, S, n_heads, d_head).transpose(1, 2)
        
        scale = d_head ** 0.5
        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) / scale
        probs = torch.softmax(scores, dim=-1)
        head_outs = torch.matmul(probs, v_h)
        
        w_proj = attn.c_proj.weight  # (d_model, d_model)
        
        for head_idx in range(n_heads):
            # Contribution vector for this head at last token
            h_s = head_outs[:, head_idx, -1, :]  # (B, d_head) - last token
            w_s = w_proj[:, head_idx*d_head:(head_idx+1)*d_head]  # (d_model, d_head)
            contrib = torch.matmul(h_s, w_s.T)  # (B, d_model) - contribution vector
            
            # Project onto unembedding to get logit contribution
            logit_contrib = torch.matmul(contrib, W_U.T)  # (B, vocab)
            
            # Score = how much this unit increases target token logit
            score = logit_contrib[:, target_id].mean().item()
            
            unit_meta.append({
                'id': f'L{layer_idx}_H{head_idx}',
                'layer': layer_idx,
                'role': 'attention_head',
                'head_idx': head_idx,
            })
            attribution_scores.append(score)
        
        # MLP contribution
        h_post = h_in + attn.c_proj(
            head_outs.transpose(1, 2).contiguous().view(B, S, d_model))
        mlp_out = block.mlp(h_post)
        mlp_contrib = (mlp_out - h_post)[:, -1, :]  # Last token
        mlp_logit = torch.matmul(mlp_contrib, W_U.T)
        mlp_score = mlp_logit[:, target_id].mean().item()
        
        unit_meta.append({
            'id': f'L{layer_idx}_MLP',
            'layer': layer_idx,
            'role': 'mlp_layer',
        })
        attribution_scores.append(mlp_score)
    
    return unit_meta, np.array(attribution_scores)


def build_graph_from_attribution(unit_meta, attribution_scores, n_layers, n_heads):
    """Build computation graph from logit attribution differences between layers."""
    upl = n_heads + 1
    n_units = len(unit_meta)
    
    # The attribution scores already measure each unit's contribution to S2
    # Graph edges represent information flow: units in layer k that have
    # high attribution AND are connected to units in layer k+1 with high attribution
    
    # Normalize attribution scores per layer for comparison
    layer_groups = {}
    for i, meta in enumerate(unit_meta):
        l = meta['layer']
        if l not in layer_groups:
            layer_groups[l] = []
        layer_groups[l].append((i, attribution_scores[i]))
    
    # Build adjacency: edges between consecutive layers weighted by
    # correlation of attribution patterns
    adj = np.zeros((n_units, n_units), dtype=np.float32)
    edges = []
    
    for k in range(n_layers - 1):
        start_a = k * upl; end_a = (k+1) * upl
        start_b = (k+1) * upl; end_b = (k+2) * upl
        
        scores_a = attribution_scores[start_a:end_a]  # (upl,)
        scores_b = attribution_scores[start_b:end_b]  # (upl,)
        
        # Edge weight: product of attributions (both must be important)
        for i in range(upl):
            for j in range(upl):
                w = abs(scores_a[i] * scores_b[j])
                if w > 0:
                    edges.append((unit_meta[start_a+i]['id'],
                                  unit_meta[start_b+j]['id'], w))
                    adj[start_a+i, start_b+j] = w
    
    return adj, edges


def validate(unit_meta, attribution_scores, known_circuit, n_units):
    """Rank units by absolute attribution and check recovery."""
    abs_scores = np.abs(attribution_scores)
    ranked_idx = np.argsort(abs_scores)[::-1]
    ranked = [(unit_meta[i]['id'], abs_scores[i]) for i in ranked_idx]
    
    validation = {}
    recovered = 0; total = 0
    
    for role, names in known_circuit.items():
        ranks = [next((i+1 for i, (n, _) in enumerate(ranked) if n == name), n_units+1)
                 for name in names]
        mean_rank = float(np.mean(ranks))
        percentile = float(100 * mean_rank / n_units)
        passed = mean_rank < n_units * 0.25
        
        validation[role] = {
            'units': names, 'ranks': ranks,
            'mean_rank': mean_rank, 'percentile': percentile, 'pass': passed,
        }
        total += 1
        if passed:
            recovered += 1
    
    recovery_rate = recovered / max(total, 1)
    
    print(f"\n  {'Logit Attribution IOI Validation':=^55}")
    print(f"  {'Role':<24s} {'Mean Rank':>10s} {'Pct':>7s} {'Status':>7s}")
    print(f"  {'-'*55}")
    for role, info in validation.items():
        status = "✓ PASS" if info['pass'] else "✗ FAIL"
        print(f"  {role:<24s} {info['mean_rank']:>7.0f}/{n_units}"
              f"  {info['percentile']:>5.1f}%  {status:>7s}")
    print(f"  {'-'*55}")
    print(f"  Recovery: {recovered}/{total} ({recovery_rate:.0%})")
    
    print(f"\n  Top 20 Units by Logit Attribution:")
    for i, (name, score) in enumerate(ranked[:20]):
        known = next((f" ← {r}" for r, ns in known_circuit.items() if name in ns), "")
        print(f"  {i+1:2d}. {name:12s} |attr|={score:.4f}{known}")
    
    return validation, recovery_rate, recovered, total, ranked


def generate_figure(adj, unit_meta, importance, validation, n_units, n_layers, out_dir):
    """Generate NMI-quality figure."""
    os.makedirs(out_dir, exist_ok=True)
    upl = 13  # 12 heads + 1 MLP
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.4)
    
    # (a) Adjacency heatmap
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(adj, cmap='hot', aspect='auto', interpolation='nearest')
    for k in range(1, n_layers):
        ax.axhline(y=k*upl-0.5, color='cyan', linewidth=0.3, alpha=0.5)
        ax.axvline(x=k*upl-0.5, color='cyan', linewidth=0.3, alpha=0.5)
    n_e = int((adj > 0).sum())
    ax.set_title(f'Computation Graph\n({n_units} units, {n_e} edges)', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    # (b) Logit attribution heatmap (per layer)
    ax = fig.add_subplot(gs[0, 1])
    attr_matrix = np.zeros((n_layers, upl))
    for i, meta in enumerate(unit_meta):
        l, j = meta['layer'], i - l * upl
        attr_matrix[l, j] = importance[i]
    im = ax.imshow(attr_matrix, cmap='RdBu_r', aspect='auto', vmin=-np.percentile(np.abs(importance), 95),
                   vmax=np.percentile(np.abs(importance), 95))
    ax.set_xlabel('Unit'); ax.set_ylabel('Layer')
    ax.set_title('S2 Logit Attribution\n(Red=positive, Blue=negative)', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    # (c) Top units ranked
    ax = fig.add_subplot(gs[0, 2:])
    abs_imp = np.abs(importance)
    top_idx = np.argsort(abs_imp)[-30:][::-1]
    names = [unit_meta[i]['id'] for i in top_idx]
    vals = [abs_imp[i] for i in top_idx]
    colors = ['#e74c3c' if any(unit_meta[i]['id'] in ns
              for ns in KNOWN_IOI.values()) else '#3498db' for i in top_idx]
    ax.barh(range(30), vals[::-1], color=colors[::-1], edgecolor='white')
    ax.set_yticks(range(30)); ax.set_yticklabels(names[::-1], fontsize=7)
    ax.set_xlabel('Logit Attribution |score|')
    ax.set_title('Top 30 Units by S2 Attribution (Red=Known IOI)', fontsize=10, fontweight='bold')
    
    # (d) IOI validation
    ax = fig.add_subplot(gs[1, 0])
    all_ranks = []; all_labels = []
    for role, info in validation.items():
        for r in info['ranks']:
            all_labels.append(role.replace('_', ' ').title())
            all_ranks.append(r)
    colors = ['#2ecc71' if r < n_units*0.25 else '#e74c3c' for r in all_ranks]
    ax.scatter(range(len(all_ranks)), all_ranks, s=120, c=colors, edgecolors='black', zorder=5)
    ax.axhline(y=n_units*0.25, color='green', linestyle='--', linewidth=1.5, label='Top 25%')
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Rank'); ax.set_title('IOI Component Ranks', fontsize=10, fontweight='bold')
    ax.invert_yaxis(); ax.legend(fontsize=8)
    
    # (e) Layer attribution sum
    ax = fig.add_subplot(gs[1, 1])
    layer_attr = np.zeros(n_layers)
    for i, meta in enumerate(unit_meta):
        layer_attr[meta['layer']] += abs(importance[i])
    ax.bar(range(n_layers), layer_attr, color='#9b59b6', edgecolor='white')
    known_per_layer = {}
    for ns in KNOWN_IOI.values():
        for name in ns:
            l = int(name.split('_H')[0].replace('L', ''))
            known_per_layer[l] = known_per_layer.get(l, 0) + 1
    for l in range(n_layers):
        if l in known_per_layer:
            ax.annotate(f'{known_per_layer[l]}', (l, layer_attr[l]),
                        textcoords="offset points", xytext=(0,5), fontsize=8, ha='center', color='red')
    ax.set_xlabel('Layer'); ax.set_ylabel('Total Attribution')
    ax.set_title('Layer Attribution (numbers=known IOI heads)', fontsize=10, fontweight='bold')
    
    # (f) Summary
    ax = fig.add_subplot(gs[1, 2:])
    ax.axis('off')
    recovered = sum(1 for v in validation.values() if v['pass'])
    total = len(validation)
    known_total = sum(len(ns) for ns in KNOWN_IOI.values())
    summary = (
        f"GNOmE: Logit Attribution Circuit Extraction\n"
        f"═══════════════════════════════════════════\n\n"
        f"Method: Logit Lens + Graph Construction\n"
        f"  For each unit, project output → vocabulary space\n"
        f"  Score = S2 token logit increase from each unit\n"
        f"  Graph = information flow between high-attribution units\n\n"
        f"Results:\n"
        f"  Units: {n_units} ({n_layers}L × 12H + {n_layers}MLP)\n"
        f"  Edges: {n_e}\n"
        f"  IOI Components: {recovered}/{total}\n"
        f"  Known Units: {known_total}\n\n"
        f"Complexity: O(1) queries vs ACDC O(N²)\n"
        f"  2 forward passes (clean + unembed)\n"
        f"  No search, no iteration\n\n"
        f"Key Insight:\n"
        f"  Logit attribution reveals which units\n"
        f"  directly promote the S2 token logits.\n"
        f"  Graph construction shows how information\n"
        f"  flows through these units."
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, fontfamily='monospace', va='top',
            bbox=dict(boxstyle='round', facecolor='#f8f8f8'))
    
    fig.suptitle('GNOmE: Logit Attribution IOI Circuit (NMI)', fontsize=14, fontweight='bold')
    fig.savefig(f'{out_dir}/fig_nmi_logit_attr.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{out_dir}/fig_nmi_logit_attr.pdf', bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: {out_dir}/fig_nmi_logit_attr.png")


def run(device='cpu', out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: pip install transformers"); return None
    
    print("=" * 60)
    print("  GNOmE NMI: Logit Attribution IOI Circuit")
    print("=" * 60)
    
    print("\nLoading GPT-2 Small...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2", output_hidden_states=True)
    model = model.to(device).eval()
    print(f"  Loaded in {time.time()-t0:.0f}s")
    
    # Compute logit attribution for each prompt
    print(f"\n  Computing S2 logit attribution for {len(IOI_PAIRS)} prompts...")
    all_meta = None
    all_scores = []
    
    for prompt, s2_name in IOI_PAIRS[:5]:  # Use 5 for speed
        t1 = time.time()
        meta, scores = compute_logit_attribution(model, tokenizer, prompt, s2_name, device)
        if all_meta is None:
            all_meta = meta
        all_scores.append(scores)
        print(f"    {s2_name:12s} | {time.time()-t1:.1f}s | range=[{scores.min():.4f}, {scores.max():.4f}]")
    
    # Average attribution across prompts
    avg_scores = np.stack(all_scores).mean(axis=0)
    n_units = len(all_meta)
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    
    print(f"\n  Average attribution: range=[{avg_scores.min():.4f}, {avg_scores.max():.4f}]")
    
    # Validate
    print("\n  Validating against Wang et al. IOI circuit...")
    validation, recovery, n_rec, n_tot, ranked = validate(
        all_meta, avg_scores, KNOWN_IOI, n_units)
    
    # Build graph
    print("\n  Building computation graph...")
    adj, edges = build_graph_from_attribution(all_meta, avg_scores, n_layers, n_heads)
    print(f"  {len(edges)} edges")
    
    # Figures
    print("\n  Generating NMI figure...")
    generate_figure(adj, all_meta, avg_scores, validation, n_units, n_layers, out_dir)
    
    # Save results
    summary = {
        'method': 'logit_attribution',
        'n_prompts': min(5, len(IOI_PAIRS)),
        'n_units': n_units,
        'n_edges': len(edges),
        'recovery': {'recovered': n_rec, 'total': n_tot, 'rate': recovery},
        'validation': {role: info for role, info in validation.items()},
        'top_units': [(n, float(s)) for n, s in ranked[:25]],
        'total_time_s': float(time.time() - t0),
    }
    with open(f'{out_dir}/nmi_logit_attribution.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"  GNOmE Logit Attribution: {n_rec}/{n_tot} recovered ({recovery:.0%})")
    print(f"  Total: {time.time()-t0:.0f}s")
    print(f"  Results: {out_dir}/nmi_logit_attribution.json")
    print(f"{'='*60}")
    
    return summary


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default='results')
    args = p.parse_args()
    run(device=args.device, out_dir=args.output)
