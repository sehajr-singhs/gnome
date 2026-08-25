#!/usr/bin/env python3
"""
GNOmE NMI: Zero-Query IOI Circuit Recovery on GPT-2 Small

Fixes the causal-IOI experiment that was stuck at 0/7 recovery. The prior
approach ranked units by graph centrality over |clean - corrupted| vectors,
which washes out the directional logit signal that defines IOI units.

This version uses mechanism-aware zero-query signals, all computable from a
SINGLE forward pass (no model interventions, no backward pass):

  * logit-diff score:  per-head last-token output projected on the
    unembedding direction u(S2) - u(S1). This is the standard IOI
    attribution (Wang et al. 2023) and identifies name movers (+) and
    negative name movers (-).
  * duplicate-token attention: attention weight from the last token to the
    first-S1 position. Identifies duplicate token heads.
  * S2 attention: attention weight from the last token to the S2 position.
    Identifies name movers and induction heads.

Ground truth: full activation patching over all 156 units (denoising
direction). This lets us report the correlation between the zero-query
score and the causal ground truth on GPT-2 scale, the headline NMI number.

Also reproduces the graph-centrality threshold sweep from the previous
session for comparison.
"""

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gpt2_manual import GPT2, load_tokenizer  # noqa: E402

OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "results"))

# Wang et al. (2023) GPT-2 Small IOI circuit components
KNOWN_IOI = {
    "duplicate_token": ["L8_H0", "L9_H6", "L9_H9"],
    "s_inhibition": ["L8_H1"],
    "name_mover": ["L10_H0"],
    "induction_head": ["L5_H1", "L6_H9"],
    "backup_name_mover": ["L9_H0", "L9_H1", "L10_H4", "L11_H10"],
    "negative_name_mover": ["L10_H7", "L11_H9"],
    "previous_token": ["L4_H11", "L5_H5"],
}

S1_NAMES = ["John", "Alice", "Sarah", "Emma", "Linda", "Karen",
            "Lisa", "Nancy", "Betty", "Sophia", "Rachel", "Laura"]
S2_NAMES = ["Mary", "Bob", "David", "James", "Michael", "William",
            "Thomas", "Richard", "Charles", "Daniel", "George", "Henry"]
DISTRACT = ["Susan", "Tom", "Mark", "Peter", "Robert", "Edward",
            "Steven", "Andrew", "Joseph", "Paul", "Frank", "Scott"]
PLACES = ["store", "park", "school", "office", "garden", "library",
          "cafe", "museum", "hotel", "market", "station", "bank"]


def build_prompts():
    """Canonical IOI templates: [S1] and [S2] went to the [place]. [S1] and ___"""
    clean, corrupt = [], []
    for i, s1 in enumerate(S1_NAMES):
        s2 = S2_NAMES[i]
        s2c = DISTRACT[i]
        place = PLACES[i]
        clean.append(f"{s1} and {s2} went to the {place}. {s1} and")
        corrupt.append(f"{s1} and {s2c} went to the {place}. {s1} and")
    return clean, corrupt


def tokenize(tokenizer, texts):
    return [tokenizer.encode(t).ids for t in texts]


def logit_diff(logits, s2_ids, s1_ids):
    """logit(S2) - logit(S1) at the final position. logits (B, V)."""
    B = logits.shape[0]
    ld = []
    for b in range(B):
        ld.append(logits[b, s2_ids[b]] - logits[b, s1_ids[b]])
    return torch.stack(ld)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    if x.std() < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / x.std()


def validate(importance, n_units, verbose=True, label=""):
    imp_map = {}
    for i, uid in enumerate(unit_ids):
        imp_map[uid] = float(importance[i])
    ranked = sorted(imp_map.items(), key=lambda kv: kv[1], reverse=True)
    validation = {}
    recovered = total = 0
    for role, names in KNOWN_IOI.items():
        ranks = [next((j + 1 for j, (n, _) in enumerate(ranked) if n == name),
                      n_units + 1) for name in names]
        mean_rank = float(np.mean(ranks))
        passed = mean_rank < n_units * 0.25
        validation[role] = {
            "units": names, "ranks": ranks, "mean_rank": mean_rank,
            "percentile": 100.0 * mean_rank / n_units, "pass": passed,
        }
        total += 1
        recovered += int(passed)
    if verbose:
        print(f"\n  [{label}] IOI recovery: {recovered}/{total}")
        for role, info in validation.items():
            mark = "PASS" if info["pass"] else "FAIL"
            print(f"    {role:<22s} mean rank {info['mean_rank']:6.1f}/{n_units}"
                  f"  ({info['percentile']:5.1f}%)  {mark}")
    return validation, recovered, total


def threshold_sweep(causal_vecs, unit_meta, n_layers, upl):
    """Graph-centrality sweep from the previous session (for comparison)."""
    n_units = len(causal_vecs)
    norms = np.linalg.norm(causal_vecs, axis=1, keepdims=True).clip(min=1e-10)
    vecs = causal_vecs / norms
    results = []
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
        adj = np.zeros((n_units, n_units), dtype=np.float32)
        for k in range(n_layers - 1):
            a = vecs[k * upl:(k + 1) * upl]
            b = vecs[(k + 1) * upl:(k + 2) * upl]
            J = np.abs(a @ b.T)
            nz = J[J > 1e-8]
            if nz.size == 0:
                continue
            thr = thresh * float(nz.mean())
            mask = J >= thr
            adj[k * upl:(k + 1) * upl, (k + 1) * upl:(k + 2) * upl] = np.where(mask, J, 0)
        n_edges = int((adj > 0).sum())
        # degree + eigenvector centrality (combined)
        deg = adj.sum(axis=0) + adj.sum(axis=1)
        if deg.max() > 0:
            deg = deg / deg.max()
        ev = np.ones(n_units) / n_units
        col_sums = adj.sum(axis=0)
        col_sums = np.where(col_sums > 0, col_sums, 1.0)
        adj_n = adj / col_sums[None, :]
        for _ in range(100):
            ev_new = adj_n.T @ ev
            if np.abs(ev_new - ev).max() < 1e-12:
                break
            ev = ev_new / (ev_new.sum() + 1e-12)
        if ev.max() > 0:
            ev = ev / ev.max()
        combined = 0.5 * deg + 0.5 * ev
        if combined.max() > 0:
            combined = combined / combined.max()
        val, rec, tot = validate(combined, n_units, verbose=False)
        results.append({
            "tau": thresh, "n_edges": n_edges,
            "density": float(n_edges / max(n_units * (n_layers - 1) * upl, 1)),
            "recovery": rec, "total": tot,
            "validation": {k: v["mean_rank"] for k, v in val.items()},
        })
    return results


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 64)
    print("  GNOmE NMI: Zero-Query IOI Circuit Recovery (GPT-2 Small)")
    print("=" * 64)

    # ---- 1. load model from cache (no transformers import) ----
    print("\nLoading GPT-2 Small from HF cache...")
    model = GPT2.from_cache("openai-community/gpt2")
    snapshot = None
    import glob as _glob
    for cand in _glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/*")):
        if os.path.isfile(os.path.join(cand, "tokenizer.json")):
            snapshot = cand
            break
    tok = load_tokenizer(snapshot)
    n_layers, n_heads = model.n_layer, model.n_head
    upl = n_heads + 1
    n_units = n_layers * upl
    unit_ids = []
    for l in range(n_layers):
        for h in range(n_heads):
            unit_ids.append(f"L{l}_H{h}")
        unit_ids.append(f"L{l}_MLP")
    print(f"  {n_layers} layers x {n_heads} heads + {n_layers} MLP = {n_units} units")

    # ---- 2. prompts ----
    clean_txt, corrupt_txt = build_prompts()
    clean_ids = tokenize(tok, clean_txt)
    corrupt_ids = tokenize(tok, corrupt_txt)
    B = len(clean_ids)
    S = max(len(x) for x in clean_ids)
    maxlen = max(S, max(len(x) for x in corrupt_ids))
    print(f"  {B} IOI prompt pairs, seq len {maxlen}")

    def pad(ids_list, pad_id):
        out = torch.full((len(ids_list), maxlen), pad_id, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            out[i, :len(ids)] = torch.tensor(ids)
        return out

    pad_id = 50256  # <|endoftext|>
    clean_t = pad(clean_ids, pad_id)
    corrupt_t = pad(corrupt_ids, pad_id)

    # S1/S2 ids and positions. S1 appears twice per prompt, once at the
    # sentence start (no leading space, a different BPE token) and once
    # mid-sequence (space-prefixed). Track both so duplicate-token attention
    # is measured from the second S1 back to the first S1.
    s1_ids = []
    s2_ids = []
    pos_first_s1 = []
    pos_second_s1 = []
    pos_s2 = []
    for i, (cids, c2ids) in enumerate(zip(clean_ids, corrupt_ids)):
        s1 = S1_NAMES[i]
        s2 = S2_NAMES[i]
        s1_tok = tok.encode(f" {s1}").ids[0]
        s2_tok = tok.encode(f" {s2}").ids[0]
        s1_ids.append(s1_tok)
        s2_ids.append(s2_tok)
        arr = np.array(cids)
        s1_ns = tok.encode(s1).ids[0]
        p1 = np.where((arr == s1_tok) | (arr == s1_ns))[0]
        pos_first_s1.append(int(p1[0]) if len(p1) else 0)
        pos_second_s1.append(int(p1[-1]) if len(p1) else int(p1[0]))
        s2_ns = tok.encode(s2).ids[0]
        p2 = np.where((arr == s2_tok) | (arr == s2_ns))[0]
        pos_s2.append(int(p2[0]) if len(p2) else 1)

    # ---- 3. clean forward: zero-query signals ----
    print("\nSingle clean forward pass (zero model queries)...")
    with torch.no_grad():
        clean = model.forward(clean_t, want_probs=True, want_full=True)
        corr = model.forward(corrupt_t, want_probs=True)

    u_S2 = model.lm_head[s2_ids].numpy()   # (B, D)
    u_S1 = model.lm_head[s1_ids].numpy()
    dir_ld = u_S2 - u_S1

    head_last = clean["head_last"].numpy()  # (B, L, H, D)
    mlp_last = clean["mlp_last"].numpy()    # (B, L, D)

    ld = np.zeros(n_units)
    dup = np.zeros(n_units)
    s2attn = np.zeros(n_units)
    prev = np.zeros(n_units)
    for l in range(n_layers):
        for h in range(n_heads):
            u = l * upl + h
            contrib = head_last[:, l, h, :]                      # (B, D)
            ld[u] = float(np.mean(np.einsum("bd,bd->b", contrib, dir_ld)))
            p = clean["probs"][l].numpy()[:, h, :, :]            # (B, S, S)
            for b in range(B):
                # duplicate-token heads attend from the second S1 to the first
                dup[u] += float(p[b, pos_second_s1[b], pos_first_s1[b]])
                # name movers attend from the last token to S2
                s2attn[u] += float(p[b, maxlen - 1, pos_s2[b]])
                # previous-token heads attend from the last token to the one before
                prev[u] += float(p[b, maxlen - 1, maxlen - 2])
            dup[u] /= B; s2attn[u] /= B; prev[u] /= B
        u = l * upl + n_heads
        contrib = mlp_last[:, l, :]
        ld[u] = float(np.mean(np.einsum("bd,bd->b", contrib, dir_ld)))

    # z-score |logit-diff| WITHIN each layer so the growing residual-stream
    # scale does not crowd out mid-layer heads, then add the attention
    # signatures (already normalized probabilities, compared globally)
    ld_layer = np.zeros(n_units)
    for l in range(n_layers):
        seg = np.abs(ld[l * upl:(l + 1) * upl])
        ld_layer[l * upl:(l + 1) * upl] = zscore(seg)
    combined = ld_layer + zscore(dup) + zscore(s2attn)

    # ---- 4. ground truth: activation patching over all units ----
    print("\nActivation patching ground truth (156 corrupted forwards)...")
    ld_corr = logit_diff(corr["last_logits"], s2_ids, s1_ids).numpy()
    imp_patch = np.zeros(n_units)
    with torch.no_grad():
        for l in range(n_layers):
            for h in range(n_heads):
                key = f"L{l}_H{h}"
                patch = {key: clean["head_full"][:, l, h, :, :]}
                out = model.forward(corrupt_t, patch=patch)
                ld_p = logit_diff(out["last_logits"], s2_ids, s1_ids).numpy()
                imp_patch[l * upl + h] = float(np.mean(ld_p - ld_corr))
            key = f"L{l}_MLP"
            patch = {key: clean["mlp_full"][:, l, :, :]}
            out = model.forward(corrupt_t, patch=patch)
            ld_p = logit_diff(out["last_logits"], s2_ids, s1_ids).numpy()
            imp_patch[l * upl + n_heads] = float(np.mean(ld_p - ld_corr))

    # ---- 5. validation ----
    print("\n" + "=" * 64)
    gt = np.abs(imp_patch)  # causal importance = magnitude of patching effect
    val_zq, rec_zq, tot = validate(combined, n_units, label="Zero-query")
    val_patch, rec_patch, _ = validate(gt, n_units, label="Patching GT")
    val_ld, rec_ld, _ = validate(np.abs(ld), n_units, label="|logit-diff| only")
    val_dup, rec_dup, _ = validate(dup, n_units, label="dup attn only")
    val_s2, rec_s2, _ = validate(s2attn, n_units, label="S2 attn only")

    # correlations vs ground truth
    def pearson_corr(x, y):
        x = np.asarray(x); y = np.asarray(y)
        return float(np.corrcoef(x, y)[0, 1])

    from scipy.stats import spearmanr as _spearman
    r_pearson = pearson_corr(combined, gt)
    r_spearman = float(_spearman(combined, gt).statistic)
    r_ld = pearson_corr(np.abs(ld), gt)

    # ---- 6. threshold sweep (previous session's approach) ----
    print("\nGraph-centrality threshold sweep (comparison)...")
    # one |clean - corrupt| activation vector per unit, in unit_ids order
    # (per layer: 12 heads then 1 MLP), averaged over the batch
    hd = np.abs(head_last - corr["head_last"].numpy()).mean(axis=0)
    hd = hd.reshape(n_layers * n_heads, -1)                            # (144, D)
    ml = np.abs(mlp_last - corr["mlp_last"].numpy()).mean(axis=0)     # (12, D)
    parts = []
    for l in range(n_layers):
        parts.append(hd[l * n_heads:(l + 1) * n_heads])
        parts.append(ml[l:l + 1])
    causal_delta = np.concatenate(parts, axis=0)                       # (156, D)
    sweep = threshold_sweep(causal_delta, unit_ids, n_layers, upl)

    total_time = time.time() - t0
    print(f"\n{'='*64}")
    print(f"  Zero-query recovery:   {rec_zq}/{tot}   (within-layer |ld|+dup+s2 attn)")
    print(f"  Patching GT recovery:  {rec_patch}/{tot}")
    print(f"  |logit-diff| only:     {rec_ld}/{tot}")
    print(f"  dup attn only:         {rec_dup}/{tot}")
    print(f"  S2 attn only:          {rec_s2}/{tot}")
    print(f"  Pearson r (zero-query vs patching GT): {r_pearson:.4f}")
    print(f"  Spearman rho:                          {r_spearman:.4f}")
    print(f"  |logit-diff| vs GT:                    {r_ld:.4f}")
    print(f"  Total time: {total_time:.0f}s")

    # ---- 7. save ----
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {
        "method": "zero_query_logitdiff_plus_attention",
        "model": "gpt2-small", "n_units": n_units,
        "n_layers": n_layers, "n_heads": n_heads,
        "n_prompt_pairs": B,
        "recovery_zero_query": {"recovered": rec_zq, "total": tot,
                                "rate": rec_zq / max(tot, 1)},
        "recovery_patching_gt": {"recovered": rec_patch, "total": tot,
                                 "rate": rec_patch / max(tot, 1)},
        "recovery_logitdiff_only": {"recovered": rec_ld, "total": tot,
                                    "rate": rec_ld / max(tot, 1)},
        "recovery_dup_attn_only": {"recovered": rec_dup, "total": tot,
                                   "rate": rec_dup / max(tot, 1)},
        "recovery_s2_attn_only": {"recovered": rec_s2, "total": tot,
                                  "rate": rec_s2 / max(tot, 1)},
        "corr_zeroquery_vs_patching": {"pearson": r_pearson,
                                       "spearman": r_spearman,
                                       "logitdiff_pearson": r_ld},
        "validation_zero_query": {k: v for k, v in val_zq.items()},
        "validation_patching": {k: v for k, v in val_patch.items()},
        "validation_dup_attn": {k: v for k, v in val_dup.items()},
        "validation_s2_attn": {k: v for k, v in val_s2.items()},
        "threshold_sweep_centrality": sweep,
        "top_units_zero_query": sorted(
            ((uid, float(s)) for uid, s in zip(unit_ids, combined)),
            key=lambda kv: kv[1], reverse=True)[:25],
        "top_units_patching": sorted(
            ((uid, float(s)) for uid, s in zip(unit_ids, imp_patch)),
            key=lambda kv: abs(kv[1]), reverse=True)[:25],
        "total_time_s": float(total_time),
    }
    out_path = os.path.join(OUT_DIR, "nmi_ioi_zeroquery.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")
