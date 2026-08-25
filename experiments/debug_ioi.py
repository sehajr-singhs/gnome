#!/usr/bin/env python3
"""Debug: dump per-unit scores for the IOI zero-query pipeline and check
tokenization / position bookkeeping that feeds the scores."""
import glob
import os

import numpy as np
import torch

from gpt2_manual import GPT2, load_tokenizer

S1_NAMES = ["John", "Alice", "Sarah", "Emma", "Linda", "Karen",
            "Lisa", "Nancy", "Betty", "Sophia", "Rachel", "Laura"]
S2_NAMES = ["Mary", "Bob", "David", "James", "Michael", "William",
            "Thomas", "Richard", "Charles", "Daniel", "George", "Henry"]
DISTRACT = ["Susan", "Tom", "Mark", "Peter", "Robert", "Edward",
            "Steven", "Andrew", "Joseph", "Paul", "Frank", "Scott"]
PLACES = ["store", "park", "school", "office", "garden", "library",
          "cafe", "museum", "hotel", "market", "station", "bank"]

model = GPT2.from_cache("openai-community/gpt2")
snap = None
for cand in glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/*")):
    if os.path.isfile(os.path.join(cand, "tokenizer.json")):
        snap = cand
        break
tok = load_tokenizer(snap)

# --- tokenization check ---
t0 = tok.encode("John and Mary went to the store. John and")
print("full prompt tokens:", t0.ids)
for name in ["John", " Mary", "John and Mary"]:
    e = tok.encode(name)
    print(f"  encode({name!r}) -> {e.ids}")
s1_tok = tok.encode(" John").ids[0]
s2_tok = tok.encode(" Mary").ids[0]
print("s1_tok", s1_tok, "s2_tok", s2_tok)
arr = np.array(t0.ids)
print("first-S1 positions:", np.where(arr == s1_tok)[0],
      "S2 positions:", np.where(arr == s2_tok)[0])

# --- scores ---
clean_txt = [f"{s1} and {s2} went to the {pl}. {s1} and"
             for s1, s2, pl in zip(S1_NAMES, S2_NAMES, PLACES)]
corrupt_txt = [f"{s1} and {s2c} went to the {pl}. {s1} and"
               for s1, s2c, pl in zip(S1_NAMES, DISTRACT, PLACES)]
clean_ids = [tok.encode(t).ids for t in clean_txt]
corrupt_ids = [tok.encode(t).ids for t in corrupt_txt]
B = len(clean_ids)
maxlen = max(len(x) for x in clean_ids)
pad_id = 50256

def pad(ids_list):
    out = torch.full((len(ids_list), maxlen), pad_id, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        out[i, :len(ids)] = torch.tensor(ids)
    return out

clean_t = pad(clean_ids)
corrupt_t = pad(corrupt_ids)

n_layers, n_heads = model.n_layer, model.n_head
upl = n_heads + 1
n_units = n_layers * upl
unit_ids = []
for l in range(n_layers):
    for h in range(n_heads):
        unit_ids.append(f"L{l}_H{h}")
    unit_ids.append(f"L{l}_MLP")

pos_first_s1, pos_s2 = [], []
for cids in clean_ids:
    arr = np.array(cids)
    p1 = np.where(arr == s1_tok)[0]
    pos_first_s1.append(int(p1[0]) if len(p1) else 0)
    p2 = np.where(arr == s2_tok)[0]
    pos_s2.append(int(p2[0]) if len(p2) else 1)

with torch.no_grad():
    clean = model.forward(clean_t, want_probs=True, want_full=True)
    corr = model.forward(corrupt_t, want_probs=True)

s2_ids = [tok.encode(f" {s}").ids[0] for s in S2_NAMES]
s1_ids = [tok.encode(f" {s}").ids[0] for s in S1_NAMES]
u_S2 = model.lm_head[s2_ids].numpy()
u_S1 = model.lm_head[s1_ids].numpy()
dir_ld = u_S2 - u_S1

head_last = clean["head_last"].numpy()  # (B, L, H, D)
mlp_last = clean["mlp_last"].numpy()

ld = np.zeros(n_units)
dup = np.zeros(n_units)
s2attn = np.zeros(n_units)
for l in range(n_layers):
    for h in range(n_heads):
        u = l * upl + h
        contrib = head_last[:, l, h, :]
        ld[u] = float(np.mean(np.einsum("bd,bd->b", contrib, dir_ld)))
        p = clean["probs"][l].numpy()[:, h, maxlen - 1, :]
        for b in range(B):
            dup[u] += float(p[b, pos_first_s1[b]])
            s2attn[u] += float(p[b, pos_s2[b]])
        dup[u] /= B
        s2attn[u] /= B
    u = l * upl + n_heads
    contrib = mlp_last[:, l, :]
    ld[u] = float(np.mean(np.einsum("bd,bd->b", contrib, dir_ld)))

# patching GT
s2_ids_t = torch.tensor(s2_ids)
s1_ids_t = torch.tensor(s1_ids)

def logit_diff(logits):
    return torch.stack([logits[b, s2_ids_t[b]] - logits[b, s1_ids_t[b]]
                        for b in range(B)])

ld_corr = logit_diff(corr["last_logits"]).numpy()
imp_patch = np.zeros(n_units)
with torch.no_grad():
    for l in range(n_layers):
        for h in range(n_heads):
            key = f"L{l}_H{h}"
            patch = {key: clean["head_full"][:, l, h, :, :]}
            out = model.forward(corrupt_t, patch=patch)
            ld_p = logit_diff(out["last_logits"]).numpy()
            imp_patch[l * upl + h] = float(np.mean(ld_p - ld_corr))
        key = f"L{l}_MLP"
        patch = {key: clean["mlp_full"][:, l, :, :]}
        out = model.forward(corrupt_t, patch=patch)
        ld_p = logit_diff(out["last_logits"]).numpy()
        imp_patch[l * upl + n_heads] = float(np.mean(ld_p - ld_corr))

KNOWN = {"duplicate_token": ["L8_H0", "L9_H6", "L9_H9"],
         "s_inhibition": ["L8_H1"],
         "name_mover": ["L10_H0"],
         "induction_head": ["L5_H1", "L6_H9"]}

print("\n=== per-unit scores (top 25 by |logit-diff|) ===")
order = np.argsort(-np.abs(ld))
for i in order[:25]:
    u = unit_ids[i]
    flag = ""
    for role, names in KNOWN.items():
        if u in names:
            flag = f"  <== {role}"
    print(f"  {u:10s} ld={ld[i]:+8.3f} |ld|={abs(ld[i]):7.3f} "
          f"dup={dup[i]:.4f} s2attn={s2attn[i]:.4f} patch={imp_patch[i]:+7.3f}{flag}")

print("\n=== per-unit scores (top 25 by patching) ===")
order = np.argsort(-np.abs(imp_patch))
for i in order[:25]:
    u = unit_ids[i]
    flag = ""
    for role, names in KNOWN.items():
        if u in names:
            flag = f"  <== {role}"
    print(f"  {u:10s} ld={ld[i]:+8.3f} dup={dup[i]:.4f} s2attn={s2attn[i]:.4f} "
          f"patch={imp_patch[i]:+7.3f}{flag}")

print("\n=== known units, all scores ===")
for role, names in KNOWN.items():
    for name in names:
        i = unit_ids.index(name)
        print(f"  {name:10s} [{role:16s}] ld={ld[i]:+8.3f} |ld|={abs(ld[i]):7.3f} "
              f"dup={dup[i]:.4f} s2attn={s2attn[i]:.4f} patch={imp_patch[i]:+7.3f}")

# correlation checks
for label, x in [("|ld|", np.abs(ld)), ("dup", dup), ("s2attn", s2attn),
                 ("ld_signed", ld)]:
    r = np.corrcoef(x, np.abs(imp_patch))[0, 1]
    print(f"\ncorr({label}, |patching|) = {r:+.4f}")
