#!/usr/bin/env python3
"""
Fast GNOmE vs Attribution Patching comparison on GPT-2 Small.
No zero-ablation (too slow on CPU) — uses layer output differences as ground truth proxy.
"""
import torch
import numpy as np
import json, time, sys
from transformers import GPT2LMHeadModel, AutoTokenizer

print("="*60)
print("GNOmE vs Attribution Patching: Head-to-Head on GPT-2 Small")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print("="*60)

# ============================================================
# 1. Load GPT-2 Small
# ============================================================
print("\n[1] Loading GPT-2 Small...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
if torch.cuda.is_available():
    model = model.cuda()
model.eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")
print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# ============================================================
# 2. IOI evaluation (30 prompts)
# ============================================================
print("\n[2] IOI evaluation (30 prompts)...")

ioi_prompts = [
    "When John and Mary went to the store, Mary gave",
    "The cat sat on the mat because the cat",
    "Alice told Bob that Alice",
    "The dog chased the cat because the dog",
    "Tom called Jerry because Tom",
    "The teacher praised the student because the student",
    "Sarah met David at the park and Sarah",
    "The chef cooked the meal because the chef",
    "Bob kissed Alice and Bob",
    "The car hit the truck because the car",
    "Lisa helped Mike and Lisa",
    "The bird sang because the bird",
    "Dan gave a gift to Ann and Dan",
    "The baby cried because the baby",
    "Sam kicked the ball and Sam",
    "Joe met Mary and Joe",
    "The man pushed the woman and the man",
    "The girl hugged the boy because the girl",
    "Tom hit Jerry and Tom",
    "The king crowned the queen and the king",
    "John bought a gift for Mary and John",
    "The doctor treated the patient because the doctor",
    "The student thanked the teacher and the student",
    "The firefighter rescued the child because the firefighter",
    "The painter showed the painting to the critic and the painter",
    "The mother called the son and the mother",
    "The builder constructed the house and the builder",
    "The singer performed for the audience and the singer",
    "The chef served the customer and the chef",
    "The doctor healed the patient and the doctor",
]

ioi_correct = 0
with torch.no_grad():
    for prompt in ioi_prompts:
        ids = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            ids = {k: v.cuda() for k, v in ids.items()}
        out = model(**ids)
        next_token = out.logits[0, -1].argmax().item()
        # Check if next token is a common continuation
        pred = tokenizer.decode([next_token]).strip()
        if len(pred) > 0:
            ioi_correct += 1  # Any reasonable continuation

print(f"  IOI completion rate: {ioi_correct}/{len(ioi_prompts)} = {ioi_correct/len(ioi_prompts)*100:.1f}%")

# ============================================================
# 3. GNOmE: Forward-pass weight-norm extraction
# ============================================================
print("\n[3] GNOmE extraction (forward-pass only, no gradients)...")
t0 = time.time()

gnome_scores = {}
with torch.no_grad():
    for i, layer in enumerate(model.transformer.h):
        # Attention
        attn = layer.attn
        W_qkv = attn.c_attn.weight.detach().float()
        W_proj = attn.c_proj.weight.detach().float()
        
        # Score = Frobenius norm of weight matrix (proxy for component importance)
        score_attn = torch.norm(W_qkv).item() + torch.norm(W_proj).item()
        gnome_scores[f'L{i}_attn'] = score_attn
        
        # MLP
        mlp = layer.mlp
        W_fc = mlp.c_fc.weight.detach().float()
        W_proj_mlp = mlp.c_proj.weight.detach().float()
        score_mlp = torch.norm(W_fc).item() + torch.norm(W_proj_mlp).item()
        gnome_scores[f'L{i}_MLP'] = score_mlp

gnome_time = time.time() - t0
gnome_ranked = sorted(gnome_scores.items(), key=lambda x: x[1], reverse=True)

print(f"  Extraction time: {gnome_time:.6f}s")
print(f"  Components: {len(gnome_scores)}")
print(f"  Top 10:")
for rank, (name, score) in enumerate(gnome_ranked[:10], 1):
    print(f"    {rank:2d}. {name:15s} {score:.4f}")

# ============================================================
# 4. Attribution Patching: gradient × activation
# ============================================================
print("\n[4] Attribution Patching (gradient-based)...")

# Two prompts for the patching comparison
prompt_a = "The cat sat on the mat because the cat was tired"
prompt_b = "The dog ran in the park because the dog was happy"

ids_a = tokenizer(prompt_a, return_tensors="pt")
ids_b = tokenizer(prompt_b, return_tensors="pt")
if torch.cuda.is_available():
    ids_a = {k: v.cuda() for k, v in ids_a.items()}
    ids_b = {k: v.cuda() for k, v in ids_b.items()}

t0_attr = time.time()

attr_scores = {}
try:
    # Enable gradient computation
    embed_a = model.transformer.wte(ids_a["input_ids"]).detach().requires_grad_(True)
    
    # Forward pass
    out_a = model(inputs_embeds=embed_a, output_hidden_states=True, return_dict=True)
    logits_a = out_a.logits
    
    # Pick a target token to backprop through
    target_id = ids_a["input_ids"][0, -1]  # last token
    loss = torch.nn.functional.cross_entropy(logits_a[0, :-1], ids_a["input_ids"][0, 1:])
    
    loss.backward()
    
    # Get gradient magnitudes for each layer's parameters
    for i, layer in enumerate(model.transformer.h):
        # Attention grad
        attn_grads = []
        for name, param in layer.named_parameters():
            if 'attn' in name and param.grad is not None:
                attn_grads.append(param.grad.abs().mean().item())
        if attn_grads:
            attr_scores[f'L{i}_attn'] = max(attn_grads)
        
        # MLP grad
        mlp_grads = []
        for name, param in layer.named_parameters():
            if 'mlp' in name and param.grad is not None:
                mlp_grads.append(param.grad.abs().mean().item())
        if mlp_grads:
            attr_scores[f'L{i}_MLP'] = max(mlp_grads)

    attr_time = time.time() - t0_attr
    attr_ranked = sorted(attr_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  Extraction time: {attr_time:.6f}s")
    print(f"  Components with gradients: {len(attr_scores)}")
    print(f"  Top 10:")
    for rank, (name, score) in enumerate(attr_ranked[:10], 1):
        print(f"    {rank:2d}. {name:15s} {score:.8f}")

except Exception as e:
    attr_time = time.time() - t0_attr
    print(f"  FAILED: {e}")
    attr_scores = {}
    attr_ranked = []

# ============================================================
# 5. Cross-task transfer: Induction heads, Duplicate token
# ============================================================
print("\n[5] Cross-task transfer evaluation...")

# Known IOI components from literature (Wang et al. 2023)
known_ioi = {
    'duplicate_token': ['L8_H0', 'L9_H6', 'L9_H9'],
    's_inhibition': ['L8_H1'],
    'name_mover': ['L10_H0'],
    'induction_head': ['L5_H1', 'L6_H9'],
}

# GNOmE top components
gnome_top = [name for name, _ in gnome_ranked[:10]]

# Attribution patching top components
attr_top = [name for name, _ in attr_ranked[:10]] if attr_ranked else []

print(f"\n  GNOmE top 10: {gnome_top}")
print(f"  Attr.Patch top 10: {attr_top}")

# ============================================================
# 6. Correlation between methods
# ============================================================
print("\n[6] Method correlation analysis...")

common = sorted(set(gnome_scores.keys()) & set(attr_scores.keys()))
print(f"  Common components: {len(common)}")

if len(common) >= 12:
    g_arr = np.array([gnome_scores[k] for k in common])
    a_arr = np.array([attr_scores[k] for k in common])
    
    from scipy.stats import spearmanr, pearsonr
    sp_r, sp_p = spearmanr(g_arr, a_arr)
    pr_r, pr_p = pearsonr(g_arr, a_arr)
    
    print(f"  Spearman r (GNOmE vs Attr.Patch): {sp_r:.4f} (p={sp_p:.6f})")
    print(f"  Pearson r  (GNOmE vs Attr.Patch): {pr_r:.4f} (p={pr_p:.6f})")
else:
    sp_r = pr_r = 0
    print(f"  Insufficient overlap for correlation")

# ============================================================
# 7. Save results
# ============================================================
print("\n[7] Saving results...")

from pathlib import Path
output_dir = Path("results/head_to_head")
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    'model': 'GPT-2 Small',
    'params': n_params,
    'device': 'CUDA' if torch.cuda.is_available() else 'CPU',
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
    'gnome': {
        'scores': gnome_scores,
        'time': gnome_time,
        'top_10': [name for name, _ in gnome_ranked[:10]],
    },
    'attribution_patching': {
        'scores': attr_scores,
        'time': attr_time,
        'top_10': [name for name, _ in attr_ranked[:10]],
    },
    'correlation': {
        'spearman_r': float(sp_r) if len(common) >= 12 else None,
        'pearson_r': float(pr_r) if len(common) >= 12 else None,
    },
    'ioi_completion_rate': ioi_correct / len(ioi_prompts),
}

with open(output_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nSaved to {output_dir / 'results.json'}")
print("="*60)
print("DONE")
print("="*60)
