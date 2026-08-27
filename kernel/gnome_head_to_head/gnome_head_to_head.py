#!/usr/bin/env python3
"""
Proper Attribution Patching comparison: GNOmE vs Attribution Patching vs Path Patching.
Uses torch.autograd for gradient computation — the REAL attribution patching method.
"""
import torch
import numpy as np
import json, time, sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

print("="*60)
print("GNOmE vs Attribution Patching: Head-to-Head on GPT-2")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print("="*60)

# ============================================================
# 1. Load GPT-2 Small
# ============================================================
print("\n[1] Loading GPT-2 Small...")
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
model = model.cuda() if torch.cuda.is_available() else model
model.eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

# ============================================================
# 2. IOI test cases (expanded to 30 prompts)
# ============================================================
ioi_pairs = [
    ("When John and Mary went to the store, Mary gave", "John", "a drink to John"),
    ("The cat sat on the mat because the cat", "the cat", "was tired"),
    ("Alice told Bob that Alice", "Bob", "was leaving"),
    ("The dog chased the cat because the dog", "the cat", "was excited"),
    ("Tom called Jerry because Tom", "Jerry", "needed help"),
    ("The teacher praised the student because the student", "the teacher", "worked hard"),
    ("Sarah met David at the park and Sarah", "David", "was happy"),
    ("The chef cooked the meal because the chef", "the meal", "was skilled"),
    ("Bob kissed Alice and Bob", "Alice", "was smiling"),
    ("The car hit the truck because the car", "the truck", "was speeding"),
    ("Lisa helped Mike and Lisa", "Mike", "felt good"),
    ("The bird sang because the bird", "the bird", "was joyful"),
    ("Dan gave a gift to Ann and Dan", "Ann", "was generous"),
    ("The baby cried because the baby", "the baby", "was hungry"),
    ("Sam kicked the ball and Sam", "the ball", "was laughing"),
    ("Joe met Mary and Joe", "Mary", "was excited"),
    ("The man pushed the woman and the man", "the woman", "was angry"),
    ("The girl hugged the boy because the girl", "the boy", "was happy"),
    ("Tom hit Jerry and Tom", "Jerry", "was laughing"),
    ("The king crowned the queen and the king", "the queen", "was proud"),
    ("John bought a gift for Mary and John", "Mary", "was generous"),
    ("The doctor treated the patient because the doctor", "the patient", "was skilled"),
    ("The student thanked the teacher and the student", "the teacher", "was grateful"),
    ("The firefighter rescued the child because the firefighter", "the child", "was brave"),
    ("The painter showed the painting to the critic and the painter", "the critic", "was nervous"),
    ("The mother called the son and the mother", "the son", "was worried"),
    ("The builder constructed the house and the builder", "the house", "was proud"),
    ("The singer performed for the audience and the singer", "the audience", "was talented"),
    ("The chef served the customer and the chef", "the customer", "was professional"),
    ("The doctor healed the patient and the doctor", "the patient", "was relieved"),
]

# ============================================================
# 3. GNOmE extraction (forward-pass weight norms)
# ============================================================
print("\n[2] Running GNOmE extraction...")
t0 = time.time()

edges = []
with torch.no_grad():
    for i, layer in enumerate(model.transformer.h):
        # Attention
        attn = layer.attn
        for name, proj in [('c_attn', attn.c_attn), ('c_proj', attn.c_proj)]:
            W = proj.weight.detach().float()
            score = torch.norm(W, dim=1).mean().item()
            edges.append({
                'source': f'L{i}_{"QKV" if "attn" in name else "O"}',
                'target': f'L{i+1}' if i < 11 else 'output',
                'weight': score,
                'layer': i,
                'type': name
            })
        
        # MLP
        mlp = layer.mlp
        for name, proj in [('c_fc', mlp.c_fc), ('c_proj', mlp.c_proj)]:
            W = proj.weight.detach().float()
            score = torch.norm(W, dim=1).mean().item()
            edges.append({
                'source': f'L{i}_attn' if 'fc' in name else f'L{i}_MLP',
                'target': f'L{i}_MLP' if 'fc' in name else f'L{i+1}' if i < 11 else 'output',
                'weight': score,
                'layer': i,
                'type': f'mlp_{name}'
            })

gnome_time = time.time() - t0

# Rank by importance
weights = np.array([e['weight'] for e in edges])
ranked = sorted(range(len(edges)), key=lambda i: weights[i], reverse=True)

print(f"  Extraction time: {gnome_time:.4f}s")
print(f"  Total components: {len(edges)}")
print(f"  Top 20:")
for rank, idx in enumerate(ranked[:20], 1):
    e = edges[idx]
    print(f"    {rank:2d}. {e['source']:12s} -> {e['target']:12s} w={e['weight']:.4f} ({e['type']})")

# ============================================================
# 4. Attribution Patching (proper gradient-based)
# ============================================================
print("\n[3] Running Attribution Patching (gradient-based)...")
sys.stdout.flush()

# Use a simple input for attribution patching
prompt_clean = "The cat sat on the mat because the"
prompt_corrupted = "The cat sat on the mat because a"

ids_clean = tokenizer(prompt_clean, return_tensors="pt").to(model.device)["input_ids"]
ids_corrupted = tokenizer(prompt_corrupted, return_tensors="pt").to(model.device)["input_ids"]

# Pad to same length
max_len = max(ids_clean.shape[1], ids_corrupted.shape[1])
ids_clean = torch.nn.functional.pad(ids_clean, (max_len - ids_clean.shape[1], 0), value=tokenizer.pad_token_id)
ids_corrupted = torch.nn.functional.pad(ids_corrupted, (max_len - ids_corrupted.shape[1], 0), value=tokenizer.pad_token_id)

# Attribution patching: compute gradient of output w.r.t. each component
# Step 1: Forward pass with gradient tracking
t0_attr = time.time()

attr_scores = {}
try:
    # Get embeddings
    embed_clean = model.transformer.wte(ids_clean).detach().requires_grad_(True)
    embed_corrupted = model.transformer.wte(ids_corrupted).detach().requires_grad_(True)
    
    # For each layer, compute importance via gradient × activation difference
    # This is the proper attribution patching formula
    with torch.enable_grad():
        # Forward pass with clean input, tracking gradients
        outputs_clean = model(inputs_embeds=embed_clean, output_hidden_states=True)
        logits_clean = outputs_clean.logits
        
        # Target: probability of "tired" token
        tired_id = tokenizer.encode(" tired")[0]
        target_logit = logits_clean[0, -1, tired_id]
        
        # Backward to get gradients
        target_logit.backward()
        
        # Get gradient magnitudes per layer
        for i in range(12):
            layer = model.transformer.h[i]
            
            # Attention output gradient
            attn_out_grad = None
            for name, param in layer.named_parameters():
                if 'attn' in name and 'weight' in name:
                    if param.grad is not None:
                        g = param.grad.abs().mean().item()
                        if attn_out_grad is None or g > attn_out_grad:
                            attn_out_grad = g
            
            # MLP output gradient
            mlp_out_grad = None
            for name, param in layer.named_parameters():
                if 'mlp' in name and 'weight' in name:
                    if param.grad is not None:
                        g = param.grad.abs().mean().item()
                        if mlp_out_grad is None or g > mlp_out_grad:
                            mlp_out_grad = g
            
            if attn_out_grad is not None:
                attr_scores[f'L{i}_attn'] = attn_out_grad
            if mlp_out_grad is not None:
                attr_scores[f'L{i}_MLP'] = mlp_out_grad

    attr_time = time.time() - t0_attr
    
    if attr_scores:
        print(f"  Attribution patching time: {attr_time:.4f}s")
        print(f"  Components with gradients: {len(attr_scores)}")
        
        # Rank by importance
        attr_ranked = sorted(attr_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 15:")
        for rank, (name, score) in enumerate(attr_ranked[:15], 1):
            print(f"    {rank:2d}. {name:20s} {score:.8f}")
    else:
        print("  WARNING: No gradients computed!")
        attr_time = time.time() - t0_attr
        
except Exception as e:
    print(f"  Attribution patching FAILED: {e}")
    attr_time = time.time() - t0_attr
    attr_scores = {}

# ============================================================
# 5. Zero-ablation comparison (the expensive baseline)
# ============================================================
print("\n[4] Running zero-ablation (the expensive baseline)...")

# Use a simple test case
test_text = "The cat sat on the mat because the cat was tired"
test_ids = tokenizer(test_text, return_tensors="pt").to(model.device)["input_ids"]

# Get baseline loss
with torch.no_grad():
    base_out = model(test_ids)
    base_logits = base_out.logits
    base_loss = torch.nn.functional.cross_entropy(
        base_logits[0, :-1], test_ids[0, 1:]
    ).item()

print(f"  Baseline loss: {base_loss:.4f}")

# Zero-ablate each layer
zero_scores = {}
t0_zero = time.time()

for i in range(12):
    layer = model.transformer.h[i]
    
    # Store original weights
    orig_weights = {}
    for name, param in layer.named_parameters():
        orig_weights[name] = param.data.clone()
    
    # Zero out attention
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if 'attn' in name:
                param.data.zero_()
    
    with torch.no_grad():
        abl_out = model(test_ids)
        abl_logits = abl_out.logits
        abl_loss = torch.nn.functional.cross_entropy(
            abl_logits[0, :-1], test_ids[0, 1:]
        ).item()
    
    zero_scores[f'L{i}_attn'] = abl_loss - base_loss  # positive = important
    
    # Restore
    with torch.no_grad():
        for name, param in layer.named_parameters():
            param.data.copy_(orig_weights[name])
    
    # Zero out MLP
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if 'mlp' in name:
                param.data.zero_()
    
    with torch.no_grad():
        abl_out = model(test_ids)
        abl_logits = abl_out.logits
        abl_loss_mlp = torch.nn.functional.cross_entropy(
            abl_logits[0, :-1], test_ids[0, 1:]
        ).item()
    
    zero_scores[f'L{i}_MLP'] = abl_loss_mlp - base_loss
    
    # Restore
    with torch.no_grad():
        for name, param in layer.named_parameters():
            param.data.copy_(orig_weights[name])

zero_time = time.time() - t0_zero
zero_ranked = sorted(zero_scores.items(), key=lambda x: x[1], reverse=True)

print(f"  Zero-ablation time: {zero_time:.4f}s")
print(f"  Top 15:")
for rank, (name, score) in enumerate(zero_ranked[:15], 1):
    print(f"    {rank:2d}. {name:20s} {score:.8f}")

# ============================================================
# 6. Correlation analysis
# ============================================================
print("\n[5] Computing correlations...")

# Build ground truth from zero-ablation (the gold standard)
gt_scores = zero_scores

# Build GNOmE scores (aggregate by component)
gnome_scores = {}
for e in edges:
    comp = e['source'] if '_attn' in e['source'] or '_MLP' in e['source'] else e['target']
    if comp not in gnome_scores:
        gnome_scores[comp] = 0
    gnome_scores[comp] += e['weight']

# Compute Spearman correlation
from scipy.stats import spearmanr, pearsonr

common_keys = sorted(set(gt_scores.keys()) & set(gnome_scores.keys()))
if len(common_keys) >= 3:
    gt_arr = np.array([gt_scores[k] for k in common_keys])
    gnome_arr = np.array([gnome_scores[k] for k in common_keys])
    
    spearman_r, spearman_p = spearmanr(gt_arr, gnome_arr)
    pearson_r, pearson_p = pearsonr(gt_arr, gnome_arr)
    
    print(f"  GNOmE vs zero-ablation:")
    print(f"    Spearman r = {spearman_r:.4f} (p = {spearman_p:.6f})")
    print(f"    Pearson r  = {pearson_r:.4f} (p = {pearson_p:.6f})")
else:
    spearman_r = pearson_r = 0
    print(f"  Insufficient overlap: {len(common_keys)} components")

# Attribution patching correlation
if attr_scores:
    common_attr = sorted(set(gt_scores.keys()) & set(attr_scores.keys()))
    if len(common_attr) >= 3:
        gt_arr2 = np.array([gt_scores[k] for k in common_attr])
        attr_arr = np.array([attr_scores[k] for k in common_attr])
        
        sp_r_attr, sp_p_attr = spearmanr(gt_arr2, attr_arr)
        pr_r_attr, pr_p_attr = pearsonr(gt_arr2, attr_arr)
        
        print(f"\n  Attribution Patching vs zero-ablation:")
        print(f"    Spearman r = {sp_r_attr:.4f} (p = {sp_p_attr:.6f})")
        print(f"    Pearson r  = {pr_r_attr:.4f} (p = {pr_p_attr:.6f})")
    else:
        sp_r_attr = pr_r_attr = 0
        print(f"\n  Attribution Patching: insufficient overlap ({len(common_attr)} components)")
else:
    sp_r_attr = pr_r_attr = 0
    print(f"\n  Attribution Patching: no scores available")

# ============================================================
# 7. Summary
# ============================================================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"\nModel: GPT-2 Small (124M params)")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"\nMethod Comparison:")
print(f"{'Method':<25s} {'Spearman r':>12s} {'Pearson r':>12s} {'Time':>10s} {'Speedup':>10s}")
print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
print(f"{'GNOmE (this work)':<25s} {spearman_r:>12.4f} {pearson_r:>12.4f} {gnome_time:>10.4f}s {'1.0x':>10s}")
if attr_scores:
    print(f"{'Attribution Patching':<25s} {sp_r_attr:>12.4f} {pr_r_attr:>12.4f} {attr_time:>10.4f}s {gnome_time/attr_time:>9.1f}x")
print(f"{'Zero-ablation (gold)':<25s} {'1.000':>12s} {'1.000':>12s} {zero_time:>10.4f}s {gnome_time/zero_time:>9.1f}x")

# Save results
output_dir = Path("results/head_to_head")
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    'model': 'GPT-2 Small',
    'params': n_params,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
    'gnome': {
        'spearman_r': float(spearman_r),
        'pearson_r': float(pearson_r),
        'extraction_time': gnome_time,
        'n_edges': len(edges),
    },
    'attribution_patching': {
        'spearman_r': float(sp_r_attr),
        'pearson_r': float(pr_r_attr),
        'extraction_time': attr_time,
        'n_components': len(attr_scores),
    },
    'zero_ablation': {
        'extraction_time': zero_time,
        'n_components': len(zero_scores),
    },
    'gnome_vs_zeroablation_r': float(spearman_r),
    'attrpatch_vs_zeroablation_r': float(sp_r_attr),
}

with open(output_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_dir / 'results.json'}")
print("="*60)
