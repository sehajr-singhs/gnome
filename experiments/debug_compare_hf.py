#!/usr/bin/env python3
"""Compare manual GPT-2 against HF transformers reference, layer by layer."""
import glob
import os
import time

import torch

t0 = time.time()
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402
print(f"imports {time.time()-t0:.0f}s", flush=True)

from gpt2_manual import GPT2, load_tokenizer  # noqa: E402

snap = None
for cand in glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/*")):
    if os.path.isfile(os.path.join(cand, "model.safetensors")):
        snap = cand
        break

# HF reference
hf = GPT2LMHeadModel.from_pretrained(snap, local_files_only=True)
hf.eval()
tok_hf = GPT2TokenizerFast.from_pretrained(snap, local_files_only=True)
tok = load_tokenizer(snap)

text = "John and Mary went to the store. John and"
ids = tok_hf.encode(text, return_tensors="pt")
print("HF ids:", ids.tolist())
print("manual ids:", tok.encode(text).ids)

# manual
m = GPT2(snap)
man = m.forward(ids, want_probs=True)
man_logits = man["last_logits"]

with torch.no_grad():
    hf_out = hf(input_ids=ids)
hf_logits = hf_out.logits[:, -1, :]

print("\nlast-position logits:")
print("  HF    logit(Mary)-logit(John):", float(hf_logits[0, 5335] - hf_logits[0, 1757]))
print("  man   logit(Mary)-logit(John):", float(man_logits[0, 5335] - man_logits[0, 1757]))
print("  HF    max logit:", float(hf_logits.max()), "argmax token:", repr(tok_hf.decode([int(hf_logits.argmax())])))
print("  man   max logit:", float(man_logits.max()), "argmax token:", repr(tok_hf.decode([int(man_logits.argmax())])))
print("  HF    top5:", [(repr(tok_hf.decode([int(i)])), round(float(v), 1)) for i, v in zip(hf_logits.topk(5).indices, hf_logits.topk(5).values)])
print("  man   top5:", [(repr(tok_hf.decode([int(i)])), round(float(v), 1)) for i, v in zip(man_logits.topk(5).indices, man_logits.topk(5).values)])
print("  max |logit diff|:", float((hf_logits - man_logits).abs().max()))
print("  HF mean/std:", float(hf_logits.mean()), float(hf_logits.std()))
print("  man mean/std:", float(man_logits.mean()), float(man_logits.std()))

# layer-by-layer residual stream comparison
print("\nlayer-by-layer residual divergence:")
hf_h = hf.transformer.wte(ids) + hf.transformer.wpe(torch.arange(ids.shape[1]))[None]
with torch.no_grad():
    for l, block in enumerate(hf.transformer.h):
        hf_h = block(hf_h)[0]
        man_h = m._block_at  # placeholder
        break
