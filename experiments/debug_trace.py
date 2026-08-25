#!/usr/bin/env python3
"""Trace divergence between manual and HF GPT-2, one sublayer at a time."""
import glob
import os

import torch
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from gpt2_manual import GPT2

snap = None
for cand in glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/*")):
    if os.path.isfile(os.path.join(cand, "model.safetensors")):
        snap = cand
        break

hf = GPT2LMHeadModel.from_pretrained(snap, local_files_only=True)
hf.eval()
tok_hf = GPT2TokenizerFast.from_pretrained(snap, local_files_only=True)
m = GPT2(snap)

ids = tok_hf.encode("John and Mary went to the store. John and", return_tensors="pt")
B, S = ids.shape
D = m.n_embd

with torch.no_grad():
    # HF embedding
    h_hf = hf.transformer.wte(ids) + hf.transformer.wpe(torch.arange(S))[None]
    h_man = m.wte[ids] + m.wpe[torch.arange(S)][None]
    print(f"embeddings: max diff {float((h_hf - h_man).abs().max()):.3e}")

    for l in range(m.n_layer):
        # HF sublayers
        block = hf.transformer.h[l]
        x_hf = block.ln_1(h_hf)
        # manual ln1
        x_man = F.layer_norm(h_man, (D,), m.blocks[l]["ln1_w"], m.blocks[l]["ln1_b"])
        print(f"\nL{l} ln1: max diff {float((x_hf - x_man).abs().max()):.3e}")

        # qkv
        qkv_hf = block.attn.c_attn(x_hf)
        qkv_man = x_man @ m.blocks[l]["c_attn_w"] + m.blocks[l]["c_attn_b"]
        print(f"L{l} qkv: max diff {float((qkv_hf - qkv_man).abs().max()):.3e}")
        q_hf, k_hf, v_hf = qkv_hf.split(D, dim=2)
        q_man, k_man, v_man = qkv_man.split(D, dim=-1)
        print(f"L{l} q: {float((q_hf - q_man).abs().max()):.3e} k: {float((k_hf - k_man).abs().max()):.3e} v: {float((v_hf - v_man).abs().max()):.3e}")

        # attention probs
        qh = q_hf.view(B, S, m.n_head, m.d_head).transpose(1, 2)
        kh = k_hf.view(B, S, m.n_head, m.d_head).transpose(1, 2)
        att = torch.matmul(qh, kh.transpose(-1, -2)) / (m.d_head ** 0.5)
        att = torch.where(torch.tril(torch.ones(S, S, dtype=torch.bool))[None, None],
                          att, torch.tensor(float("-inf")))
        probs_hf = torch.softmax(att, dim=-1)
        # manual attn (run block partially)
        probs_man = m._block_at(h_man, l)[2]
        print(f"L{l} probs: max diff {float((probs_hf - probs_man).abs().max()):.3e}")

        # HF full attn output
        vh = v_hf.view(B, S, m.n_head, m.d_head).transpose(1, 2)
        attn_hf = block.attn.c_proj((torch.matmul(probs_hf, vh).transpose(1, 2)
                                     .reshape(B, S, D)))
        # manual head contributions -> attn_out is h1 - h_in
        _, head_contribs, _, _ = m._block_at(h_man, l)
        attn_man = torch.stack(head_contribs, dim=1).sum(dim=1)
        print(f"L{l} attn_out: max diff {float((attn_hf - attn_man).abs().max()):.3e}")

        # MLP
        mlp_hf = block.mlp(block.ln_2(h_hf + attn_hf))
        mlp_man = m._block_at(h_man, l)[3]
        print(f"L{l} mlp_out: max diff {float((mlp_hf - mlp_man).abs().max()):.3e}")

        h_hf = h_hf + attn_hf + mlp_hf
        h_man, _, _, _ = m._block_at(h_man, l)
        print(f"L{l} residual: max diff {float((h_hf - h_man).abs().max()):.3e}")

    lnf_hf = hf.transformer.ln_f(h_hf)
    lnf_man = F.layer_norm(h_man, (D,), m.ln_f_w, m.ln_f_b)
    print(f"\nln_f: max diff {float((lnf_hf - lnf_man).abs().max()):.3e}")
    logits_hf = hf.lm_head(lnf_hf)
    logits_man = lnf_man @ m.lm_head.T
    print(f"logits: max diff {float((logits_hf - logits_man).abs().max()):.3e}")
