"""
Minimal GPT-2 forward pass from cached safetensors weights.

Bypasses `import transformers`, which hangs in this environment due to
pathologically slow importlib.metadata / dill file reads. Uses only
torch + safetensors + tokenizers, all of which import cleanly.

Provides:
  * GPT2.load_from_cache() -> GPT2  (reads ~/.cache/huggingface/hub)
  * forward(input_ids, patch=None) with full attention probabilities
    and per-head last-token output contributions (after c_proj).
  * patch=(layer, head_or_mlp) semantics: pass a dict of unit id ->
    tensor to override that unit's post-projection output during a run.
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

try:
    from safetensors import safe_open
    HAS_SAFE = True
except Exception:  # pragma: no cover
    HAS_SAFE = False


def _find_cached(name: str) -> str:
    base = os.path.expanduser("~/.cache/huggingface/hub")
    for model_dir in (f"models--{name.replace('/', '--')}",):
        snap = os.path.join(base, model_dir, "snapshots")
        if not os.path.isdir(snap):
            continue
        for snapshot in os.listdir(snap):
            p = os.path.join(snap, snapshot)
            if os.path.isfile(os.path.join(p, "model.safetensors")):
                return p
    return ""


class GPT2:
    """Tiny manual GPT-2 (causal LM) with per-head introspection."""

    def __init__(self, snapshot_dir: str):
        cfg_path = os.path.join(snapshot_dir, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        self.n_layer = cfg["n_layer"]
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.n_pos = cfg.get("n_positions", 1024)
        self.vocab = cfg.get("vocab_size", 50257)
        self.d_head = self.n_embd // self.n_head

        state = {}
        st_path = os.path.join(snapshot_dir, "model.safetensors")
        if HAS_SAFE:
            with safe_open(st_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state[k] = f.get_tensor(k)
        else:  # pragma: no cover
            raise RuntimeError("safetensors required")

        self.wte = state["wte.weight"].float()
        self.wpe = state["wpe.weight"].float()
        self.ln_f_w = state["ln_f.weight"].float()
        self.ln_f_b = state["ln_f.bias"].float()
        # lm_head tied to wte (present in HF checkpoints)
        self.lm_head = state.get("lm_head.weight", self.wte).float()

        self.blocks = []
        for i in range(self.n_layer):
            b = {
                "ln1_w": state[f"h.{i}.ln_1.weight"].float(),
                "ln1_b": state[f"h.{i}.ln_1.bias"].float(),
                "c_attn_w": state[f"h.{i}.attn.c_attn.weight"].float(),
                "c_attn_b": state[f"h.{i}.attn.c_attn.bias"].float(),
                "c_proj_w": state[f"h.{i}.attn.c_proj.weight"].float(),
                "c_proj_b": state[f"h.{i}.attn.c_proj.bias"].float(),
                "ln2_w": state[f"h.{i}.ln_2.weight"].float(),
                "ln2_b": state[f"h.{i}.ln_2.bias"].float(),
                "c_fc_w": state[f"h.{i}.mlp.c_fc.weight"].float(),
                "c_fc_b": state[f"h.{i}.mlp.c_fc.bias"].float(),
                "c_mlp_w": state[f"h.{i}.mlp.c_proj.weight"].float(),
                "c_mlp_b": state[f"h.{i}.mlp.c_proj.bias"].float(),
            }
            self.blocks.append(b)

    @staticmethod
    def from_cache(name: str = "openai-community/gpt2") -> "GPT2":
        d = _find_cached(name)
        if not d:
            raise FileNotFoundError(f"gpt2 not cached under {name}")
        return GPT2(d)

    # ------------------------------------------------------------------
    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def _block_at(self, h_in, layer_idx, patch=None, unit_ids=None):
        b = self.blocks[layer_idx]
        B, S, D = h_in.shape

        # --- attention ---
        # GPT-2 safetensors store linear weights as (in, out): use x @ W
        x = F.layer_norm(h_in, (D,), b["ln1_w"], b["ln1_b"])
        qkv = x @ b["c_attn_w"] + b["c_attn_b"]
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, S, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_head, self.d_head).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=h_in.device))
        att = att.masked_fill(~causal[None, None], float("-inf"))
        probs = torch.softmax(att, dim=-1)                      # (B, H, S, S)
        head_outs = torch.matmul(probs, v)                      # (B, H, S, dH)
        head_outs = head_outs.transpose(1, 2).reshape(B, S, self.n_head * self.d_head)
        # per-head post-projection contributions
        w_proj = b["c_proj_w"]                                  # (D, D)
        head_contribs = []
        for h_idx in range(self.n_head):
            h_slice = head_outs[..., h_idx * self.d_head:(h_idx + 1) * self.d_head]
            w_slice = w_proj[h_idx * self.d_head:(h_idx + 1) * self.d_head]
            contrib = h_slice @ w_slice                           # (B, S, D)
            if patch is not None:
                key = f"L{layer_idx}_H{h_idx}"
                if key in patch:
                    contrib = patch[key]  # already (B, S, D)
            head_contribs.append(contrib)
        attn_out = torch.stack(head_contribs, dim=1).sum(dim=1) + b["c_proj_b"]  # (B, S, D)
        h1 = h_in + attn_out

        # --- mlp ---
        y = F.layer_norm(h1, (D,), b["ln2_w"], b["ln2_b"])
        y = y @ b["c_fc_w"] + b["c_fc_b"]
        y = self._gelu(y)
        mlp_out = y @ b["c_mlp_w"] + b["c_mlp_b"]
        if patch is not None:
            key = f"L{layer_idx}_MLP"
            if key in patch:
                mlp_out = patch[key]  # already (B, S, D)
        h2 = h1 + mlp_out

        return h2, head_contribs, probs, mlp_out

    def forward(self, input_ids, patch=None, want_probs=False, want_full=False):
        """
        input_ids: (B, S) long. patch: {unit_id: (B, S, D) tensor} or None.
        Returns dict with logits (B, S, V), last_logits (B, V),
        head_last (B, L, H, D) and mlp_last (B, L, D) post-projection
        last-token contributions, optionally probs (list of (B,H,S,S)),
        and optionally head_full (B, L, H, S, D) / mlp_full (B, L, S, D).
        """
        B, S = input_ids.shape
        device = input_ids.device
        h = self.wte[input_ids] + self.wpe[torch.arange(S, device=device)][None]
        head_last = []
        mlp_last = []
        head_full = [] if want_full else None
        mlp_full = [] if want_full else None
        probs_all = [] if want_probs else None

        for layer_idx in range(self.n_layer):
            h, head_contribs, probs, mlp_out = self._block_at(h, layer_idx, patch)
            # last-token per-head contributions (B, H, D)
            last = torch.stack([c[:, -1, :] for c in head_contribs], dim=1)
            head_last.append(last)
            mlp_last.append(mlp_out[:, -1, :])
            if want_probs:
                probs_all.append(probs)
            if want_full:
                head_full.append(torch.stack(head_contribs, dim=1))  # (B,H,S,D)
                mlp_full.append(mlp_out)

        h = F.layer_norm(h, (self.n_embd,), self.ln_f_w, self.ln_f_b)
        logits = h @ self.lm_head.T                              # (B, S, V)

        out = {
            "logits": logits,
            "last_logits": logits[:, -1, :],
            "head_last": torch.stack(head_last, dim=1),         # (B, L, H, D)
            "mlp_last": torch.stack(mlp_last, dim=1),           # (B, L, D)
            "probs": probs_all,                                 # list of (B,H,S,S)
        }
        if want_full:
            out["head_full"] = torch.stack(head_full, dim=1)   # (B, L, H, S, D)
            out["mlp_full"] = torch.stack(mlp_full, dim=1)     # (B, L, S, D)
        return out


def load_tokenizer(snapshot_dir: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(snapshot_dir, "tokenizer.json"))
