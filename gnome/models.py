"""Models with an explicit block structure.

Every model exposes ``blocks()``, a list of differentiable callables where
block k maps unit-layer k to unit-layer k+1, and ``unit_dims``, the
dimension of each unit layer. Circuit-graph extraction (extraction.py)
differentiates each block and reads the mean-abs Jacobian as the edge
weight between consecutive unit layers. Residual connections are folded
into the block they close, so the graph stays layered and every edge is a
real computational path.

Two transformer layouts are supported:

* ``seq_mode=False`` (modular tasks): two one-hot tokens (a, b) of width
  p, concatenated. Attention mixes the two tokens only.
* ``seq_mode=True`` (boolean tasks): the input is a sequence of n_in
  one-hot bit tokens (0/1); attention runs over all input bits, so input
  dependence is *function-dependent* (attention gates which bits a token
  reads), in contrast to the dense MLP whose units saturate to full input
  support.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPCircuit(nn.Module):
    """d_in -> h -> h -> d_out with ReLU between. Four unit layers."""

    def __init__(self, d_in: int, h: int = 64, d_out: int = 2):
        super().__init__()
        self.d_in, self.h, self.d_out = d_in, h, d_out
        self.l1 = nn.Linear(d_in, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, d_out)
        self.relu = nn.ReLU()
        # six unit layers: x -> l1 -> relu -> l2 -> relu -> l3 -> out
        self.unit_dims = [d_in, h, h, h, h, d_out]
        # Jacobian kind per block: "linear" (exact W), "elementwise"
        # (diagonal), anything else uses autograd.
        self.block_kinds = ["linear", "elementwise", "linear",
                            "elementwise", "linear"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.l2(self.relu(self.l1(x))))
        return self.l3(h)

    def blocks(self):
        l1, l2, l3, relu = self.l1, self.l2, self.l3, self.relu
        return [
            lambda x: l1(x),        # u0 -> u1
            lambda x: relu(x),      # u1 -> u2
            lambda x: l2(x),        # u2 -> u3
            lambda x: relu(x),      # u3 -> u4
            lambda x: l3(x),        # u4 -> u5
        ]


class OneLayerTransformer(nn.Module):
    """1-layer attention + MLP transformer.

    Two layouts (see module docstring):

    * two-token (modular): input (n, 2p) -> embed per token -> attn over
      2 tokens -> per-token MLP -> mean-pool -> unembed.
    * sequence (boolean): input (n, n_in) bits -> one-hot per bit
      (n, n_in, 2) -> embed -> attn over n_in tokens -> per-token MLP ->
      mean-pool -> unembed.

    Blocks (both layouts):
        u0 (raw input) --embed--> u1 (n_tok*d)
        u1 --attn+residual--> u2 (n_tok*d)
        u2 --MLP+residual--> u3 (n_tok*d)
        u3 --pool+unembed--> u4 (d_out)
    """

    def __init__(self, p: int, d_model: int = 32, d_ff: int = 64,
                 n_heads: int = 2, d_out: int = 2, seq_mode: bool = False,
                 n_tokens: int | None = None, n_layers: int = 1):
        super().__init__()
        self.p, self.d_model, self.d_ff, self.n_heads = p, d_model, d_ff, n_heads
        self.seq_mode = seq_mode
        self.n_tokens = n_tokens if seq_mode else 2
        self.n_layers = n_layers
        assert d_model % n_heads == 0
        self.embed = nn.Linear(p, d_model)   # p = one-hot width per token
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.unembed = nn.Linear(d_model, d_out)
        nt = self.n_tokens
        # transitions: embed, (attn, mlp) x n_layers, readout
        self.unit_dims = [nt * p]
        for _ in range(n_layers):
            self.unit_dims += [nt * d_model, nt * d_model]
        self.unit_dims += [nt * d_model, d_out]
        self.block_kinds = (["linear"] +
                            ["attention", "mlp"] * n_layers +
                            ["linear"])

    # -- tokenization ------------------------------------------------------
    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(n, n_in) -> (n, n_tok, p) token feature sequence.

        Two-token mode: (n, 2p) -> (n, 2, p) one-hot blocks.
        Sequence mode: each bit is its own 1-dim token (n, n_in, 1); raw
        bits keep the tokenization *differentiable* so Jacobian extraction
        sees real paths, and each token still reads exactly one input bit.
        """
        if not self.seq_mode:
            return x.view(-1, 2, self.p)
        return x.unsqueeze(2)

    def _embed_block(self, x: torch.Tensor) -> torch.Tensor:
        # (n, n_tok, p) -> (n, n_tok, d) -> (n, n_tok*d)
        tok = self._tokens(x)
        return self.embed(tok).reshape(x.shape[0], -1)

    def _attn_block(self, z: torch.Tensor) -> torch.Tensor:
        # z (n, n_tok*d) -> residual attention over tokens
        nt = self.n_tokens
        z2 = z.view(-1, nt, self.d_model)
        q, k, v = self.q(z2), self.k(z2), self.v(z2)
        h = self.d_model // self.n_heads
        q = q.view(-1, nt, self.n_heads, h).transpose(1, 2)  # (n, H, nt, h)
        k = k.view(-1, nt, self.n_heads, h).transpose(1, 2)
        v = v.view(-1, nt, self.n_heads, h).transpose(1, 2)
        scores = q @ k.transpose(2, 3) / (h ** 0.5)          # (n, H, nt, nt)
        att = torch.softmax(scores, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(-1, nt, self.d_model)
        out = self.o(out)
        return (z2 + out).reshape(-1, nt * self.d_model)      # residual

    def _mlp_block(self, z: torch.Tensor) -> torch.Tensor:
        nt = self.n_tokens
        z2 = z.view(-1, nt, self.d_model)
        h = self.ff2(self.relu(self.ff1(z2)))
        return (z2 + h).reshape(-1, nt * self.d_model)

    def _readout_block(self, z: torch.Tensor) -> torch.Tensor:
        nt = self.n_tokens
        z2 = z.view(-1, nt, self.d_model)
        pooled = z2.mean(dim=1)                              # (n, d)
        return self.unembed(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._embed_block(x)
        for _ in range(self.n_layers):
            z = self._attn_block(z)
            z = self._mlp_block(z)
        return self._readout_block(z)

    def blocks(self):
        blks = [self._embed_block]
        for _ in range(self.n_layers):
            blks += [self._attn_block, self._mlp_block]
        blks.append(self._readout_block)
        return blks


def build_model(kind: str, task, **kw) -> nn.Module:
    if kind == "mlp":
        return MLPCircuit(task.n_input, kw.get("h", 64), task.n_output)
    if kind == "transformer":
        seq_mode = task.family == "boolean"
        return OneLayerTransformer(
            p=1 if seq_mode else task.p,
            d_model=kw.get("d_model", 32),
            d_ff=kw.get("d_ff", 64),
            n_heads=kw.get("n_heads", 2),
            d_out=task.n_output,
            seq_mode=seq_mode,
            n_tokens=task.n_input if seq_mode else None,
            n_layers=2 if seq_mode else 1,
        )
    raise ValueError(kind)
