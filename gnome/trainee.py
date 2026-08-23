"""Small trainable transformer for learning mechanistic circuits.

We train a tiny 2-layer transformer on tasks with known solutions
(induction heads, Indirect Object Identification). This lets us:
  1. Extract the computation graph from the trained model
  2. Identify which heads/mlp layers are important
  3. Compare GNOmE's graph-based method against path patching

Architecture: 2 layers, 4 heads, 128-dim (~460K params — trains in ~2 min on CPU).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MultiHeadAttention(nn.Module):
    """Standard MHA with exposed per-head outputs for circuit extraction."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        head_out = torch.matmul(attn_probs, v)  # (B, H, S, d_head)
        concat = head_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(concat)


class FeedForward(nn.Module):
    """2-layer MLP with GeLU."""

    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.ln1(x))
        h = h + self.ff(self.ln2(h))
        return h


class SmallTransformer(nn.Module):
    """Small transformer for learning mechanistic circuits.

    ~460K params (2 layers, 4 heads, 128-dim).
    """

    def __init__(self, vocab_size: int = 64, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, max_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)

        for block in self.blocks:
            h = block(h)
        h = self.ln_final(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
            )
        return logits, loss

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -------------------------------------------------------------------
# Task generators
# -------------------------------------------------------------------

def _gen_ioi_batch(n, seq_len, vocab_size, device):
    """Generate IOI training examples.

    Pattern: [S1][AND][S2][WENT][filler...][S1][AND][?]
    Target at end: S2

    The model must copy S2 from position 2 to the final position,
    which requires S1-tracking, S2-tracking, and name-mover heads.
    """
    AND, WENT = 1, 2
    X = torch.zeros(n, seq_len, dtype=torch.long, device=device)
    Y = torch.zeros(n, seq_len, dtype=torch.long, device=device)
    for i in range(n):
        s1 = torch.randint(4, vocab_size, (1,)).item()
        s2 = torch.randint(4, vocab_size, (1,)).item()
        filler_len = seq_len - 6
        fillers = torch.randint(4, vocab_size, (filler_len,))
        seq = [s1, AND, s2, WENT] + fillers.tolist() + [s1, AND, 0]
        seq = seq[:seq_len]
        X[i] = torch.tensor(seq, dtype=torch.long)
        # Y[t] = what X[t+1] is (next-token prediction)
        Y[i, :-1] = X[i, 1:]
        Y[i, -1] = s2  # ground truth for the final position
    return X, Y


def _gen_induction_batch(n, seq_len, vocab_size, device):
    """Generate induction head training examples.

    Pattern: [A][B][filler...][A][?]
    Target at end: B

    The model must recognize that [A] at position 0 predicts [B] at
    position 1, and copy that to the final position.
    """
    X = torch.zeros(n, seq_len, dtype=torch.long, device=device)
    Y = torch.zeros(n, seq_len, dtype=torch.long, device=device)
    for i in range(n):
        seq = torch.randint(4, vocab_size, (seq_len,))
        A = torch.randint(4, vocab_size, (1,)).item()
        B = torch.randint(4, vocab_size, (1,)).item()
        # Place A at position 0, B at position 1, A again near end
        seq[0] = A
        seq[1] = B
        seq[-2] = A  # second occurrence of A
        X[i] = seq
        Y[i, :-1] = X[i, 1:]
        Y[i, -1] = B  # should predict B after the second A
    return X, Y


# -------------------------------------------------------------------
# Training loops
# -------------------------------------------------------------------

def _accuracy_at_target(logits, targets):
    """Compute accuracy at the position that matters.

    Loss uses logits[:, :-1] vs targets[:, 1:].
    So logits[:, t] predicts targets[:, t+1].
    The 'final target' is targets[:, -1], which is predicted by logits[:, -2].
    """
    pred = logits[:, -2, :].argmax(dim=-1)  # what model predicts for last target
    true = targets[:, -1]                   # the actual last target
    return (pred == true).float().mean().item()


def train_on_ioi(
    model: SmallTransformer,
    vocab_size: int = 64,
    seq_len: int = 12,
    n_train: int = 4000,
    n_val: int = 1000,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Train the small transformer on the IOI task."""
    model = model.to(device)

    train_X, train_Y = _gen_ioi_batch(n_train, seq_len, vocab_size, device)
    val_X, val_Y = _gen_ioi_batch(n_val, seq_len, vocab_size, device)

    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_Y), batch_size=batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            _, loss = model(xb, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        model.eval()
        val_losses, accs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits, loss = model(xb, yb)
                val_losses.append(loss.item())
                accs.append(_accuracy_at_target(logits, yb))

        avg_train = sum(losses) / len(losses)
        avg_val = sum(val_losses) / len(val_losses)
        avg_acc = sum(accs) / len(accs)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(avg_acc)

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}: train={avg_train:.4f} val={avg_val:.4f} acc={avg_acc:.3f}")

    return {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "history": history,
    }


def train_on_induction(
    model: SmallTransformer,
    vocab_size: int = 64,
    seq_len: int = 12,
    n_train: int = 4000,
    n_val: int = 1000,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Train on the induction task."""
    model = model.to(device)

    train_X, train_Y = _gen_induction_batch(n_train, seq_len, vocab_size, device)
    val_X, val_Y = _gen_induction_batch(n_val, seq_len, vocab_size, device)

    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_Y), batch_size=batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            _, loss = model(xb, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        model.eval()
        val_losses, accs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits, loss = model(xb, yb)
                val_losses.append(loss.item())
                accs.append(_accuracy_at_target(logits, yb))

        avg_train = sum(losses) / len(losses)
        avg_val = sum(val_losses) / len(val_losses)
        avg_acc = sum(accs) / len(accs)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(avg_acc)

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}: train={avg_train:.4f} val={avg_val:.4f} acc={avg_acc:.3f}")

    return {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "history": history,
    }
