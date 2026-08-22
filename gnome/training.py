"""Training loops for base models (MLP / transformer) and the GNN."""

from __future__ import annotations

import torch
import torch.nn as nn


def train_classifier(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 400,
    lr: float = 1e-3,
    batch: int = 64,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Train a classification model on (X, y). Returns final metrics."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = X.shape[0]
    best = {"acc": 0.0, "loss": float("inf")}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot_loss, tot_correct = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb, yb = X[idx], y[idx]
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            tot_correct += (out.argmax(1) == yb).sum().item()
        acc = tot_correct / n
        if acc > best["acc"]:
            best = {"acc": acc, "loss": tot_loss / n}
        if verbose and (ep + 1) % 100 == 0:
            print(f"  ep {ep+1}/{epochs} loss {tot_loss/n:.4f} acc {acc:.4f}")
    model.eval()
    with torch.no_grad():
        out = model(X)
        acc = (out.argmax(1) == y).float().mean().item()
    return {"accuracy": float(acc), "best_accuracy": float(best["acc"]),
            "final_loss": float(tot_loss / n)}


def train_gnn(
    model: nn.Module,
    data: dict,
    role_targets: torch.Tensor,
    graph_targets: torch.Tensor | None,
    epochs: int = 300,
    lr: float = 1e-3,
    seed: int = 0,
    w_graph: float = 0.5,
    verbose: bool = False,
) -> dict:
    """Train CircuitGNN. role_targets: (N,) ints; graph_targets: (G, T)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = role_targets.shape[0]
    G = data["n_graphs"]
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data)
        loss = nn.functional.cross_entropy(out["roles"], role_targets)
        if graph_targets is not None and "graph" in out:
            loss = loss + w_graph * nn.functional.mse_loss(out["graph"],
                                                           graph_targets)
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 50 == 0:
            acc = (out["roles"].argmax(1) == role_targets).float().mean().item()
            print(f"  ep {ep+1} loss {loss.item():.4f} role_acc {acc:.4f}")
    model.eval()
    with torch.no_grad():
        out = model(data)
        role_acc = (out["roles"].argmax(1) == role_targets).float().mean().item()
        res = {"role_accuracy": float(role_acc), "N": int(N), "G": int(G)}
        if graph_targets is not None and "graph" in out:
            mae = (out["graph"] - graph_targets).abs().mean().item()
            res["graph_mae"] = float(mae)
    return res
