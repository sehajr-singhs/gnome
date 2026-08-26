"""Circuit-graph extraction via blockwise Jacobian attribution.

Given a model with an explicit block structure (models.py), we build a
layered graph whose nodes are the model's *units* (one node per neuron in
each unit layer) and whose edges carry the mean-absolute Jacobian of each
block:

    W_k[i, j] = E_x | d u_{k+1}[i] / d u_k[j] |

averaged over a batch of inputs. A block with zero Jacobian on an edge is
not a computational path, so thresholding W_k yields the *extracted
circuit graph*: the subgraph of the model that actually computes the task.

The extraction is:
* faithful  -- edges are real, measurable derivatives, not attention or
               saliency heuristics;
* layered   -- residual connections are folded into the block they close,
               so the graph is a DAG with unit layers as ranks;
* cheap     -- one batched autograd pass per block, no iterative pruning.

The threshold is set per layer as a multiple of the layer's mean edge
weight (``rel_thresh``), which keeps the extraction scale-free across
layers of different magnitudes.
"""

from __future__ import annotations

import numpy as np
import torch


def blockwise_jacobians(
    model,
    X: torch.Tensor,
    batch: int = 256,
) -> list[np.ndarray]:
    """Mean-absolute Jacobian per block.

    Args:
        model: nn.Module exposing ``blocks()`` (callables u_k -> u_{k+1})
            and ``unit_dims`` (list of layer widths).
        X: input tensor (n, d_in).
        batch: rows per autograd pass.

    Returns:
        list of (d_{k+1}, d_k) float matrices of mean-abs Jacobians.
    """
    blocks = list(model.blocks())
    kinds = getattr(model, "block_kinds", ["autograd"] * len(blocks))
    Ws: list[np.ndarray] = []
    n = X.shape[0]
    for k, blk in enumerate(blocks):
        d_out, d_in = model.unit_dims[k + 1], model.unit_dims[k]
        kind = kinds[k] if k < len(kinds) else "autograd"
        outs: list[torch.Tensor] = []
        if kind == "linear":
            # exact Jacobian: input-independent, computed once on the first
            # batch and reused (mean-abs |W| is constant across inputs).
            xb = X[:batch].detach().requires_grad_(True)
            out = blk(xb)
            W = np.zeros((d_out, d_in), dtype=np.float64)
            for r in range(out.shape[1]):
                g = torch.autograd.grad(
                    out[:, r].sum(), xb, retain_graph=True, allow_unused=True
                )[0]
                if g is not None:
                    W[r] = g.detach().abs().mean(dim=0).cpu().numpy()
            Ws.append(W.astype(np.float32))
            with torch.no_grad():
                X = blk(X).detach()
            continue
        acc = np.zeros((d_out, d_in), dtype=np.float64)
        n_seen = 0
        for i in range(0, n, batch):
            # the block's own input: previous block's output, detached and
            # made a leaf so the Jacobian below is d u_{k+1} / d u_k.
            xb = X[i : i + batch].detach().requires_grad_(True)
            out = blk(xb)
            outs.append(out.detach())
            if kind == "elementwise":
                # diagonal Jacobian: mean |f'(u_j)| per unit.
                g = torch.autograd.grad(
                    out.sum(), xb, retain_graph=True, allow_unused=True
                )[0]
                if g is not None:
                    diag_len = min(d_out, d_in, g.shape[-1])
                    acc[np.arange(diag_len), np.arange(diag_len)] += (
                        g.detach().abs().mean(dim=0).cpu().numpy()[:diag_len])
                n_seen += 1
                continue
            # out: (b, d_out). Accumulate mean-abs Jacobian row by row.
            for r in range(out.shape[1]):
                g = torch.autograd.grad(
                    out[:, r].sum(), xb, retain_graph=True, allow_unused=True
                )[0]
                if g is None:
                    continue
                acc[r] += g.detach().abs().mean(dim=0).cpu().numpy()
            n_seen += 1
        if n_seen:
            acc /= n_seen
        Ws.append(acc.astype(np.float32))
        X = torch.cat(outs, dim=0)  # feed the next block its true inputs
    return Ws


def threshold_edges(Ws: list[np.ndarray], rel_thresh: float = 0.5) -> list[list[tuple[int, int, float]]]:
    """Per-layer relative thresholding of Jacobian matrices.

    For each block k, keep edge (i, j) iff
        W_k[i, j] >= rel_thresh * mean(W_k[W_k > 0])
    i.e. the edge is at least ``rel_thresh`` times the layer's mean
    nonzero path strength. Returns per-layer edge lists of
    (src_unit, dst_unit, weight).
    """
    edges: list[list[tuple[int, int, float]]] = []
    for W in Ws:
        nz = W[W > 0]
        if nz.size == 0:
            edges.append([])
            continue
        thr = rel_thresh * float(nz.mean())
        m = W >= thr
        layer_edges = [
            (int(j), int(i), float(W[i, j]))
            for i, j in zip(*np.where(m))
        ]
        edges.append(layer_edges)
    return edges


def extract_circuit_graph(
    model,
    X: torch.Tensor,
    rel_thresh: float = 0.5,
    batch: int = 256,
) -> dict:
    """Full extraction: blockwise Jacobians -> thresholded layered graph.

    Returns a dict with:
        nodes:   [{"id": "u<layer>_<unit>", "layer": k, "role": ...}]
        edges:   list of (src_id, dst_id, weight)
        Ws:      raw per-layer Jacobian matrices
        unit_dims: model.unit_dims
    """
    Ws = blockwise_jacobians(model, X, batch=batch)
    unit_dims = list(model.unit_dims)
    nodes: list[dict] = []
    ids: list[list[str]] = []
    for k, d in enumerate(unit_dims):
        layer_ids = [f"u{k}_{j}" for j in range(d)]
        ids.append(layer_ids)
        for j in range(d):
            role = "input" if k == 0 else ("output" if k == len(unit_dims) - 1 else "hidden")
            nodes.append({"id": layer_ids[j], "layer": k, "role": role})
    edges_out: list[tuple[str, str, float]] = []
    for k, layer_edges in enumerate(threshold_edges(Ws, rel_thresh)):
        for (src, dst, w) in layer_edges:
            edges_out.append((ids[k][src], ids[k + 1][dst], w))
    return {"nodes": nodes, "edges": edges_out, "Ws": Ws,
            "unit_dims": unit_dims, "rel_thresh": rel_thresh}
