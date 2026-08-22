"""Graph networks that read explicability directly from circuit graphs.

The core GNOmE claim: a model's computation is a graph, and a graph
network is the natural readout of that graph. ``CircuitGNN`` is a
pure-PyTorch GCN/GAT stack that consumes an extracted circuit graph --
node features (layer + degree) and edge features (Jacobian strength) --
and predicts either

* node roles (input / hidden / output), or
* graph-level explicability metrics (sparsity, modularity, depth).

Everything is implemented by hand (no torch-geometric dependency) so the
method is reproducible on any torch install.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph batching helpers
# ---------------------------------------------------------------------------

def graph_to_batch(graphs: list[dict], n_layers: int | None = None,
                   positional: bool = True) -> dict:
    """Pack a list of extracted circuit graphs into one batched tensor set.

    Node features (positional=True): one-hot layer index ++ normalized
    out-degree ++ normalized in-degree.
    Node features (positional=False): degree and edge-weight statistics
    ONLY --- no layer information, so depth/role must be inferred from
    structure. Features: out-deg norm, in-deg norm, mean outgoing edge
    weight, mean incoming edge weight.
    Edge features: log(1 + Jacobian weight).
    Returns {"x", "edge_index", "edge_attr", "batch", "n_graphs"}.
    """
    if n_layers is None:
        n_layers = max((max(nd["layer"] for nd in g["nodes"]) for g in graphs),
                       default=1) + 1
    xs, es, ews, bs = [], [], [], []
    for gi, g in enumerate(graphs):
        node_map = {nd["id"]: i for i, nd in enumerate(g["nodes"])}
        N = len(g["nodes"])
        deg_out = [0] * N
        deg_in = [0] * N
        w_out = [0.0] * N
        w_in = [0.0] * N
        for e in g["edges"]:
            w = e[2] if len(e) > 2 else 1.0
            deg_out[node_map[e[0]]] += 1
            deg_in[node_map[e[1]]] += 1
            w_out[node_map[e[0]]] += w
            w_in[node_map[e[1]]] += w
        max_deg = max(deg_out + deg_in, default=1)
        feats = []
        for nd in g["nodes"]:
            i = node_map[nd["id"]]
            if positional:
                layer_oh = [0.0] * n_layers
                layer_oh[min(nd["layer"], n_layers - 1)] = 1.0
                feats.append(layer_oh + [deg_out[i] / max_deg,
                                         deg_in[i] / max_deg])
            else:
                mo = w_out[i] / deg_out[i] if deg_out[i] else 0.0
                mi = w_in[i] / deg_in[i] if deg_in[i] else 0.0
                feats.append([deg_out[i] / max_deg, deg_in[i] / max_deg,
                              mo, mi])
        xs.append(torch.tensor(feats, dtype=torch.float32))
        for e in g["edges"]:
            es.append((node_map[e[0]], node_map[e[1]]))
            ews.append(math.log1p(e[2]) if len(e) > 2 else 0.0)
        bs.extend([gi] * N)
    n_feat = xs[0].shape[1] if xs else 1
    x = torch.cat(xs, dim=0) if xs else torch.zeros(0, n_feat)
    edge_index = torch.tensor(es, dtype=torch.long).t().contiguous() if es \
        else torch.zeros(2, 0, dtype=torch.long)
    edge_attr = torch.tensor(ews, dtype=torch.float32).unsqueeze(1) if ews \
        else torch.zeros(0, 1)
    batch = torch.tensor(bs, dtype=torch.long)
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr,
            "batch": batch, "n_graphs": len(graphs)}


def _normalize_adj(edge_index: torch.Tensor, edge_attr: torch.Tensor,
                   n_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-normalized adjacency (D^-1 A) so messages are stable."""
    if edge_index.shape[1] == 0:
        return edge_index, edge_attr
    src, dst = edge_index[0], edge_index[1]
    deg = torch.zeros(n_nodes, device=edge_index.device)
    deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    deg = deg.clamp(min=1.0)
    w = edge_attr.squeeze(1) / deg[dst]
    return edge_index, w.unsqueeze(1)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """Mean-aggregation GCN with edge weights (D^-1 A X W)."""

    def __init__(self, in_f: int, out_f: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, n_nodes):
        ei, ea = _normalize_adj(edge_index, edge_attr, n_nodes)
        out = torch.zeros_like(x)
        if ei.shape[1] > 0:
            src, dst = ei[0], ei[1]
            msg = x[src] * ea                       # (E, F)
            out = out.index_add(0, dst, msg)
        out = out + x                               # self-loop
        return self.dropout(F.relu(self.lin(out)))


class GATLayer(nn.Module):
    """Graph attention: alpha from concat(src,dst) features, per edge."""

    def __init__(self, in_f: int, out_f: int, n_heads: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_f // n_heads
        assert self.head_dim * n_heads == out_f
        self.lin = nn.Linear(in_f, out_f, bias=False)
        self.att = nn.Parameter(torch.empty(2 * self.head_dim))
        nn.init.xavier_uniform_(self.att.view(1, -1))
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, n_nodes):
        h = self.lin(x).view(-1, self.n_heads, self.head_dim)  # (N, H, D)
        out = torch.zeros_like(h)
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            hs, hd = h[src], h[dst]
            alpha = self.leaky((torch.cat([hs, hd], dim=-1) * self.att).sum(-1))
            # softmax over incoming edges per node
            max_a = torch.zeros(n_nodes, self.n_heads, device=x.device)
            max_a = max_a.scatter_reduce(0, dst.unsqueeze(1).expand(-1, self.n_heads),
                                         alpha, reduce="amax", include_self=False)
            exp = torch.exp(alpha - max_a[dst])
            denom = torch.zeros(n_nodes, self.n_heads, device=x.device)
            denom = denom.scatter_add(0, dst.unsqueeze(1).expand(-1, self.n_heads), exp)
            attn = exp / (denom[dst].clamp(min=1e-8))
            msg = hs * attn.unsqueeze(-1) * edge_attr.unsqueeze(-1).clamp(min=1e-3)
            out = out.scatter_add(0, dst.unsqueeze(1).unsqueeze(-1).expand(-1, self.n_heads, self.head_dim), msg)
        out = out + h  # self-loop
        out = out.reshape(-1, self.n_heads * self.head_dim)
        return self.dropout(F.relu(out))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CircuitGNN(nn.Module):
    """GNN reading circuit graphs.

    Args:
        n_layers: number of unit layers (feature dict size).
        hidden: hidden width.
        n_roles: number of node-role classes (3: input/hidden/output).
        n_graph_targets: number of graph-level regression targets
            (explicability metrics), or 0 to disable the graph head.
        kind: "gcn" or "gat".
    """

    def __init__(self, n_layers: int, hidden: int = 64, n_roles: int = 3,
                 n_graph_targets: int = 0, kind: str = "gcn",
                 depth: int = 2, dropout: float = 0.1, in_f: int | None = None):
        super().__init__()
        if in_f is None:
            in_f = n_layers + 2
        self.kind = kind
        self.n_graph_targets = n_graph_targets
        self.in_proj = nn.Linear(in_f, hidden)
        self.gnn_layers = nn.ModuleList()
        for _ in range(depth):
            if kind == "gat":
                self.gnn_layers.append(GATLayer(hidden, hidden, n_heads=2,
                                                dropout=dropout))
            else:
                self.gnn_layers.append(GCNLayer(hidden, hidden, dropout=dropout))
        self.role_head = nn.Linear(hidden, n_roles)
        if n_graph_targets > 0:
            self.pool = nn.Linear(hidden, hidden)
            self.graph_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_graph_targets))

    def forward(self, data: dict, return_graph: bool = False):
        x = self.in_proj(data["x"])
        ei, ea = data["edge_index"], data["edge_attr"]
        n = data["x"].shape[0]
        for layer in self.gnn_layers:
            x = layer(x, ei, ea, n)
        role_logits = self.role_head(x)
        out = {"roles": role_logits}
        if self.n_graph_targets > 0:
            pooled = torch.zeros(data["n_graphs"], x.shape[1], device=x.device)
            pooled = pooled.index_add(0, data["batch"], self.pool(x))
            counts = torch.bincount(data["batch"], minlength=data["n_graphs"])
            pooled = pooled / counts.unsqueeze(1).clamp(min=1.0)  # mean pool
            out["graph"] = self.graph_head(pooled)
        if return_graph:
            out["node_feats"] = x
        return out


def build_gnn(n_layers: int, n_graph_targets: int = 0, kind: str = "gcn",
              hidden: int = 64) -> CircuitGNN:
    return CircuitGNN(n_layers=n_layers, hidden=hidden, n_roles=3,
                      n_graph_targets=n_graph_targets, kind=kind)
