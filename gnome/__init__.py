"""GNOmE: Graph Networks for Mechanistic Explicability.

A research codebase that treats a trained model's computation as a graph,
extracts that graph from the weights, measures explicability as graph
properties, and learns graph networks that read explicability directly
from the circuit graph.

Project layout:
    gnome/circuits.py    ground-truth circuit synthesis (known graphs)
    gnome/models.py      MLP and 1-layer transformer with explicit blocks
    gnome/extraction.py  blockwise Jacobian circuit-graph extraction
    gnome/metrics.py     explicability + recovery metrics
    gnome/graphnets.py   pure-torch GCN/GAT over circuit graphs
    gnome/training.py    training loop
"""

from .circuits import (  # noqa: F401
    CircuitTask,
    ModularTask,
    RandomBooleanCircuit,
    make_task,
)
from .models import MLPCircuit, OneLayerTransformer  # noqa: F401
from .extraction import extract_circuit_graph  # noqa: F401
from .metrics import (  # noqa: F401
    explicability_metrics,
    mes_score,
    recovery_wiring_overlap,
)
from .graphnets import CircuitGNN, GATLayer, GCNLayer  # noqa: F401

__version__ = "0.1.0"
