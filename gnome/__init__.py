"""GNOmE: Graph Networks for Mechanistic Explicability.

A research codebase that treats a trained model's computation as a graph,
extracts that graph from the weights, measures explicability as graph
properties, and learns graph networks that read explicability directly
from the circuit graph.

Project layout:
    gnome/circuits.py       ground-truth circuit synthesis (known graphs)
    gnome/models.py         MLP and 1-layer transformer with explicit blocks
    gnome/extraction.py     blockwise Jacobian circuit-graph extraction
    gnome/extract_small.py  circuit extraction from SmallTransformer
    gnome/real_models.py    GPT-2 and ResNet-18 circuit extraction
    gnome/llama_extract.py  Llama/Mistral/Gemma circuit extraction
    gnome/metrics.py        explicability + recovery metrics
    gnome/graphnets.py      pure-torch GCN/GAT over circuit graphs
    gnome/trainee.py        small transformer training for IOI/induction
    gnome/acdc_benchmark.py GNOmE vs ACDC vs Attribution Patching
    gnome/auto_discovery.py automated circuit discovery pipeline
    gnome/training.py       training loop
"""

from .circuits import (  # noqa: F401
    CircuitTask,
    ModularTask,
    RandomBooleanCircuit,
    make_task,
)
from .models import MLPCircuit, OneLayerTransformer  # noqa: F401
from .extraction import extract_circuit_graph  # noqa: F401
from .extract_small import extract_circuit, compute_head_importance  # noqa: F401
from .metrics import (  # noqa: F401
    explicability_metrics,
    mes_score,
    recovery_wiring_overlap,
)
from .graphnets import CircuitGNN, GATLayer, GCNLayer  # noqa: F401
from .acdc_benchmark import CircuitBenchmark, run_full_benchmark  # noqa: F401
from .auto_discovery import AutoCircuitDiscovery, GraphCircuitScorer  # noqa: F401

__version__ = "0.2.0"
