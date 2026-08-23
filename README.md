# GNOmE — Graph Networks for Mechanistic Explicability

**Zero-Query Circuit Extraction from Trained Neural Networks**

[![Nature Machine Intelligence](https://img.shields.io/badge/Nature%20Machine%20Intelligence-paper-blue)](figs/gnome_nmi.pdf)
[![IEEE](https://img.shields.io/badge/IEEE-conference-green)](figs/gnome_ieee.pdf)
[![Project Page](https://img.shields.io/badge/Project-Page-orange)](https://sehajr-singhs.github.io/gnome/)

## Overview

GNOmE extracts the computational graph from a trained neural network in a single forward pass, then reads it with a GNN to predict per-head importance **without any model interventions**.

### Key Results

| Method | Correlation (r) | Precision@3 | Query Cost |
|--------|----------------|-------------|------------|
| **GNOmE** | **0.748 ± 0.082** | **0.667** | **O(1)** |
| Path Patching | −0.365 ± 0.218 | 0.278 | O(n_L · n_H) |

### Cross-Model Transfer

| Transfer | Correlation |
|----------|------------|
| IOI → Induction | **0.954** |
| Induction → IOI | **0.963** |
| LOO CV | **0.864 ± 0.201** |

## Installation

```bash
pip install torch numpy matplotlib scikit-learn
```

## Usage

```python
from gnome.trainee import SmallTransformer, train_on_ioi
from gnome.extract_small import extract_circuit, compute_head_importance

# Train a small transformer
model = SmallTransformer(vocab_size=8, d_model=64, n_heads=4, n_layers=2)
info = train_on_ioi(model, epochs=80)

# Extract computation graph
circuit = extract_circuit(model, vocab_size=8, seq_len=8)

# Compute head importance from graph structure (zero queries!)
gnome_importance = graph_centrality(circuit['adj_matrix'])
```

## Reproduce Results

```bash
# Phase 1: Train models + extract circuits
python experiments/nmi_full.py
```

## Citation

```bibtex
@article{singh2025gnome,
  title={GNOmE: Graph Networks for Mechanistic Explicability},
  author={Singh, Sehaj},
  year={2025}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
