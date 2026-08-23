# GNOmE — Graph Networks for Mechanistic Explicability

> **G**raph **N**etworks f**O**r **M**echanistic **E**xplicability.
> A model's computation is a graph; we extract it, measure it, and teach a
> graph network to read it.

GNOmE makes mechanistic interpretability *measurable and learnable*:

1. **Extract.** From a trained transformer, build the *circuit graph* — one node
   per attention head and MLP layer, edges weighted by Jacobian contribution
   strengths between consecutive layers. Extraction costs a single forward pass,
   O(N·L) versus O(N²) for path patching.
2. **Measure.** Zero-ablation attribution and path patching — two independent
   intervention methods — agree at r ≈ 0.51 on which heads matter. The
   computation graph encodes enough structure to predict unit importance.
3. **Predict.** A graph neural network trained on computation graphs predicts
   head importance — how much removing each unit hurts the model — **without
   running any intervention experiments**. Leave-one-out across independently
   trained models achieves Pearson r = 0.475.

## Headline results (all from `results/nmi_benchmark.json` + `results/nmi_full.json`)

| Finding | Value |
|---|---|
| Induction task val accuracy | 95.0–96.7% (3 seeds, 2-layer 4-head transformers) |
| Attribution ↔ path patching agreement | r = 0.508–0.529 (3 seeds) |
| Cross-model GNN transfer (leave-one-out) | r = 0.475 mean (range 0.449–0.520) |
| Extraction cost vs path patching | 1 forward pass vs N units (O(N·L) vs O(N²)) |
| Graph sparsity at τ=0.05 | 25/25 edges survive |
| Extraction speed | <0.1s per model (head-level granularity) |

## Why this matters

Attention maps, saliency, and patching-based circuits all try to answer graph
questions — which paths, which dependencies, which modules — with non-graph
instruments. GNOmE extracts the computation as a graph from real derivatives
and shows that a GNN trained on graph structure alone can predict which units
matter, **without running any interventions**. The graph is not a metaphor for
the computation. It is the computation — and it encodes function.

## Reproduce

```bash
pip install -r requirements.txt       # torch, numpy, networkx, matplotlib
python experiments/run_nmi.py --epochs 25 --n-seeds 3
python benchmarks/make_nmi_figures.py
cd manuscript && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

Everything runs on CPU in ~5 minutes.

## Repository layout

```
gnome/
  gnome/
    circuits.py       ground-truth circuit synthesis (boolean DAGs, modular Fourier)
    models.py         base models (MLP, transformer) exposing block structure
    extraction.py     blockwise Jacobian attribution → layered circuit graph
    metrics.py        explicability metrics + recovery + MES composite
    graphnets.py      pure-PyTorch GCN/GAT readout models
    trainee.py        2-layer transformer trained on IOI / induction tasks
    extract_small.py  head-level extraction from SmallTransformer
    training.py       training loops
  experiments/
    run_nmi.py        train models, extract, benchmark, cross-model GNN
    run_benchmark.py  original synthetic benchmark
  benchmarks/
    make_nmi_figures.py   NMI paper figures
    make_figures.py       original figures
  manuscript/
    paper.tex         Nature Machine Intelligence (Letters) draft
    paper_ieee.tex    IEEE conference-format version
    references.bib
  index.html          project page (GitHub Pages, served from repo root)
  results/            committed result JSONs
  figs/               committed figures
```

## Honesty notes

- All numbers from committed scripts on CPU (PyTorch); nothing simulated.
- The GNN uses **zero positional features** — the signal comes from graph
  structure alone.
- Cross-architecture transfer fails (r ≈ 0.32) — structural reading is
  architecture-specific. This is a negative result, reported as a result.
- IOI task accuracy ≈ 9% (near chance) — 2-layer model too small; this is
  expected and informative about model capacity vs. circuit complexity.

## Author

Sehaj Randhir Singh — independent researcher.

## Papers

- Nature Machine Intelligence (Letters) draft: `manuscript/paper.pdf`
- IEEE conference format: `manuscript/paper_ieee.pdf`
- Project page: `index.html` (GitHub Pages)
