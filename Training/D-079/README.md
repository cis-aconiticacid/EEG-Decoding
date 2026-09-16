# D-079: Coordinate and Early-Fusion Ablation

Ablate coordinate bias and add a learned patch-by-channel table during continuation from D066.

## Experiment contract

- Training purpose: Ablate coordinate bias and add a learned patch-by-channel table during continuation from D066.
- Objective: Image-latent MSE plus 0.1 symmetric contrastive loss; not cross-entropy classification.
- Subject scope: Subject 0 only.
- Temporal-effect control: Not excluded: the historical first-30/last-20 source-order split is retained.
- Status: B0/B1/B2 warm-start arms completed through epoch 120.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`scripts/d079_run_no_coord_early_fusion.py`](scripts/d079_run_no_coord_early_fusion.py)
- [`src/d079_architecture.py`](src/d079_architecture.py)
- [`config/d079_no_coord_early_fusion.json`](config/d079_no_coord_early_fusion.json)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
