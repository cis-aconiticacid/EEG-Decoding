# D-080: Coordinate Ablations from Random Initialization

Repeat B0/B1/B2 without pretrained EEG-model weights to separate architecture effects from warm-start effects.

## Experiment contract

- Training purpose: Repeat B0/B1/B2 without pretrained EEG-model weights to separate architecture effects from warm-start effects.
- Objective: Image-latent MSE plus 0.1 symmetric contrastive loss; not cross-entropy classification.
- Subject scope: Subject 0 only.
- Temporal-effect control: Not excluded: training uses the first 30 source-order images per class and diagnostics use the last 20.
- Status: Three fixed 70-epoch arms completed.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`scripts/d080_run_from_scratch.py`](scripts/d080_run_from_scratch.py)
- [`src/d080_from_scratch.py`](src/d080_from_scratch.py)
- [`config/d080_from_scratch_no_coord.json`](config/d080_from_scratch_no_coord.json)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
