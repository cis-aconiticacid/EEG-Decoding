# D-064: Masked Frequency Pretraining and Linear Probe

Measure how frequency-patch masking affects a self-supervised encoder and frozen linear-probe performance.

## Experiment contract

- Training purpose: Measure how frequency-patch masking affects a self-supervised encoder and frozen linear-probe performance.
- Objective: Masked reconstruction pretraining followed by an 80-way linear probe; probe training uses cross-entropy.
- Subject scope: Subject 0 only.
- Temporal-effect control: Partially controlled inside the first 30: positions 0--23 fit and 24--29 validate; the official last 20 are excluded from evaluation.
- Status: Mask-ratio and seed sweep completed.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`scripts/run_d064_mask_ratio.py`](scripts/run_d064_mask_ratio.py)
- [`config/d064_local_mask_ratio.json`](config/d064_local_mask_ratio.json)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
