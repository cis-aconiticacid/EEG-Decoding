# D-085: Exploratory Riemannian SVM Record

Preserve an exploratory covariance/tangent-space subject-classification result from a non-EEG-ImageNet dataset.

## Experiment contract

- Training purpose: Preserve an exploratory covariance/tangent-space subject-classification result from a non-EEG-ImageNet dataset.
- Objective: Covariance features, tangent-space projection, and SVM classification.
- Subject scope: Per-subject evaluation on the exploratory external dataset.
- Temporal-effect control: Not applicable to the EEG-ImageNet first-30/last-20 protocol.
- Status: Historical record only; no public runnable source is retained in this folder.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- No runnable public source is retained for this historical record.

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
