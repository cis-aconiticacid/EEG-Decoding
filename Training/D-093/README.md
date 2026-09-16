# D-093: Stage-1 Classification Baseline

Train an 80-class classifier on the D089 encoder and compare classification with latent-retrieval experiments.

## Experiment contract

- Training purpose: Train an 80-class classifier on the D089 encoder and compare classification with latent-retrieval experiments.
- Objective: 80-way cross-entropy with label smoothing 0.05.
- Subject scope: Subject 0 only.
- Temporal-effect control: Not excluded: 27/3 development trials are drawn from the first 30, followed by refit on all 30 and one test on the last 20.
- Status: Completed with one official test evaluation.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`scripts/run_d093_stage1_classification_baseline.py`](scripts/run_d093_stage1_classification_baseline.py)
- [`src/d093_stage1_classifier.py`](src/d093_stage1_classifier.py)
- [`config/d093_stage1_classification_baseline.json`](config/d093_stage1_classification_baseline.json)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
