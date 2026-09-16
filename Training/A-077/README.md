# A-077: Pairwise Identity and Temporal-Position Diagnostics

`A` stands for **Analysis**: this numbered record evaluates diagnostic structure in EEG features rather than training a neural decoding model.

Test whether PSD differences predict subject identity (T1) or class-block position proximity (T3).

## Experiment contract

- Training purpose: Test whether PSD differences predict subject identity (T1) or class-block position proximity (T3).
- Objective: RBF SVC and random-forest binary classification; no neural embedding objective.
- Subject scope: All sixteen subjects.
- Temporal-effect control: Mixed: T1 uses image-disjoint 48/16/16 splits; T3 deliberately measures position-order effects. T2 is omitted because no session label exists.
- Status: T1 and T3 completed; T2 intentionally unavailable.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`scripts/a077_prepare_pair_tasks.py`](scripts/a077_prepare_pair_tasks.py)
- [`scripts/a077_run_models.py`](scripts/a077_run_models.py)
- [`config/a077_pair_tasks.json`](config/a077_pair_tasks.json)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
