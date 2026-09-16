# D-088: Convolution-Routing Comparison

Compare four local/global convolution routes under masked pretraining and supervised fine-tuning.

## Experiment contract

- Training purpose: Compare four local/global convolution routes under masked pretraining and supervised fine-tuning.
- Objective: Masked JEPA-style pretraining, then 80-way cross-entropy classification.
- Subject scope: Subject 0 only.
- Temporal-effect control: Not excluded: 27/3 development trials come from the first 30 source-order images; the last 20 form a one-time test.
- Status: Four-route benchmark completed.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`scripts/run_d088_subject0_conv_routing.py`](scripts/run_d088_subject0_conv_routing.py)
- [`src/d088_dual_attention.py`](src/d088_dual_attention.py)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
