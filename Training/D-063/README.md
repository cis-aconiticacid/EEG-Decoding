# D-063: Raw and Local-Frequency Classification Probe

Compare raw-waveform, frequency-only, and fused inputs for 80-class EEG classification.

## Experiment contract

- Training purpose: Compare raw-waveform, frequency-only, and fused inputs for 80-class EEG classification.
- Objective: Ordinary 80-way cross-entropy classification.
- Subject scope: Subject 0 only.
- Temporal-effect control: Not excluded: the first 30 source-order images per class train the probe and the last 20 form the test set.
- Status: Three-arm, three-seed probe completed.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`scripts/run_d063_local_frequency_probe.py`](scripts/run_d063_local_frequency_probe.py)
- [`config/d063_local_frequency_probe.json`](config/d063_local_frequency_probe.json)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
