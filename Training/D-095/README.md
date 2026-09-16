# D-095: Hybrid Temporal/Spatial Classifier

Combine aligned waveform and local-frequency tokens in a hybrid temporal/spatial classifier.

## Experiment contract

- Training purpose: Combine aligned waveform and local-frequency tokens in a hybrid temporal/spatial classifier.
- Objective: Ordinary 80-way cross-entropy without label smoothing.
- Subject scope: Subject 0 only.
- Temporal-effect control: Not excluded: the first 30 source-order images per class train the model and the last 20 form the official test.
- Status: Fixed 100-epoch baseline and ablations completed.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`model.py`](model.py)
- [`run.py`](run.py)
- [`config/config.json`](config/config.json)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
