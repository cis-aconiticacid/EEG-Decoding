# D-094: Multi-Scale Temporal/Spatial CNN

Train the requested multi-scale temporal and spatial CNN for 80-class EEG decoding.

## Experiment contract

- Training purpose: Train the requested multi-scale temporal and spatial CNN for 80-class EEG decoding.
- Objective: 80-way cross-entropy with label smoothing 0.05.
- Subject scope: Subject 0 only.
- Temporal-effect control: Development only: 27/3 trials from the first 30; the official last 20 receive zero classifier forwards.
- Status: Development run completed; official test remains sealed.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`scripts/run_d094_multiscale_cnn_stage1.py`](scripts/run_d094_multiscale_cnn_stage1.py)
- [`src/d094_multiscale_cnn.py`](src/d094_multiscale_cnn.py)
- [`config/d094_multiscale_cnn_stage1.json`](config/d094_multiscale_cnn_stage1.json)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
