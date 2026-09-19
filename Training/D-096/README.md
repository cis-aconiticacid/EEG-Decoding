# D-096: Position-Aware Waveform/Frequency Cross-Attention

Replace D-095's scalar frequency gate with directional cross-attention while preserving its waveform frontend, frequency frontend, temporal/spatial trunk, readout, and fixed training contract.

## Experiment contract

- Training purpose: Test whether waveform queries can retrieve useful local-frequency context without collapsing both sources through an immediate elementwise sum.
- Objective: Ordinary 80-way cross-entropy without label smoothing.
- Subject scope: Subject 0 only.
- Temporal-effect control: Not excluded; the first 30 source-order images per class train the model and the last 20 form the official test.
- Status: One fixed 100-epoch A100 run completed; test accuracy was 46.8125% (749/1600).

## Cross-attention contract

- Waveform tokens are queries; frequency tokens are keys and values.
- Attention runs over 16 temporal patches independently for each of 62 electrodes.
- Electrode identity, unit-sphere montage coordinates, and temporal position condition query/key matching.
- Values contain normalized frequency content only.
- A learned residual scale starts at 0.1, preserving a waveform-dominant initialization.
- The original four-block temporal/spatial trunk and classifier remain unchanged.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`model.py`](model.py)
- [`run.py`](run.py)
- [`config/config.json`](config/config.json)

Raw EEG data, generated runs, and checkpoints are excluded from Git. Use [`../../scripts/download_eeg_imagenet.py`](../../scripts/download_eeg_imagenet.py) to download and verify the official archives explicitly, or let the shared loader fetch the required archive on first use.
