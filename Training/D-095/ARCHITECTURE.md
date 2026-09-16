# D-095 Architecture

![D-095 architecture](architecture.png)

The image above is rendered from [`ARCHITECTURE.tex`](ARCHITECTURE.tex).

## Data flow

1. EEG 62 x 400 — 40--440 ms
2. Waveform multiscale conv — + aligned 12--80-Hz FFT
3. Learned gated fusion — 192-D tokens
4. 4 temporal/spatial attention blocks
5. Electrode pooling — 3072-to-80 classifier

## Training interface

- Objective: Ordinary 80-way cross-entropy without label smoothing.
- Subject scope: Subject 0 only.
- Temporal-effect control: Not excluded: the first 30 source-order images per class train the model and the last 20 form the official test.

## Authoritative implementation

- [`model.py`](model.py)
- [`run.py`](run.py)
- [`config/config.json`](config/config.json)
