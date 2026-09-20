# D-098: One Independent Model per Participant

Train the D-097 winning architecture independently for all sixteen EEG-ImageNet participants. Every participant gets separate normalization statistics, model initialization, optimizer state, checkpoint, predictions, and reports; no parameters are shared across participants.

## Experiment contract

- Architecture: waveform-query-frequency cross-attention with the three retained FFT bins below 25 Hz masked and post-cross-attention `+P` retained.
- Objective: ordinary 80-way cross-entropy without label smoothing.
- Participants: IDs 0--15, one model per participant.
- Split: first 30 source-order images per available class for training; remaining 20 for one endpoint test.
- Budget: seed 17, fixed 100 epochs, AdamW, identical hyperparameters for every participant.
- Archives: IDs 0--7 use `EEG-ImageNet_1.pth`; IDs 8--15 use `EEG-ImageNet_2.pth`.

## Source-data exception

Participant 2 has 3,950 trials in the verified official Part 1 archive because class index 1 (`n07749192`) is entirely absent. Participant 12 has 3,900 trials in verified Part 2 because class indices 29 (`n04249415`) and 71 (`n03452741`) are absent. D-098 does not synthesize or borrow trials. Their models use 79 and 78 available classes respectively while retaining the shared 80-logit head. Both available-class macro accuracy and all-80-output macro accuracy are recorded.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`model.py`](model.py)
- [`run.py`](run.py)
- [`config/config.json`](config/config.json)

Raw data, generated runs, predictions, and checkpoints are excluded from Git.

## Completed result

All sixteen independent 100-epoch A100 runs completed. Mean participant accuracy is 37.0685%, pooled sample accuracy is 37.1026%, and individual participant accuracy ranges from 15.2564% to 57.5000%. See [Results analysis](RESULTS_ANALYSIS.md) for the full table, data hashes, missing-class handling, and artifact locations.
