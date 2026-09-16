# D-078: A0--A3 Continuation and Augmentation Study

Continue the D066 latent model while comparing four waveform-augmentation arms.

## Experiment contract

- Training purpose: Continue the D066 latent model while comparing four waveform-augmentation arms.
- Objective: Image-latent MSE plus 0.1 symmetric contrastive loss; not cross-entropy classification.
- Subject scope: Subject 0 only.
- Temporal-effect control: Not excluded: the historical first-30/last-20 source-order split is retained.
- Status: Four warm-start continuation arms completed through epoch 120.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`scripts/d078_run_continuation.py`](scripts/d078_run_continuation.py)
- [`src/d078_augmentation.py`](src/d078_augmentation.py)
- [`config/d078_a100_continuation.json`](config/d078_a100_continuation.json)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
