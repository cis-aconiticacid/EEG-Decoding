# D-083: Synthetic CUDA Feasibility Check

Verify that the D081 joint waveform/frequency pipeline can execute its two training stages on CUDA.

## Experiment contract

- Training purpose: Verify that the D081 joint waveform/frequency pipeline can execute its two training stages on CUDA.
- Objective: Stage-A masked reconstruction and Stage-B latent alignment on synthetic tensors.
- Subject scope: No real subject data; synthetic feasibility only.
- Temporal-effect control: Not applicable: no EEG-ImageNet train/test split is used.
- Status: Feasibility check passed; this is not an accuracy experiment.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`scripts/d083_local_gpu_feasibility.py`](scripts/d083_local_gpu_feasibility.py)
- [`src/d081_joint_waveform_frequency.py`](src/d081_joint_waveform_frequency.py)
- [`src/d081_training.py`](src/d081_training.py)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
