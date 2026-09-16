# D-089: Masked Waveform Pretraining and Retrieval

Pretrain convolution-routing encoders with masked waveform reconstruction, then align them to image latents for retrieval.

## Experiment contract

- Training purpose: Pretrain convolution-routing encoders with masked waveform reconstruction, then align them to image latents for retrieval.
- Objective: Masked waveform value/slope loss followed by latent MSE, cosine, and contrastive losses; no classification head.
- Subject scope: Subject 0 only.
- Temporal-effect control: Not excluded: 27/3 development trials are drawn from the first 30; the official last 20 are sealed until final evaluation.
- Status: Checkpointed implementation; no public scalar result record.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`scripts/run_d089_local_curve_pretrain.py`](scripts/run_d089_local_curve_pretrain.py)
- [`scripts/run_d089_corrected_masked_retrieval.py`](scripts/run_d089_corrected_masked_retrieval.py)
- [`src/d089_masked_retrieval.py`](src/d089_masked_retrieval.py)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
