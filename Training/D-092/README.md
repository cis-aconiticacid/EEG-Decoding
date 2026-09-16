# D-092: Sample-Efficient Latent Post-Training

Adapt the D089 local-convolution encoder to image latents with a small negative-target set and staged unfreezing.

## Experiment contract

- Training purpose: Adapt the D089 local-convolution encoder to image latents with a small negative-target set and staged unfreezing.
- Objective: Latent MSE, cosine, contrastive, and consistency losses; no classification head.
- Subject scope: Subject 0 only.
- Temporal-effect control: Development only: 27/3 trials from the first 30; official last-20 EEG is never forwarded.
- Status: Checkpointed implementation; no public scalar result record.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`scripts/run_d092_sample_efficient_posttrain.py`](scripts/run_d092_sample_efficient_posttrain.py)
- [`src/d092_sample_efficient_latent.py`](src/d092_sample_efficient_latent.py)
- [`config/d092_sample_efficient_posttrain.json`](config/d092_sample_efficient_posttrain.json)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
