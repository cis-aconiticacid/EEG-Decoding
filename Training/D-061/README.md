# D-061: Multi-Subject Paper-Latent Alignment

Learn an EEG representation aligned to frozen image latents and evaluate nearest-image retrieval.

## Experiment contract

- Training purpose: Learn an EEG representation aligned to frozen image latents and evaluate nearest-image retrieval.
- Objective: Image-latent MSE plus 0.1 symmetric contrastive loss; not cross-entropy classification.
- Subject scope: Sixteen subjects, fitted independently for each subject/task combination.
- Temporal-effect control: Not excluded: training uses the first 30 source-order images per class and evaluation uses the last 20.
- Status: Completed historical runs are documented.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`scripts/run_d061_paper_latent.py`](scripts/run_d061_paper_latent.py)
- [`config/d061_paper_latent_s17.json`](config/d061_paper_latent_s17.json)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
