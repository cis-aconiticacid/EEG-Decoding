# D-066: Latent Fine-Tuning from a Masked EEG Encoder

Fine-tune the masked EEG encoder to predict image latents and evaluate image-gallery retrieval.

## Experiment contract

- Training purpose: Fine-tune the masked EEG encoder to predict image latents and evaluate image-gallery retrieval.
- Objective: Image-latent MSE plus 0.1 symmetric contrastive loss; not cross-entropy classification.
- Subject scope: Subject 0 only, with three random seeds.
- Temporal-effect control: Not excluded: the first 30 source-order images per class train the model and the last 20 are evaluated.
- Status: Three seed runs completed at epoch 70.

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`scripts/run_d066_latent_finetune.py`](scripts/run_d066_latent_finetune.py)
- [`config/d066_latent_finetune.json`](config/d066_latent_finetune.json)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
