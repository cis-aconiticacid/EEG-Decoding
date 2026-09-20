# D-097: Frequency Mask x Post-Cross-Position Factorial Ablation

Measure the separate and combined effects of masking retained frequency bins below 25 Hz and removing the post-cross-attention position injection.

## Experiment contract

- Training purpose: Measure the separate and combined effects of masking retained frequency bins below 25 Hz and removing the post-cross-attention position injection.
- Objective: Ordinary 80-way cross-entropy without label smoothing.
- Subject scope: Subject 0 only.
- Temporal-effect control: Not excluded: the first 30 source-order images per class train the model and the last 20 form the official test.
- Status: Four fixed 100-epoch A100 groups use the same seed and training protocol; results are recorded after endpoint completion.

## Two controlled factors

1. **Frequency mask:** retain all 17 bins from 12--80 Hz, or set the retained bins below 25 Hz to zero before `Linear(17, 192)`. At 1 kHz with a 256-point FFT, this masks 15.625, 19.53125, and 23.4375 Hz.
2. **Post-cross-attention position:** keep or remove the single `z_s + P` addition immediately after cross-attention and before the four Transformer blocks. Position context remains in `Q=LN(z_w+P)` and `K=LN(z_f+P)` in all groups.

| Group | Frequency input | Post-cross `+P` |
|---|---|---|
| `full_frequency_with_position` | all 17 retained bins | yes |
| `masked_below25_with_position` | bins below 25 Hz zeroed | yes |
| `full_frequency_without_post_cross_position` | all 17 retained bins | no |
| `masked_below25_without_post_cross_position` | bins below 25 Hz zeroed | no |

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

- [`model.py`](model.py)
- [`run.py`](run.py)
- [`config/full_frequency_with_position.json`](config/full_frequency_with_position.json)
- [`config/masked_below25_with_position.json`](config/masked_below25_with_position.json)
- [`config/full_frequency_without_post_cross_position.json`](config/full_frequency_without_post_cross_position.json)
- [`config/masked_below25_without_post_cross_position.json`](config/masked_below25_without_post_cross_position.json)

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
