# D-097 Results Analysis

## Recorded outcome

Four seed-17, 100-epoch A100 runs used the same split, optimizer, microbatch probe, and one-time endpoint evaluation. Every group reached 100% training accuracy, used microbatch 64, and had 2,297,634 trainable parameters.

| Group | Mask `<25 Hz` | Post-cross `+P` | Test accuracy | Test loss | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| `full_frequency_with_position` | no | yes | 47.0000% (752/1600) | 4.826310 | 304.875 |
| `masked_below25_with_position` | yes | yes | 48.5625% (777/1600) | 4.588695 | 305.385 |
| `full_frequency_without_post_cross_position` | no | no | 43.3125% (693/1600) | 4.980443 | 305.197 |
| `masked_below25_without_post_cross_position` | yes | no | 41.3750% (662/1600) | 4.820108 | 305.167 |

## Factorial comparison

Using accuracy as the endpoint, the `<25 Hz` mask effect is **+1.5625 percentage points** when post-cross `+P` is retained, but **−1.9375 points** when it is removed. The post-cross `+P` effect is **+3.6875 points** with all frequencies and **+7.1875 points** with the `<25 Hz` mask. The difference-in-differences interaction is **+3.5000 points**, meaning the observed benefit of masking is larger in the `+P` condition by 3.5 points. The corresponding loss interaction is −0.077280, with lower loss favored.

Within this one fixed subject/split/seed/budget, retaining the post-cross position injection was more important than the frequency mask: removing `+P` reduced accuracy by 3.6875 points in the full-frequency condition and by 7.1875 points in the masked condition. The frequency mask helped only when `+P` was retained and hurt when `+P` was removed.

## Interpretation boundary

These are descriptive single-seed factorial effects, not replicated causal estimates; the interaction could be seed-specific. Attention weights remain routing diagnostics rather than causal attribution. All four groups have identical model and runner SHA-256 values, identical train/test split hashes, and each endpoint test was forwarded exactly once.

## Artifact availability

Raw EEG data, generated runs, and checkpoints are excluded from Git. The completed remote artifacts remain on `gpumachine1` under `/root/eeg-decoding-d097/Training/D-097/outputs/factorial/`. The public repository contains only source, configuration, and this analysis.
