# D-098 Results Analysis

## Per-participant endpoints

All models use seed 17 and the same fixed 100-epoch training contract. Each row comes from that participant's completed endpoint `result.json`; the held-out test split was forwarded exactly once.

| Participant | Archive | Train / test | Present classes | Correct | Test accuracy | Test loss | Runtime (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Part 1 | 2400 / 1600 | 80 | 773 | 48.3125% | 4.501676 | 305.984 |
| 1 | Part 1 | 2400 / 1600 | 80 | 554 | 34.6250% | 5.740657 | 305.825 |
| 2 | Part 1 | 2370 / 1580 | 79 | 588 | 37.2152% | 4.235509 | 303.441 |
| 3 | Part 1 | 2400 / 1600 | 80 | 701 | 43.8125% | 4.911870 | 305.852 |
| 4 | Part 1 | 2400 / 1600 | 80 | 413 | 25.8125% | 5.409041 | 305.650 |
| 5 | Part 1 | 2400 / 1600 | 80 | 654 | 40.8750% | 4.978152 | 306.019 |
| 6 | Part 1 | 2400 / 1600 | 80 | 695 | 43.4375% | 4.256651 | 306.133 |
| 7 | Part 1 | 2400 / 1600 | 80 | 443 | 27.6875% | 7.042597 | 306.275 |
| 8 | Part 2 | 2400 / 1600 | 80 | 920 | 57.5000% | 3.186159 | 300.136 |
| 9 | Part 2 | 2400 / 1600 | 80 | 596 | 37.2500% | 5.097136 | 300.174 |
| 10 | Part 2 | 2400 / 1600 | 80 | 550 | 34.3750% | 4.827823 | 300.052 |
| 11 | Part 2 | 2400 / 1600 | 80 | 702 | 43.8750% | 3.258587 | 300.097 |
| 12 | Part 2 | 2340 / 1560 | 78 | 238 | 15.2564% | 6.755784 | 292.501 |
| 13 | Part 2 | 2400 / 1600 | 80 | 570 | 35.6250% | 4.557504 | 300.019 |
| 14 | Part 2 | 2400 / 1600 | 80 | 468 | 29.2500% | 5.326570 | 300.016 |
| 15 | Part 2 | 2400 / 1600 | 80 | 611 | 38.1875% | 4.274770 | 300.491 |

## Aggregate summary

- Unweighted mean participant accuracy: **37.0685%**.
- Median participant accuracy: **37.2326%**.
- Population standard deviation across participants: **9.5491 percentage points**.
- Range: **15.2564%** (participant 12) to **57.5000%** (participant 8).
- Pooled sample accuracy: **9,476 / 25,540 = 37.1026%**.
- All sixteen final training accuracies were 100%; this is fit-set accuracy, not evidence of equal generalization.

For participants with all 80 classes, ordinary accuracy, present-class macro accuracy, and all-80-output macro accuracy coincide because each test class has 20 examples. Participant 2's present-class macro accuracy is 37.2152% and all-80-output macro accuracy is 36.7500%; participant 12's values are 15.2564% and 14.8750% respectively.

## Data provenance

The two archives were downloaded from the user-supplied official Tsinghua Cloud links:

- `EEG-ImageNet_1.pth`: 7,944,350,936 bytes; SHA-256 `57e187c0515587f2b7283f41dc439df4ba8ac5fe106b8c3e6ee850d2a9a6d8d9`.
- `EEG-ImageNet_2.pth`: 7,931,918,744 bytes; SHA-256 `0e54b706d5f06a56bc590fa5dabed06d640ff4e28fed97db766b012ce24c452e`.

Participant 2 is missing class index 1 (`n07749192`) from Part 1. Participant 12 is missing class indices 29 (`n04249415`) and 71 (`n03452741`) from Part 2. No trials were synthesized, borrowed, or duplicated to fill these omissions.

## Interpretation boundary

Each value is a single-seed, participant-specific model result. Across-participant variability can be summarized descriptively, but it is not a repeated-seed uncertainty estimate.

## Artifact availability

Raw EEG data, generated runs, predictions, and checkpoints are excluded from Git. The completed participant 0--7 artifacts are under `/root/eeg-decoding-d098/Training/D-098/outputs/participants/` on `gpumachine1`; participant 8--15 artifacts use the same path on `gpumachine2`. Every participant directory contains `epoch100.pt`, `test_predictions.jsonl`, and `result.json`.
