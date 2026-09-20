# D-099 Results Analysis

## Controlled comparison

D-099 differs from the D-098 participant-12 model only by setting `frequency_mask_below_hz` from `25.0` to `null`. It restores the 15.625, 19.53125, and 23.4375 Hz inputs to the frequency projection. Both conditions use seed 17, the same 2,297,634-parameter model, 2,340 training trials, 1,560 test trials, post-cross-attention `+P`, and a fixed 100-epoch budget.

| Checkpoint | D-098 masked correct | D-098 masked accuracy | D-099 full-frequency correct | D-099 full-frequency accuracy | Accuracy change |
|---:|---:|---:|---:|---:|---:|
| Epoch 25 | 211 | 13.5256% | 257 | 16.4744% | +2.9487 pp |
| Epoch 50 | 233 | 14.9359% | **280** | **17.9487%** | **+3.0128 pp** |
| Epoch 75 | **246** | **15.7692%** | 268 | 17.1795% | +1.4103 pp |
| Epoch 100 | 238 | 15.2564% | 279 | 17.8846% | +2.6282 pp |

The full-frequency condition improves accuracy at every saved checkpoint. Its best checkpoint is epoch 50, with 34 more correct predictions and +2.1795 percentage points relative to the best masked checkpoint. The improvement is meaningful for this run but does not lift participant 12 above 20% accuracy.

## Loss and overfitting signal

| Checkpoint | Masked test loss | Full-frequency test loss |
|---:|---:|---:|
| Epoch 25 | 3.800616 | **3.730679** |
| Epoch 50 | 5.701667 | **5.458108** |
| Epoch 75 | 6.566755 | **6.371029** |
| Epoch 100 | 6.755782 | **6.439300** |

Restoring the low-frequency bins lowers test loss at every checkpoint, but both conditions show rising test loss after epoch 25 while training accuracy approaches 100%. The frequency mask was harmful for participant 12, yet removing it does not resolve the broader overfitting pattern.

## Interpretation boundary

Epoch 25, 50, 75, and 100 were all evaluated on the official test split at the user's explicit request. Choosing epoch 50 from these values is retrospective test-set selection and must not be reported as an untouched held-out estimate. The fixed epoch-100 result remains the protocol-faithful endpoint.

## Remote artifacts

The completed checkpoints, training curve, predictions, and result record are under `/root/eeg-decoding-d099/Training/D-099/outputs/participant_12/` on `gpumachine2`. Generated model artifacts are excluded from Git.
