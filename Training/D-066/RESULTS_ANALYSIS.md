# D-066 Results Analysis

## Recorded outcome

- Recorded test class accuracies were 37.125%, 38.312%, and 36.125% for seeds 17, 23, and 41.
- Same-gallery training accuracy was about 94%, indicating a large train/test gap.

## Interpretation

This is embedding alignment with retrieval readout. The 30/20 source-order split leaves temporal-order effects possible.

## Artifact availability

Raw EEG data, generated run directories, and model checkpoints are not published in this repository. When a collaborator needs a checkpoint, it should be supplied separately under `checkpoints/D-066/`. The metrics above are retained as the experiment record; this documentation pass did not rerun training or evaluation.
