# D-080 Results Analysis

## Recorded outcome

- At epoch 0, recorded test accuracies were 0.312% for B0, 1.375% for B1, and 1.125% for B2; these are initialization diagnostics, not trained endpoints.
- The complete epoch-70 results remain in the local result archive.

## Interpretation

This design removes trained-model inheritance but still uses the source-order 30/20 evaluation protocol.

## Artifact availability

Raw EEG data, generated run directories, and model checkpoints are not published in this repository. When a collaborator needs a checkpoint, it should be supplied separately under `checkpoints/D-080/`. The metrics above are retained as the experiment record; this documentation pass did not rerun training or evaluation.
