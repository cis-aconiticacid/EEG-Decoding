# D-094 Results Analysis

## Recorded outcome

- The selected epoch-50 development checkpoint achieved 9.167% validation accuracy (22/240), macro accuracy 9.167%, and loss 4.8633.
- The official last-20-per-class test set was not evaluated.

## Interpretation

The development score exceeds the 1.25% chance reference but is not a held-out test result.

## Artifact availability

Raw EEG data, generated run directories, and model checkpoints are not published in this repository. When a collaborator needs a checkpoint, it should be supplied separately under `checkpoints/D-094/`. The metrics above are retained as the experiment record; this documentation pass did not rerun training or evaluation.
