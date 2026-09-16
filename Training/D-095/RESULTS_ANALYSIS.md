# D-095 Results Analysis

## Recorded outcome

- Recorded test accuracy: 42.188% for unified192, 45.875% with the 18--28 Hz exclusion, 14.750% without the frequency branch, and 48.688% with the optional post-fusion projection.
- All reported runs used one seed and reached 100% training accuracy.

## Interpretation

The projection-enabled variant has the highest recorded score, but single-seed ablations do not establish a robust causal effect.

## Artifact availability

Raw EEG data, generated run directories, and model checkpoints are not published in this repository. When a collaborator needs a checkpoint, it should be supplied separately under `checkpoints/D-095/`. The metrics above are retained as the experiment record; this documentation pass did not rerun training or evaluation.
