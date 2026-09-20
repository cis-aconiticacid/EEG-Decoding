# D-097 Results Analysis

## Recorded outcome

- All groups retain position in Q/K; only the post-cross z_s+P injection is toggled.
- The frequency factor zeros 15.625, 19.53125, and 23.4375 Hz bins before frequency projection.
- The four one-seed endpoint metrics are reported together with descriptive main effects and interaction.

## Interpretation

This is a controlled implementation ablation within D-096; it does not establish a general causal effect beyond the fixed subject, split, seed, and budget.

## Artifact availability

Raw EEG data, generated run directories, and model checkpoints are not published in this repository. When a collaborator needs a checkpoint, it should be supplied separately under `checkpoints/D-097/`. The metrics above are retained as the experiment record; this documentation pass did not rerun training or evaluation.
