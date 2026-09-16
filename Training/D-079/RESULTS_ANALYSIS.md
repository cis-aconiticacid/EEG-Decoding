# D-079 Results Analysis

## Recorded outcome

- The indexed parent checkpoint metrics were 37.125% for B0, 19.875% for B1, and 2.938% for B2 before their continuation updates.
- The local result archive, not this public repository, contains the complete epoch-120 comparison.

## Interpretation

This is a warm-start architecture ablation. Parent-state and source-order effects must be considered when comparing arms.

## Artifact availability

Raw EEG data, generated run directories, and model checkpoints are not published in this repository. When a collaborator needs a checkpoint, it should be supplied separately under `checkpoints/D-079/`. The metrics above are retained as the experiment record; this documentation pass did not rerun training or evaluation.
