# D-088 Results Analysis

## Recorded outcome

- The indexed summary records best development loss 2.87269 for local convolution and 3.14535 for global convolution.
- Corresponding stored test losses were 5.71287 and 5.50517; route accuracies are not reproduced in the public tree.

## Interpretation

Development selection and final testing are separated, but the split still follows source order.

## Artifact availability

Raw EEG data, generated run directories, and model checkpoints are not published in this repository. When a collaborator needs a checkpoint, it should be supplied separately under `checkpoints/D-088/`. The metrics above are retained as the experiment record; this documentation pass did not rerun training or evaluation.
