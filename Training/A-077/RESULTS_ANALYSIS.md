# A-077 Results Analysis

## Recorded outcome

- T1 test balanced accuracy was 89.550% for the SVC and 86.150% for the random forest.
- T3 random-forest test balanced accuracy was 54.900%, close to but above the 50% reference.
- T2 has no score because the archive contains no verified session identifier.

## Interpretation

T1 supports subject-identifying signal in the constructed PSD pairs. T3 explicitly probes order proximity and must not be presented as session decoding.

## Artifact availability

Raw EEG data, generated run directories, and model checkpoints are not published in this repository. When a collaborator needs a checkpoint, it should be supplied separately under `checkpoints/A-077/`. The metrics above are retained as the experiment record; this documentation pass did not rerun training or evaluation.
