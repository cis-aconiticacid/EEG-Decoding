# D-096 Results Analysis

## Recorded outcome

- Local unit and shape validation is complete.
- The A100 training endpoint has not yet been recorded.

## Planned comparison

The primary comparison is D-095 `unified192` versus D-096 under the same subject, split, seed, optimizer, 100-epoch budget, and one-time test policy. Attention entropy and diagonal mass are recorded as routing diagnostics, not as causal attribution.

## Interpretation

Any single-seed difference is an available-run observation. A robust architectural claim requires repeated seeds and should distinguish improved optimization from a reproducible cross-modal fusion effect.

## Artifact availability

Raw EEG data, generated runs, and model checkpoints are excluded from Git. Remote outputs remain on the authorized training host unless separately packaged.
