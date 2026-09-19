# D-096 Results Analysis

## Recorded outcome

- One seed-17, 100-epoch A100 run completed in 304.704 seconds with microbatch 64.
- Training accuracy reached 100% with final training loss 0.000583.
- The single official test forward classified 749 of 1,600 trials correctly: 46.8125% accuracy, 46.8125% macro class accuracy, and 4.83625 cross-entropy loss.
- The model has 2,297,634 trainable parameters.
- The learned cross-attention residual scale moved from 0.1 to 0.121882.
- Mean cross-attention entropy was 2.67268 nats, or 96.40% of the maximum `log(16)` entropy. Mean diagonal attention mass was 0.06277, close to the uniform reference `1/16 = 0.0625`.

## Comparison

D-096 exceeded D-095 `unified192` by 4.625 percentage points (46.8125% versus 42.1875%) and the 18--28 Hz exclusion run by 0.9375 points (46.8125% versus 45.875%). It remained 1.875 points below D-095 `postfusion192` (46.8125% versus 48.6875%). All are single-seed available-run observations.

## Interpretation

The result is consistent with cross-attention improving over the D-095 scalar-gate baseline in this run, but it does not establish that cross-attention is the cause of the gain. The nearly uniform averaged routing statistics show little learned time-selective alignment at the endpoint. Repeated seeds and controlled parameter-matched ablations are required before making a robust architectural claim. Attention weights remain routing diagnostics rather than causal attribution.

## Artifact availability

Raw EEG data, generated runs, and model checkpoints are excluded from Git. Remote outputs remain on the authorized training host unless separately packaged.
