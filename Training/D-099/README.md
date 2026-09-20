# D-099: Participant 12 Full-Frequency Control

D-099 repeats the D-098 participant-12 run with one controlled change: all 17 retained 12--80 Hz frequency bins are supplied to cross-attention. The D-098 mask that zeroed 15.625, 19.53125, and 23.4375 Hz is disabled. The waveform branch, post-cross-attention position addition, seed, split, optimizer, and 100-epoch budget are unchanged.

Participant 12 has 78 available classes in the verified official `EEG-ImageNet_2.pth` archive. Class indices 29 and 71 are absent; no trials are synthesized or borrowed.

All 100 training epochs completed on an A100. The fixed epoch-100 endpoint reached 17.8846% (279/1560), compared with 15.2564% for the D-098 masked endpoint. The user-requested retrospective checkpoint comparison found epoch 50 highest at 17.9487% (280/1560), versus 15.7692% for the best masked checkpoint.

See [Results analysis](RESULTS_ANALYSIS.md) and the [rendered architecture](ARCHITECTURE.md). The checkpoint comparison uses the official test split repeatedly and is therefore exploratory rather than validation-based model selection.
