# Brainstorm: EEG Decoding Experiments

This repository is the public, English-language record for a numbered sequence of EEG-ImageNet training and diagnostic experiments. Start with the [Training experiment index](Training/README.md); the current maintained model is [D-096](Training/D-096/README.md).

## Repository layout

```text
eegdecoding/
├── data/                 # Local data only; never published
├── checkpoints/          # Local model weights only; never published
├── Training/
│   ├── README.md         # Experiment table and navigation
│   ├── CURRENT.md        # Current experiment pointer
│   └── D-###/ or A-###/   # One self-contained experiment per directory
├── src/eegdecoding/      # Code shared by two or more experiments
├── scripts/              # Repository-level maintenance tools
├── .gitignore
└── README.md
```

Local-only work trees, virtual environments, data, checkpoints, generated reports, runs, outputs, and artifacts are excluded by `.gitignore`. In particular, **the `data/` directory must not be uploaded to GitHub**.

## Experiment contents

Each public `Training/D-###/` or `Training/A-###/` directory contains:


- `README.md`: purpose, loss type, subject scope, temporal-effect policy, and status;
- `ARCHITECTURE.md`: the rendered model/data-flow explanation;
- `ARCHITECTURE.tex`: the editable TikZ source;
- `architecture.png`: the image rendered from the TeX source;
- `RESULTS_ANALYSIS.md`: recorded metrics, interpretation, and evidence limits;
- experiment-owned `config/`, `scripts/`, `src/`, or tests when applicable.
- Generated data, weights, caches, and runtime outputs do not belong in the public experiment bundle.

## Data behavior

All shared EEG inputs resolve from the repository-level `data/` directory. The common loader in [`src/eegdecoding/data.py`](src/eegdecoding/data.py) downloads a missing official EEG-ImageNet archive into `data/` and verifies its byte length and SHA-256 digest. For an explicit setup step, run `python scripts/download_eeg_imagenet.py --part 1`; use `--part all` for both archives or `--verify-only` to check existing files without downloading.
## Checkpoints

Checkpoints are intentionally not published. Store locally supplied weights under `checkpoints/D-###/`, matching the experiment number. Documentation names the expected experiment directory but does not link to absent local files.

## Maintenance rules

1. Use a unique, increasing `D-###` directory for every experiment. Do not reuse an identifier.
2. Keep experiment-specific code, configuration, and documentation inside that experiment. Move a module to `src` only when at least two active experiments genuinely share it.(Before that you need to check if the folder could be merge into one D-xx)
3. Resolve paths from `__file__`, never from the shell's current directory, a user-specific absolute path, or a private root marker.
4. Keep every public filename, document, diagram label, code comment, error message, and generated report in English.
5. When you add to a new Number, Besure that you have sucessfully update the navigation README.md (Especially in the subfolder!!!)
6. Do not copy below the current path!
7. Update `ARCHITECTURE.tex`, rerender `architecture.png`, and review `ARCHITECTURE.md` whenever the model or data flow changes.
8. Generated data, weights, caches, and runtime outputs do not belong in the public experiment bundle.
