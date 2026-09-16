# Brainstorm: EEG Decoding Experiments

This repository is the public, English-language record for a numbered sequence of EEG-ImageNet training and diagnostic experiments. Start with the [Training experiment index](Training/README.md); the current maintained model is [D-095](Training/D-095/README.md).

## Repository layout

```text
eegdecoding/
├── data/                 # Local data only; never published
├── checkpoints/          # Local model weights only; never published
├── Training/
│   ├── README.md         # Experiment table and navigation
│   ├── CURRENT.md        # Current experiment pointer
│   └── D-###/ or A-###/  # One self-contained experiment per directory
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

Generated data, weights, caches, and runtime outputs do not belong in the public experiment bundle.

## Data behavior

All shared EEG inputs resolve from the repository-level `data/` directory. The common loader in [`src/eegdecoding/data.py`](src/eegdecoding/data.py) downloads a missing official EEG-ImageNet archive into `data/`, verifies its expected byte count and SHA-256 digest, and only then exposes it to training code. A failed verification never overwrites a previously existing file.

Some later experiments can also use optional prepared arrays under `data/prepared/`. When those arrays are absent, supported entry points fall back to the verified raw archive. Image-latent experiments additionally require separately obtained image assets or teacher targets under `data/`; their exact paths are recorded in each experiment's configuration.

## Checkpoints

Checkpoints are intentionally not published. Store locally supplied weights under `checkpoints/D-###/` or `checkpoints/A-###/`, matching the experiment number. Documentation names the expected experiment directory but does not link to absent local files.

## Maintenance rules

1. Use a unique, increasing `D-###` directory for every experiment. Do not reuse an identifier.
2. Keep experiment-specific code, configuration, and documentation inside that experiment. Move a module to `src/eegdecoding/` only when at least two active experiments genuinely share it.
3. Resolve paths from `__file__`, never from the shell's current directory, a user-specific absolute path, or a private root marker.
4. Read shared data only from the repository-level `data/` directory. Never commit data, prepared arrays, teacher assets, run outputs, or checkpoints.
5. Keep every public filename, document, diagram label, code comment, error message, and generated report in English.
6. State whether the objective is cross-entropy classification, self-supervised reconstruction, latent alignment, retrieval, or a classical classifier.
7. State the subject scope and explicitly document whether a first-30/last-20 source-order split is used. Do not claim that temporal effects were removed unless the split design actually removes them.
8. Distinguish development metrics, historical diagnostics, and held-out test results. A checkpoint is not itself a result.
9. Update `ARCHITECTURE.tex`, rerender `architecture.png`, and review `ARCHITECTURE.md` whenever the model or data flow changes.
10. Run the publication checks described in [`TODO.md`](TODO.md) before uploading.

## Common commands

```powershell
# Install the package and test dependencies.
python -m pip install -e ".[dev]"

# Rerender every TeX architecture diagram and rebuild the English indexes.
python scripts/render_training_documentation.py

# Run experiment-local tests.
python -m pytest

# Run the current model's smoke check without starting training.
python Training/D-095/run.py --smoke
```

Training commands and required optional assets are documented inside each experiment directory.
