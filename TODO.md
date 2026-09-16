# Publication Readiness Checklist

This checklist records the repository-wide cleanup performed before publication.

- [x] Keep only the selected numbered experiments under `Training/`.
- [x] Give every experiment an English README, architecture explanation, TeX source, rendered PNG, and results analysis.
- [x] Add training purpose, objective type, subject scope, and temporal-effect policy to the experiment index.
- [x] Move experiment-owned configuration and source into the matching `D-###` directory.
- [x] Resolve shared data through the repository-level `data/` directory.
- [x] Exclude `data/`, `checkpoints/`, all local-only work trees, generated runs, reports, outputs, and artifacts from publication.
- [x] Remove public references to private files, local absolute paths, and machine-specific GPU identifiers.
- [x] Replace non-English public prose and generated report text with English.
- [x] Remove broken checkpoint junctions from the public Training tree.
- [x] Validate JSON, Python syntax, Markdown navigation, TeX rendering, privacy patterns, and ignore rules.

If a future change invalidates any item, reopen it and record the repair in the relevant experiment documentation.
