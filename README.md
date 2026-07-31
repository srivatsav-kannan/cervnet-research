# CervNet Research

CervNet is the unified research workspace for computer-assisted analysis of
lateral cervical-spine radiographs. Previously fragmented experiments are now
kept in named folders here; deployable inference remains separate in
[`cervnet-api`](https://github.com/srivatsav-kannan/cervnet-api).

> **Research use only.** This code is not a medical device, is not clinically
> validated, and must not be used to diagnose or treat patients.

## Canonical status

The supported, dependency-light foundation is the installable package under
`src/cervnet`. It provides:

- validated input and label conventions for binary CS/healthy experiments;
- reusable geometry utilities for the 23-keypoint quantitative pipeline;
- environment-based paths instead of machine-specific absolute paths; and
- tests for deterministic components.

There is not yet one validated end-to-end clinical model. The newest preserved
direction combines EfficientNetB7 binary classification, keypoint localization,
quantitative measurements, preprocessing, and saliency experiments. It is kept
under `prototypes/current-combined/` and is still prototype code.

## Repository map

| Area | Status | Location |
|---|---|---|
| Shared research package | Canonical foundation | `src/cervnet/` |
| Latest combined experiments | Newest prototype | `prototypes/current-combined/` |
| Broad binary/localization work | Historical prototype | `prototypes/historical-binary-and-localization/` |
| Five-class curvature | Related historical prototype | `prototypes/five-class-curvature/` |
| Quantitative/SAINT/LightGBM work | Historical prototype | `prototypes/quantitative-modeling/` |
| Early ViT fracture work | Related early prototype | `prototypes/vit-fracture/` |
| Inference service | Separate deployable repository | `cervnet-api` |

Each former project now has its own folder under [`prototypes/`](prototypes/).
The old repositories remain private archives because their full histories
included a sample radiograph and patient-derived output. This public repository
begins from a reviewed snapshot with that material excluded.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

For deep-learning experiments:

```bash
pip install -e ".[ml]"
export CERVNET_DATA_DIR=/absolute/path/to/datasets
export CERVNET_ARTIFACT_DIR=/absolute/path/to/model-artifacts
```

Datasets and trained weights are intentionally not committed. See
[`docs/DATA.md`](docs/DATA.md) and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before running an
experiment.

## Documentation

- [`prototypes/README.md`](prototypes/README.md) — exact status of each prototype
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — components and boundaries
- [`docs/RESEARCH_HISTORY.md`](docs/RESEARCH_HISTORY.md) — mapping from former repositories
- [`docs/DATA.md`](docs/DATA.md) — dataset assumptions and privacy rules
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — migration to repeatable runs
- [`docs/RESULTS.md`](docs/RESULTS.md) — claims supported by current evidence
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — medical, dataset, and engineering limits

## Relationship to the API

This repository owns experimentation, evaluation, and provenance. The separate
API repository owns only a stable HTTP contract and frozen inference artifacts.
A model should move to the API only after preprocessing, label order, threshold,
artifact checksum, and evaluation record have been frozen.
