# CervNet Research

CervNet is a research workspace for computer-assisted analysis of lateral
cervical-spine radiographs. It brings the previously fragmented CervNet experiments
into one repository while keeping the deployable inference API separate.

> **Research use only.** This code is not a medical device, is not clinically
> validated, and must not be used to diagnose or treat patients.

## What is canonical

The supported research interface is the installable package under `src/cervnet`.
It currently provides:

- validated input and label conventions for binary CS/healthy experiments;
- reusable geometry utilities for the 23-keypoint quantitative pipeline;
- environment-based paths instead of machine-specific absolute paths; and
- a small test suite for deterministic, dependency-light components.

The newest experimental snapshot that preceded this cleanup remains at the repository
root. It includes EfficientNetB7 binary classification, keypoint localization,
quantitative measurements, preprocessing, and saliency experiments. Those scripts are
retained for provenance, but several are notebook-style programs with hard-coded local
paths and should not be treated as a reproducible release.

## Repository map

| Area | Status | Location |
|---|---|---|
| Shared research package | Canonical foundation | `src/cervnet/` |
| Binary CS/healthy classification | Newest experimental direction | root `train.py`, `test.py` |
| Keypoint localization and measurements | Newest experimental direction | `ssd/`, `quantitative algorithm/` |
| Tabular/multimodal quantitative models | Experimental | `quantitative model/` and `archive/quantitative/` |
| Historical localization/classification work | Archived with history | `archive/cervical-dataset/` |
| Five-class curvature experiments | Archived with history | `archive/curvature/` |
| Early ViT fracture prototype | Archived with history | `archive/vit-fracture-prototype/` |
| Inference service | Separate repository | `cervnet-api` |

The four `archive/` directories preserve the code snapshots that were imported from the
fragmented repositories. Their original repositories and the full consolidation history
remain private because that history included a sample radiograph and other
patient-derived output. This public repository begins from a reviewed, clean snapshot.

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

The datasets and trained weights are intentionally not committed. See
[`docs/DATA.md`](docs/DATA.md) for expected layouts and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before running an experiment.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — components and boundaries
- [`docs/RESEARCH_HISTORY.md`](docs/RESEARCH_HISTORY.md) — what each old repository contained
- [`docs/DATA.md`](docs/DATA.md) — dataset assumptions and privacy rules
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — how to turn snapshots into repeatable runs
- [`docs/RESULTS.md`](docs/RESULTS.md) — what is and is not currently supported by evidence
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — medical, dataset, and engineering limitations

## Relationship to the API

The research repository owns experimentation, evaluation, and provenance. The separate
`cervnet-api` repository owns only a stable HTTP contract and packaged inference
artifacts. Models should move from research to the API only after their exact
preprocessing, label order, threshold, and evaluation record have been frozen.
