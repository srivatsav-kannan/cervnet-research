# Data

## Source assumptions

The historical code refers to a cervical-spine X-ray atlas, JSON landmark annotations,
spreadsheet measurements, and locally constructed CS/healthy image directories. Those
assets are not included here. A sample radiograph found during publication review was
removed from the public snapshot. Licensing and redistribution terms must be checked
before sharing any source or derived data.

## Privacy and versioning

- Do not commit radiographs, patient identifiers, spreadsheets, or extracted metadata.
- Keep datasets outside the repository and reference them with `CERVNET_DATA_DIR`.
- Store a de-identified manifest containing opaque sample IDs, checksums, source,
  inclusion criteria, and split assignment.
- Create train/validation/test splits by patient or study—not by image—to avoid leakage.
- Never infer missing demographic or clinical labels from filenames.

## Proposed local layout

```text
$CERVNET_DATA_DIR/
  manifests/
    dataset-v1.csv
  images/
    <opaque-sample-id>.png
  landmarks/
    <opaque-sample-id>.json
  tabular/
    measurements-v1.parquet
```

Generated images and augmented variants must inherit the source sample's split.
