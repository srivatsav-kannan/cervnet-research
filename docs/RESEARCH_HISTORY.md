# Research history

The prior repository names reflected stages of development, not stable components.
They are now represented as historical branches inside this repository.

| Former repository | Period represented | Interpretation |
|---|---|---|
| `CervicalDataset` | 2024–2025 | Broad exploration of preprocessing, classification, feature fusion, bounding boxes, SSD, YOLO, U-Net, keypoints, and documented result images |
| `CervicalCurvature` | 2025 | ResNet50 five-class curvature experiments: lordotic, straight, two sigmoid patterns, and kyphotic |
| `CervicalQuantFinal` | 2025 | Quantitative features, LightGBM, image-to-stat regression, and SAINT tabular/multimodal experiments |
| `cervical-fracture-test` | 2024 | Early Hugging Face ViT fracture-classification prototype; not part of the current CS/healthy pipeline |
| `CervicalNew` | 2025 onward | Most recent combined work on EfficientNetB7 classification, keypoints, quantitative measures, preprocessing, and saliency |

Reviewed snapshots live as separate projects under `prototypes/`. “Prototype”
means preserved and discoverable, not validated or production-ready.
Machine-specific paths, copied experiments, and known metric inconsistencies
remain visible so the scientific record is not silently rewritten.

| Former repository | Public folder |
|---|---|
| `CervicalNew` | `prototypes/current-combined/` |
| `CervicalDataset` | `prototypes/historical-binary-and-localization/` |
| `CervicalCurvature` | `prototypes/five-class-curvature/` |
| `CervicalQuantFinal` | `prototypes/quantitative-modeling/` |
| `cervical-fracture-test` | `prototypes/vit-fracture/` |

## Canonical interpretation

There is not yet one validated end-to-end clinical model. The newest research direction
is a binary CS/healthy image classifier complemented by a 23-keypoint quantitative
pipeline. The quantitative repository explores whether tabular measurements, predicted
measurements, or multimodal fusion can improve that task. Curvature and fracture work
are related experiments, not direct replacements for the binary pipeline.
