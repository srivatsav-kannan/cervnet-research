# CervNet prototypes

These folders preserve distinct research directions that were previously split
across several repositories. They are separated by project so provenance is
clear, but they are not supported packages or validated clinical pipelines.

| Folder | Former source | What it contains | Status |
|---|---|---|---|
| `current-combined/` | `CervicalNew` | EfficientNetB7 classification, SSD keypoints, quantitative measurements, preprocessing, saliency | Newest experimental snapshot |
| `historical-binary-and-localization/` | `CervicalDataset` | Broad classification, feature fusion, bounding boxes, SSD, YOLO, U-Net, and keypoint exploration | Historical |
| `five-class-curvature/` | `CervicalCurvature` | Five-class ResNet50 curvature experiments | Related historical work |
| `quantitative-modeling/` | `CervicalQuantFinal` | LightGBM, regression, SAINT, and multimodal/tabular experiments | Historical |
| `vit-fracture/` | `cervical-fracture-test` | Early Hugging Face ViT fracture classifier | Early related prototype |

The snapshots intentionally retain some notebook-style scripts, hard-coded
paths, stale dependency assumptions, and historical figures. Promote reusable
work into `src/cervnet/` only after replacing local paths with configuration,
pinning dependencies and data splits, documenting metrics, and adding tests.

Datasets, model weights, patient-derived images, and private credentials do not
belong in this repository.
