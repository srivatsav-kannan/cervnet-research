# Architecture

## Boundary

CervNet research answers: *Which model, preprocessing path, and quantitative method
should be evaluated?* The separate `cervnet-api` answers: *How is one frozen model
served safely and consistently?*

No training dataset, patient record, private credential, or ad-hoc notebook dependency
belongs in the API. No HTTP or deployment concern belongs in the research package.

## Research layers

1. **Data preparation** — organization, augmentation, CLAHE, normalization, and crop
   experiments.
2. **Localization** — bounding-box and 23-keypoint experiments using SSD, YOLO, and
   U-Net-derived approaches.
3. **Representation** — EfficientNetB7 and ResNet50 image features.
4. **Prediction** — binary CS/healthy classification, five-class curvature
   classification, LightGBM, SVM, SAINT, and regression experiments.
5. **Quantification** — deterministic measurements derived from predicted landmarks.
6. **Evaluation and explainability** — classification reports, ROC analysis,
   saliency, and Grad-CAM snapshots.

Only the shared configuration and deterministic geometry foundation has been promoted
to `src/cervnet` so far. Promotion of a model pipeline requires a frozen dataset
manifest, split, preprocessing contract, artifact checksum, and evaluation record.

## Model promotion contract

A research model is ready for the API only when the following are recorded together:

- immutable artifact checksum and framework version;
- image size, color space, normalization, and crop behavior;
- output label order and threshold;
- validation cohort and split identity;
- metrics with confidence intervals where feasible; and
- known failure modes and intended-use statement.

