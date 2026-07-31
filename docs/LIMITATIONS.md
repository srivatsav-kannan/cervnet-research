# Limitations

- The repository represents research prototypes, not a clinically validated system.
- Dataset provenance, licensing, cohort composition, and external validity require
  formal documentation.
- Historical splitting and augmentation code must be audited for patient-level leakage.
- Several old evaluation scripts mix binary and five-class logic; their reported metrics
  must not be relied upon without correction and rerun.
- Image preprocessing differs between experiments, including raw scaling, ResNet
  preprocessing, CLAHE, normalization, and cropping.
- Models may learn acquisition-site markers, annotations, or demographic proxies rather
  than pathology.
- Saliency maps do not establish causal reasoning or clinical correctness.
- Performance on one atlas or internal dataset does not imply performance across
  scanners, institutions, populations, or disease prevalence.
- A clinician-designed intended-use statement, risk analysis, external validation, and
  regulatory review would be required before any patient-facing use.

