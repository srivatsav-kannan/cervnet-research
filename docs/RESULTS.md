# Results

Historical scripts and notebooks contain printed metrics and figures, but this cleanup
does not present those values as verified results. The current repository lacks the
frozen datasets, exact artifacts, and independent rerun needed to reproduce them.

## Evidence status

| Experiment family | Evidence present | Current status |
|---|---|---|
| Binary image classification | Training/evaluation scripts and saved notebook output | Requires clean rerun |
| SVM over CNN features | Code snapshots | Requires artifact and preprocessing verification |
| Five-class curvature | Training/evaluation script and figures | Archived; requires clean rerun |
| 23-keypoint localization | Training/inference scripts | Requires annotation manifest and held-out metrics |
| Quantitative LightGBM/SAINT | Code, notebooks, and histories | Experimental; requires split and leakage audit |
| Saliency/Grad-CAM | Visualization scripts | Qualitative only; not evidence of clinical validity |

Verified results should be added only with the dataset manifest ID, source commit,
artifact checksum, sample count, confidence interval, and evaluation script command.

