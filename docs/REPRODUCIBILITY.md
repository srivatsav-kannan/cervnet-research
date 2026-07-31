# Reproducibility

The preserved experiments document useful work, but they are not yet fully reproducible:
paths are machine-specific, package versions vary, several notebooks contain saved
outputs, and trained artifacts are absent.

For each experiment promoted from the archive:

1. assign an experiment ID and state the hypothesis;
2. pin dataset manifest and split checksums;
3. move paths and parameters into a checked-in config;
4. pin framework and package versions;
5. set and record Python, NumPy, and framework seeds;
6. log preprocessing, class balance, augmentation, and sampling;
7. save artifact, config, source commit, and metrics together; and
8. rerun evaluation from a clean environment before reporting results.

The root scripts should be treated as source material during this migration, not invoked
as an automated pipeline without review.

