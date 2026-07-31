"""Pure NumPy geometry used by the 23-keypoint quantitative pipeline.

The original quantitative script mixes model loading, filesystem access, plotting,
and measurements. These functions isolate deterministic calculations so they can be
tested and reused by training, evaluation, and eventually the API.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

KEYPOINT_NAMES = (
    "C2 centroid",
    "C2 bottom left",
    "C2 bottom right",
    "C3 top left",
    "C3 top right",
    "C3 bottom left",
    "C3 bottom right",
    "C4 top left",
    "C4 top right",
    "C4 bottom left",
    "C4 bottom right",
    "C5 top left",
    "C5 top right",
    "C5 bottom left",
    "C5 bottom right",
    "C6 top left",
    "C6 top right",
    "C6 bottom left",
    "C6 bottom right",
    "C7 top left",
    "C7 top right",
    "C7 bottom left",
    "C7 bottom right",
)


def as_keypoints(points: ArrayLike) -> NDArray[np.float64]:
    """Return a validated `(23, 2)` floating-point keypoint array."""

    result = np.asarray(points, dtype=np.float64)
    if result.shape != (23, 2):
        raise ValueError(f"Expected keypoints with shape (23, 2); received {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Keypoints must contain only finite coordinates")
    return result


def line_angle_degrees(first: ArrayLike, second: ArrayLike) -> float:
    """Return the signed angle from one two-point line to another in degrees."""

    line_a = np.asarray(first, dtype=np.float64)
    line_b = np.asarray(second, dtype=np.float64)
    if line_a.shape != (2, 2) or line_b.shape != (2, 2):
        raise ValueError("Each line must have shape (2, 2)")

    vector_a = line_a[1] - line_a[0]
    vector_b = line_b[1] - line_b[0]
    if np.linalg.norm(vector_a) == 0 or np.linalg.norm(vector_b) == 0:
        raise ValueError("Cannot calculate an angle from a zero-length line")

    cross = np.cross(vector_a, vector_b)
    dot = np.dot(vector_a, vector_b)
    return float(np.degrees(np.arctan2(cross, dot)))


def adjacent_row_widths(points: ArrayLike) -> dict[str, float]:
    """Measure the eleven paired endplate widths encoded after the C2 centroid."""

    keypoints = as_keypoints(points)
    pairs = keypoints[1:].reshape(11, 2, 2)
    labels = (
        "C2 inferior",
        "C3 superior",
        "C3 inferior",
        "C4 superior",
        "C4 inferior",
        "C5 superior",
        "C5 inferior",
        "C6 superior",
        "C6 inferior",
        "C7 superior",
        "C7 inferior",
    )
    return {
        label: float(np.linalg.norm(pair[1] - pair[0]))
        for label, pair in zip(labels, pairs, strict=True)
    }


def adjacent_endplate_angles(points: ArrayLike) -> dict[str, float]:
    """Calculate signed angles between each neighboring pair of endplates."""

    keypoints = as_keypoints(points)
    endplates = keypoints[1:].reshape(11, 2, 2)
    names = (
        "C2-C3",
        "C3 superior-inferior",
        "C3-C4",
        "C4 superior-inferior",
        "C4-C5",
        "C5 superior-inferior",
        "C5-C6",
        "C6 superior-inferior",
        "C6-C7",
        "C7 superior-inferior",
    )
    return {
        name: line_angle_degrees(endplates[index], endplates[index + 1])
        for index, name in enumerate(names)
    }

