import numpy as np
import pytest

from cervnet.geometry import (
    adjacent_endplate_angles,
    adjacent_row_widths,
    as_keypoints,
    line_angle_degrees,
)


def sample_keypoints() -> np.ndarray:
    points = np.zeros((23, 2), dtype=float)
    for index in range(11):
        points[1 + index * 2] = (0, index)
        points[2 + index * 2] = (10, index)
    return points


def test_keypoint_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="shape"):
        as_keypoints(np.zeros((22, 2)))


def test_parallel_endplates_have_equal_width_and_zero_angle() -> None:
    points = sample_keypoints()

    assert set(adjacent_row_widths(points).values()) == {10.0}
    assert set(adjacent_endplate_angles(points).values()) == {0.0}


def test_signed_line_angle() -> None:
    horizontal = np.array([[0, 0], [1, 0]])
    vertical = np.array([[0, 0], [0, 1]])

    assert line_angle_degrees(horizontal, vertical) == pytest.approx(90.0)


def test_zero_length_line_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        line_angle_degrees(np.zeros((2, 2)), np.array([[0, 0], [1, 0]]))

