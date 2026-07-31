"""Shared, dependency-light foundations for CervNet research.

Deep-learning frameworks are imported only inside the modules that need them, so
geometry and configuration tests can run on a normal development machine.
"""

from cervnet.config import Settings
from cervnet.geometry import (
    KEYPOINT_NAMES,
    adjacent_endplate_angles,
    adjacent_row_widths,
    line_angle_degrees,
)

__all__ = [
    "KEYPOINT_NAMES",
    "Settings",
    "adjacent_endplate_angles",
    "adjacent_row_widths",
    "line_angle_degrees",
]

