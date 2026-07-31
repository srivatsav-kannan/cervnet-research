"""Filesystem configuration for reproducible CervNet experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Locations supplied by the environment instead of developer-specific paths."""

    data_dir: Path
    artifact_dir: Path

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load and validate the two paths required by research entry points."""

        data_value = os.getenv("CERVNET_DATA_DIR")
        artifact_value = os.getenv("CERVNET_ARTIFACT_DIR")
        missing = [
            name
            for name, value in (
                ("CERVNET_DATA_DIR", data_value),
                ("CERVNET_ARTIFACT_DIR", artifact_value),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing required environment variable(s): " + ", ".join(missing)
            )

        return cls(
            data_dir=Path(data_value).expanduser().resolve(),
            artifact_dir=Path(artifact_value).expanduser().resolve(),
        )

