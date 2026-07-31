from pathlib import Path

import pytest

from cervnet.config import Settings


def test_settings_load_paths_from_environment(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setenv("CERVNET_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CERVNET_ARTIFACT_DIR", str(artifact_dir))

    settings = Settings.from_environment()

    assert settings.data_dir == data_dir
    assert settings.artifact_dir == artifact_dir


def test_settings_report_missing_variables(monkeypatch) -> None:
    monkeypatch.delenv("CERVNET_DATA_DIR", raising=False)
    monkeypatch.delenv("CERVNET_ARTIFACT_DIR", raising=False)

    with pytest.raises(RuntimeError, match="CERVNET_DATA_DIR"):
        Settings.from_environment()

