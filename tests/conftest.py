"""Shared fixtures. Every test gets a throwaway project directory."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from rotary_archive import db  # noqa: E402
from rotary_archive.config import load_config  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path: Path):
    """An isolated project rooted at tmp_path, with the real config.toml."""
    shutil.copy(PROJECT_ROOT / "config.toml", tmp_path / "config.toml")
    cfg = load_config(tmp_path)
    cfg.paths.ensure()
    return cfg


@pytest.fixture
def conn(project):
    connection = db.connect(project.paths.database)
    yield connection
    connection.close()
