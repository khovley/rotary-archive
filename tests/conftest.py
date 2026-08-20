"""Shared fixtures. Every test gets a throwaway project directory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from rotary_archive import db  # noqa: E402
from rotary_archive.config import load_config  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path: Path):
    """An isolated project rooted at tmp_path, with the real config.toml.

    Vision is forced off. The shipped config has it on - it is what makes
    segmentation work - but a test that reaches a live model is not a test:
    it is slow, it costs money, it needs credentials, and its result depends
    on something outside the repository. Tests that exercise the reading path
    pass a FakeProvider explicitly.
    """
    text = (PROJECT_ROOT / "config.toml").read_text()
    text = text.replace("use_vision = true", "use_vision = false")
    (tmp_path / "config.toml").write_text(text)

    cfg = load_config(tmp_path)
    assert cfg.segment.get("use_vision") is False, "test config still reads pages"
    cfg.paths.ensure()
    return cfg


@pytest.fixture
def conn(project):
    connection = db.connect(project.paths.database)
    yield connection
    connection.close()
