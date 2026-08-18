"""Configuration loading.

Finds config.toml by walking up from the current directory, so the CLI works
from anywhere inside the project. All paths in the returned config are
resolved to absolute paths against the project root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]

CONFIG_FILENAME = "config.toml"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    root: Path
    inbox: Path
    originals: Path
    items: Path
    derivatives: Path
    exports: Path
    site: Path
    database: Path

    def ensure(self) -> None:
        """Create every directory the pipeline writes to."""
        for p in (
            self.inbox,
            self.originals,
            self.items,
            self.derivatives,
            self.exports,
            self.site,
        ):
            p.mkdir(parents=True, exist_ok=True)
        self.database.parent.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        """Path as stored in the database: relative to root, POSIX separators."""
        path = Path(path)
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def absolute(self, stored: str) -> Path:
        """Inverse of `relative`."""
        p = Path(stored)
        return p if p.is_absolute() else (self.root / p)


@dataclass(frozen=True)
class Config:
    paths: Paths
    segment: dict[str, Any] = field(default_factory=dict)
    rectify: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    site: dict[str, Any] = field(default_factory=dict)
    publish: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` looking for config.toml."""
    if env := os.environ.get("ROTARY_ARCHIVE_ROOT"):
        root = Path(env).expanduser().resolve()
        if (root / CONFIG_FILENAME).is_file():
            return root
        raise ConfigError(f"ROTARY_ARCHIVE_ROOT={root} has no {CONFIG_FILENAME}")

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / CONFIG_FILENAME).is_file():
            return candidate
    raise ConfigError(
        f"No {CONFIG_FILENAME} found in {current} or any parent directory. "
        "Run from inside the project, or set ROTARY_ARCHIVE_ROOT."
    )


def load_config(root: Path | None = None) -> Config:
    root = (root or find_project_root()).resolve()
    with (root / CONFIG_FILENAME).open("rb") as fh:
        data = tomllib.load(fh)

    raw_paths = data.get("paths", {})

    def resolve(key: str, default: str) -> Path:
        value = Path(raw_paths.get(key, default)).expanduser()
        return value if value.is_absolute() else (root / value)

    paths = Paths(
        root=root,
        inbox=resolve("inbox", "inbox"),
        originals=resolve("originals", "masters/originals"),
        items=resolve("items", "masters/items"),
        derivatives=resolve("derivatives", "derivatives"),
        exports=resolve("exports", "exports"),
        site=resolve("site", "site"),
        database=resolve("database", "archive.db"),
    )

    return Config(
        paths=paths,
        segment=data.get("segment", {}),
        rectify=data.get("rectify", {}),
        review=data.get("review", {}),
        llm=data.get("llm", {}),
        site=data.get("site", {}),
        publish=data.get("publish", {}),
        raw=data,
    )
