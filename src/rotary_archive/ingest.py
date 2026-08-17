"""Ingest: pull photos out of the inbox into the archive.

Content-addressed by SHA-256, so re-dropping the same photo is a no-op and the
inbox can stay messy. iPhone HEIC files are decoded via pillow-heif and stored
as-is; EXIF orientation is recorded here and applied at read time rather than
baked into the master, keeping the original bytes untouched.

GPS is deliberately dropped. A club archive has no reason to publish the
coordinates of the room the photos were taken in.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from PIL import ExifTags, Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:  # pragma: no cover
    HEIF_AVAILABLE = False

from . import db

SUPPORTED_SUFFIXES = {
    ".heic",
    ".heif",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

# EXIF tags worth keeping. Everything else (notably GPS) is discarded.
_KEEP_EXIF = {
    "Make",
    "Model",
    "LensModel",
    "DateTimeOriginal",
    "DateTimeDigitized",
    "DateTime",
    "Orientation",
    "ExifImageWidth",
    "ExifImageHeight",
    "FNumber",
    "ExposureTime",
    "ISOSpeedRatings",
    "FocalLength",
}

_TAG_NAMES = {v: k for k, v in ExifTags.TAGS.items()}


@dataclass
class IngestResult:
    ingested: list[str]
    skipped_duplicate: list[Path]
    unreadable: list[tuple[Path, str]]

    @property
    def total_seen(self) -> int:
        return len(self.ingested) + len(self.skipped_duplicate) + len(self.unreadable)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash file contents. Streamed so a 5GB drop doesn't blow up memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def find_photos(inbox: Path) -> list[Path]:
    """Every supported image under inbox, recursively, sorted for determinism."""
    if not inbox.exists():
        return []
    return sorted(
        p
        for p in inbox.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not p.name.startswith(".")
    )


def _parse_exif_datetime(value: Any) -> str | None:
    """EXIF uses 'YYYY:MM:DD HH:MM:SS'. Return ISO 8601, or None if unparseable."""
    if not value or not isinstance(value, str):
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d"):
        try:
            return datetime.strptime(value.strip(), fmt).isoformat(timespec="seconds")
        except ValueError:
            continue
    return None


def read_metadata(path: Path) -> dict[str, Any]:
    """Dimensions, capture time, and a filtered EXIF subset.

    Dimensions are reported post-orientation, matching what the segmentation
    stage will see once ImageOps.exif_transpose has been applied.
    """
    with Image.open(path) as img:
        raw = img.getexif() or {}
        exif: dict[str, Any] = {}
        for name in _KEEP_EXIF:
            tag = _TAG_NAMES.get(name)
            if tag is None or tag not in raw:
                continue
            value = raw[tag]
            if isinstance(value, bytes):
                value = value.decode("utf-8", "replace")
            exif[name] = value if isinstance(value, (int, float, str)) else str(value)

        oriented = ImageOps.exif_transpose(img)
        width, height = oriented.size
        if oriented is not img:
            oriented.close()

    captured_at = (
        _parse_exif_datetime(exif.get("DateTimeOriginal"))
        or _parse_exif_datetime(exif.get("DateTimeDigitized"))
        or _parse_exif_datetime(exif.get("DateTime"))
    )
    return {
        "width": width,
        "height": height,
        "captured_at": captured_at,
        "exif": exif or None,
    }


def load_oriented(path: Path) -> Image.Image:
    """Open an image with EXIF orientation applied. The rest of the pipeline
    only ever sees upright pixels."""
    img = Image.open(path)
    return ImageOps.exif_transpose(img) or img


def ingest_inbox(
    conn: sqlite3.Connection,
    paths: Any,
    *,
    move: bool = False,
    progress: Any = None,
) -> IngestResult:
    """Ingest everything in the inbox.

    `move` deletes the inbox copy after a successful archive copy. Off by
    default - a failed run should never be able to lose a source photo.
    """
    result = IngestResult(ingested=[], skipped_duplicate=[], unreadable=[])
    candidates = find_photos(paths.inbox)

    for source in candidates:
        if progress is not None:
            progress(source)
        try:
            sha = sha256_file(source)
        except OSError as exc:
            result.unreadable.append((source, f"read failed: {exc}"))
            continue

        if db.photo_exists(conn, sha):
            result.skipped_duplicate.append(source)
            if move:
                source.unlink(missing_ok=True)
            continue

        try:
            meta = read_metadata(source)
        except Exception as exc:  # Pillow raises a wide variety here
            result.unreadable.append((source, f"not a readable image: {exc}"))
            continue

        stored = paths.originals / f"{sha}{source.suffix.lower()}"
        try:
            if not stored.exists():
                shutil.copy2(source, stored)
        except OSError as exc:
            result.unreadable.append((source, f"copy failed: {exc}"))
            continue

        with db.transaction(conn):
            db.insert_photo(
                conn,
                sha256=sha,
                original_name=source.name,
                stored_path=paths.relative(stored),
                width=meta["width"],
                height=meta["height"],
                captured_at=meta["captured_at"],
                exif=meta["exif"],
                size_bytes=stored.stat().st_size,
            )
        result.ingested.append(sha)

        if move:
            source.unlink(missing_ok=True)

    return result
