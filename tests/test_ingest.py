"""Ingest: hashing, dedupe, EXIF, and orientation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rotary_archive import db
from rotary_archive.ingest import (
    find_photos,
    ingest_inbox,
    load_oriented,
    read_metadata,
    sha256_file,
)


def write_image(path: Path, width: int = 40, height: int = 30, colour=(200, 40, 40)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), colour).save(path)
    return path


def test_sha256_matches_for_identical_bytes(tmp_path):
    a = write_image(tmp_path / "a.png")
    b = tmp_path / "b.png"
    b.write_bytes(a.read_bytes())
    assert sha256_file(a) == sha256_file(b)


def test_sha256_differs_for_different_content(tmp_path):
    a = write_image(tmp_path / "a.png", colour=(10, 10, 10))
    b = write_image(tmp_path / "b.png", colour=(250, 250, 250))
    assert sha256_file(a) != sha256_file(b)


def test_find_photos_filters_and_sorts(tmp_path):
    write_image(tmp_path / "b.jpg")
    write_image(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / ".hidden.jpg").write_bytes(b"")

    found = [p.name for p in find_photos(tmp_path)]
    assert found == ["a.png", "b.jpg"]


def test_find_photos_recurses(tmp_path):
    write_image(tmp_path / "nested" / "deep" / "x.jpg")
    assert len(find_photos(tmp_path)) == 1


def test_find_photos_on_missing_dir_returns_empty(tmp_path):
    assert find_photos(tmp_path / "nope") == []


def test_read_metadata_reports_dimensions(tmp_path):
    path = write_image(tmp_path / "x.png", width=64, height=48)
    meta = read_metadata(path)
    assert (meta["width"], meta["height"]) == (64, 48)


def test_exif_orientation_is_applied_on_load(tmp_path):
    """A portrait image tagged 'rotate 90' must come back upright.

    iPhone photos routinely carry orientation in metadata rather than pixels;
    if this is not applied, every quad the detector produces is transposed.
    """
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (100, 50), (120, 120, 120))
    exif = image.getexif()
    exif[274] = 6  # Orientation: rotate 90 CW
    image.save(path, exif=exif)

    with Image.open(path) as raw:
        assert raw.size == (100, 50)          # stored dimensions

    with load_oriented(path) as oriented:
        assert oriented.size == (50, 100)     # display dimensions

    # read_metadata reports post-orientation dimensions, matching what
    # segmentation will actually see.
    assert (read_metadata(path)["width"], read_metadata(path)["height"]) == (50, 100)


def test_ingest_copies_and_records(project, conn):
    write_image(project.paths.inbox / "one.png", 20, 10)
    result = ingest_inbox(conn, project.paths)

    assert len(result.ingested) == 1
    photo = db.get_photo(conn, result.ingested[0])
    assert photo["original_name"] == "one.png"
    assert (photo["width"], photo["height"]) == (20, 10)
    assert photo["status"] == "ingested"
    assert project.paths.absolute(photo["stored_path"]).exists()


def test_ingest_is_idempotent(project, conn):
    write_image(project.paths.inbox / "one.png")

    first = ingest_inbox(conn, project.paths)
    second = ingest_inbox(conn, project.paths)

    assert len(first.ingested) == 1
    assert len(second.ingested) == 0
    assert len(second.skipped_duplicate) == 1
    assert db.counts(conn)["photos"] == 1


def test_ingest_deduplicates_same_content_different_names(project, conn):
    """The same photo dropped twice under different names is one archive item."""
    a = write_image(project.paths.inbox / "IMG_001.png")
    (project.paths.inbox / "copy of IMG_001.png").write_bytes(a.read_bytes())

    result = ingest_inbox(conn, project.paths)
    assert len(result.ingested) == 1
    assert len(result.skipped_duplicate) == 1


def test_ingest_reports_unreadable_files_without_aborting(project, conn):
    write_image(project.paths.inbox / "good.png")
    (project.paths.inbox / "broken.jpg").write_bytes(b"this is not a JPEG")

    result = ingest_inbox(conn, project.paths)

    assert len(result.ingested) == 1          # the good one still got through
    assert len(result.unreadable) == 1
    assert result.unreadable[0][0].name == "broken.jpg"


def test_ingest_move_removes_source(project, conn):
    path = write_image(project.paths.inbox / "one.png")
    ingest_inbox(conn, project.paths, move=True)
    assert not path.exists()


def test_ingest_without_move_keeps_source(project, conn):
    path = write_image(project.paths.inbox / "one.png")
    ingest_inbox(conn, project.paths, move=False)
    assert path.exists()


def test_exif_retention_is_an_allowlist(tmp_path):
    """Only explicitly named tags survive.

    An allowlist rather than a GPS blocklist: a club archive has no reason to
    publish where a photo was taken, and a blocklist would leak whatever new
    tag a future phone starts writing.
    """
    from rotary_archive.ingest import _KEEP_EXIF

    # GPS can only survive by being named, and it is not named. Asserting on
    # the allowlist rather than round-tripping a real GPS IFD keeps this test
    # about our filtering instead of about Pillow's IFD handling.
    assert "GPSInfo" not in _KEEP_EXIF
    assert not any("GPS" in name for name in _KEEP_EXIF)

    path = tmp_path / "tagged.jpg"
    image = Image.new("RGB", (30, 30), (90, 90, 90))
    exif = image.getexif()
    exif[271] = "TestMake"          # Make - on the allowlist
    exif[315] = "Some Photographer"  # Artist - not on the allowlist
    exif[33432] = "Copyright 1962"   # Copyright - not on the allowlist
    image.save(path, exif=exif)

    kept = read_metadata(path)["exif"] or {}
    assert kept.get("Make") == "TestMake"
    assert set(kept).issubset(_KEEP_EXIF)
    assert "Some Photographer" not in str(kept)


def test_exif_keys_are_names_not_numeric_tags(tmp_path):
    """Stored EXIF uses readable names, so a numeric tag that slipped through
    the allowlist would be obvious in the database rather than opaque."""
    path = tmp_path / "named.jpg"
    image = Image.new("RGB", (30, 30), (90, 90, 90))
    exif = image.getexif()
    exif[271] = "TestMake"
    exif[272] = "TestModel"
    image.save(path, exif=exif)

    kept = read_metadata(path)["exif"] or {}
    assert set(kept) == {"Make", "Model"}
    assert all(isinstance(key, str) for key in kept)
