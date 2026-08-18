"""Adding photos: upload sanitisation, waiting counts, and the pipeline job.

The upload path takes a filename from whatever was dragged onto the page, so
it is treated as untrusted even though the server only listens on localhost.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from PIL import Image

from rotary_archive import db
from rotary_archive.review.server import (
    UPLOAD_SUFFIXES,
    PipelineJob,
    inbox_state,
    safe_upload_name,
)
from synthetic import make_table_shot


_photo_seed = 0


def write_photo(path: Path):
    """A photo with content unique to this call.

    Distinct pixels matter: the archive is content-addressed, so two
    byte-identical files are one photo by design, and a fixture that wrote the
    same image twice would be testing deduplication rather than whatever the
    test meant to check.
    """
    global _photo_seed
    _photo_seed += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (40, 30), (180, 60, 60))
    image.putpixel((0, 0), (_photo_seed % 256, (_photo_seed * 7) % 256, 11))
    image.save(path)
    return path


# ------------------------------------------------------- name sanitisation --


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("IMG_1234.HEIC", "IMG_1234.HEIC"),
        ("holiday snap.jpg", "holiday snap.jpg"),
        ("Scan 001.tiff", "Scan 001.tiff"),
    ],
)
def test_ordinary_names_pass_through(raw, expected):
    assert safe_upload_name(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "../../../../etc/passwd.jpg",
        "../secrets.png",
        "/etc/hosts.jpg",
        "subdir/nested.jpg",
        "..\\..\\windows\\evil.jpg",
    ],
)
def test_directory_components_are_stripped(raw):
    """A traversal attempt must land in the inbox as a plain filename, never
    outside it."""
    safe = safe_upload_name(raw)
    assert "/" not in safe and "\\" not in safe
    assert not safe.startswith("..")
    assert Path(safe).name == safe


@pytest.mark.parametrize("raw", ["payload.sh", "notes.txt", "archive.db", "x.py"])
def test_non_photo_extensions_are_refused(raw):
    """The inbox should only ever hold files ingest can actually read."""
    with pytest.raises(ValueError, match="not a photo"):
        safe_upload_name(raw)


@pytest.mark.parametrize("raw", ["", "   ", ".", "..", ".hidden.jpg"])
def test_unusable_names_are_refused(raw):
    with pytest.raises(ValueError):
        safe_upload_name(raw)


def test_every_accepted_suffix_is_one_ingest_understands():
    from rotary_archive.ingest import SUPPORTED_SUFFIXES

    assert UPLOAD_SUFFIXES <= SUPPORTED_SUFFIXES


# ------------------------------------------------------------ inbox state --


def test_empty_inbox(project):
    state = inbox_state(project)
    assert state["present"] == 0
    assert state["waiting"] == 0
    assert state["exact"] is True
    assert state["path"] == str(project.paths.inbox)


def test_new_photos_are_counted_as_waiting(project, conn):
    write_photo(project.paths.inbox / "a.jpg")
    write_photo(project.paths.inbox / "b.jpg")
    assert inbox_state(project)["waiting"] == 2


def test_already_ingested_photos_are_not_waiting(project, conn):
    """Without --move, ingested files stay in the inbox. Counting those as
    waiting would nag forever."""
    from rotary_archive.ingest import ingest_inbox

    write_photo(project.paths.inbox / "a.jpg")
    ingest_inbox(conn, project.paths)

    state = inbox_state(project)
    assert state["present"] == 1
    assert state["waiting"] == 0


def test_mixed_inbox_counts_only_the_new_ones(project, conn):
    from rotary_archive.ingest import ingest_inbox

    write_photo(project.paths.inbox / "old.jpg")
    ingest_inbox(conn, project.paths)
    write_photo(project.paths.inbox / "new.jpg")

    state = inbox_state(project)
    assert state["present"] == 2
    assert state["waiting"] == 1


def test_non_images_are_ignored(project, conn):
    write_photo(project.paths.inbox / "a.jpg")
    (project.paths.inbox / "notes.txt").write_text("not a photo")
    assert inbox_state(project)["present"] == 1


def test_huge_inbox_skips_hashing_and_says_so(project, conn, monkeypatch):
    """Past a few hundred files, hashing every one to answer 'how many are
    new' costs more than the precision is worth."""
    import rotary_archive.review.server as server

    fake = [project.paths.inbox / f"{i}.jpg" for i in range(500)]
    monkeypatch.setattr(server, "find_photos", lambda p: fake, raising=False)
    monkeypatch.setattr(
        "rotary_archive.ingest.find_photos", lambda p: fake
    )

    state = inbox_state(project)
    assert state["present"] == 500
    assert state["exact"] is False


# -------------------------------------------------------------- pipeline ---


def wait_for(job: PipelineJob, timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = job.snapshot()
        if snap["state"] not in ("running",):
            return snap
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_job_runs_the_whole_pipeline(project, conn):
    make_table_shot(project.paths.inbox / "table.jpg", n_items=4, seed=3)

    job = PipelineJob()
    assert job.start(project) is True
    snap = wait_for(job)

    assert snap["state"] == "done", snap
    assert snap["counts"]["ingested"] == 1
    assert snap["counts"]["items"] == 4
    assert snap["counts"]["rectified"] == 4

    stats = db.counts(conn)
    assert stats["photos"] == 1
    assert stats["items"] == 4


def test_only_one_job_runs_at_a_time(project):
    """Both stages write the same database and image directories; two runs
    would race over both."""
    make_table_shot(project.paths.inbox / "table.jpg", n_items=6, seed=7)

    job = PipelineJob()
    assert job.start(project) is True
    second = job.start(project)          # while the first is still going
    wait_for(job)

    assert second is False, "a second concurrent run was allowed to start"


def test_job_can_be_run_again_after_finishing(project):
    make_table_shot(project.paths.inbox / "a.jpg", n_items=1, seed=2)
    job = PipelineJob()
    job.start(project)
    wait_for(job)

    make_table_shot(project.paths.inbox / "b.jpg", n_items=1, seed=5)
    assert job.start(project) is True
    snap = wait_for(job)
    assert snap["counts"]["ingested"] == 1


def test_unreadable_files_are_reported_but_do_not_fail_the_run(project):
    """One corrupt file in a batch of hundreds must not stop the rest."""
    make_table_shot(project.paths.inbox / "good.jpg", n_items=4, seed=3)
    (project.paths.inbox / "broken.jpg").write_bytes(b"this is not a JPEG")

    job = PipelineJob()
    job.start(project)
    snap = wait_for(job)

    assert snap["state"] == "done"
    assert snap["counts"]["ingested"] == 1
    assert snap["counts"]["unreadable"] == 1
    assert "broken.jpg" in (snap["error"] or "")


def test_running_with_an_empty_inbox_is_harmless(project):
    job = PipelineJob()
    job.start(project)
    snap = wait_for(job)
    assert snap["state"] == "done"
    assert snap["counts"]["ingested"] == 0


def test_snapshot_is_safe_before_anything_runs():
    snap = PipelineJob().snapshot()
    assert snap["state"] == "idle"
    assert snap["elapsed"] is None
    assert snap["counts"] == {}


def test_job_reports_elapsed_time(project):
    make_table_shot(project.paths.inbox / "a.jpg", n_items=1, seed=2)
    job = PipelineJob()
    job.start(project)
    snap = wait_for(job)
    assert isinstance(snap["elapsed"], float) and snap["elapsed"] >= 0


def test_identical_files_under_different_names_are_one_photo(project, conn):
    """Content addressing, seen from the inbox panel.

    A member re-exporting the same photo under a new name should not add a
    second copy, and the panel should not report it as waiting.
    """
    from rotary_archive.ingest import ingest_inbox

    original = write_photo(project.paths.inbox / "IMG_0001.jpg")
    ingest_inbox(conn, project.paths)
    (project.paths.inbox / "IMG_0001 copy.jpg").write_bytes(original.read_bytes())

    state = inbox_state(project)
    assert state["present"] == 2
    assert state["waiting"] == 0, "a byte-identical copy is not a new photo"
