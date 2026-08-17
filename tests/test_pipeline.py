"""End-to-end pipeline and database behaviour.

Covers the properties that make the workflow usable on a real collection:
resumability, idempotency, and that a re-crop invalidates its own approval.
"""

from __future__ import annotations

import json

import pytest

from rotary_archive import db
from rotary_archive.ingest import ingest_inbox
from rotary_archive.rectify import rectify_pending
from rotary_archive.segment import segment_pending
from synthetic import make_table_shot


@pytest.fixture
def loaded(project, conn):
    """Two table shots ingested, segmented, and rectified."""
    make_table_shot(project.paths.inbox / "table_a.jpg", n_items=6, seed=7)
    make_table_shot(project.paths.inbox / "table_b.jpg", n_items=4, seed=3)

    ingest_inbox(conn, project.paths)
    segment_pending(conn, project.paths, project.segment, flag_below=0.80)
    rectify_pending(conn, project.paths, project.rectify)
    return project, conn


def test_full_pipeline_produces_expected_items(loaded):
    project, conn = loaded
    stats = db.counts(conn)
    assert stats["photos"] == 2
    assert stats["items"] == 10           # 6 + 4, matching ground truth
    assert stats["items_by_status"]["rectified"] == 10


def test_every_item_has_a_master_and_three_derivatives(loaded):
    project, conn = loaded
    for item in db.items_with_status(conn, "rectified"):
        assert project.paths.absolute(item["master_path"]).exists()
        derivatives = db.derivatives_for_item(conn, item["id"])
        assert len(derivatives) == 3
        for derivative in derivatives:
            assert project.paths.absolute(derivative["path"]).exists()


def test_stages_are_resumable(loaded):
    """Re-running each stage does no work and changes nothing - the property
    that lets an interrupted run be restarted without re-billing or
    duplicating."""
    project, conn = loaded
    before = db.counts(conn)

    assert ingest_inbox(conn, project.paths).ingested == []
    assert segment_pending(conn, project.paths, project.segment) == []
    assert rectify_pending(conn, project.paths, project.rectify) == []

    assert db.counts(conn) == before


def test_item_ids_are_stable_across_resegmentation(loaded):
    """IDs derive from the photo hash and sequence, so a re-segment of an
    unchanged photo reproduces the same identifiers."""
    project, conn = loaded
    before = sorted(i["id"] for i in db.items_with_status(conn, "rectified"))

    segment_pending(conn, project.paths, project.segment, force=True)
    after = sorted(i["id"] for i in db.items_with_status(conn, "detected"))

    assert before == after


def test_recrop_invalidates_approval(loaded):
    """An approval applies to the image that was approved. Changing the crop
    makes it a different image, so it must go back for confirmation."""
    project, conn = loaded
    item = db.items_with_status(conn, "rectified")[0]

    with db.transaction(conn):
        db.set_item_status(conn, item["id"], "approved")
    assert db.get_item(conn, item["id"])["status"] == "approved"

    quad = json.loads(item["quad"])
    shrunk = [[x + 5, y + 5] for x, y in quad]
    with db.transaction(conn):
        db.update_item_quad(conn, item["id"], shrunk)

    assert db.get_item(conn, item["id"])["status"] == "detected"


def test_recrop_preserves_the_original_detection(loaded):
    """quad_detected is never overwritten, so review can always offer
    'reset to detected'."""
    project, conn = loaded
    item = db.items_with_status(conn, "rectified")[0]
    original = json.loads(item["quad_detected"])

    with db.transaction(conn):
        db.update_item_quad(conn, item["id"], [[1, 2], [3, 4], [5, 6], [7, 8]])

    refreshed = db.get_item(conn, item["id"])
    assert json.loads(refreshed["quad_detected"]) == original
    assert json.loads(refreshed["quad"]) == [[1, 2], [3, 4], [5, 6], [7, 8]]


def test_deleting_a_photo_cascades_to_items_and_derivatives(loaded):
    project, conn = loaded
    photo = db.all_photos(conn)[0]
    item_id = db.items_for_photo(conn, photo["sha256"])[0]["id"]

    with db.transaction(conn):
        conn.execute("DELETE FROM photos WHERE sha256 = ?", (photo["sha256"],))

    assert db.get_item(conn, item_id) is None
    assert db.derivatives_for_item(conn, item_id) == []


def test_review_log_records_decisions(loaded):
    project, conn = loaded
    item = db.items_with_status(conn, "rectified")[0]

    with db.transaction(conn):
        db.log_review(conn, item_id=item["id"], action="approve", actor="tester")

    rows = conn.execute(
        "SELECT * FROM review_log WHERE item_id = ?", (item["id"],)
    ).fetchall()
    assert [r["action"] for r in rows] == ["approve"]
    assert rows[0]["actor"] == "tester"


def test_analyses_view_returns_only_the_newest(project, conn):
    """Re-analysing appends rather than overwrites; the view must expose the
    live row so a better model can be re-run without losing history."""
    make_table_shot(project.paths.inbox / "t.jpg", n_items=1, seed=2)
    ingest_inbox(conn, project.paths)
    segment_pending(conn, project.paths, project.segment)
    item_id = db.items_with_status(conn, "detected")[0]["id"]

    for title in ("first pass", "second pass"):
        with db.transaction(conn):
            db.supersede_analyses(conn, item_id)
            conn.execute(
                "INSERT INTO analyses (item_id, provider, model, created_at, "
                "title, raw_json) VALUES (?, 'test', 'm', ?, ?, '{}')",
                (item_id, db.utcnow(), title),
            )

    assert db.current_analysis(conn, item_id)["title"] == "second pass"
    total = conn.execute(
        "SELECT COUNT(*) FROM analyses WHERE item_id = ?", (item_id,)
    ).fetchone()[0]
    assert total == 2, "history must be retained, not overwritten"


def test_next_item_seq_appends_after_existing(loaded):
    project, conn = loaded
    photo = db.all_photos(conn)[0]
    existing = len(db.items_for_photo(conn, photo["sha256"]))
    assert db.next_item_seq(conn, photo["sha256"]) == existing
