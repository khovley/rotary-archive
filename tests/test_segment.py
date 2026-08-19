"""Segmentation, measured against synthetic scenes with known ground truth.

The headline metric is item *count*: a slightly loose crop is fixable in the
review UI, but an item the detector never proposed is silently lost from the
archive. Crop tightness is checked separately via IoU.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotary_archive import geometry
from rotary_archive.segment import (
    Candidate,
    _drop_swallowers,
    segment_image,
)
from synthetic import make_table_shot

IOU_PASS = 0.80


def quad(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def candidate(x0, y0, x1, y1, area_frac=0.1, method="edge"):
    return Candidate(
        quad=quad(x0, y0, x1, y1), confidence=0.9, method=method, area_frac=area_frac
    )


# --------------------------------------------------------------- unit tests --


# Candidate holds a numpy array, so `x in list` would invoke elementwise
# comparison and raise. Compare by identity instead.
def contains(items, target):
    return any(item is target for item in items)


def test_drop_swallowers_removes_a_quad_covering_two_items():
    """A blob that engulfed two separate items must not survive - if it did,
    the widest-wins merge would pick it and cost us an item."""
    left = candidate(0, 0, 40, 100, area_frac=0.10)
    right = candidate(60, 0, 100, 100, area_frac=0.10)
    swallower = candidate(0, 0, 100, 100, area_frac=0.40)

    kept = _drop_swallowers([swallower, left, right])

    assert not contains(kept, swallower)
    assert contains(kept, left) and contains(kept, right)


def test_drop_swallowers_keeps_a_quad_containing_one_item():
    """Containing a single smaller detection is normal - that is just the
    content-crop of the same object, not two objects."""
    inner = candidate(10, 10, 90, 90, area_frac=0.10)
    outer = candidate(0, 0, 100, 100, area_frac=0.30)

    kept = _drop_swallowers([outer, inner])
    assert contains(kept, outer)


def test_drop_swallowers_ignores_similarly_sized_neighbours():
    a = candidate(0, 0, 100, 100, area_frac=0.30)
    b = candidate(10, 10, 90, 90, area_frac=0.25)   # not < 0.65 * a
    assert len(_drop_swallowers([a, b])) == 2


# ------------------------------------------------------ end-to-end scenes --


SCENES = [
    pytest.param(dict(n_items=6, seed=7), 6, id="6-dark"),
    pytest.param(dict(n_items=8, seed=11), 8, id="8-dark"),
    pytest.param(dict(n_items=4, seed=3), 4, id="4-dark"),
    pytest.param(dict(n_items=10, seed=13), 10, id="10-dense"),
    pytest.param(dict(n_items=6, seed=9, tilt=0.03, max_angle=20), 6, id="6-tilted"),
    pytest.param(dict(n_items=6, seed=17, background=128), 6, id="6-mid-grey"),
    pytest.param(dict(n_items=6, seed=5, background=235), 6, id="6-light"),
]


@pytest.fixture(scope="module")
def seg_config():
    from rotary_archive.config import load_config
    from pathlib import Path

    return load_config(Path(__file__).resolve().parents[1]).segment


@pytest.mark.parametrize("scene,expected", SCENES)
def test_finds_every_item(tmp_path, seg_config, scene, expected):
    path = tmp_path / "scene.jpg"
    make_table_shot(path, **scene)
    result = segment_image(path, seg_config)
    assert len(result.candidates) == expected


@pytest.mark.parametrize("scene,expected", SCENES)
def test_crops_are_tight(tmp_path, seg_config, scene, expected):
    """Every ground-truth item is matched by a detection at IoU >= 0.8.

    The light-background scene is the documented worst case - low contrast
    between cream paper and a pale table - so it is allowed one loose crop,
    which must be flagged (asserted separately below).
    """
    path = tmp_path / "scene.jpg"
    truth = make_table_shot(path, **scene)
    result = segment_image(path, seg_config)

    ious = [
        max((geometry.bbox_iou(t.quad, c.quad) for c in result.candidates), default=0.0)
        for t in truth
    ]
    allowed_misses = 1 if scene.get("background", 28) > 200 else 0
    misses = sum(1 for i in ious if i < IOU_PASS)
    assert misses <= allowed_misses, f"IoUs: {[round(i, 3) for i in ious]}"


def test_loose_crops_are_flagged(tmp_path, seg_config):
    """The confidence score must actually track crop quality.

    This is the property that makes the review workflow work: if bad crops
    scored high, the "flagged only" view would hide exactly what needs looking
    at. An earlier pass-agreement heuristic got this backwards.
    """
    path = tmp_path / "light.jpg"
    truth = make_table_shot(path, n_items=6, seed=5, background=235)
    result = segment_image(path, seg_config)

    for item in truth:
        iou, cand = max(
            ((geometry.bbox_iou(item.quad, c.quad), c) for c in result.candidates),
            key=lambda pair: pair[0],
        )
        if iou < IOU_PASS:
            assert cand.confidence < 0.80, (
                f"loose crop (IoU {iou:.3f}) scored {cand.confidence:.2f} "
                "and would not be surfaced for review"
            )


def test_single_item_close_up(tmp_path, seg_config):
    path = tmp_path / "single.jpg"
    make_table_shot(path, n_items=1, seed=2)
    result = segment_image(path, seg_config)
    assert len(result.candidates) == 1


def test_blank_frame_falls_back_to_whole_image(tmp_path, seg_config):
    """An empty background yields no contours; rather than dropping the photo
    we hand back the whole frame and flag it."""
    import cv2

    path = tmp_path / "blank.jpg"
    cv2.imwrite(str(path), np.full((600, 800, 3), 30, np.uint8))

    result = segment_image(path, seg_config)
    assert len(result.candidates) == 1
    assert result.needs_review
    assert result.candidates[0].method == "whole_frame"


def test_reading_order_is_top_to_bottom_then_left_to_right(tmp_path, seg_config):
    path = tmp_path / "order.jpg"
    make_table_shot(path, n_items=6, seed=7)
    result = segment_image(path, seg_config)

    centres = [c.quad.mean(axis=0) for c in result.candidates]
    rows = [(round(c[1] / 500), c[0]) for c in centres]
    assert rows == sorted(rows), f"out of order: {rows}"


def test_small_gaps_between_items_are_not_bridged(tmp_path, seg_config):
    """Morphology must not weld neighbouring items into one blob.

    A closing kernel wide enough to reach across the gap people actually leave
    between clippings merged eight of them into a single 50%-of-frame shape
    that no rectangle test would accept, so the photo produced no items at all.
    Hole-filling is done by redrawing contours solid instead, which closes a
    hole of any size without reaching sideways to a neighbour.
    """
    path = tmp_path / "tight.jpg"
    # A dense grid packs items close together, which is where bridging shows.
    truth = make_table_shot(path, n_items=10, seed=13)
    result = segment_image(path, seg_config)

    assert len(result.candidates) == len(truth)
    biggest = max(c.area_frac for c in result.candidates)
    assert biggest < 0.25, (
        f"largest detection covers {biggest:.0%} of the frame - items look merged"
    )


def test_irregular_clippings_are_not_rejected(tmp_path, seg_config):
    """A clipping cut around a headline is L-shaped, not rectangular.

    The solidity and fill filters existed to throw out hands and cast shadows,
    but at their original settings they also threw out genuine clippings whose
    outline juts out to keep a headline or photo attached.
    """
    import cv2
    import numpy as np

    canvas = np.full((1400, 1100, 3), 26, np.uint8)
    # An L: a wide headline band with a narrower column hanging below it.
    cv2.rectangle(canvas, (150, 200), (950, 420), (232, 228, 216), -1)
    cv2.rectangle(canvas, (150, 420), (560, 1150), (232, 228, 216), -1)
    for y in range(460, 1120, 34):
        cv2.rectangle(canvas, (185, y), (520, y + 12), (40, 40, 40), -1)
    cv2.rectangle(canvas, (190, 240), (900, 300), (30, 30, 30), -1)
    path = tmp_path / "ell.jpg"
    cv2.imwrite(str(path), canvas)

    result = segment_image(path, seg_config)

    assert len(result.candidates) == 1, "the L-shaped clipping was not found"
    quad = result.candidates[0].quad
    x0, y0, x1, y1 = geometry.quad_bbox(quad)
    # The crop must contain the whole item, headline band included, rather
    # than trimming to the tidy rectangular column.
    assert x0 <= 175 and y0 <= 225
    assert x1 >= 925 and y1 >= 1125


# ------------------------------------------------- relationship persistence --


def test_write_links_persists_all_three_relationships(tmp_path):
    """The vision pass can assert three different things about how items
    relate, and they are stored three different ways because they mean three
    different things to the site: a merge, a cross-reference, and a hide."""
    from rotary_archive import db
    from rotary_archive.segment import _write_links
    from rotary_archive.vision_segment import Region

    conn = db.connect(tmp_path / "t.db")
    with db.transaction(conn):
        db.insert_photo(
            conn, sha256="a" * 64, original_name="p.jpg", stored_path="p.jpg",
            width=100, height=100, captured_at=None, exif=None, size_bytes=1,
        )
        ids = []
        for seq in range(4):
            item_id = f"{'a' * 12}-{seq:02d}"
            db.insert_item(
                conn, item_id=item_id, photo_sha256="a" * 64, seq=seq,
                quad=[[0, 0], [1, 0], [1, 1], [0, 1]],
                detection_confidence=0.9, detection_method="vision",
            )
            ids.append(item_id)

    regions = [
        Region(box=(0, 0, 1, 1)),
        Region(box=(0, 0, 1, 1), part_of=0, part_reason="continues the story"),
        Region(box=(0, 0, 1, 1), related_to=[0], related_reason="same event"),
        Region(box=(0, 0, 1, 1), duplicate_of=0),
    ]
    with db.transaction(conn):
        _write_links(conn, regions, ids)

    rows = {row["id"]: row for row in db.items_for_photo(conn, "a" * 64)}
    assert rows[ids[1]]["part_of_item_id"] == ids[0]
    assert rows[ids[1]]["part_reason"] == "continues the story"
    assert rows[ids[3]]["duplicate_of_item_id"] == ids[0]

    # The cross-reference is stored both ways round: which item the model
    # happened to list first must not decide what a reader can navigate to.
    assert [r["related_item_id"] for r in db.related_items(conn, ids[2])] == [ids[0]]
    assert [r["related_item_id"] for r in db.related_items(conn, ids[0])] == [ids[2]]
    # And it did not merge them.
    assert rows[ids[2]]["part_of_item_id"] is None


def test_write_links_ignores_indexes_past_the_item_list(tmp_path):
    """Defence in depth: the parser already drops these, but _write_links
    turns an index into a foreign key and must not trust its input."""
    from rotary_archive import db
    from rotary_archive.segment import _write_links
    from rotary_archive.vision_segment import Region

    conn = db.connect(tmp_path / "t.db")
    with db.transaction(conn):
        db.insert_photo(
            conn, sha256="b" * 64, original_name="p.jpg", stored_path="p.jpg",
            width=100, height=100, captured_at=None, exif=None, size_bytes=1,
        )
        db.insert_item(
            conn, item_id="b" * 12 + "-00", photo_sha256="b" * 64, seq=0,
            quad=[[0, 0], [1, 0], [1, 1], [0, 1]],
            detection_confidence=0.9, detection_method="vision",
        )

    with db.transaction(conn):
        _write_links(
            conn,
            [Region(box=(0, 0, 1, 1), part_of=7, related_to=[9], duplicate_of=5)],
            ["b" * 12 + "-00"],
        )

    row = db.items_for_photo(conn, "b" * 64)[0]
    assert row["part_of_item_id"] is None
    assert row["duplicate_of_item_id"] is None
    assert db.related_items(conn, row["id"]) == []


@pytest.mark.parametrize("kwargs,expect", [
    ({"part_of": 0, "link_confidence": 0.25}, "belongs with another item"),
    ({"duplicate_of": 0, "link_confidence": 0.4}, "duplicate"),
    ({"related_to": [0], "link_confidence": 0.5}, "linked to another item"),
])
def test_an_uncertain_relationship_is_flagged_for_review(kwargs, expect):
    """A shaky crop costs a moment to fix. A shaky *relationship* publishes two
    objects as one entry, or hides one from the site entirely - a claim about
    the club's history. It gets flagged even when the box is perfect."""
    from rotary_archive.segment import _uncertain_link
    from rotary_archive.vision_segment import Region

    note = _uncertain_link(Region(box=(0, 0, 1, 1), **kwargs), 0.80)
    assert note is not None and expect in note


def test_a_confident_relationship_is_not_flagged():
    from rotary_archive.segment import _uncertain_link
    from rotary_archive.vision_segment import Region

    region = Region(box=(0, 0, 1, 1), part_of=0, link_confidence=0.95)
    assert _uncertain_link(region, 0.80) is None


def test_an_item_asserting_nothing_is_not_flagged_by_link_confidence():
    """link_confidence defaults low and means nothing when no link is claimed;
    flagging on it alone would flag every standalone item on the table."""
    from rotary_archive.segment import _uncertain_link
    from rotary_archive.vision_segment import Region

    assert _uncertain_link(Region(box=(0, 0, 1, 1), link_confidence=0.1), 0.80) is None
