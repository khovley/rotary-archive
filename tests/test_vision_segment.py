"""LLM-guided segmentation.

The division of labour is the thing worth protecting here: the model decides
what is on the table and how many pieces there are, and the pixels decide
where each one ends. These tests pin both halves, and the boundary between
them - a model that returns nonsense coordinates must not be able to produce
a nonsense crop.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from fake_provider import FakeProvider
from rotary_archive import geometry
from rotary_archive.vision_segment import (
    GRID,
    MEDIA,
    VISION_LONG_EDGE,
    Region,
    _parse_regions,
    _to_pixels,
    prepare_image,
    propose_regions,
    refine_regions,
    segment_with_vision,
)
from synthetic import make_table_shot


def vision_provider(items):
    """A provider that answers the segmentation prompt with `items`."""
    return FakeProvider(responder=lambda job: {"items": items})


def entry(box, **kw):
    base = {
        "box": box,
        "kind": "newspaper_clipping",
        "medium": "newsprint",
        "headline": "A Headline",
        "date_hint": "",
        "shape_note": "",
        "panels": 1,
        "clipped_by_frame": False,
        "part_of": -1,
        "part_reason": "",
        "related_to": [],
        "related_reason": "",
        "duplicate_of": -1,
        "link_confidence": 0.9,
        "confidence": 0.9,
    }
    base.update(kw)
    return base


# ------------------------------------------------------------ coordinates --


def test_to_pixels_maps_the_grid_onto_the_image():
    assert _to_pixels([0, 0, GRID, GRID], 1000, 500) == (0.0, 0.0, 999.0, 499.0)
    left, top, right, bottom = _to_pixels([500, 500, 1000, 1000], 800, 600)
    assert (left, top) == (400.0, 300.0)


def test_to_pixels_repairs_inverted_corners():
    """A model that names the corners in the wrong order still gets a box."""
    assert _to_pixels([600, 400, 200, 100], 1000, 1000) == (200.0, 100.0, 600.0, 400.0)


@pytest.mark.parametrize(
    "box", [None, "big", [1, 2, 3], [0, 0, 0, 0], ["a", 0, 1, 1], [10, 10, 10.2, 90]]
)
def test_to_pixels_rejects_unusable_boxes(box):
    assert _to_pixels(box, 1000, 1000) is None


def test_boxes_are_clamped_to_the_image():
    """A box running off the edge is clipped, not allowed to index outside."""
    left, top, right, bottom = _to_pixels([-200, -50, 1400, 1200], 400, 300)
    assert (left, top) == (0.0, 0.0)
    assert (right, bottom) == (399.0, 299.0)


# ----------------------------------------------------------------- parsing --


def test_unusable_entries_are_dropped_not_fatal():
    regions = _parse_regions(
        [entry([0, 0, 100, 100]), "not a dict", {"box": None}, entry([200, 200, 400, 400])],
        1000, 1000,
    )
    assert len(regions) == 2


def test_percentage_confidence_is_normalised():
    assert _parse_regions([entry([0, 0, 100, 100], confidence=90)], 1000, 1000)[0].confidence == 0.9


def test_unknown_kind_falls_back_rather_than_inventing_a_type():
    regions = _parse_regions([entry([0, 0, 100, 100], kind="sculpture")], 1000, 1000)
    assert regions[0].kind == "other"


@pytest.mark.parametrize("part_of", [5, -3, 0])
def test_a_grouping_that_points_nowhere_is_discarded(part_of):
    """part_of is an index into this photo's own items. Out of range, or an
    item claiming to be part of itself, means the model lost track - and a bad
    link would silently bury one clipping inside another on the site."""
    regions = _parse_regions(
        [entry([0, 0, 100, 100], part_of=part_of, part_reason="continues"),
         entry([200, 200, 400, 400])],
        1000, 1000,
    )
    assert regions[0].part_of == -1
    assert regions[0].part_reason == ""


def test_a_valid_grouping_survives():
    regions = _parse_regions(
        [entry([0, 0, 100, 100], part_of=1, part_reason="same byline"),
         entry([200, 200, 400, 400])],
        1000, 1000,
    )
    assert regions[0].part_of == 1
    assert regions[0].part_reason == "same byline"


# -------------------------------------------------------------- the image --


def test_the_model_is_shown_a_downscaled_copy(tmp_path):
    """A 12MP master costs several times the image tokens for detail the model
    cannot use, so it is resized - but reported dimensions stay full-size, or
    every box would land in the wrong place."""
    path = tmp_path / "big.jpg"
    make_table_shot(path, n_items=4, seed=3)
    image, full_w, full_h = prepare_image(path)

    assert max(image.shape[:2]) == VISION_LONG_EDGE
    assert (full_w, full_h) == (3024, 4032)


# ------------------------------------------------------------ end to end --


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    path = tmp_path_factory.mktemp("vision") / "scene.jpg"
    truth = make_table_shot(path, n_items=6, seed=7)
    return path, truth


def to_grid(quad, width, height):
    x0, y0, x1, y1 = geometry.quad_bbox(quad)
    return [x0 * GRID / width, y0 * GRID / height, x1 * GRID / width, y1 * GRID / height]


def test_refining_tightens_the_model_s_boxes(scene):
    """The whole reason refine_regions exists.

    The model is asked for boxes it can only estimate, so they come back loose
    and offset. Refining takes the boundary from the pixels instead, using the
    model's box only to say which piece of paper is meant. Loosening every
    ground-truth box by 4% of the frame simulates that error.
    """
    path, truth = scene
    bgr = cv2.imread(str(path))
    height, width = bgr.shape[:2]
    slop = 0.04 * width

    loose = [
        Region(box=(
            max(0.0, geometry.quad_bbox(t.quad)[0] - slop),
            max(0.0, geometry.quad_bbox(t.quad)[1] - slop),
            min(width - 1.0, geometry.quad_bbox(t.quad)[2] + slop),
            min(height - 1.0, geometry.quad_bbox(t.quad)[3] + slop),
        ))
        for t in truth
    ]

    def median_iou(regions):
        return float(np.median([
            max(geometry.bbox_iou(t.quad, r.quad()) for r in regions) for t in truth
        ]))

    before = median_iou(loose)
    after = median_iou(refine_regions(bgr, loose))

    assert after > before
    assert after >= 0.90, f"refined to only {after:.3f}"


def test_every_region_is_marked_refined_or_left_alone(scene):
    path, truth = scene
    bgr = cv2.imread(str(path))
    height, width = bgr.shape[:2]
    regions = [Region(box=tuple(geometry.quad_bbox(t.quad))) for t in truth]

    for region in refine_regions(bgr, regions):
        assert region.refined is True


def test_a_box_over_bare_table_is_left_as_the_model_gave_it(scene):
    """Refining must not invent a boundary where there is no paper. The box
    stays put and stays unrefined, so review can see it was never confirmed."""
    path, _ = scene
    bgr = cv2.imread(str(path))
    empty = Region(box=(5.0, 5.0, 90.0, 90.0))

    refined = refine_regions(bgr, [empty])[0]
    assert refined.refined is False
    assert refined.box == (5.0, 5.0, 90.0, 90.0)


def test_identification_comes_from_the_model_and_geometry_from_the_pixels(scene):
    """The contract of segment_with_vision, in one test."""
    path, truth = scene
    _, width, height = prepare_image(path)

    provider = vision_provider([
        entry(to_grid(t.quad, width, height), headline=f"Item {i}")
        for i, t in enumerate(truth)
    ])
    result = segment_with_vision(path, provider)

    assert result.ok
    assert len(result.regions) == len(truth)
    assert [r.headline for r in result.regions] == [f"Item {i}" for i in range(len(truth))]

    ious = [max(geometry.bbox_iou(t.quad, r.quad()) for r in result.regions) for t in truth]
    assert min(ious) >= 0.80, f"IoUs: {[round(i, 3) for i in ious]}"


def test_refine_can_be_switched_off(scene):
    path, truth = scene
    _, width, height = prepare_image(path)
    boxes = [to_grid(t.quad, width, height) for t in truth]

    result = segment_with_vision(path, vision_provider([entry(b) for b in boxes]), refine=False)
    assert not any(r.refined for r in result.regions)


def test_a_provider_failure_is_reported_not_raised(scene):
    path, _ = scene
    provider = FakeProvider(responder=lambda job: {"nonsense": True})

    result = propose_regions(path, provider)
    assert not result.ok
    assert result.regions == []


def test_the_temporary_copy_is_always_cleaned_up(scene):
    """The resized copy is written next to the source, because the claude_cli
    provider cannot read outside the project. That makes leaving one behind an
    ingest problem, not just clutter - it would be picked up as a new photo."""
    path, _ = scene

    def explode(job):
        raise RuntimeError("provider died")

    with pytest.raises(RuntimeError):
        propose_regions(path, FakeProvider(responder=explode))

    assert list(path.parent.glob(".rotary-vision-*")) == []


# ---------------------------------------------- the three relationship kinds --

# These three are deliberately separate, and the tests below pin the
# difference. Using the wrong one damages the record: part_of publishes two
# objects as one entry under one title, and duplicate_of hides one from the
# site entirely.


def two(**second):
    return _parse_regions(
        [entry([0, 0, 100, 100]), entry([200, 200, 400, 400], **second)],
        1000, 1000,
    )


def test_related_to_survives_and_does_not_merge():
    regions = two(related_to=[0], related_reason="the charity's own leaflet")
    assert regions[1].related_to == [0]
    assert regions[1].related_reason == "the charity's own leaflet"
    # It must not have quietly become a merge.
    assert regions[1].part_of == -1


def test_duplicate_of_survives():
    assert two(duplicate_of=0)[1].duplicate_of == 0


@pytest.mark.parametrize("field,bad", [
    ("part_of", 9), ("duplicate_of", 9), ("part_of", 1), ("duplicate_of", 1),
])
def test_a_link_pointing_nowhere_is_dropped(field, bad):
    """Out of range, or an item pointing at itself. A bad link is worse than
    no link - one buries an item inside another, the other hides it."""
    regions = two(**{field: bad})
    assert getattr(regions[1], field) == -1


def test_related_indexes_are_cleaned_and_deduplicated():
    regions = two(related_to=[0, 0, 1, 7, -3], related_reason="same night")
    assert regions[1].related_to == [0]


def test_related_reason_is_dropped_when_no_link_survives():
    regions = two(related_to=[42], related_reason="points at nothing")
    assert regions[1].related_to == []
    assert regions[1].related_reason == ""


def test_a_merge_wins_over_a_duplicate_claim():
    """An item cannot both continue another and be a copy of one. The merge is
    the stronger claim, and letting both stand would hide the item twice."""
    regions = two(part_of=0, part_reason="continues", duplicate_of=0)
    assert regions[1].part_of == 0
    assert regions[1].duplicate_of == -1


# ------------------------------------------------- the descriptive fields ----


def test_medium_is_captured_and_unknown_values_fall_back():
    """The signal that separates a glossy brochure from the newsprint it lies
    on: same subject, different object, different publisher."""
    regions = _parse_regions(
        [entry([0, 0, 100, 100], medium="glossy_print"),
         entry([200, 200, 300, 300], medium="papyrus")],
        1000, 1000,
    )
    assert regions[0].medium == "glossy_print"
    assert regions[1].medium == "other"
    assert "glossy_print" in MEDIA


def test_panels_records_a_folded_item_without_splitting_it():
    """A programme opened flat is one object showing two panels. The crease is
    a strong straight edge and must not become a crop boundary."""
    region = _parse_regions([entry([0, 0, 500, 300], panels=2)], 1000, 1000)[0]
    assert region.panels == 2


@pytest.mark.parametrize("value", [0, -4, None, "two"])
def test_panels_is_never_less_than_one(value):
    assert _parse_regions([entry([0, 0, 100, 100], panels=value)], 1000, 1000)[0].panels == 1


def test_clipped_by_frame_and_date_hint_are_carried():
    region = _parse_regions(
        [entry([0, 0, 100, 100], clipped_by_frame=True, date_hint="Nov. 29, 1982")],
        1000, 1000,
    )[0]
    assert region.clipped_by_frame is True
    assert region.date_hint == "Nov. 29, 1982"


def test_link_confidence_is_separate_from_box_confidence():
    """An uncaptioned photograph under an article is probably that article's
    photograph, but position is weak evidence. The boundary can be certain
    while the attribution is not."""
    region = _parse_regions(
        [entry([0, 0, 100, 100]),
         entry([0, 200, 100, 300], part_of=0, part_reason="sits below it",
               confidence=0.95, link_confidence=0.3)],
        1000, 1000,
    )[1]
    assert region.confidence == 0.95
    assert region.link_confidence == 0.3


# ------------------------------------------ boxes the model placed badly ----

# The model sometimes answers past the edge of the 0-1000 grid. Clamping the
# result *after* converting to pixels pulled the bottom back to the last row
# while leaving the top beyond it, so the box came out inverted - and because a
# degenerate box is also one refine_regions refuses to touch, it passed
# straight through to the crop. Real photographs produced negative-height quads
# and nineteen-pixel slivers this way.


@pytest.mark.parametrize("box,why", [
    ([100, 1200, 500, 1210], "entirely past the bottom of the grid"),
    ([100, 1100, 400, 1500], "starts and ends past the grid"),
    ([1200, 100, 1300, 400], "entirely past the right of the grid"),
    ([0, 500, 1000, 503], "a sliver too thin to be paper"),
    ([400, 400, 402, 900], "a sliver too narrow to be paper"),
])
def test_a_box_that_cannot_be_placed_is_rejected(box, why):
    assert _to_pixels(box, 3024, 4032) is None, why


@pytest.mark.parametrize("box", [
    [200, 1400, 600, 900],     # inverted and out of range
    [-300, -200, 400, 400],    # negative coordinates
    [900, 900, 1400, 1400],    # runs off the bottom-right corner
])
def test_a_salvageable_box_stays_the_right_way_up(box):
    """Partly off the image is ordinary - an item at the edge of the frame.
    It gets trimmed, and must never come back inverted."""
    left, top, right, bottom = _to_pixels(box, 3024, 4032)
    assert right > left and bottom > top
    assert 0 <= left and 0 <= top
    assert right <= 3023 and bottom <= 4031


def test_dropped_regions_are_counted_not_swallowed():
    """A dropped region is a real piece of paper nobody has catalogued. The
    count has to reach the review page or the photo looks complete."""
    result = _parse_regions
    dropped: list[int] = []
    regions = result(
        [entry([0, 0, 200, 200]), entry([100, 1200, 500, 1210]), entry([300, 300, 500, 500])],
        1000, 1000, dropped,
    )
    assert len(regions) == 2
    assert len(dropped) == 1
