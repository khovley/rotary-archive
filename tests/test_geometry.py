"""Geometry helpers.

Corner ordering is the load-bearing one: every downstream stage assumes
TL, TR, BR, BL, and a silent mis-order produces a mirrored or rotated crop
rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotary_archive import geometry

SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]


def test_order_corners_is_order_independent():
    """Any permutation of the same four points yields the same ordering."""
    expected = geometry.order_corners(SQUARE)
    rng = np.random.default_rng(0)
    for _ in range(12):
        shuffled = list(SQUARE)
        rng.shuffle(shuffled)
        assert np.allclose(geometry.order_corners(shuffled), expected)


def test_order_corners_assigns_tl_tr_br_bl():
    ordered = geometry.order_corners([[10, 10], [0, 10], [0, 0], [10, 0]])
    assert np.allclose(ordered[0], [0, 0])    # top-left
    assert np.allclose(ordered[1], [10, 0])   # top-right
    assert np.allclose(ordered[2], [10, 10])  # bottom-right
    assert np.allclose(ordered[3], [0, 10])   # bottom-left


def test_order_corners_survives_rotation():
    """A rotated square must still order consistently - detection hands us
    quads at arbitrary angles."""
    theta = np.radians(20)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = (np.array(SQUARE, dtype=float) - 5) @ rot.T + 5

    ordered = geometry.order_corners(rotated)
    assert ordered[0][1] < ordered[2][1]   # TL above BR
    assert ordered[0][0] < ordered[1][0]   # TL left of TR


def test_quad_area_shoelace():
    assert geometry.quad_area(SQUARE) == pytest.approx(100.0)
    assert geometry.quad_area([[0, 0], [4, 0], [4, 3], [0, 3]]) == pytest.approx(12.0)


def test_bbox_iou_identical_and_disjoint():
    assert geometry.bbox_iou(SQUARE, SQUARE) == pytest.approx(1.0)
    far = [[100, 100], [110, 100], [110, 110], [100, 110]]
    assert geometry.bbox_iou(SQUARE, far) == pytest.approx(0.0)


def test_bbox_iou_half_overlap():
    other = [[5, 0], [15, 0], [15, 10], [5, 10]]
    # intersection 50, union 150
    assert geometry.bbox_iou(SQUARE, other) == pytest.approx(50 / 150)


def test_bbox_containment_is_asymmetric():
    outer = [[0, 0], [100, 0], [100, 100], [0, 100]]
    inner = [[10, 10], [20, 10], [20, 20], [10, 20]]
    assert geometry.bbox_containment(inner, outer) == pytest.approx(1.0)
    assert geometry.bbox_containment(outer, inner) == pytest.approx(0.01)


def test_scale_quad_round_trips():
    scaled = geometry.scale_quad(SQUARE, 3.0)
    assert np.allclose(geometry.scale_quad(scaled, 1 / 3.0), SQUARE)


def test_clip_quad_keeps_corners_in_bounds():
    clipped = geometry.clip_quad([[-5, -5], [200, -5], [200, 200], [-5, 200]], 100, 50)
    assert clipped[:, 0].min() >= 0 and clipped[:, 0].max() <= 99
    assert clipped[:, 1].min() >= 0 and clipped[:, 1].max() <= 49


def test_expand_quad_grows_area_without_moving_centre():
    expanded = geometry.expand_quad(SQUARE, 2.0)
    assert geometry.quad_area(expanded) > geometry.quad_area(SQUARE)
    assert np.allclose(
        np.asarray(expanded).mean(axis=0), np.asarray(SQUARE, float).mean(axis=0),
        atol=1e-4,
    )


def test_quad_edge_lengths_and_aspect():
    wide = [[0, 0], [40, 0], [40, 10], [0, 10]]
    w, h = geometry.quad_edge_lengths(wide)
    assert w == pytest.approx(40.0)
    assert h == pytest.approx(10.0)
    assert geometry.aspect_ratio(wide) == pytest.approx(4.0)
    # Aspect ratio is orientation independent.
    tall = [[0, 0], [10, 0], [10, 40], [0, 40]]
    assert geometry.aspect_ratio(tall) == pytest.approx(4.0)


def test_degenerate_quad_has_infinite_aspect():
    assert geometry.aspect_ratio([[0, 0], [10, 0], [10, 0], [0, 0]]) == float("inf")
