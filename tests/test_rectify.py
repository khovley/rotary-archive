"""Rectification: perspective warp, fine deskew, and derivative generation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from rotary_archive.rectify import (
    apply_orientation,
    estimate_skew,
    rotate_image,
    warp_quad,
    write_derivatives,
)


def ruled_page(width=600, height=400, angle=0.0):
    """A white page with horizontal rules, optionally rotated.

    Ruled lines stand in for text baselines, which is what the deskew
    estimator actually locks onto.
    """
    img = np.full((height, width, 3), 245, np.uint8)
    for y in range(40, height - 20, 28):
        cv2.rectangle(img, (30, y), (width - 30, y + 6), (30, 30, 30), -1)
    return rotate_image(img, angle) if angle else img


# ------------------------------------------------------------------ warp --


def test_warp_quad_produces_expected_dimensions():
    source = np.zeros((500, 500, 3), np.uint8)
    quad = [[100, 100], [400, 100], [400, 300], [100, 300]]
    out = warp_quad(source, quad)
    assert out.shape[1] == pytest.approx(300, abs=1)
    assert out.shape[0] == pytest.approx(200, abs=1)


def test_warp_quad_recovers_a_rotated_rectangle():
    """The whole point of the warp: content laid at an angle comes back square."""
    page = ruled_page(400, 300)
    canvas = np.full((700, 700, 3), 20, np.uint8)

    angle = 15.0
    theta = np.radians(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src = np.array([[0, 0], [399, 0], [399, 299], [0, 299]], np.float32)
    dst = ((src - src.mean(axis=0)) @ rot.T + [350, 350]).astype(np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped_in = cv2.warpPerspective(page, matrix, (700, 700))
    canvas[warped_in.sum(axis=2) > 0] = warped_in[warped_in.sum(axis=2) > 0]

    recovered = warp_quad(canvas, dst)
    assert abs(estimate_skew(recovered)) < 1.0
    assert recovered.shape[1] > recovered.shape[0]      # landscape preserved


def test_warp_quad_accepts_unordered_corners():
    source = np.zeros((400, 400, 3), np.uint8)
    ordered = warp_quad(source, [[50, 50], [350, 50], [350, 250], [50, 250]])
    jumbled = warp_quad(source, [[350, 250], [50, 50], [50, 250], [350, 50]])
    assert ordered.shape == jumbled.shape


# ----------------------------------------------------------------- skew --


@pytest.mark.parametrize("angle", [-3.5, -2.4, -1.5, -0.8, -0.3, 0.3, 0.8, 1.5, 2.4, 3.5])
def test_estimate_skew_negates_the_applied_rotation(angle):
    """Accuracy and sign convention together.

    `rotate_image` takes an OpenCV angle, where positive is counter-clockwise.
    `estimate_skew` returns the correction to apply, which is the negative of
    the tilt that was introduced. That inversion is what makes
    `rotate_image(img, estimate_skew(img))` cancel the tilt rather than double
    it, so it is worth pinning explicitly.

    The 0.1 degree tolerance is deliberately tight: the projection-profile
    search resolves to 0.02 degrees. A Hough-based estimator could not hold
    this bound - it reported exactly 0.0 for genuinely tilted scans, which is
    why this one replaced it.
    """
    assert estimate_skew(ruled_page(angle=angle)) == pytest.approx(-angle, abs=0.1)


def test_estimate_skew_resolves_sub_degree_tilt():
    """The regression guard for the estimator swap.

    Post-warp corrections are fractions of a degree. An estimator that
    silently returns 0.0 here would leave every scan visibly crooked while
    reporting success.
    """
    measured = estimate_skew(ruled_page(angle=0.6))
    assert measured != 0.0
    assert measured == pytest.approx(-0.6, abs=0.1)


@pytest.mark.parametrize(
    "angle", [-3.5, -2.4, -2.0, -0.8, -0.3, 0.3, 0.8, 2.0, 2.4, 3.5]
)
def test_deskew_reduces_residual_to_a_quarter_degree(angle):
    """The end-to-end property that matters: after correction, content is
    straight enough to read as scanned rather than photographed.

    Both signs are exercised, because a sign error would still pass a
    single-direction test half the time.

    A quarter degree rather than a tenth because these fixtures are rotated
    into place and then rotated back, so they carry two rounds of interpolation
    softening - harder than a real item, which is resampled once. Measured
    residuals on the actual pipeline sit at a median of 0.06 degrees.
    """
    page = ruled_page(angle=angle)
    corrected = rotate_image(page, estimate_skew(page))
    assert abs(estimate_skew(corrected)) < 0.25


def test_estimate_skew_returns_zero_without_evidence():
    """A flat colour patch has no lines; rotating on noise costs a resample
    and gains nothing."""
    assert estimate_skew(np.full((300, 300, 3), 128, np.uint8)) == 0.0


def test_estimate_skew_ignores_tiny_images():
    assert estimate_skew(np.zeros((10, 10, 3), np.uint8)) == 0.0


def test_estimate_skew_never_exceeds_the_search_range():
    """The returned angle is bounded, so a pathological item cannot produce a
    wild rotation that throws content out of frame."""
    img = np.full((300, 300, 3), 240, np.uint8)
    cv2.line(img, (0, 0), (299, 299), (20, 20, 20), 4)
    assert abs(estimate_skew(img, max_angle=5.0)) <= 5.0


def test_correcting_twice_does_not_drift():
    """A second correction pass on already-straight content must be a no-op.

    Each rotation resamples the pixels, so a pipeline that kept nudging would
    soften the image for nothing. This is also why the deskew is a single
    measurement rather than an iterative loop: with an accurate estimator the
    loop only accumulated resampling drift.
    """
    page = ruled_page(angle=2.0)
    once = rotate_image(page, estimate_skew(page))
    residual = estimate_skew(once)
    assert abs(residual) < 0.25

    # Applying the residual again must not make things worse.
    twice = rotate_image(once, residual)
    assert abs(estimate_skew(twice)) < 0.25


# ----------------------------------------------------------- orientation --


def test_rotate_image_expands_canvas_so_nothing_is_cropped():
    rotated = rotate_image(np.zeros((100, 200, 3), np.uint8), 30.0)
    assert rotated.shape[0] > 100 and rotated.shape[1] > 200


def test_rotate_image_is_a_noop_for_zero():
    img = np.zeros((10, 20, 3), np.uint8)
    assert rotate_image(img, 0.0) is img


@pytest.mark.parametrize(
    "degrees,expected", [(0, (40, 80)), (90, (80, 40)), (180, (40, 80)), (270, (80, 40))]
)
def test_apply_orientation_quarter_turns(degrees, expected):
    out = apply_orientation(np.zeros((40, 80, 3), np.uint8), degrees)
    assert out.shape[:2] == expected


def test_apply_orientation_wraps_past_360():
    img = np.zeros((40, 80, 3), np.uint8)
    assert apply_orientation(img, 450).shape[:2] == apply_orientation(img, 90).shape[:2]


# ---------------------------------------------------------- derivatives --


def test_write_derivatives_writes_each_size(tmp_path):
    rows = write_derivatives(
        np.full((1200, 1600, 3), 200, np.uint8), "item-1", tmp_path,
        [1600, 800, 320], 80,
    )
    assert {r["long_edge"] for r in rows} == {1600, 800, 320}
    for row in rows:
        assert row["path"].exists()
        assert max(row["width"], row["height"]) == row["long_edge"]


def test_write_derivatives_preserves_aspect_ratio(tmp_path):
    rows = write_derivatives(
        np.full((500, 1000, 3), 200, np.uint8), "item-2", tmp_path, [500], 80
    )
    assert rows[0]["width"] / rows[0]["height"] == pytest.approx(2.0, abs=0.02)


def test_write_derivatives_does_not_upscale(tmp_path):
    """An upscaled derivative is bytes with no extra information in them."""
    rows = write_derivatives(
        np.full((200, 300, 3), 200, np.uint8), "item-3", tmp_path, [1600], 80
    )
    assert (rows[0]["width"], rows[0]["height"]) == (300, 200)
