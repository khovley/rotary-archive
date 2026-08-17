"""Quad geometry helpers shared by segmentation, rectification, and the UI.

A "quad" throughout this project is four (x, y) points in source-image pixel
coordinates, ordered top-left, top-right, bottom-right, bottom-left.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

Quad = np.ndarray  # shape (4, 2), float32


def order_corners(points: Sequence[Sequence[float]]) -> Quad:
    """Put four arbitrary corners into TL, TR, BR, BL order.

    Uses the standard sum/difference trick: the top-left corner has the
    smallest x+y, bottom-right the largest; top-right has the smallest y-x,
    bottom-left the largest. This is robust to any input ordering and to
    moderate rotation, which is exactly what contour detection hands us.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    total = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(total)]  # TL
    ordered[2] = pts[np.argmax(total)]  # BR

    diff = np.diff(pts, axis=1).ravel()  # y - x
    ordered[1] = pts[np.argmin(diff)]  # TR
    ordered[3] = pts[np.argmax(diff)]  # BL
    return ordered


def quad_area(quad: Sequence[Sequence[float]]) -> float:
    """Polygon area via the shoelace formula."""
    pts = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def quad_bbox(quad: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    pts = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def bbox_iou(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """Intersection-over-union of two quads' bounding boxes.

    Bounding-box IoU rather than true polygon IoU: items on a table are
    axis-alignable and near-rectangular, so the approximation is accurate
    enough for de-duplication and far cheaper.
    """
    ax0, ay0, ax1, ay1 = quad_bbox(a)
    bx0, by0, bx1, by1 = quad_bbox(b)

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def bbox_containment(
    inner: Sequence[Sequence[float]], outer: Sequence[Sequence[float]]
) -> float:
    """Fraction of `inner`'s bounding box that falls inside `outer`'s.

    Unlike IoU this is asymmetric, which is what lets us ask "did this large
    detection swallow that small one?" - a question IoU answers poorly when
    the two differ greatly in size.
    """
    ix0, iy0, ix1, iy1 = quad_bbox(inner)
    ox0, oy0, ox1, oy1 = quad_bbox(outer)

    cx0, cy0 = max(ix0, ox0), max(iy0, oy0)
    cx1, cy1 = min(ix1, ox1), min(iy1, oy1)
    overlap = max(0.0, cx1 - cx0) * max(0.0, cy1 - cy0)

    inner_area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    return float(overlap / inner_area) if inner_area > 0 else 0.0


def scale_quad(quad: Sequence[Sequence[float]], factor: float) -> Quad:
    """Map a quad between the detection-scale image and full resolution."""
    return (np.asarray(quad, dtype=np.float32).reshape(4, 2) * float(factor)).astype(
        np.float32
    )


def clip_quad(quad: Sequence[Sequence[float]], width: int, height: int) -> Quad:
    """Keep corners inside the image. Contour dilation can push them out by a
    pixel or two, which warpPerspective would happily sample as black."""
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, width - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, height - 1))
    return pts


def expand_quad(quad: Sequence[Sequence[float]], margin_px: float) -> Quad:
    """Push corners outward from the centroid by roughly `margin_px`.

    Detected edges tend to sit just inside the physical item, shaving off the
    outermost row of pixels. A small outward nudge keeps the full item.
    """
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    centre = pts.mean(axis=0)
    vectors = pts - centre
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return (pts + vectors / lengths * float(margin_px)).astype(np.float32)


def quad_edge_lengths(quad: Sequence[Sequence[float]]) -> tuple[float, float]:
    """(width, height) for a TL,TR,BR,BL quad, taking the longer of each
    opposing pair so perspective doesn't shrink the output."""
    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    tl, tr, br, bl = pts
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return float(width), float(height)


def aspect_ratio(quad: Sequence[Sequence[float]]) -> float:
    """Long side over short side, always >= 1."""
    w, h = quad_edge_lengths(quad)
    if w <= 0 or h <= 0:
        return float("inf")
    return max(w, h) / min(w, h)
