"""Segmentation: find each individual item in a table shot.

Two independent detection passes run over a downscaled copy of the photo and
their results are merged:

  * edge pass      - bilateral filter then Canny. Strong on crisp borders;
                     the bilateral filter is what stops newsprint texture from
                     generating thousands of spurious contours.
  * threshold pass - adaptive threshold. Catches low-contrast items (a faded
                     sepia photo on a cream tablecloth) the edge pass misses.

Neither pass is reliable alone, and their failure modes are different enough
that the union plus IoU de-duplication beats either. Everything is then
filtered on area, aspect ratio, and solidity to throw out shadows, table
edges, and stray hands.

Detection runs at ~2000px for speed; quads are scaled back to full resolution
before anything is written to the database, so cropping is always done from
the master pixels.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from . import db, geometry
from .geometry import Quad
from .ingest import load_oriented


@dataclass
class Candidate:
    quad: Quad                    # full-resolution coordinates
    confidence: float
    method: str
    area_frac: float

    def as_list(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in self.quad]


@dataclass
class SegmentResult:
    photo_sha256: str
    candidates: list[Candidate] = field(default_factory=list)
    note: str | None = None
    needs_review: bool = False


# ------------------------------------------------------------ preparation ---


def load_for_detection(
    path: Path, work_long_edge: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (full_bgr, work_bgr, scale).

    `scale` multiplies work-image coordinates back up to full resolution.
    Reading goes through Pillow so HEIC and EXIF orientation are handled in one
    place; OpenCV then works on upright RGB->BGR pixels.
    """
    with load_oriented(path) as pil_img:
        rgb = np.asarray(pil_img.convert("RGB"))
    full = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    h, w = full.shape[:2]
    long_edge = max(h, w)
    if long_edge <= work_long_edge:
        return full, full, 1.0

    factor = work_long_edge / long_edge
    work = cv2.resize(
        full, (max(1, int(w * factor)), max(1, int(h * factor))),
        interpolation=cv2.INTER_AREA,
    )
    return full, work, 1.0 / factor


# ------------------------------------------------------------------ passes ---


def _edge_mask(gray: np.ndarray) -> np.ndarray:
    """Canny edges, closed into continuous item borders."""
    smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    # Otsu's threshold gives a per-image Canny range, so exposure varies
    # between shots without needing hand-tuned constants.
    otsu, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(smooth, max(10.0, otsu * 0.5), otsu)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.dilate(closed, kernel, iterations=1)


def _background_mask(bgr: np.ndarray) -> np.ndarray:
    """Everything that isn't the table.

    Estimates the background colour from the outer border ring of the frame -
    which, if the item is laid out per the shooting guide, is pure background -
    then marks every pixel far from it in LAB space.

    This is the pass that gets the *outer* edge of a light item on a light
    background right. The edge and threshold passes latch onto the printed
    content and crop inside the paper; measuring distance-from-background
    instead finds the paper itself. It makes no assumption about whether the
    background is darker or lighter than the items, so one code path covers
    black posterboard and a white tablecloth alike.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    band = max(4, int(min(h, w) * 0.03))

    ring = np.concatenate(
        [
            lab[:band].reshape(-1, 3),
            lab[-band:].reshape(-1, 3),
            lab[:, :band].reshape(-1, 3),
            lab[:, -band:].reshape(-1, 3),
        ]
    )
    background = np.median(ring, axis=0)

    # Weight L down: a shadow changes lightness a lot but chroma little, and we
    # do not want shadows counted as foreground.
    delta = lab - background
    distance = np.sqrt(
        0.5 * delta[..., 0] ** 2 + delta[..., 1] ** 2 + delta[..., 2] ** 2
    )
    distance = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    distance = cv2.GaussianBlur(distance, (7, 7), 0)

    _, mask = cv2.threshold(distance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Opening removes speckle. Closing is kept deliberately small: it bridges
    # gaps, and the gaps that matter here are the ones *between* items. A
    # 13x13 kernel over three iterations reaches roughly 40px, which is wider
    # than the space people leave between clippings - it welded eight of them
    # into one 50%-of-frame blob that no shape test could accept.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Interior holes are filled by redrawing each outline solid, which is the
    # right tool for the job: it closes a hole of any size without reaching
    # across the gap to a neighbour, as a larger kernel would.
    filled = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _threshold_mask(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold, cleaned into solid item blobs."""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    block = max(31, (min(gray.shape[:2]) // 16) | 1)  # odd, scales with image
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, 7,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)


def _contour_to_quad(contour: np.ndarray) -> tuple[Quad, str]:
    """Best 4-corner fit for a contour.

    A true quadrilateral from approxPolyDP preserves perspective (shooting at
    an angle), so it is preferred. minAreaRect is the fallback and only models
    rotation, which still beats an axis-aligned box.
    """
    perimeter = cv2.arcLength(contour, True)
    for epsilon_frac in (0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(contour, epsilon_frac * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return geometry.order_corners(approx.reshape(4, 2)), "poly"
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    return geometry.order_corners(box), "rect"


def _candidates_from_mask(
    mask: np.ndarray,
    method: str,
    frame_area: float,
    cfg: dict[str, Any],
) -> list[Candidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = frame_area * float(cfg.get("min_area_frac", 0.015))
    max_area = frame_area * float(cfg.get("max_area_frac", 0.90))
    max_aspect = float(cfg.get("max_aspect_ratio", 8.0))
    min_solidity = float(cfg.get("min_solidity", 0.80))

    out: list[Candidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < min_solidity:
            continue

        quad, fit = _contour_to_quad(contour)
        if geometry.aspect_ratio(quad) > max_aspect:
            continue

        # How much of the fitted quad the contour actually fills. A clean
        # rectangular item scores near 1.0; a blob that merely happens to have
        # four extreme points scores low, and is likely two touching items or
        # a shadow.
        qarea = geometry.quad_area(quad)
        fill = float(area / qarea) if qarea > 0 else 0.0
        if fill < float(cfg.get("min_fill", 0.42)):
            continue

        # Confidence blends shape quality with fit type. It drives which items
        # the review UI surfaces first, so it needs to be monotonic in
        # "how sure are we this is one clean rectangular object".
        confidence = 0.55 * min(1.0, fill) + 0.30 * min(1.0, solidity)
        confidence += 0.15 if fit == "poly" else 0.05
        out.append(
            Candidate(
                quad=quad,
                confidence=round(min(1.0, confidence), 4),
                method=method,
                area_frac=float(area / frame_area),
            )
        )
    return out


def _estimate_background_lab(bgr: np.ndarray) -> np.ndarray:
    """Median LAB colour of the frame's outer border ring."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    band = max(4, int(min(h, w) * 0.03))
    ring = np.concatenate(
        [
            lab[:band].reshape(-1, 3),
            lab[-band:].reshape(-1, 3),
            lab[:, :band].reshape(-1, 3),
            lab[:, -band:].reshape(-1, 3),
        ]
    )
    return np.median(ring, axis=0).astype(np.float32)


def _boundary_score(
    bgr: np.ndarray, quad: Quad, background_lab: np.ndarray
) -> float:
    """How confident we are that this quad is the item's true outer edge.

    Asks one question directly: do the pixels just *outside* the quad look like
    the table? If they still look like the item, we cropped inside it and are
    about to lose part of the object.

    This replaces the earlier pass-agreement heuristic, which was actively
    misleading - when one pass finds the paper edge and another finds the
    printed content, they disagree precisely because the wider one is correct.
    Agreement measures consensus, not accuracy, and the two come apart exactly
    in the low-contrast case where the score matters most.
    """
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    diag = float(np.linalg.norm(pts[2] - pts[0]))
    band = max(3.0, diag * 0.02)

    inner = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(inner, pts.astype(np.int32), 255)
    outer = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(
        outer, geometry.expand_quad(pts, band * 2.0).astype(np.int32), 255
    )
    ring = cv2.subtract(outer, cv2.dilate(inner, np.ones((3, 3), np.uint8)))

    ring_pixels = lab[ring > 0]
    inner_pixels = lab[cv2.erode(inner, np.ones((5, 5), np.uint8)) > 0]
    if ring_pixels.size == 0 or inner_pixels.size == 0:
        return 0.5

    def distance(pixels: np.ndarray) -> float:
        delta = pixels - background_lab
        return float(
            np.median(
                np.sqrt(
                    0.5 * delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2
                )
            )
        )

    ring_distance = distance(ring_pixels)
    item_distance = distance(inner_pixels)

    if item_distance < 1e-6:
        # Item is indistinguishable from the table; nothing to be confident in.
        return 0.35

    # 0 when the ring looks exactly like the table (clean cut), 1 when it looks
    # exactly like the item (we cropped inside it).
    leak = float(np.clip(ring_distance / item_distance, 0.0, 1.0))
    return float(np.clip(1.0 - leak, 0.0, 1.0))


def _split_touching(
    mask: np.ndarray, frame_area: float, cfg: dict[str, Any]
) -> list[Candidate]:
    """Rescue pass: pull apart items that merged into one blob.

    When items are laid out touching or overlapping - which is how people
    actually arrange a table, whatever the shooting guide says - every pass
    finds one ragged mass instead of several rectangles, and the shape filters
    reject it. The result is no candidates at all and a useless whole-frame
    crop.

    A distance transform peaks at the centre of each item and dips where two
    meet, so thresholding it yields one seed per item; watershed then grows
    those seeds back over the mask. The splits are approximate, and where one
    item genuinely covers another the hidden part is simply gone - no algorithm
    recovers it. But approximate boxes a human can drag are far more use than
    nothing, so everything this produces is flagged for review.

    Runs only when the normal passes came up empty. It must never disturb the
    well-behaved case.
    """
    filled = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    distance = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    if distance.max() <= 0:
        return []

    # 0.35 of the peak is deliberately low: a high threshold splits one item
    # into several, which costs a human more time than a missed split.
    _, seeds = cv2.threshold(distance, 0.35 * distance.max(), 255, cv2.THRESH_BINARY)
    seeds = seeds.astype(np.uint8)

    count, markers = cv2.connectedComponents(seeds)
    if count <= 2:
        return []   # one blob and background: nothing to split

    markers = markers + 1
    unknown = cv2.subtract(filled, seeds)
    markers[unknown > 0] = 0
    cv2.watershed(cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR), markers)

    min_area = frame_area * float(cfg.get("min_area_frac", 0.015))
    max_area = frame_area * float(cfg.get("max_area_frac", 0.90))
    max_aspect = float(cfg.get("max_aspect_ratio", 8.0))

    out: list[Candidate] = []
    for label in range(2, count + 1):
        region = np.uint8(markers == label) * 255
        pieces, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not pieces:
            continue
        contour = max(pieces, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        quad, _ = _contour_to_quad(contour)
        if geometry.aspect_ratio(quad) > max_aspect:
            continue

        out.append(
            Candidate(
                quad=quad,
                # Low by construction: these are guesses that need a human.
                confidence=0.4,
                method="split",
                area_frac=float(area / frame_area),
            )
        )
    return out


def _drop_swallowers(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Remove detections that have engulfed two or more separate items.

    Adjacent items joined by a drop shadow, or a pair sitting close together,
    can be closed into a single blob by the morphology in any of the passes.
    Because the merge step prefers the widest quad, such a blob would win its
    group and silently cost us an item - the one failure mode that is not
    recoverable in review, since the lost item never appears at all.

    The test is direct: if a candidate contains two smaller candidates that do
    not overlap each other, it is describing two things, not one.
    """
    keep: list[Candidate] = []
    for outer in candidates:
        contained = [
            inner
            for inner in candidates
            if inner is not outer
            and inner.area_frac < outer.area_frac * 0.65
            and geometry.bbox_containment(inner.quad, outer.quad) > 0.85
        ]
        # Only count contained items that are distinct from one another.
        distinct: list[Candidate] = []
        for cand in sorted(contained, key=lambda c: c.area_frac, reverse=True):
            if all(geometry.bbox_iou(cand.quad, d.quad) < 0.20 for d in distinct):
                distinct.append(cand)

        if len(distinct) < 2:
            keep.append(outer)
    return keep


def _merge_candidates(
    candidates: Sequence[Candidate], iou_threshold: float
) -> list[Candidate]:
    """Greedy non-maximum suppression by bounding-box IoU.

    Both passes usually find the same item; whichever found it more
    convincingly wins, and the survivor is marked 'merged' so the review UI
    can show that two independent methods agreed.
    """
    survivors = _drop_swallowers(candidates)

    # Group overlapping detections, then decide per group. Seeding the groups
    # with the largest candidates first means a group forms around the widest
    # interpretation of an item rather than around a content-only crop.
    ordered = sorted(survivors, key=lambda c: c.area_frac, reverse=True)

    groups: list[list[Candidate]] = []
    for cand in ordered:
        target = next(
            (
                g
                for g in groups
                if any(geometry.bbox_iou(cand.quad, m.quad) > iou_threshold for m in g)
            ),
            None,
        )
        if target is None:
            groups.append([cand])
        else:
            target.append(cand)

    kept: list[Candidate] = []
    for group in groups:
        # Prefer the largest quad in the group. The passes fail in opposite
        # directions - edge and threshold crop inside the paper to the printed
        # content, background can over-grow into a shadow - but cropping inside
        # destroys content while cropping wide only adds a margin a human can
        # trim. Widest-wins is the recoverable error.
        winner = max(group, key=lambda c: c.area_frac)
        if len({c.method for c in group}) > 1:
            winner.method = "merged"
        kept.append(winner)

    # Reading order: top to bottom, then left to right. Bucket rows by centroid
    # rather than topmost corner - rotated neighbours have very different top
    # edges but near-identical centres, and bucketing on the corner splits a
    # visual row in two.
    if kept:
        centroids = [c.quad.mean(axis=0) for c in kept]
        heights = [geometry.quad_edge_lengths(k.quad)[1] for k in kept]
        row_tol = max(1.0, float(np.median(heights)) * 0.75)
        order = sorted(
            range(len(kept)),
            key=lambda i: (round(centroids[i][1] / row_tol), centroids[i][0]),
        )
        kept = [kept[i] for i in order]
    return kept


def _whole_frame_candidate(width: int, height: int, reason: str) -> Candidate:
    quad = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    return Candidate(quad=quad, confidence=0.5, method=reason, area_frac=1.0)


# ---------------------------------------------------------------- pipeline ---


def segment_image(path: Path, cfg: dict[str, Any]) -> SegmentResult:
    """Detect every item in one source photo. Pure function - no database."""
    work_long_edge = int(cfg.get("work_long_edge", 2000))
    full, work, scale = load_for_detection(path, work_long_edge)
    full_h, full_w = full.shape[:2]

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    frame_area = float(gray.shape[0] * gray.shape[1])

    raw = [
        *_candidates_from_mask(_background_mask(work), "background", frame_area, cfg),
        *_candidates_from_mask(_edge_mask(gray), "edge", frame_area, cfg),
        *_candidates_from_mask(_threshold_mask(gray), "threshold", frame_area, cfg),
    ]
    merged = _merge_candidates(raw, float(cfg.get("iou_merge_threshold", 0.35)))

    # Nothing survived the shape filters. Before giving up on the photo and
    # cropping the whole frame, try splitting the foreground: items laid out
    # touching each other merge into one ragged mass that no rectangle test
    # will accept.
    split_rescue = False
    if not merged:
        background = _background_mask(work)
        coverage = float(background.mean()) / 255.0
        if 0.05 < coverage < 0.95:
            rescued = _split_touching(background, frame_area, cfg)
            if len(rescued) > 1:
                merged = _merge_candidates(
                    rescued, float(cfg.get("iou_merge_threshold", 0.35))
                )
                split_rescue = True

    # Final confidence is dominated by whether the quad sits on the item's real
    # outer edge - the shape metrics from the detection pass only describe how
    # tidy the contour was, which says nothing about whether it is the right
    # contour.
    background_lab = _estimate_background_lab(work)
    for cand in merged:
        support = _boundary_score(work, cand.quad, background_lab)
        cand.confidence = round(0.70 * support + 0.30 * cand.confidence, 4)

    result = SegmentResult(photo_sha256="")

    # Scale to full resolution, nudge outward a touch, and clip back in bounds.
    margin = max(2.0, 0.002 * max(full_w, full_h))
    for cand in merged:
        quad = geometry.scale_quad(cand.quad, scale)
        quad = geometry.expand_quad(quad, margin)
        cand.quad = geometry.clip_quad(quad, full_w, full_h)

    single_frac = float(cfg.get("single_item_frac", 0.85))

    if split_rescue:
        # Every box here is a guess from a merged blob, so all of them go in
        # front of a human rather than into the archive unseen.
        for cand in merged:
            cand.confidence = min(cand.confidence, 0.45)
        result.candidates = merged
        result.note = (
            f"items were touching, so {len(merged)} rough crops were split out "
            "- check and adjust each one"
        )
        result.needs_review = True
        return result

    if not merged:
        # Nothing found. Most likely a single item shot close up, filling the
        # frame - a background-less photo has no edges to find. Treat the whole
        # frame as one item and flag it so a human confirms.
        result.candidates = [_whole_frame_candidate(full_w, full_h, "whole_frame")]
        result.note = "no items detected; using whole frame"
        result.needs_review = True
        return result

    if len(merged) == 1 and merged[0].area_frac >= single_frac:
        merged[0].method = "whole_frame"
        merged[0].confidence = max(merged[0].confidence, 0.75)
        result.note = "single item filling the frame"

    result.candidates = merged

    # Low-confidence individual items get flagged for review below; a photo
    # where everything is uncertain gets a note so it sorts to the top.
    if merged and max(c.confidence for c in merged) < 0.6:
        result.note = (result.note or "") + " low-confidence detection"
        result.needs_review = True

    return result


def segment_photo_row(
    conn: sqlite3.Connection,
    paths: Any,
    photo: sqlite3.Row,
    cfg: dict[str, Any],
    *,
    flag_below: float = 0.70,
) -> SegmentResult:
    """Segment one photo row and persist the resulting items."""
    source = paths.absolute(photo["stored_path"])
    result = segment_image(source, cfg)
    result.photo_sha256 = photo["sha256"]

    with db.transaction(conn):
        db.delete_items_for_photo(conn, photo["sha256"])
        for seq, cand in enumerate(result.candidates):
            flagged = cand.confidence < flag_below or result.needs_review
            reason = None
            if flagged:
                reason = (
                    result.note
                    or f"detection confidence {cand.confidence:.2f} below {flag_below:.2f}"
                )
            db.insert_item(
                conn,
                item_id=db.make_item_id(photo["sha256"], seq),
                photo_sha256=photo["sha256"],
                seq=seq,
                quad=cand.as_list(),
                detection_confidence=cand.confidence,
                detection_method=cand.method,
                needs_human_review=flagged,
                review_reason=reason,
            )
        db.set_photo_status(conn, photo["sha256"], "segmented", result.note)

    return result


def segment_pending(
    conn: sqlite3.Connection,
    paths: Any,
    cfg: dict[str, Any],
    *,
    flag_below: float = 0.70,
    force: bool = False,
    progress: Any = None,
) -> list[SegmentResult]:
    """Segment every photo that hasn't been segmented yet (or all, with force)."""
    photos = (
        db.all_photos(conn) if force else db.photos_with_status(conn, "ingested")
    )
    results = []
    for photo in photos:
        if progress is not None:
            progress(photo)
        results.append(
            segment_photo_row(conn, paths, photo, cfg, flag_below=flag_below)
        )
    return results
