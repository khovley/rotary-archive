"""Rectify: turn a detected quad into a flat, upright archival image.

Two rotations happen here, and they do different jobs:

  * The perspective warp removes the gross geometry - the item was lying at an
    angle and photographed from slightly off-axis. This is driven entirely by
    the four detected corners.
  * The fine deskew removes what is left, typically under two degrees. Corner
    detection is only accurate to a pixel or two, and on a 3000px edge that is
    still enough residual tilt to make newsprint look photographed rather than
    scanned. This pass measures the actual text baselines and straightens to
    them.

Output is a full-resolution JPEG master plus WebP derivatives. Masters stay on
disk and are never published; derivatives are what the site ships.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from . import db, geometry
from .ingest import load_oriented


@dataclass
class RectifyResult:
    item_id: str
    master_path: Path
    width: int
    height: int
    fine_skew_deg: float
    derivatives: list[dict[str, Any]]


def _read_source_bgr(path: Path) -> np.ndarray:
    with load_oriented(path) as pil_img:
        rgb = np.asarray(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def warp_quad(image: np.ndarray, quad: Sequence[Sequence[float]]) -> np.ndarray:
    """Perspective-correct the region bounded by `quad` into a flat rectangle."""
    src = geometry.order_corners(quad).astype(np.float32)
    width, height = geometry.quad_edge_lengths(src)
    out_w, out_h = max(1, int(round(width))), max(1, int(round(height)))

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image, matrix, (out_w, out_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


DESKEW_WORK_EDGE = 900


def _ink_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask of the dark marks on an item, downscaled for the search."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    h, w = gray.shape[:2]
    if max(h, w) > DESKEW_WORK_EDGE:
        factor = DESKEW_WORK_EDGE / max(h, w)
        gray = cv2.resize(
            gray, (max(1, int(w * factor)), max(1, int(h * factor))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _profile_score(mask: np.ndarray, angle: float) -> float:
    """How sharply the mask's rows separate at this rotation.

    Rotating text into true horizontal makes every row either dense with ink
    or empty, so the row-sum profile becomes a comb of tall peaks. Variance of
    that profile therefore peaks at the correct angle. Squaring the row
    differences rewards sharp transitions and keeps the score sensitive at
    fractions of a degree, where a Hough accumulator's angular bins cannot
    tell one candidate from the next.
    """
    if abs(angle) > 1e-9:
        h, w = mask.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        # Constant zero border, not BORDER_REPLICATE: replicating edge pixels
        # would smear ink into the corners the rotation opens up and reward
        # larger angles for a reason that has nothing to do with alignment.
        rotated = cv2.warpAffine(
            mask, matrix, (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    else:
        rotated = mask

    profile = rotated.sum(axis=1, dtype=np.float64)
    if profile.size < 3:
        return 0.0
    return float(np.diff(profile).var())


def estimate_skew(image: np.ndarray, max_angle: float = 5.0) -> float:
    """Residual rotation in degrees; apply it with `rotate_image` to correct.

    Uses a projection profile with a coarse-to-fine search rather than a Hough
    transform. Hough resolves lines well but quantises angle by the
    accumulator's theta step, which is far too blunt here - the corrections
    that matter after perspective warping are fractions of a degree, and a
    Hough estimate would report exactly 0.0 for a visibly tilted scan.

    Returns 0.0 when there is not enough ink to measure. Doing nothing beats
    rotating on noise, since every rotation costs a resample.
    """
    if image.ndim == 3:
        if min(image.shape[:2]) < 40:
            return 0.0
    elif min(image.shape[:2]) < 40:
        return 0.0

    mask = _ink_mask(image)
    # Too little or too much ink means no usable line structure - a blank
    # patch or a solid black frame both score meaninglessly.
    coverage = float(mask.mean()) / 255.0
    if coverage < 0.01 or coverage > 0.95:
        return 0.0

    best = 0.0
    span, step = float(max_angle), 0.5
    for _ in range(3):  # 0.5deg -> 0.1deg -> 0.02deg
        lo, hi = best - span, best + span
        candidates = np.arange(lo, hi + step / 2, step)
        candidates = candidates[np.abs(candidates) <= max_angle + 1e-9]
        if candidates.size == 0:
            break
        scores = [_profile_score(mask, float(a)) for a in candidates]
        best = float(candidates[int(np.argmax(scores))])
        span, step = step, step / 5.0

    return 0.0 if abs(best) < 1e-3 else round(best, 3)


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate about the centre, expanding the canvas so no content is cut off."""
    if abs(angle_deg) < 1e-6:
        return image

    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)

    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))
    matrix[0, 2] += new_w / 2.0 - centre[0]
    matrix[1, 2] += new_h / 2.0 - centre[1]

    return cv2.warpAffine(
        image, matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def apply_orientation(image: np.ndarray, rotation: int) -> np.ndarray:
    """Apply a quarter-turn correction chosen by a human in review."""
    rotation %= 360
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def write_derivatives(
    image: np.ndarray,
    item_id: str,
    out_dir: Path,
    sizes: Sequence[int],
    quality: int,
) -> list[dict[str, Any]]:
    """Write WebP derivatives at each long-edge size.

    Sizes larger than the source are skipped rather than upscaled - an
    upscaled derivative is bytes with no information in them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = image.shape[:2]
    long_edge = max(h, w)

    written: list[dict[str, Any]] = []
    for size in sorted(set(int(s) for s in sizes), reverse=True):
        if size >= long_edge:
            scaled, out_w, out_h = image, w, h
        else:
            factor = size / long_edge
            out_w, out_h = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
            scaled = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)

        path = out_dir / f"{item_id}-{size}.webp"
        cv2.imwrite(str(path), scaled, [cv2.IMWRITE_WEBP_QUALITY, int(quality)])
        written.append(
            {
                "long_edge": size,
                "path": path,
                "width": out_w,
                "height": out_h,
                "bytes": path.stat().st_size if path.exists() else None,
            }
        )
    return written


def rectify_item(
    conn: sqlite3.Connection,
    paths: Any,
    item: sqlite3.Row,
    cfg: dict[str, Any],
    *,
    source_image: np.ndarray | None = None,
) -> RectifyResult:
    """Rectify one item and persist master + derivatives.

    `source_image` lets a caller decode a photo once and rectify all of its
    items from the same in-memory array, which matters: these are 12-megapixel
    HEICs and decoding is the dominant cost.
    """
    photo = db.get_photo(conn, item["photo_sha256"])
    if photo is None:
        raise LookupError(f"item {item['id']} references a missing photo")

    if source_image is None:
        source_image = _read_source_bgr(paths.absolute(photo["stored_path"]))

    quad = np.asarray(json.loads(item["quad"]), dtype=np.float32)
    warped = warp_quad(source_image, quad)

    max_skew = float(cfg.get("max_fine_skew_deg", 5.0))
    min_skew = float(cfg.get("min_fine_skew_deg", 0.15))
    skew = estimate_skew(warped, max_angle=max_skew)
    if abs(skew) >= min_skew:
        warped = rotate_image(warped, skew)
    else:
        skew = 0.0

    rotation = int(item["rotation_applied"] or 0)
    if rotation:
        warped = apply_orientation(warped, rotation)

    paths.items.mkdir(parents=True, exist_ok=True)
    master_path = paths.items / f"{item['id']}.jpg"
    cv2.imwrite(
        str(master_path), warped,
        [cv2.IMWRITE_JPEG_QUALITY, int(cfg.get("master_quality", 95))],
    )

    derivatives = write_derivatives(
        warped,
        item["id"],
        paths.derivatives,
        cfg.get("derivative_sizes", [1600, 800, 320]),
        int(cfg.get("derivative_quality", 82)),
    )

    height, width = warped.shape[:2]
    with db.transaction(conn):
        db.set_item_rectified(
            conn, item["id"],
            master_path=paths.relative(master_path),
            width=width, height=height, fine_skew_deg=round(skew, 4),
        )
        db.replace_derivatives(
            conn, item["id"],
            [{**d, "path": paths.relative(d["path"])} for d in derivatives],
        )

    return RectifyResult(
        item_id=item["id"],
        master_path=master_path,
        width=width,
        height=height,
        fine_skew_deg=round(skew, 4),
        derivatives=derivatives,
    )


def rectify_pending(
    conn: sqlite3.Connection,
    paths: Any,
    cfg: dict[str, Any],
    *,
    force: bool = False,
    progress: Any = None,
) -> list[RectifyResult]:
    """Rectify every item awaiting it, decoding each source photo only once."""
    statuses = (
        ["detected", "rectified", "analyzed", "approved"] if force else ["detected"]
    )
    items = db.items_with_status(conn, statuses)

    # Group by source photo so one decode serves all of that photo's items.
    by_photo: dict[str, list[sqlite3.Row]] = {}
    for item in items:
        by_photo.setdefault(item["photo_sha256"], []).append(item)

    results: list[RectifyResult] = []
    for photo_sha, photo_items in by_photo.items():
        photo = db.get_photo(conn, photo_sha)
        if photo is None:
            continue
        source = _read_source_bgr(paths.absolute(photo["stored_path"]))
        for item in photo_items:
            if progress is not None:
                progress(item)
            results.append(
                rectify_item(conn, paths, item, cfg, source_image=source)
            )
        db.set_photo_status(conn, photo_sha, "rectified")

    return results
