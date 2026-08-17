"""Generate synthetic table shots for testing segmentation.

Real photos are the only true test, but a synthetic generator gives us
ground-truth corner positions, which real photos don't have without manual
labelling. It models the things that actually break detection: paper texture,
uneven lighting, drop shadows, rotation, perspective, and low contrast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PlacedItem:
    """An item with its ground-truth quad in image coordinates (TL,TR,BR,BL)."""

    quad: np.ndarray
    kind: str


def _paper_texture(h: int, w: int, base: int, rng: np.random.Generator) -> np.ndarray:
    """A sheet of aged paper: base tone, fine grain, gentle blotching."""
    img = np.full((h, w, 3), base, dtype=np.float32)
    img += rng.normal(0, 3.5, (h, w, 3))

    # Low-frequency staining, as on old newsprint.
    blot = rng.normal(0, 1, (max(2, h // 24), max(2, w // 24))).astype(np.float32)
    blot = cv2.resize(blot, (w, h), interpolation=cv2.INTER_CUBIC)
    img += (blot * 6.0)[..., None]

    # Warm the tone slightly: less blue than red, like yellowed paper.
    img[..., 0] *= 0.93
    img[..., 1] *= 0.97
    return np.clip(img, 0, 255).astype(np.uint8)


def _add_text_lines(img: np.ndarray, rng: np.random.Generator) -> None:
    """Rows of dark dashes standing in for body text."""
    h, w = img.shape[:2]
    y = int(h * 0.14)
    line_h = max(3, h // 42)
    while y < h - line_h * 2:
        x = int(w * 0.07)
        right = int(w * rng.uniform(0.82, 0.95))
        while x < right:
            word = int(rng.integers(w // 22, w // 8))
            word = min(word, right - x)
            if word <= 1:
                break
            cv2.rectangle(
                img, (x, y), (x + word, y + line_h),
                (int(rng.integers(30, 70)),) * 3, -1,
            )
            x += word + max(3, w // 45)
        y += int(line_h * 2.4)


def _make_clipping(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    img = _paper_texture(h, w, int(rng.integers(196, 226)), rng)
    # Headline bar.
    cv2.rectangle(
        img, (int(w * 0.07), int(h * 0.05)), (int(w * 0.82), int(h * 0.11)),
        (35, 35, 35), -1,
    )
    _add_text_lines(img, rng)
    return img


def _make_photograph(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A mid-grey print with a white border - low contrast against light
    backgrounds, which is the hard case."""
    img = _paper_texture(h, w, 235, rng)
    inset = max(4, min(h, w) // 14)
    tone = int(rng.integers(95, 150))
    cv2.rectangle(img, (inset, inset), (w - inset, h - inset), (tone,) * 3, -1)
    # A couple of vague shapes so it isn't a flat rectangle.
    for _ in range(3):
        cx = int(rng.integers(inset, max(inset + 1, w - inset)))
        cy = int(rng.integers(inset, max(inset + 1, h - inset)))
        r = int(rng.integers(max(3, h // 14), max(4, h // 6)))
        shade = int(np.clip(tone + rng.integers(-45, 45), 0, 255))
        cv2.circle(img, (cx, cy), r, (shade,) * 3, -1)
    img[:] = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def _perspective_quad(
    cx: float, cy: float, w: float, h: float, angle_deg: float,
    tilt: float, rng: np.random.Generator,
) -> np.ndarray:
    """Corners of a rotated, slightly perspective-warped rectangle."""
    hw, hh = w / 2.0, h / 2.0
    corners = np.array(
        [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32
    )
    if tilt > 0:
        jitter = rng.uniform(-tilt, tilt, size=(4, 2)) * np.array([w, h])
        corners = corners + jitter.astype(np.float32)

    theta = math.radians(angle_deg)
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float32,
    )
    return (corners @ rot.T + np.array([cx, cy], dtype=np.float32)).astype(np.float32)


def make_table_shot(
    path: Path,
    *,
    n_items: int = 6,
    width: int = 3024,
    height: int = 4032,
    background: int = 28,
    seed: int = 7,
    tilt: float = 0.012,
    max_angle: float = 12.0,
) -> list[PlacedItem]:
    """Render a table shot to `path` and return ground-truth quads.

    Defaults model the recommended setup: dark matte background, items laid out
    in a loose grid with a visible gap, shot roughly straight down.
    """
    rng = np.random.default_rng(seed)

    # Background: matte board with grain and a soft light gradient.
    canvas = np.full((height, width, 3), background, dtype=np.float32)
    canvas += rng.normal(0, 4.0, (height, width, 3))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gradient = 18.0 * (1.0 - ((xx / width - 0.4) ** 2 + (yy / height - 0.35) ** 2))
    canvas += gradient[..., None]
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    cols = 2 if n_items > 1 else 1
    rows = math.ceil(n_items / cols)
    cell_w, cell_h = width / cols, height / rows

    placed: list[PlacedItem] = []
    for idx in range(n_items):
        r, c = divmod(idx, cols)
        cx = (c + 0.5) * cell_w + rng.uniform(-cell_w * 0.03, cell_w * 0.03)
        cy = (r + 0.5) * cell_h + rng.uniform(-cell_h * 0.03, cell_h * 0.03)

        kind = "clipping" if idx % 2 == 0 else "photograph"
        iw = cell_w * rng.uniform(0.58, 0.72)
        ih = cell_h * rng.uniform(0.52, 0.68)
        if kind == "photograph":
            ih = min(ih, iw * rng.uniform(0.7, 1.0))

        angle = float(rng.uniform(-max_angle, max_angle))
        quad = _perspective_quad(cx, cy, iw, ih, angle, tilt, rng)

        src_h, src_w = int(round(ih)), int(round(iw))
        tile = (
            _make_clipping(src_h, src_w, rng)
            if kind == "clipping"
            else _make_photograph(src_h, src_w, rng)
        )
        src_quad = np.array(
            [[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src_quad, quad)

        warped = cv2.warpPerspective(tile, matrix, (width, height))
        mask = cv2.warpPerspective(
            np.full((src_h, src_w), 255, np.uint8), matrix, (width, height)
        )

        # Drop shadow, offset down-right. This is the single most common cause
        # of a detector merging two adjacent items or over-growing one.
        shadow = cv2.GaussianBlur(mask, (41, 41), 0).astype(np.float32) / 255.0
        shadow = np.roll(shadow, (14, 14), axis=(0, 1))
        canvas = (canvas.astype(np.float32) * (1.0 - 0.45 * shadow[..., None]))
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

        canvas[mask > 0] = warped[mask > 0]
        placed.append(PlacedItem(quad=quad, kind=kind))

    # Camera noise last, over everything.
    final = np.clip(
        canvas.astype(np.float32) + rng.normal(0, 2.5, canvas.shape), 0, 255
    ).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), final, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return placed
