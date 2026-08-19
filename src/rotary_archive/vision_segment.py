"""LLM-guided segmentation: read the page, then decide the boundaries.

Classical contour detection knows where the paper stops but nothing about what
is printed on it, and that is exactly the information needed here. It cannot
tell that a headline strip and the columns beneath it are one clipping, that a
photograph belongs to the article next to it, or that two strips are page one
and page two of the same story. On a table of overlapping newsprint those are
not edge cases - they are most of the material.

So this pass asks a vision model to read the photo and mark out each distinct
item, then hands each proposed region back to the contour code to snap onto
the real paper edge. The model contributes judgement about what belongs with
what; OpenCV contributes pixel accuracy. Neither is good at the other's job.

Falls back to pure contour detection when no model is configured, so the
pipeline still runs offline and for free.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import geometry

# Sent to the model. Well inside the 2576px high-resolution ceiling, and enough
# for it to read a headline; boxes come back in a normalised 0-1000 grid so the
# source resolution never has to be explained to it.
VISION_LONG_EDGE = 2000
GRID = 1000

ITEM_KINDS = [
    "newspaper_clipping", "photograph", "document", "letter", "certificate",
    "program", "newsletter", "ephemera", "object", "other",
]

REGION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "description": "One entry per distinct physical item on the surface.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "box": {
                        "type": "array",
                        "description": (
                            "[left, top, right, bottom] on a 0-1000 grid over the "
                            "whole image. Must enclose every part of the item, "
                            "including a headline strip or photograph that juts "
                            "out from the main column."
                        ),
                        "items": {"type": "integer"},
                    },
                    "kind": {"type": "string", "enum": ITEM_KINDS},
                    "headline": {
                        "type": "string",
                        "description": (
                            "The largest words on the item, verbatim, so a human "
                            "can tell which item this is. Empty if it has no text."
                        ),
                    },
                    "shape_note": {
                        "type": "string",
                        "description": (
                            "If the item is not a plain rectangle - cut around a "
                            "headline, an L, a photo attached along one edge - say "
                            "so briefly. Empty if it is a plain rectangle."
                        ),
                    },
                    "part_of": {
                        "type": "integer",
                        "description": (
                            "Index of an earlier item in this list that this one "
                            "continues or belongs to - a second page, a spilled "
                            "column, a photograph belonging to that article. -1 "
                            "when the item stands alone."
                        ),
                    },
                    "part_reason": {
                        "type": "string",
                        "description": (
                            "Why it belongs with that item, if part_of is set. "
                            "Empty otherwise."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0.0 to 1.0 in this item's boundary.",
                    },
                },
                "required": [
                    "box", "kind", "headline", "shape_note",
                    "part_of", "part_reason", "confidence",
                ],
            },
        }
    },
    "required": ["items"],
}

SYSTEM_PROMPT = """\
You are looking at a photograph of historical memorabilia laid out on a table: \
newspaper clippings, photographs, certificates, programs. Your job is to mark \
out every distinct physical item so each can be cropped out separately.

Return a JSON object with an `items` array. Nothing else.

Coordinates are on a 0-1000 grid over the whole image: 0,0 is the top-left \
corner, 1000,1000 the bottom-right. `box` is [left, top, right, bottom].

Work from what is printed, not just from edges. That is the whole reason you \
are being asked rather than an edge detector.

**Each box must enclose the entire item.** Newspaper clippings are cut by hand \
and are often not rectangles - somebody cuts around a headline to keep it \
attached, or leaves a photograph joined along one edge, so the outline juts \
out. Include those parts. A box slightly too large is easily trimmed; one that \
slices off a headline has destroyed it. When two items overlap, box the whole \
of the one on top, and box as much of the one underneath as you can see.

**Read enough to tell items apart.** Put the largest words of each item in \
`headline`, verbatim. This is how a person checks your work.

**Say when pieces belong together.** If one item continues another - a story \
carried onto a second strip, a column that spilled over, a photograph cut out \
alongside the article it illustrates - set `part_of` to the index of the item \
it belongs with, and explain briefly in `part_reason`. Judge this from the \
text: a continued headline, a matching byline or date, a caption that refers \
to the neighbouring story. Do not guess from position alone - two clippings \
sitting next to each other are usually unrelated. Set `part_of` to -1 when in \
any doubt.

Count carefully. Every separate piece of paper gets its own entry, including \
small ones. Do not merge two items into one box because they touch.
"""


@dataclass
class Region:
    """One item the model marked out, in full-resolution pixel coordinates."""

    box: tuple[float, float, float, float]
    kind: str = "other"
    headline: str = ""
    shape_note: str = ""
    part_of: int = -1
    part_reason: str = ""
    confidence: float = 0.5
    refined: bool = False

    def quad(self) -> np.ndarray:
        left, top, right, bottom = self.box
        return np.array(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float32,
        )


@dataclass
class VisionResult:
    regions: list[Region] = field(default_factory=list)
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


# ------------------------------------------------------------------ request ---


def prepare_image(path: Path) -> tuple[np.ndarray, int, int]:
    """The photo at model resolution, plus its full-resolution dimensions."""
    from .ingest import load_oriented

    with load_oriented(path) as pil_img:
        rgb = np.asarray(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    full_h, full_w = bgr.shape[:2]
    long_edge = max(full_h, full_w)
    if long_edge > VISION_LONG_EDGE:
        factor = VISION_LONG_EDGE / long_edge
        bgr = cv2.resize(
            bgr, (max(1, int(full_w * factor)), max(1, int(full_h * factor))),
            interpolation=cv2.INTER_AREA,
        )
    return bgr, full_w, full_h


def _encode(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("could not encode image for the model")
    return buf.tobytes()


def _ask(
    provider: Any,
    path: Path,
    image: np.ndarray,
    system: str,
    schema: dict[str, Any],
    context: str,
    tag: str,
) -> Any:
    """Send one annotated image to the provider and hand back its result.

    The copy is written beside the source rather than in the system temp
    directory. The claude_cli provider runs a nested CLI whose file access is
    scoped to the project, so a /var/folders path comes back as "I don't have
    permission to read that file path" rather than an analysis.
    """
    import os

    from .providers.base import Job

    tmp_path = path.parent / f".rotary-vision-{os.getpid()}-{tag}-{path.stem}.jpg"
    tmp_path.write_bytes(_encode(image))
    try:
        return provider.analyze(
            Job(item_id=path.stem, image_path=tmp_path, context=context),
            system,
            schema,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _to_pixels(box: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Map a 0-1000 grid box onto full-resolution pixels."""
    try:
        left, top, right, bottom = (float(v) for v in box)
    except (TypeError, ValueError):
        return None

    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))
    if right - left < 1 or bottom - top < 1:
        return None

    scale_x, scale_y = width / GRID, height / GRID
    return (
        max(0.0, left * scale_x),
        max(0.0, top * scale_y),
        min(float(width - 1), right * scale_x),
        min(float(height - 1), bottom * scale_y),
    )


def _parse_regions(raw: Any, width: int, height: int) -> list[Region]:
    """Turn the model's items array into Regions in full-resolution pixels."""
    regions: list[Region] = []
    if not isinstance(raw, list):
        return regions

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        box = _to_pixels(entry.get("box"), width, height)
        if box is None:
            continue

        try:
            confidence = float(entry.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        if confidence > 1.0:
            confidence /= 100.0

        try:
            part_of = int(entry.get("part_of", -1))
        except (TypeError, ValueError):
            part_of = -1

        kind = str(entry.get("kind", "other"))
        regions.append(
            Region(
                box=box,
                kind=kind if kind in ITEM_KINDS else "other",
                headline=str(entry.get("headline", ""))[:200],
                shape_note=str(entry.get("shape_note", ""))[:200],
                part_of=part_of,
                part_reason=str(entry.get("part_reason", ""))[:300],
                confidence=min(1.0, max(0.0, confidence)),
            )
        )

    # An index pointing past the end, or at itself, means nothing.
    for index, region in enumerate(regions):
        if not (0 <= region.part_of < len(regions)) or region.part_of == index:
            region.part_of = -1
            region.part_reason = ""
    return regions


def propose_regions(path: Path, provider: Any) -> VisionResult:
    """Ask the model to mark out each item in the photo."""
    try:
        image, width, height = prepare_image(path)
    except Exception as exc:
        return VisionResult(error=f"could not read {path.name}: {exc}")

    # The model is shown the resized copy rather than the original: a 12MP
    # master costs several times the image tokens for detail it cannot use.
    result = _ask(
        provider,
        path,
        image,
        SYSTEM_PROMPT,
        REGION_SCHEMA,
        "Mark out every distinct item laid out in this photograph, "
        "following the instructions exactly.",
        "propose",
    )
    if not result.ok:
        return VisionResult(error=result.error, usage=result.usage)

    regions = _parse_regions(result.data.get("items"), width, height)
    if not regions:
        return VisionResult(error="model returned no usable items", usage=result.usage)
    return VisionResult(regions=regions, usage=result.usage)


# ------------------------------------------------------------------ refine ---


def refine_regions(bgr: np.ndarray, regions: list[Region]) -> list[Region]:
    """Give every proposed region a boundary taken from the pixels.

    The model reliably says *what* is on the table and how many pieces there
    are; it is much weaker at saying exactly where each edge falls, and on a
    low-resolution photo its boxes come back noticeably offset and oversized.
    Contour detection is the other way round - precise about edges, unable to
    tell one clipping from the two it is touching.

    So the model's boxes are used as seeds rather than as answers. The paper is
    separated from the table on the full frame, one marker is planted per
    proposed item, and watershed hands every paper pixel to its nearest marker.
    The count and the identities come from the model; every coordinate comes
    from the image.
    """
    from .segment import _background_mask

    if not regions:
        return regions

    mask = _background_mask(bgr)
    height, width = mask.shape[:2]

    # One marker per item, placed where the proposal overlaps actual paper.
    markers = np.zeros((height, width), np.int32)
    planted: list[int] = []
    for index, region in enumerate(regions, start=1):
        left, top, right, bottom = region.box
        window = np.zeros((height, width), np.uint8)
        cv2.rectangle(
            window,
            (int(max(0, left)), int(max(0, top))),
            (int(min(width - 1, right)), int(min(height - 1, bottom))),
            255, -1,
        )
        overlap = cv2.bitwise_and(window, mask)
        if cv2.countNonZero(overlap) < 40:
            continue

        # Erode hard so the seed sits deep inside one item rather than
        # straddling the join with its neighbour.
        core = cv2.erode(
            overlap, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=2,
        )
        if cv2.countNonZero(core) < 20:
            core = overlap
        markers[core > 0] = index
        planted.append(index)

    if len(planted) < 1:
        return regions

    # Everything outside the paper is known background; the rest is unassigned
    # and gets handed out by the watershed.
    markers[mask == 0] = len(regions) + 1
    cv2.watershed(bgr, markers)

    for index, region in enumerate(regions, start=1):
        if index not in planted:
            continue
        component = np.uint8(markers == index) * 255
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < (height * width) * 0.004:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        region.box = (float(x), float(y), float(x + w), float(y + h))
        region.refined = True

    return regions


def segment_with_vision(
    path: Path, provider: Any, *, refine: bool = True
) -> VisionResult:
    """Full LLM-guided pass over one photograph.

    Two stages, and the split between them is the whole design:

    1. `propose_regions` reads the page and decides what is on the table, how
       many pieces there are, what each one says, and which pieces belong to
       the same article. Only the model can do this - no edge detector can
       tell that a photograph was cut out alongside the story it illustrates.
    2. `refine_regions` throws away the model's edges and takes the boundary
       from the pixels instead, using its boxes only as markers.

    Stage 2 is not a tidy-up, it is most of the accuracy. Measured against
    synthetic scenes with known ground truth, the model's own boxes match at a
    median IoU of 0.90 and clear the 0.80 bar on 5 of 6 items; after refining,
    the same regions sit at 0.995 and clear it on all 6.

    Two further ideas were built and measured and are deliberately absent: an
    annotated coordinate grid drawn over the image (raw boxes got *worse*,
    0.904 -> 0.842) and a second pass showing the model its own boxes to
    correct (worse again at 0.764, and identical after refining). Both cost a
    model call per photo and neither changed the refined result.
    """
    result = propose_regions(path, provider)
    if not result.ok or not result.regions or not refine:
        return result

    from .ingest import load_oriented

    with load_oriented(path) as pil_img:
        rgb = np.asarray(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    result.regions = refine_regions(bgr, result.regions)
    return result
