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

# What the item is printed on. This is the signal that separates a glossy
# ShelterBox brochure lying on top of a newspaper story about ShelterBox: same
# subject, different object, different publisher. Merging them would have the
# archive claim the local paper printed the brochure.
MEDIA = [
    "newsprint", "glossy_print", "photographic_print", "typescript",
    "manuscript", "card_stock", "other",
]


def _field(kind: str, description: str, **extra: Any) -> dict[str, Any]:
    return {"type": kind, "description": description, **extra}


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
                            "out from the main column, and every panel of a "
                            "folded item lying open."
                        ),
                        "items": {"type": "integer"},
                    },
                    "kind": {"type": "string", "enum": ITEM_KINDS},
                    "medium": {"type": "string", "enum": MEDIA},
                    "headline": _field(
                        "string",
                        "The largest words on the item, verbatim, so a human can "
                        "tell which item this is. Empty if it has no text.",
                    ),
                    "date_hint": _field(
                        "string",
                        "Any date printed on the item, copied as it appears. This "
                        "is often what tells you two items belong to the same "
                        "occasion. Empty if none is printed.",
                    ),
                    "shape_note": _field(
                        "string",
                        "If the item is not a plain rectangle - cut around a "
                        "headline, an L, columns of unequal length, a photo "
                        "attached along one edge - say so briefly.",
                    ),
                    "panels": _field(
                        "integer",
                        "How many panels or pages of this one physical object are "
                        "visible. A folded program lying open shows 2. Use 1 for "
                        "a flat single sheet. Never split an object at its fold.",
                    ),
                    "clipped_by_frame": _field(
                        "boolean",
                        "True if any part of the item runs off the edge of the "
                        "photograph, so what is missing cannot be recovered.",
                    ),
                    "part_of": _field(
                        "integer",
                        "MERGES this item into another. Index of an item in this "
                        "list that this one continues: a story carried onto a "
                        "second strip, a spilled column, a photograph cut from "
                        "the same article. -1 when it stands alone.",
                    ),
                    "part_reason": _field(
                        "string", "Why it continues that item. Empty otherwise."
                    ),
                    "related_to": {
                        "type": "array",
                        "description": (
                            "LINKS WITHOUT MERGING. Indexes of items that document "
                            "the same subject or occasion but are separate objects "
                            "in their own right - a ticket and the programme for "
                            "the same night, a brochure beside an article about "
                            "the same charity. Empty when nothing else relates."
                        ),
                        "items": {"type": "integer"},
                    },
                    "related_reason": _field(
                        "string", "What connects them. Empty if related_to is empty."
                    ),
                    "duplicate_of": _field(
                        "integer",
                        "Index of an item this is a second copy of - the same page "
                        "cut out twice. -1 when it is not a duplicate.",
                    ),
                    "link_confidence": _field(
                        "number",
                        "0.0 to 1.0 in part_of, related_to and duplicate_of. High "
                        "when printed evidence supports it - a continued headline, "
                        "a matching date, an identical page. Low when it rests on "
                        "the items merely lying next to each other.",
                    ),
                    "confidence": _field(
                        "number", "0.0 to 1.0 in this item's boundary."
                    ),
                },
                "required": [
                    "box", "kind", "medium", "headline", "date_hint",
                    "shape_note", "panels", "clipped_by_frame", "part_of",
                    "part_reason", "related_to", "related_reason",
                    "duplicate_of", "link_confidence", "confidence",
                ],
            },
        }
    },
    "required": ["items"],
}

SYSTEM_PROMPT = """\
You are looking at a photograph of historical memorabilia laid out on a \
surface: newspaper clippings, photographs, programmes, tickets, certificates, \
letters. Your job is to mark out every distinct physical item so each can be \
cropped out and catalogued separately.

Return a JSON object with an `items` array. Nothing else.

Coordinates are on a 0-1000 grid over the whole image: 0,0 is the top-left \
corner, 1000,1000 the bottom-right. `box` is [left, top, right, bottom].

Read the items before you decide where their edges are. That is the whole \
reason you are being asked rather than an edge detector.


## One box per physical object

Not per article, and not per page. Ask yourself: if I picked this up off the \
table, what would come up in my hand?

**A folded item lying open is one object.** A programme opened flat showing \
Part I and Part II is a single sheet of paper. The crease down the middle is a \
straight, high-contrast line and it is tempting to cut there. Do not. One box \
around both panels, and set `panels` to 2.

**Two things that touch are still two things.** Items in a scrapbook photo are \
usually laid edge to edge or overlapping with no surface visible between them. \
Where one piece of paper ends and the next begins there is a seam - a shadow, \
a change in paper shade, a change in column width, a headline starting again. \
Box the whole of the item on top, and as much of the one underneath as you can \
see.

**A change of medium means a different object.** Newsprint is grey, matte and \
coarse; a brochure is glossy, often in full colour; a photographic print has \
its own surface again. If the material changes, you have crossed from one item \
to the next, however closely they overlap.

**Include the parts that jut out.** Clippings are cut by hand around what \
matters - somebody cuts around a headline to keep it attached, or leaves a \
photograph joined along one edge, or the columns of an article end at \
different depths so the outline is a staircase. All of it is the item. A box \
slightly too large is trimmed in seconds; one that slices off a headline has \
destroyed it.


## Three ways items can belong together

These are different, and using the wrong one damages the record.

**`part_of` merges.** Use it only when the two pieces are *the same document*: \
a story continued on a second strip, a column that spilled over, a photograph \
cut from the article it illustrated. The archive will publish them as one \
entry with one title.

**`related_to` links without merging.** Use it when two items document the \
same subject or the same occasion but are separate things in their own right: \
a ticket and the programme for that night, a charity's own brochure lying \
beside a newspaper story about that charity. Each keeps its own catalogue \
entry, its own kind, its own date. Merging these would attribute one \
publisher's work to another, which is simply false.

The test is authorship, not subject. Would the same person have printed both, \
as parts of one thing? Then `part_of`. Different origins that happen to be \
about the same event? Then `related_to`.

**`duplicate_of`** is for the same page cut out twice. Both copies are kept - \
they may be cropped or lit differently - but only one should reach the site.


## Say how sure you are, separately

`confidence` is about the *box*. `link_confidence` is about `part_of`, \
`related_to` and `duplicate_of`.

Set `link_confidence` high only when something printed supports the link - a \
continued headline, a matching byline or date, an identical page, a caption \
naming the neighbouring story. Set it low when the link rests on the two items \
merely lying next to each other. An uncaptioned photograph sitting under an \
article is *probably* that article's photograph, but position alone is weak \
evidence, especially when every item on the table is about a similar subject. \
Say so with a low number rather than asserting it.

Set `part_of` to -1, `duplicate_of` to -1, and `related_to` to empty whenever \
you are in real doubt. A missing link costs a reader one click. A wrong one \
puts a false claim in the club's history.


## Also

**Copy any printed date** into `date_hint`, exactly as it appears. Dates are \
usually what reveals that a ticket, a programme and a clipping belong to one \
occasion. An item can carry two - when a paper was published and when the \
event happened. Prefer the event date if both are legible.

**Flag anything running off the edge** of the photograph with \
`clipped_by_frame`. That content cannot be recovered from this shot and \
someone needs to know to take another.

Count carefully. Every separate piece gets its own entry, including small ones \
- a ticket stub, a caption strip, a photograph on its own. Do not merge two \
items into one box because they touch.
"""


@dataclass
class Region:
    """One item the model marked out, in full-resolution pixel coordinates."""

    box: tuple[float, float, float, float]
    kind: str = "other"
    medium: str = "other"
    headline: str = ""
    date_hint: str = ""
    shape_note: str = ""
    panels: int = 1
    clipped_by_frame: bool = False
    # part_of merges this item into another; related_to links without merging;
    # duplicate_of marks a second copy. See SYSTEM_PROMPT for why the three are
    # kept apart.
    part_of: int = -1
    part_reason: str = ""
    related_to: list[int] = field(default_factory=list)
    related_reason: str = ""
    duplicate_of: int = -1
    link_confidence: float = 0.5
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
    # Items the model described but whose coordinates could not be placed on
    # the image. Surfaced rather than swallowed: a dropped region means a real
    # piece of paper nobody has catalogued, and the photo needs a human.
    dropped: int = 0

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


# Below this, in grid units, a region is not a piece of paper. Half a percent
# of the frame is a 20px sliver on a 4000px photograph.
MIN_GRID_EXTENT = 5


def _to_pixels(box: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Map a 0-1000 grid box onto full-resolution pixels.

    Everything is clamped in *grid* space, before any conversion. Clamping
    afterwards was a real defect: a model that answered past the edge of the
    grid - top=1200 where the grid ends at 1000 - had its bottom pulled back to
    the last pixel while its top was left beyond it, and the box came out
    inverted. Those reached the crop stage as negative-height quads and
    nineteen-pixel slivers, and because a degenerate box is also one the
    refinement step refuses to touch, they sailed through untouched and became
    the crops nobody could explain.
    """
    try:
        left, top, right, bottom = (float(v) for v in box)
    except (TypeError, ValueError):
        return None

    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))

    left, right, top, bottom = (
        min(max(value, 0.0), float(GRID)) for value in (left, right, top, bottom)
    )
    if right - left < MIN_GRID_EXTENT or bottom - top < MIN_GRID_EXTENT:
        return None

    scale_x, scale_y = width / GRID, height / GRID
    return (
        left * scale_x,
        top * scale_y,
        min(float(width - 1), right * scale_x),
        min(float(height - 1), bottom * scale_y),
    )


def _parse_regions(
    raw: Any, width: int, height: int, dropped: list[int] | None = None
) -> list[Region]:
    """Turn the model's items array into Regions in full-resolution pixels."""
    regions: list[Region] = []
    if not isinstance(raw, list):
        return regions

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        box = _to_pixels(entry.get("box"), width, height)
        if box is None:
            if dropped is not None:
                dropped.append(1)
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
        medium = str(entry.get("medium", "other"))
        regions.append(
            Region(
                box=box,
                kind=kind if kind in ITEM_KINDS else "other",
                medium=medium if medium in MEDIA else "other",
                headline=str(entry.get("headline", ""))[:200],
                date_hint=str(entry.get("date_hint", ""))[:80],
                shape_note=str(entry.get("shape_note", ""))[:200],
                panels=max(1, _as_int(entry.get("panels"), 1)),
                clipped_by_frame=bool(entry.get("clipped_by_frame", False)),
                part_of=part_of,
                part_reason=str(entry.get("part_reason", ""))[:300],
                related_to=[
                    value for value in
                    (_as_int(v, -1) for v in _as_sequence(entry.get("related_to")))
                    if value >= 0
                ],
                related_reason=str(entry.get("related_reason", ""))[:300],
                duplicate_of=_as_int(entry.get("duplicate_of"), -1),
                link_confidence=_as_confidence(entry.get("link_confidence"), 0.5),
                confidence=min(1.0, max(0.0, confidence)),
            )
        )

    _validate_links(regions)
    return regions


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _as_confidence(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 1.0:
        number /= 100.0
    return min(1.0, max(0.0, number))


def _validate_links(regions: list[Region]) -> None:
    """Drop links that point nowhere.

    Every index the model returns is an index into this photo's own item list.
    One pointing past the end, or at itself, means it lost track - and a bad
    link is worse than no link, because part_of buries one item inside another
    and duplicate_of hides one from the site entirely.
    """
    count = len(regions)

    def valid(index: int, self_index: int) -> bool:
        return 0 <= index < count and index != self_index

    for index, region in enumerate(regions):
        if not valid(region.part_of, index):
            region.part_of = -1
            region.part_reason = ""
        if not valid(region.duplicate_of, index):
            region.duplicate_of = -1
        region.related_to = sorted(
            {other for other in region.related_to if valid(other, index)}
        )
        if not region.related_to:
            region.related_reason = ""

        # An item cannot both continue another and be a copy of one. When the
        # model says both, the merge is the stronger claim and wins.
        if region.part_of >= 0:
            region.duplicate_of = -1


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

    dropped: list[int] = []
    regions = _parse_regions(result.data.get("items"), width, height, dropped)
    if not regions:
        return VisionResult(error="model returned no usable items", usage=result.usage)
    return VisionResult(regions=regions, usage=result.usage, dropped=len(dropped))


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
