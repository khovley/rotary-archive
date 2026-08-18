"""Analyse stage: read each item with a vision model and catalogue it.

Runs after rectify. For each item it sends one image to the configured
provider, validates the response against the schema's expectations, and writes
a new row to `analyses` - appending rather than overwriting, so re-running with
a better model later keeps the earlier reading for comparison.

Also extracts people, organisations, places, and topics into a deduplicated
`entities` table. That is what makes "everything mentioning Harold Pratt" a
query rather than a full-text scan, and it is why the site can offer a person
index at all.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import db
from .providers import AnalysisResult, Job, VisionProvider
from .schema_item import (
    DATE_PRECISIONS,
    DATE_SOURCES,
    ITEM_SCHEMA,
    ITEM_TYPES,
    ORIENTATIONS,
    PRESENTATIONS,
    SYSTEM_PROMPT,
    build_context,
)

ENTITY_FIELDS = {
    "people": "person",
    "organizations": "organization",
    "places": "place",
    "topics": "topic",
}

# Guards against a model returning a wall of text in a field meant for a line.
MAX_LENGTHS = {
    "title": 300,
    "summary": 4000,
    "alt_text": 600,
    "review_reason": 600,
    "date_note": 1000,
    "condition_notes": 1000,
    "rotary_context": 2000,
}


@dataclass
class AnalyzeSummary:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    flagged: int = 0
    errors: list[tuple[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


# --------------------------------------------------------------- validation --


def _as_text(value: Any, limit: int | None = None) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    return text[:limit] if limit else text


def _as_list(value: Any) -> list[str]:
    """Coerce to a clean list of strings, order-preserving and deduplicated.

    Models occasionally return a comma-joined string where a list was asked
    for; splitting it is friendlier than discarding the content.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [part for part in re.split(r"[;,]", value)]
    if not isinstance(value, (list, tuple)):
        return []

    seen: set[str] = set()
    out: list[str] = []
    for entry in value:
        text = _as_text(entry, 200)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _as_enum(value: Any, allowed: Sequence[str], fallback: str) -> str:
    text = _as_text(value).lower().replace("-", "_").replace(" ", "_")
    return text if text in allowed else fallback


def normalise(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce a raw model response into the shape the database expects.

    Providers without schema enforcement return near-misses - a string where a
    list belongs, a percentage where a fraction belongs, an invented enum
    value. Rather than reject those outright and lose the analysis, coerce what
    is coercible and fall back to a safe default otherwise. Anything that had
    to fall back is a reason to flag the item.
    """
    clean: dict[str, Any] = {}

    clean["item_type"] = _as_enum(data.get("item_type"), ITEM_TYPES, "other")
    clean["presentation"] = _as_enum(
        data.get("presentation"), PRESENTATIONS, "image"
    )
    clean["date_precision"] = _as_enum(
        data.get("date_precision"), DATE_PRECISIONS, "unknown"
    )
    clean["date_source"] = _as_enum(data.get("date_source"), DATE_SOURCES, "unknown")
    clean["orientation_hint"] = _as_enum(
        data.get("orientation_hint"), ORIENTATIONS, "upright"
    )

    for field, limit in MAX_LENGTHS.items():
        clean[field] = _as_text(data.get(field), limit)
    clean["full_text"] = _as_text(data.get("full_text"))
    clean["date_value"] = _as_text(data.get("date_value"), 40)

    for field in ENTITY_FIELDS:
        clean[field] = _as_list(data.get(field))

    try:
        legibility = int(data.get("legibility", 3))
    except (TypeError, ValueError):
        legibility = 3
    clean["legibility"] = min(5, max(1, legibility))

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    # Some models answer 85 when asked for 0-1.
    if confidence > 1.0:
        confidence = confidence / 100.0
    clean["confidence"] = min(1.0, max(0.0, confidence))

    clean["needs_human_review"] = bool(data.get("needs_human_review", False))

    # An empty date must never claim to be printed, whatever the model said.
    if not clean["date_value"]:
        clean["date_source"] = "unknown"
        clean["date_precision"] = "unknown"

    return clean


def consistency_flags(clean: dict[str, Any]) -> list[str]:
    """Reasons to put this item in front of a human, beyond the model's own.

    The model judges its own confidence; these are the cross-field checks it
    cannot make about itself - a transcription that contradicts the item type,
    a date asserted as printed with no supporting text, an item it says is
    barely readable.
    """
    reasons: list[str] = []

    if clean["legibility"] <= 2:
        reasons.append("low legibility")
    if clean["confidence"] < 0.6:
        reasons.append(f"low model confidence ({clean['confidence']:.2f})")
    if clean["date_source"] == "inferred":
        reasons.append("date inferred rather than printed")
    if clean["orientation_hint"] != "upright":
        reasons.append(f"may need rotating ({clean['orientation_hint']})")
    if clean["presentation"] in ("text", "both") and not clean["full_text"]:
        reasons.append("marked as text but no transcription returned")
    if clean["date_source"] == "printed" and not clean["full_text"]:
        reasons.append("date claimed as printed but nothing was transcribed")
    if not clean["title"]:
        reasons.append("no title")

    return reasons


# ------------------------------------------------------------------ entities --


def slugify(value: str) -> str:
    """Stable key for entity deduplication and URLs.

    Case, accents, and punctuation are all normalised away, so "O'Brien",
    "OBrien", and "o'brien" collapse to one person rather than three.

    Apostrophes are elided rather than treated as separators. As separators
    they split "O'Brien" into "o-brien" while "OBrien" stays "obrien",
    silently making two people out of one - the exact failure this function
    exists to prevent.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = decomposed.encode("ascii", "ignore").decode()
    without_apostrophes = re.sub(r"['’ʼ`]", "", ascii_only)
    slug = re.sub(r"[^a-z0-9]+", "-", without_apostrophes.lower()).strip("-")
    return slug or re.sub(r"\s+", "-", value.strip().lower())


def upsert_entity(conn: sqlite3.Connection, kind: str, name: str) -> int | None:
    slug = slugify(name)
    if not slug:
        return None

    row = conn.execute(
        "SELECT id FROM entities WHERE kind = ? AND slug = ?", (kind, slug)
    ).fetchone()
    if row:
        return int(row["id"])

    cursor = conn.execute(
        "INSERT INTO entities (kind, name, slug, created_at) VALUES (?, ?, ?, ?)",
        (kind, name.strip(), slug, db.utcnow()),
    )
    return int(cursor.lastrowid)


def link_entities(
    conn: sqlite3.Connection, item_id: str, clean: dict[str, Any]
) -> int:
    """Replace this item's model-derived entity links.

    Only rows with source='model' are cleared - a name a human added by hand
    must survive a re-analysis, or correcting the archive would be pointless.
    """
    conn.execute(
        "DELETE FROM item_entities WHERE item_id = ? AND source = 'model'",
        (item_id,),
    )

    linked = 0
    for field, kind in ENTITY_FIELDS.items():
        for name in clean.get(field, []):
            entity_id = upsert_entity(conn, kind, name)
            if entity_id is None:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO item_entities (item_id, entity_id, source) "
                "VALUES (?, ?, 'model')",
                (item_id, entity_id),
            )
            linked += 1
    return linked


# -------------------------------------------------------------- persistence --


def store_analysis(
    conn: sqlite3.Connection,
    paths: Any,
    item_id: str,
    result: AnalysisResult,
    clean: dict[str, Any],
    extra_reasons: Sequence[str],
) -> None:
    needs_review = bool(clean["needs_human_review"] or extra_reasons)
    reasons = list(extra_reasons)
    if clean["review_reason"]:
        reasons.insert(0, clean["review_reason"])
    reason_text = "; ".join(reasons)[:600] or None

    with db.transaction(conn):
        db.supersede_analyses(conn, item_id)
        conn.execute(
            """
            INSERT INTO analyses (
                item_id, provider, model, created_at, superseded,
                item_type, title, summary, full_text,
                date_value, date_precision, date_source, date_note,
                presentation, legibility, condition_notes, alt_text,
                rotary_context, orientation_hint, confidence,
                needs_human_review, review_reason, raw_json, usage_json
            ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id, result.provider, result.model, db.utcnow(),
                clean["item_type"], clean["title"], clean["summary"],
                clean["full_text"], clean["date_value"], clean["date_precision"],
                clean["date_source"], clean["date_note"], clean["presentation"],
                clean["legibility"], clean["condition_notes"], clean["alt_text"],
                clean["rotary_context"], clean["orientation_hint"],
                clean["confidence"], int(needs_review), reason_text,
                json.dumps(clean, ensure_ascii=False),
                json.dumps(result.usage) if result.usage else None,
            ),
        )
        link_entities(conn, item_id, clean)
        db.set_item_status(
            conn, item_id, "analyzed",
            needs_human_review=needs_review, review_reason=reason_text,
        )

    write_export(paths, item_id, clean)


def write_export(paths: Any, item_id: str, clean: dict[str, Any]) -> Path | None:
    """Write a Markdown rendering for items whose value is in their words.

    This is the "does it stay a photo, or become a document" decision made
    concrete: an image-only item gets no export, while a clipping gets a
    readable article the site can show alongside the scan.
    """
    if clean["presentation"] not in ("text", "both") or not clean["full_text"]:
        return None

    paths.exports.mkdir(parents=True, exist_ok=True)
    path = paths.exports / f"{item_id}.md"

    front: list[str] = ["---", f'title: "{clean["title"]}"']
    if clean["date_value"]:
        front.append(f"date: {clean['date_value']}")
        front.append(f"date_source: {clean['date_source']}")
    front.append(f"item_type: {clean['item_type']}")
    for field in ("people", "organizations", "places", "topics"):
        if clean.get(field):
            joined = ", ".join(clean[field])
            front.append(f"{field}: [{joined}]")
    front.append("---")

    body = [clean["title"], "=" * max(3, len(clean["title"])), ""]
    if clean["summary"]:
        body += [f"*{clean['summary']}*", ""]
    body += [clean["full_text"].rstrip(), ""]
    if clean["rotary_context"]:
        body += ["---", "", clean["rotary_context"], ""]

    path.write_text("\n".join(front) + "\n\n" + "\n".join(body), encoding="utf-8")
    return path


# ------------------------------------------------------------------ jobs ----


def pick_image(
    conn: sqlite3.Connection, paths: Any, item_id: str, preferred_size: int
) -> Path | None:
    """The derivative to send. Falls back through smaller sizes, then master.

    Sending the 1600px derivative rather than the master is deliberate: it is
    well inside the high-resolution ceiling, costs roughly half the image
    tokens of a full-size upload, and loses no legibility on a crop that has
    already been rectified.
    """
    derivatives = db.derivatives_for_item(conn, item_id)
    by_size = {int(d["long_edge"]): d for d in derivatives}

    for size in sorted(by_size, key=lambda s: (abs(s - preferred_size), -s)):
        candidate = paths.absolute(by_size[size]["path"])
        if candidate.exists():
            return candidate

    item = db.get_item(conn, item_id)
    if item and item["master_path"]:
        master = paths.absolute(item["master_path"])
        if master.exists():
            return master
    return None


def build_jobs(
    conn: sqlite3.Connection,
    paths: Any,
    items: Iterable[sqlite3.Row],
    preferred_size: int,
) -> tuple[list[Job], list[tuple[str, str]]]:
    jobs: list[Job] = []
    skipped: list[tuple[str, str]] = []

    photo_cache: dict[str, sqlite3.Row] = {}
    counts: dict[str, int] = {}

    for item in items:
        image = pick_image(conn, paths, item["id"], preferred_size)
        if image is None:
            skipped.append((item["id"], "no image on disk; run `rotary rectify`"))
            continue

        sha = item["photo_sha256"]
        if sha not in photo_cache:
            photo_cache[sha] = db.get_photo(conn, sha)
            counts[sha] = len(db.items_for_photo(conn, sha))
        photo = photo_cache[sha]

        jobs.append(
            Job(
                item_id=item["id"],
                image_path=image,
                context=build_context(
                    captured_at=photo["captured_at"] if photo else None,
                    source_photo=photo["original_name"] if photo else None,
                    neighbours=max(0, counts.get(sha, 1) - 1),
                ),
            )
        )
    return jobs, skipped


def items_to_analyze(
    conn: sqlite3.Connection,
    *,
    force: bool,
    limit: int | None,
    item_ids: Sequence[str] | None = None,
) -> list[sqlite3.Row]:
    """Rectified items without a live analysis, oldest first.

    Rejected items are excluded - there is no reason to pay to read something
    already discarded.

    `item_ids` restricts the selection to named items and implies force, which
    is what the review UI's per-item re-analyse needs: without it, asking to
    re-read one item would sweep up every other unanalysed item in the archive
    and spend real money doing it.
    """
    params: list[Any] = []
    if item_ids is not None:
        if not item_ids:
            return []
        placeholders = ",".join("?" * len(item_ids))
        sql = (
            f"SELECT * FROM items WHERE id IN ({placeholders}) "
            "AND master_path IS NOT NULL ORDER BY id"
        )
        params = list(item_ids)
    elif force:
        sql = (
            "SELECT * FROM items WHERE status != 'rejected' "
            "AND master_path IS NOT NULL ORDER BY id"
        )
    else:
        sql = (
            "SELECT i.* FROM items i "
            "LEFT JOIN current_analyses a ON a.item_id = i.id "
            "WHERE i.status != 'rejected' AND i.master_path IS NOT NULL "
            "AND a.id IS NULL ORDER BY i.id"
        )
    rows = conn.execute(sql, params).fetchall()
    return rows[:limit] if limit else rows


def analyze_items(
    conn: sqlite3.Connection,
    paths: Any,
    provider: VisionProvider,
    llm_config: dict[str, Any],
    *,
    force: bool = False,
    limit: int | None = None,
    item_ids: Sequence[str] | None = None,
    progress: Any = None,
) -> AnalyzeSummary:
    summary = AnalyzeSummary()

    items = items_to_analyze(conn, force=force, limit=limit, item_ids=item_ids)
    jobs, skipped = build_jobs(
        conn, paths, items, int(llm_config.get("analyze_image_size", 1600))
    )
    for item_id, reason in skipped:
        summary.failed += 1
        summary.errors.append((item_id, reason))

    if not jobs:
        return summary

    summary.attempted = len(jobs)
    schema = ITEM_SCHEMA
    max_concurrency = int(llm_config.get("max_concurrency", 4))

    for result in provider.analyze_many(
        jobs, SYSTEM_PROMPT, schema,
        max_concurrency=max_concurrency, progress=progress,
    ):
        # Providers emit status-only results while a batch is in flight; those
        # carry no item and must not be counted as analyses.
        if not result.item_id:
            continue

        if not result.ok:
            summary.failed += 1
            summary.errors.append((result.item_id, result.error or "unknown error"))
            continue

        clean = normalise(result.data)
        reasons = consistency_flags(clean)
        store_analysis(conn, paths, result.item_id, result, clean, reasons)

        summary.succeeded += 1
        if clean["needs_human_review"] or reasons:
            summary.flagged += 1

    return summary
