"""Build the published archive from SQLite.

The site is a pure export: everything it shows comes from the database, so it
can be rebuilt from scratch at any time and never becomes a second source of
truth that can drift.

Two decisions shape the output:

  * **Data ships as a JavaScript assignment, not a JSON file.** Browsers block
    `fetch()` against `file://`, so a JSON file would make the site work when
    served and silently fail when opened from disk. A `window.ARCHIVE = {...}`
    assignment works in both, which matters for checking a build before it goes
    anywhere near the club's host.
  * **Only web derivatives are copied.** Masters stay on Ken's disk. They are
    the irreplaceable copy and there is no reason to put them on a web server.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .. import db

# Items in these states are never published: one was thrown away, the others
# have not been read yet and would appear as untitled blanks.
UNPUBLISHABLE = {"rejected", "detected", "rectified"}

ASSET_NAMES = ("style.css", "app.js")


@dataclass
class BuildSummary:
    items: int = 0
    unapproved: int = 0
    media_files: int = 0
    media_bytes: int = 0
    entities: int = 0
    decades: list[str] = field(default_factory=list)
    output: Path | None = None

    @property
    def media_mb(self) -> float:
        return round(self.media_bytes / (1024 * 1024), 1)


# ------------------------------------------------------------------ dates ---


def decade_of(date_value: str) -> str:
    """'1962-07-14' -> '1960s'. Empty string when there is no usable year."""
    if not date_value or len(date_value) < 4 or not date_value[:4].isdigit():
        return ""
    return f"{int(date_value[:4]) // 10 * 10}s"


def year_of(date_value: str) -> str:
    if not date_value or len(date_value) < 4 or not date_value[:4].isdigit():
        return ""
    return date_value[:4]


def display_date(date_value: str, precision: str) -> str:
    """Render a date at no more precision than the evidence supports.

    Showing '1962-01-01' for something only known to be from 1962 would invent
    a specificity the archive does not have.
    """
    if not date_value:
        return ""
    parts = date_value.split("-")
    year = parts[0]

    if precision == "decade":
        return f"{int(year) // 10 * 10}s" if year.isdigit() else year
    if precision == "year" or len(parts) == 1:
        return year
    months = (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    )
    try:
        month = months[int(parts[1]) - 1]
    except (ValueError, IndexError):
        return year
    if precision == "month" or len(parts) == 2:
        return f"{month} {year}"
    try:
        return f"{int(parts[2])} {month} {year}"
    except ValueError:
        return f"{month} {year}"


# ------------------------------------------------------------------ export ---


def publishable_items(
    conn: sqlite3.Connection, *, approved_only: bool
) -> list[sqlite3.Row]:
    """Items with a live analysis that are fit to publish."""
    if approved_only:
        clause = "i.status = 'approved'"
    else:
        placeholders = ",".join("?" * len(UNPUBLISHABLE))
        clause = f"i.status NOT IN ({placeholders})"

    params = [] if approved_only else sorted(UNPUBLISHABLE)
    return conn.execute(
        f"""
        SELECT i.*, a.item_type, a.title, a.summary, a.full_text,
               a.date_value, a.date_precision, a.date_source, a.date_note,
               a.presentation, a.legibility, a.condition_notes, a.alt_text,
               a.visual_description,
               a.rotary_context, a.confidence AS analysis_confidence
          FROM items i
          JOIN current_analyses a ON a.item_id = i.id
         WHERE {clause}
         ORDER BY COALESCE(NULLIF(a.date_value, ''), '9999'), i.id
        """,
        params,
    ).fetchall()


def entity_map(conn: sqlite3.Connection) -> dict[str, list[dict[str, str]]]:
    """item_id -> its entities, as {kind, name, slug}."""
    rows = conn.execute(
        "SELECT ie.item_id, e.kind, e.name, e.slug FROM item_entities ie "
        "JOIN entities e ON e.id = ie.entity_id ORDER BY e.kind, e.name"
    ).fetchall()
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["item_id"], []).append(
            {"kind": row["kind"], "name": row["name"], "slug": row["slug"]}
        )
    return grouped


def derivative_map(conn: sqlite3.Connection) -> dict[str, dict[int, sqlite3.Row]]:
    rows = conn.execute("SELECT * FROM derivatives").fetchall()
    grouped: dict[str, dict[int, sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(row["item_id"], {})[int(row["long_edge"])] = row
    return grouped


def build_records(
    conn: sqlite3.Connection,
    items: Iterable[sqlite3.Row],
    entities: dict[str, list[dict[str, str]]],
    derivatives: dict[str, dict[int, sqlite3.Row]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for item in items:
        sizes = derivatives.get(item["id"], {})
        item_entities = entities.get(item["id"], [])

        def names(kind: str) -> list[dict[str, str]]:
            return [
                {"name": e["name"], "slug": e["slug"]}
                for e in item_entities
                if e["kind"] == kind
            ]

        records.append(
            {
                "id": item["id"],
                "type": item["item_type"] or "other",
                "title": item["title"] or "Untitled",
                "summary": item["summary"] or "",
                "text": item["full_text"] or "",
                "date": item["date_value"] or "",
                "date_display": display_date(
                    item["date_value"] or "", item["date_precision"] or "unknown"
                ),
                # Carried through so the site can show a deduced date
                # differently from one printed on the item. Flattening the two
                # would let a guess read as a fact.
                "date_source": item["date_source"] or "unknown",
                "date_note": item["date_note"] or "",
                "decade": decade_of(item["date_value"] or ""),
                "year": year_of(item["date_value"] or ""),
                "presentation": item["presentation"] or "image",
                "alt": item["alt_text"] or item["title"] or "Archive item",
                # Search leans on this for photographs, which have no
                # transcription to match against.
                "visual": item["visual_description"] or "",
                "condition": item["condition_notes"] or "",
                "rotary": item["rotary_context"] or "",
                "legibility": item["legibility"],
                "people": names("person"),
                "orgs": names("organization"),
                "places": names("place"),
                "topics": names("topic"),
                "sizes": sorted(sizes),
                "w": item["master_width"],
                "h": item["master_height"],
                # Which item this one belongs with, if any. Resolved into
                # `pages` by fold_groups before the site is written.
                "part_of": item["part_of_item_id"] or "",
                "part_reason": item["part_reason"] or "",
                "duplicate_of": item["duplicate_of_item_id"] or "",
                "pages": [],
                # Filled by attach_related once the surviving set is known.
                "related": [],
            }
        )
    return records


def fold_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Publish an article carried across several clippings as one entry.

    A story continued on a second strip, or a photograph cut out alongside the
    article it illustrates, is one thing to a reader and should read as one
    thing on the site - with the extra scans as further pages, and their text
    folded into the searchable body so a phrase from the second column still
    finds the article.

    A child whose parent did not make it into this build is promoted back to a
    top-level entry rather than dropped. That case is ordinary, not
    exceptional: parent and child are approved separately, and losing a
    clipping because its neighbour was still pending would be silent.
    """
    # Duplicates drop out first, before any page is folded in. Doing it the
    # other way round loses data: a page folded into a copy that is then
    # discarded goes with it. A page whose parent was a duplicate is instead
    # orphaned here and promoted below, which is recoverable.
    #
    # They drop at build time rather than at query time so the timeline, the
    # search index and the entity counts all agree with each other.
    everything = {record["id"]: record for record in records}
    records = [
        record for record in records
        if not (
            record["duplicate_of"]
            and record["duplicate_of"] != record["id"]
            and record["duplicate_of"] in everything
        )
    ]

    by_id = {record["id"]: record for record in records}
    absorbed: set[str] = set()

    for record in records:
        parent_id = record["part_of"]
        parent = by_id.get(parent_id) if parent_id != record["id"] else None
        if parent is None:
            continue

        absorbed.add(record["id"])
        parent["pages"].append(
            {
                "id": record["id"],
                "alt": record["alt"],
                "sizes": record["sizes"],
                "w": record["w"],
                "h": record["h"],
                "note": record["part_reason"],
            }
        )
        if record["text"]:
            parent["text"] = f"{parent['text']}\n\n{record['text']}".strip()

    return [record for record in records if record["id"] not in absorbed]


def attach_related(
    records: list[dict[str, Any]], links: list[sqlite3.Row]
) -> list[dict[str, Any]]:
    """Hang each record's related items off it, for the site to cross-link.

    This is the relationship that does *not* merge. Both items keep their own
    entry, their own title and their own date; the link only means a reader
    looking at one should be able to find the other. Links pointing at an item
    that did not make it into this build - still pending, rejected, or folded
    in as a page - are dropped rather than published as dead ends.
    """
    published = {record["id"]: record for record in records}

    for link in links:
        source = published.get(link["item_id"])
        target = published.get(link["related_item_id"])
        if source is None or target is None or source is target:
            continue
        source["related"].append(
            {
                "id": target["id"],
                "title": target["title"],
                "type": target["type"],
                "date_display": target["date_display"],
                "reason": link["reason"] or "",
                "sizes": target["sizes"],
                "alt": target["alt"],
            }
        )

    for record in records:
        record["related"].sort(key=lambda entry: (entry["date_display"], entry["title"]))
    return records


def build_entity_index(records: list[dict[str, Any]]) -> dict[str, list[dict]]:
    """Every distinct entity with the items that mention it.

    This is what turns "everything mentioning Harold Pratt" into a lookup
    rather than a scan, and it is why the site can offer a person index at all.
    """
    index: dict[str, dict[str, dict[str, Any]]] = {
        "person": {}, "organization": {}, "place": {}, "topic": {}
    }
    field_to_kind = {
        "people": "person", "orgs": "organization",
        "places": "place", "topics": "topic",
    }

    for record in records:
        for field_name, kind in field_to_kind.items():
            for entity in record[field_name]:
                bucket = index[kind].setdefault(
                    entity["slug"],
                    {"name": entity["name"], "slug": entity["slug"], "items": []},
                )
                bucket["items"].append(record["id"])

    return {
        kind: sorted(
            entries.values(),
            key=lambda e: (-len(e["items"]), e["name"].lower()),
        )
        for kind, entries in index.items()
    }


def build_timeline(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records grouped decade -> year, oldest first, undated last.

    Undated material is kept rather than hidden: a club archive has plenty of
    it, and burying it would make the timeline look more complete than it is.
    """
    decades: dict[str, dict[str, list[str]]] = {}
    for record in records:
        decade = record["decade"] or "Undated"
        year = record["year"] or "Undated"
        decades.setdefault(decade, {}).setdefault(year, []).append(record["id"])

    def decade_key(name: str) -> tuple[int, str]:
        return (9999, name) if name == "Undated" else (int(name[:-1]), name)

    return [
        {
            "decade": decade,
            "count": sum(len(ids) for ids in years.values()),
            "years": [
                {"year": year, "items": years[year]}
                for year in sorted(years, key=lambda y: (y == "Undated", y))
            ],
        }
        for decade, years in sorted(decades.items(), key=lambda kv: decade_key(kv[0]))
    ]


# ------------------------------------------------------------------ media ---


def copy_media(
    paths: Any,
    records: list[dict[str, Any]],
    derivatives: dict[str, dict[int, sqlite3.Row]],
    out_dir: Path,
) -> tuple[int, int]:
    """Copy web derivatives into the site. Masters are never published."""
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = total_bytes = 0

    # A record's own scan plus any further pages folded into it. The pages
    # are no longer top-level records, so iterating records alone would ship
    # a page whose <img> points at a file that was never copied.
    wanted = [
        (entry["id"], entry["sizes"])
        for record in records
        for entry in (record, *record.get("pages", []))
    ]

    for item_id, sizes in wanted:
        for size in sizes:
            row = derivatives[item_id][size]
            source = paths.absolute(row["path"])
            if not source.exists():
                continue
            target = out_dir / source.name
            shutil.copyfile(source, target)
            copied += 1
            total_bytes += target.stat().st_size

    return copied, total_bytes


# ------------------------------------------------------------------ render ---


def _asset(name: str) -> str:
    from importlib import resources

    return (
        resources.files("rotary_archive.build").joinpath("assets", name).read_text()
    )


def _template(name: str) -> str:
    from importlib import resources

    return (
        resources.files("rotary_archive.build").joinpath("templates", name).read_text()
    )


def build_site(
    conn: sqlite3.Connection,
    paths: Any,
    site_config: dict[str, Any],
    *,
    approved_only: bool = False,
    clean: bool = True,
) -> BuildSummary:
    out = paths.site
    if clean and out.exists():
        # Only ever remove directories this builder owns. A configured `site`
        # pointing somewhere unexpected should not take neighbouring files
        # with it.
        for child in ("assets", "data", "media"):
            shutil.rmtree(out / child, ignore_errors=True)
        for stale in out.glob("*.html"):
            stale.unlink()
    out.mkdir(parents=True, exist_ok=True)

    items = publishable_items(conn, approved_only=approved_only)
    entities = entity_map(conn)
    derivatives = derivative_map(conn)
    records = fold_groups(build_records(conn, items, entities, derivatives))
    records = attach_related(records, db.all_item_links(conn))

    entity_index = build_entity_index(records)
    timeline = build_timeline(records)

    unapproved = sum(1 for item in items if item["status"] != "approved")

    payload = {
        "generated_at": db.utcnow(),
        "club": site_config.get("club_name", "Rotary Club"),
        "tagline": site_config.get("tagline", ""),
        "contact": site_config.get("contact", ""),
        "sizes": {
            "detail": int(site_config.get("detail_image_size", 1600)),
            "card": int(site_config.get("card_image_size", 800)),
            "thumb": int(site_config.get("thumb_image_size", 320)),
        },
        "counts": {
            "items": len(records),
            "people": len(entity_index["person"]),
            "places": len(entity_index["place"]),
            "organizations": len(entity_index["organization"]),
            "topics": len(entity_index["topic"]),
            "dated": sum(1 for r in records if r["date"]),
        },
        "items": records,
        "entities": entity_index,
        "timeline": timeline,
    }

    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data" / "archive.js").write_text(
        "window.ARCHIVE = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )

    (out / "assets").mkdir(parents=True, exist_ok=True)
    for name in ASSET_NAMES:
        (out / "assets" / name).write_text(_asset(name), encoding="utf-8")

    club = site_config.get("club_name", "Rotary Club")
    tagline = site_config.get("tagline", "")
    (out / "index.html").write_text(
        _template("index.html")
        .replace("{{CLUB}}", club)
        .replace("{{TAGLINE}}", tagline),
        encoding="utf-8",
    )
    (out / "embed.html").write_text(
        _template("embed.html").replace("{{CLUB}}", club), encoding="utf-8"
    )

    media_files, media_bytes = copy_media(
        paths, records, derivatives, out / "media"
    )

    return BuildSummary(
        items=len(records),
        unapproved=unapproved,
        media_files=media_files,
        media_bytes=media_bytes,
        entities=sum(len(v) for v in entity_index.values()),
        decades=[block["decade"] for block in timeline],
        output=out,
    )
