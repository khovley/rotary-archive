"""SQLite access layer.

The database is the source of truth for the whole pipeline; the published site
is a pure export from it. Every stage records per-row status so any command can
be re-run and will skip work that is already done.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA_VERSION = "1"


def utcnow() -> str:
    """Timestamp string used for every `*_at` column."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _schema_sql() -> str:
    return resources.files("rotary_archive").joinpath("schema.sql").read_text()


def connect(database: Path, *, create: bool = True) -> sqlite3.Connection:
    """Open (and if needed initialise) the archive database."""
    database = Path(database)
    if not database.exists() and not create:
        raise FileNotFoundError(f"No archive database at {database}")
    database.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(database, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_schema_sql())
    _migrate(conn)
    conn.execute(
        "INSERT INTO schema_meta(key, value) VALUES('version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (SCHEMA_VERSION,),
    )
    return conn


# Columns added after the first release. CREATE TABLE IF NOT EXISTS will not
# add them to a database that already exists, so they are applied here.
_ADDED_COLUMNS = {
    "items": [
        ("part_of_item_id", "TEXT REFERENCES items(id) ON DELETE SET NULL"),
        ("part_reason", "TEXT"),
        ("headline", "TEXT"),
    ],
}


# Indexes over columns the migration adds. They cannot live in schema.sql: on a
# database created before the column existed, CREATE TABLE IF NOT EXISTS is a
# no-op, so the script would try to index a column that is not there yet.
_ADDED_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_items_part_of ON items(part_of_item_id)",
]


def _migrate(conn: sqlite3.Connection) -> None:
    for table, columns in _ADDED_COLUMNS.items():
        existing = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
        }
        for name, decl in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    for statement in _ADDED_INDEXES:
        conn.execute(statement)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Explicit transaction. isolation_level=None means we drive BEGIN ourselves."""
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


# ----------------------------------------------------------------- photos ---


def photo_exists(conn: sqlite3.Connection, sha256: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM photos WHERE sha256 = ?", (sha256,)
    ).fetchone()
    return row is not None


def insert_photo(
    conn: sqlite3.Connection,
    *,
    sha256: str,
    original_name: str,
    stored_path: str,
    width: int | None,
    height: int | None,
    captured_at: str | None,
    exif: dict[str, Any] | None,
    size_bytes: int | None,
) -> None:
    conn.execute(
        """
        INSERT INTO photos (sha256, original_name, stored_path, width, height,
                            captured_at, exif_json, bytes, ingested_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ingested')
        """,
        (
            sha256,
            original_name,
            stored_path,
            width,
            height,
            captured_at,
            json.dumps(exif) if exif else None,
            size_bytes,
            utcnow(),
        ),
    )


def photos_with_status(
    conn: sqlite3.Connection, status: str | Sequence[str]
) -> list[sqlite3.Row]:
    statuses = [status] if isinstance(status, str) else list(status)
    placeholders = ",".join("?" * len(statuses))
    return conn.execute(
        f"SELECT * FROM photos WHERE status IN ({placeholders}) "
        "ORDER BY COALESCE(captured_at, ingested_at), sha256",
        statuses,
    ).fetchall()


def all_photos(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM photos ORDER BY COALESCE(captured_at, ingested_at), sha256"
    ).fetchall()


def get_photo(conn: sqlite3.Connection, sha256: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM photos WHERE sha256 = ?", (sha256,)
    ).fetchone()


def set_photo_status(
    conn: sqlite3.Connection, sha256: str, status: str, note: str | None = None
) -> None:
    conn.execute(
        "UPDATE photos SET status = ?, segment_note = COALESCE(?, segment_note) "
        "WHERE sha256 = ?",
        (status, note, sha256),
    )


# ------------------------------------------------------------------ items ---


def make_item_id(photo_sha256: str, seq: int) -> str:
    return f"{photo_sha256[:12]}-{seq:02d}"


def delete_items_for_photo(conn: sqlite3.Connection, photo_sha256: str) -> int:
    """Used when re-segmenting a photo. Cascades to derivatives/analyses."""
    cur = conn.execute("DELETE FROM items WHERE photo_sha256 = ?", (photo_sha256,))
    return cur.rowcount


def insert_item(
    conn: sqlite3.Connection,
    *,
    item_id: str,
    photo_sha256: str,
    seq: int,
    quad: Sequence[Sequence[float]],
    detection_confidence: float,
    detection_method: str,
    needs_human_review: bool = False,
    review_reason: str | None = None,
    headline: str | None = None,
) -> None:
    quad_json = json.dumps([[float(x), float(y)] for x, y in quad])
    now = utcnow()
    conn.execute(
        """
        INSERT INTO items (id, photo_sha256, seq, quad, quad_detected,
                           detection_confidence, detection_method,
                           needs_human_review, review_reason, headline,
                           status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'detected', ?, ?)
        """,
        (
            item_id,
            photo_sha256,
            seq,
            quad_json,
            quad_json,
            float(detection_confidence),
            detection_method,
            int(needs_human_review),
            review_reason,
            headline,
            now,
            now,
        ),
    )


def set_item_part_of(
    conn: sqlite3.Connection, item_id: str, parent_id: str | None, reason: str | None
) -> None:
    """Link an item to the one it continues, or clear the link."""
    conn.execute(
        "UPDATE items SET part_of_item_id = ?, part_reason = ?, updated_at = ? "
        "WHERE id = ?",
        (parent_id, reason, utcnow(), item_id),
    )


def items_for_photo(conn: sqlite3.Connection, photo_sha256: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM items WHERE photo_sha256 = ? ORDER BY seq",
        (photo_sha256,),
    ).fetchall()


def items_with_status(
    conn: sqlite3.Connection, status: str | Sequence[str]
) -> list[sqlite3.Row]:
    statuses = [status] if isinstance(status, str) else list(status)
    placeholders = ",".join("?" * len(statuses))
    return conn.execute(
        f"SELECT * FROM items WHERE status IN ({placeholders}) ORDER BY id",
        statuses,
    ).fetchall()


def get_item(conn: sqlite3.Connection, item_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()


def update_item_quad(
    conn: sqlite3.Connection, item_id: str, quad: Sequence[Sequence[float]]
) -> None:
    """Replace an item's crop.

    Resetting status to 'detected' is deliberate: an approval applies to the
    image that was approved, and a new crop is a different image. The item goes
    back through rectify and needs confirming again. `quad_detected` is left
    untouched so the review UI can always offer "reset to detected".
    """
    conn.execute(
        "UPDATE items SET quad = ?, status = 'detected', updated_at = ? WHERE id = ?",
        (json.dumps([[float(x), float(y)] for x, y in quad]), utcnow(), item_id),
    )


def set_item_rectified(
    conn: sqlite3.Connection,
    item_id: str,
    *,
    master_path: str,
    width: int,
    height: int,
    fine_skew_deg: float,
) -> None:
    conn.execute(
        """
        UPDATE items
           SET master_path = ?, master_width = ?, master_height = ?,
               fine_skew_deg = ?, status = 'rectified', updated_at = ?
         WHERE id = ?
        """,
        (master_path, width, height, fine_skew_deg, utcnow(), item_id),
    )


def set_item_status(
    conn: sqlite3.Connection,
    item_id: str,
    status: str,
    *,
    needs_human_review: bool | None = None,
    review_reason: str | None = None,
) -> None:
    if needs_human_review is None:
        conn.execute(
            "UPDATE items SET status = ?, updated_at = ? WHERE id = ?",
            (status, utcnow(), item_id),
        )
    else:
        conn.execute(
            "UPDATE items SET status = ?, needs_human_review = ?, "
            "review_reason = ?, updated_at = ? WHERE id = ?",
            (status, int(needs_human_review), review_reason, utcnow(), item_id),
        )


def next_item_seq(conn: sqlite3.Connection, photo_sha256: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), -1) + 1 AS n FROM items WHERE photo_sha256 = ?",
        (photo_sha256,),
    ).fetchone()
    return int(row["n"])


# ------------------------------------------------------------ derivatives ---


def replace_derivatives(
    conn: sqlite3.Connection, item_id: str, rows: Iterable[dict[str, Any]]
) -> None:
    conn.execute("DELETE FROM derivatives WHERE item_id = ?", (item_id,))
    conn.executemany(
        "INSERT INTO derivatives (item_id, long_edge, path, width, height, bytes) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                item_id,
                r["long_edge"],
                r["path"],
                r["width"],
                r["height"],
                r.get("bytes"),
            )
            for r in rows
        ],
    )


def derivatives_for_item(conn: sqlite3.Connection, item_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM derivatives WHERE item_id = ? ORDER BY long_edge DESC",
        (item_id,),
    ).fetchall()


# -------------------------------------------------------------- analyses ---


def supersede_analyses(conn: sqlite3.Connection, item_id: str) -> None:
    conn.execute(
        "UPDATE analyses SET superseded = 1 WHERE item_id = ? AND superseded = 0",
        (item_id,),
    )


def current_analysis(conn: sqlite3.Connection, item_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM current_analyses WHERE item_id = ?", (item_id,)
    ).fetchone()


# -------------------------------------------------------------- review log ---


def log_review(
    conn: sqlite3.Connection,
    *,
    item_id: str | None,
    action: str,
    detail: str | None = None,
    actor: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO review_log (item_id, action, detail, actor, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (item_id, action, detail, actor, utcnow()),
    )


# ------------------------------------------------------------------ stats ---


def counts(conn: sqlite3.Connection) -> dict[str, Any]:
    def scalar(sql: str, params: Sequence[Any] = ()) -> int:
        row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    photo_status = {
        r["status"]: r["n"]
        for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM photos GROUP BY status"
        )
    }
    item_status = {
        r["status"]: r["n"]
        for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM items GROUP BY status"
        )
    }
    return {
        "photos": scalar("SELECT COUNT(*) FROM photos"),
        "photos_by_status": photo_status,
        "items": scalar("SELECT COUNT(*) FROM items"),
        "items_by_status": item_status,
        "flagged": scalar(
            "SELECT COUNT(*) FROM items WHERE needs_human_review = 1 "
            "AND status NOT IN ('approved', 'rejected')"
        ),
        "analyses": scalar("SELECT COUNT(*) FROM analyses WHERE superseded = 0"),
    }
