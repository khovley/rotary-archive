"""Local review server.

Deliberately stdlib-only: no Flask, no FastAPI. This binds to localhost, is
launched by the user, and serves one person at a time - a framework would add
dependencies without buying anything.

Routes:
    GET  /                        the UI
    GET  /static/<file>           UI assets
    GET  /api/photos              photo list with per-photo item summaries
    GET  /api/photo/<sha>         one photo, its items, and their quads
    GET  /media/photo/<sha>       source photo, downscaled for the browser
    GET  /media/item/<id>         item derivative
    POST /api/items/decide        approve or reject a batch of items
    POST /api/item/<id>/quad      replace a crop and re-rectify it
    POST /api/item/<id>/rotate    apply a quarter turn and re-rectify
    POST /api/photo/<sha>/item    add an item a human drew by hand
    POST /api/item/<id>/delete    remove a spurious detection
    GET  /api/inbox               how many new photos are waiting, and where
    POST /api/inbox/open          reveal the inbox folder in the file manager
    POST /api/inbox/upload        write one uploaded photo into the inbox
    GET  /api/process             progress of the running pipeline job
    POST /api/process             start ingest -> segment -> rectify
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from .. import db
from ..config import Config

# Serving the full 12MP source to the browser for every photo makes the UI
# crawl; this is plenty to draw crop overlays against.
PREVIEW_LONG_EDGE = 1400

_preview_cache: dict[str, bytes] = {}
_cache_lock = threading.Lock()

# Photo suffixes the inbox accepts on upload. Deliberately the same set ingest
# understands, so a file that lands here is one the pipeline can actually read.
UPLOAD_SUFFIXES = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024   # a 200MB single photo is already absurd


class PipelineJob:
    """Runs ingest -> segment -> rectify on a background thread.

    One job at a time, by design. The stages write to the same SQLite database
    and the same image directories; two concurrent runs would race over both.
    The UI polls `snapshot()` rather than holding a connection open, so a
    closed browser tab cannot abandon a half-finished run.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.reset()

    def reset(self) -> None:
        self.state = "idle"        # idle | running | done | error
        self.stage = ""
        self.message = ""
        self.error: str | None = None
        self.counts: dict[str, int] = {}
        self.started_at: float | None = None
        self.finished_at: float | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = None
            if self.started_at:
                end = self.finished_at or time.monotonic()
                elapsed = round(end - self.started_at, 1)
            return {
                "state": self.state,
                "stage": self.stage,
                "message": self.message,
                "error": self.error,
                "counts": dict(self.counts),
                "elapsed": elapsed,
            }

    def _set(self, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, value)

    def start(self, cfg: Config) -> bool:
        if self.running:
            return False
        with self._lock:
            self.reset()
            self.state = "running"
            self.stage = "starting"
            self.started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, args=(cfg,), daemon=True
        )
        self._thread.start()
        return True

    def _run(self, cfg: Config) -> None:
        from ..ingest import ingest_inbox
        from ..rectify import rectify_pending
        from ..segment import segment_pending

        conn = db.connect(cfg.paths.database)
        try:
            self._set(stage="ingest", message="Reading photos from the inbox...")
            ingested = ingest_inbox(
                conn, cfg.paths,
                progress=lambda p: self._set(message=f"Reading {p.name}"),
            )
            self._set(counts={**self.counts,
                              "ingested": len(ingested.ingested),
                              "duplicates": len(ingested.skipped_duplicate),
                              "unreadable": len(ingested.unreadable)})

            self._set(stage="segment", message="Finding items in each photo...")
            flag_below = float(cfg.review.get("flag_below_confidence", 0.80))
            results = segment_pending(
                conn, cfg.paths, cfg.segment, flag_below=flag_below,
                progress=lambda p: self._set(
                    message=f"Finding items in {p['original_name']}"
                ),
            )
            found = sum(len(r.candidates) for r in results)
            self._set(counts={**self.counts, "items": found})

            self._set(stage="rectify", message="Cropping and straightening...")
            rectified = rectify_pending(
                conn, cfg.paths, cfg.rectify,
                progress=lambda i: self._set(message=f"Cropping {i['id']}"),
            )
            self._set(counts={**self.counts, "rectified": len(rectified)})

            stats = db.counts(conn)
            self._set(
                state="done", stage="done", message="Finished.",
                finished_at=time.monotonic(),
                counts={**self.counts, "flagged": stats["flagged"]},
            )

            notices = [f"{p.name}: {why}" for p, why in ingested.unreadable]
            if ingested.low_resolution:
                names = ", ".join(n for n, _ in ingested.low_resolution[:3])
                notices.append(
                    f"{len(ingested.low_resolution)} photo(s) are low-resolution "
                    f"({names}) - these look like photo-library exports. Add the "
                    "unmodified originals instead; there is far more detail in them."
                )
            if notices:
                self._set(message="Finished, with something worth reading.",
                          error="; ".join(notices[:5]))
        except Exception as exc:
            self._set(
                state="error", stage="error", error=f"{type(exc).__name__}: {exc}",
                message="The run stopped early.", finished_at=time.monotonic(),
            )
        finally:
            conn.close()


_job = PipelineJob()


def inbox_state(cfg: Config) -> dict[str, Any]:
    """What is sitting in the inbox, and how much of it is new.

    "Waiting" counts only photos not already in the archive. Without --move,
    ingested files stay in the inbox, and reporting those as waiting would nag
    forever. Hashing is the only honest way to tell, so this is capped: past a
    few hundred files the count stops being worth the disk read and the UI
    just says "many".
    """
    from ..ingest import find_photos, sha256_file

    present = find_photos(cfg.paths.inbox)
    result: dict[str, Any] = {
        "path": str(cfg.paths.inbox),
        "present": len(present),
        "waiting": 0,
        "exact": True,
    }
    if not present:
        return result

    if len(present) > 400:
        result["waiting"] = len(present)
        result["exact"] = False
        return result

    conn = db.connect(cfg.paths.database, create=False)
    try:
        result["waiting"] = sum(
            1 for path in present if not db.photo_exists(conn, sha256_file(path))
        )
    finally:
        conn.close()
    return result


def safe_upload_name(raw: str) -> str:
    """A filename that cannot escape the inbox.

    Uploads arrive over localhost from our own page, but the name is still
    attacker-shaped input: it comes from the filesystem of whatever was
    dragged in. Taking only the basename kills traversal, and the suffix
    allowlist keeps the inbox to files ingest can actually read.
    """
    # Backslashes are not separators on POSIX, so a name carrying a Windows
    # path would survive Path().name intact. Normalising them first means a
    # file dragged from a Windows share is treated the same everywhere.
    name = Path(raw.replace("\\", "/")).name.strip()
    if not name or name in (".", "..") or name.startswith("."):
        raise ValueError(f"unusable filename: {raw!r}")
    if Path(name).suffix.lower() not in UPLOAD_SUFFIXES:
        raise ValueError(
            f"{name}: not a photo this archive reads "
            f"({', '.join(sorted(UPLOAD_SUFFIXES))})"
        )
    return name


def _guess_media_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".gif": "image/gif", ".tif": "image/tiff",
        ".tiff": "image/tiff", ".heic": "image/heic", ".heif": "image/heif",
    }.get(path.suffix.lower(), "application/octet-stream")


def _static_file(name: str) -> bytes | None:
    try:
        return (
            resources.files("rotary_archive.review")
            .joinpath("static", name)
            .read_bytes()
        )
    except (FileNotFoundError, NotADirectoryError, ModuleNotFoundError):
        return None


def _photo_preview(cfg: Config, photo: sqlite3.Row) -> bytes:
    """JPEG preview of a source photo, cached in memory per run."""
    sha = photo["sha256"]
    with _cache_lock:
        if sha in _preview_cache:
            return _preview_cache[sha]

    from ..ingest import load_oriented

    path = cfg.paths.absolute(photo["stored_path"])
    with load_oriented(path) as pil_img:
        rgb = np.asarray(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    if max(h, w) > PREVIEW_LONG_EDGE:
        factor = PREVIEW_LONG_EDGE / max(h, w)
        bgr = cv2.resize(
            bgr, (max(1, int(w * factor)), max(1, int(h * factor))),
            interpolation=cv2.INTER_AREA,
        )

    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    data = buf.tobytes() if ok else b""
    with _cache_lock:
        _preview_cache[sha] = data
    return data


ANALYSIS_FIELDS = (
    "item_type", "title", "summary", "full_text",
    "date_value", "date_precision", "date_source", "date_note",
    "presentation", "legibility", "condition_notes", "alt_text",
    "rotary_context", "orientation_hint", "confidence",
)

# Fields a human may correct in the review UI. Everything else is the model's
# reading and is left alone.
EDITABLE_FIELDS = frozenset(
    {
        "item_type", "title", "summary", "full_text",
        "date_value", "date_precision", "date_source",
        "presentation", "condition_notes", "alt_text", "rotary_context",
    }
)


def _entities_for(conn: sqlite3.Connection, item_id: str) -> dict[str, list[str]]:
    rows = conn.execute(
        "SELECT e.kind, e.name FROM item_entities ie "
        "JOIN entities e ON e.id = ie.entity_id "
        "WHERE ie.item_id = ? ORDER BY e.kind, e.name",
        (item_id,),
    ).fetchall()
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(row["kind"], []).append(row["name"])
    return grouped


def _item_payload(conn: sqlite3.Connection, cfg: Config, item: sqlite3.Row) -> dict:
    analysis = db.current_analysis(conn, item["id"])
    payload = {
        "id": item["id"],
        "seq": item["seq"],
        "photo": item["photo_sha256"],
        "quad": json.loads(item["quad"]),
        "quad_detected": json.loads(item["quad_detected"]),
        "confidence": item["detection_confidence"],
        "method": item["detection_method"],
        "status": item["status"],
        "flagged": bool(item["needs_human_review"]),
        "reason": item["review_reason"],
        "rotation": item["rotation_applied"],
        "skew": item["fine_skew_deg"],
        "width": item["master_width"],
        "height": item["master_height"],
        "has_master": bool(item["master_path"]),
        # What the segmentation pass read off the item, and whether it judged
        # this piece to belong with another one. Shown in review so a wrong
        # grouping can be spotted before it reaches the site.
        "headline": item["headline"],
        "part_of": item["part_of_item_id"],
        "part_reason": item["part_reason"],
        "analysis": None,
    }

    if analysis is not None:
        payload["analysis"] = {
            **{field: analysis[field] for field in ANALYSIS_FIELDS},
            "provider": analysis["provider"],
            "model": analysis["model"],
            "created_at": analysis["created_at"],
            "entities": _entities_for(conn, item["id"]),
        }

    return payload


def _photo_payload(conn: sqlite3.Connection, cfg: Config, photo: sqlite3.Row) -> dict:
    items = db.items_for_photo(conn, photo["sha256"])
    return {
        "sha256": photo["sha256"],
        "name": photo["original_name"],
        "width": photo["width"],
        "height": photo["height"],
        "captured_at": photo["captured_at"],
        "status": photo["status"],
        "note": photo["segment_note"],
        "items": [_item_payload(conn, cfg, i) for i in items],
    }


class Handler(BaseHTTPRequestHandler):
    cfg: Config = None  # type: ignore[assignment]

    server_version = "RotaryArchiveReview/0.1"

    # ------------------------------------------------------------ plumbing --

    def log_message(self, fmt: str, *args: Any) -> None:
        # The default handler spams stdout with one line per asset request.
        return

    def _send(
        self, status: int, body: bytes, content_type: str, *, cache: bool = False
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Cache-Control", "public, max-age=3600" if cache else "no-store"
        )
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status=status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def _conn(self) -> sqlite3.Connection:
        return db.connect(self.cfg.paths.database, create=False)

    def _wants_full(self) -> bool:
        """True when the caller asked for the original rather than a preview."""
        from urllib.parse import parse_qs

        return "1" in parse_qs(urlparse(self.path).query).get("full", [])

    # ---------------------------------------------------------------- GET --

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            self._route_get(path)
        except FileNotFoundError as exc:
            self._error(404, str(exc))
        except Exception as exc:  # keep the server alive across UI bugs
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _route_get(self, path: str) -> None:
        if path in ("/", "/index.html"):
            body = _static_file("index.html")
            if body is None:
                raise FileNotFoundError("index.html missing from the package")
            return self._send(200, body, "text/html; charset=utf-8")

        if path.startswith("/static/"):
            name = path[len("/static/") :]
            if "/" in name or ".." in name:
                return self._error(400, "bad asset path")
            body = _static_file(name)
            if body is None:
                raise FileNotFoundError(name)
            kind = {
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
            }.get(Path(name).suffix, "application/octet-stream")
            return self._send(200, body, kind)

        if path == "/api/inbox":
            return self._json(inbox_state(self.cfg))

        if path == "/api/process":
            return self._json(_job.snapshot())

        if path == "/api/photos":
            conn = self._conn()
            try:
                photos = [
                    _photo_payload(conn, self.cfg, p) for p in db.all_photos(conn)
                ]
                stats = db.counts(conn)
            finally:
                conn.close()
            return self._json(
                {
                    "photos": photos,
                    "stats": stats,
                    "flag_below": float(
                        self.cfg.review.get("flag_below_confidence", 0.80)
                    ),
                }
            )

        if path.startswith("/api/photo/"):
            sha = path[len("/api/photo/") :]
            conn = self._conn()
            try:
                photo = db.get_photo(conn, sha)
                if photo is None:
                    raise FileNotFoundError(sha)
                return self._json(_photo_payload(conn, self.cfg, photo))
            finally:
                conn.close()

        if path.startswith("/media/photo/"):
            sha = path[len("/media/photo/") :]
            conn = self._conn()
            try:
                photo = db.get_photo(conn, sha)
                if photo is None:
                    raise FileNotFoundError(sha)
                # The grid uses a 1400px preview; the lightbox asks for the
                # original, because the whole point of opening it is to see
                # detail the preview threw away.
                if self._wants_full():
                    source = self.cfg.paths.absolute(photo["stored_path"])
                    if source.exists():
                        return self._send(
                            200, source.read_bytes(),
                            _guess_media_type(source), cache=True,
                        )
                data = _photo_preview(self.cfg, photo)
            finally:
                conn.close()
            return self._send(200, data, "image/jpeg", cache=True)

        if path.startswith("/media/item/"):
            item_id = path[len("/media/item/") :]
            conn = self._conn()
            try:
                derivatives = db.derivatives_for_item(conn, item_id)
                item = db.get_item(conn, item_id)
            finally:
                conn.close()

            # The lightbox wants the archival master; the grid wants something
            # small. Serving the master to a grid of 200 thumbnails would move
            # hundreds of megabytes for no visible gain.
            if self._wants_full() and item is not None and item["master_path"]:
                master = self.cfg.paths.absolute(item["master_path"])
                if master.exists():
                    return self._send(
                        200, master.read_bytes(),
                        _guess_media_type(master), cache=True,
                    )

            # Prefer the 800px derivative for the review grid; fall back through
            # whatever exists, then to the master.
            chosen: Path | None = None
            media_type = "image/webp"
            for want in (800, 1600, 320):
                match = next(
                    (d for d in derivatives if d["long_edge"] == want), None
                )
                if match:
                    chosen = self.cfg.paths.absolute(match["path"])
                    break
            if chosen is None and item is not None and item["master_path"]:
                chosen = self.cfg.paths.absolute(item["master_path"])
                media_type = "image/jpeg"
            if chosen is None or not chosen.exists():
                raise FileNotFoundError(item_id)
            return self._send(200, chosen.read_bytes(), media_type, cache=True)

        raise FileNotFoundError(path)

    # --------------------------------------------------------------- POST --

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            self._route_post(path)
        except FileNotFoundError as exc:
            self._error(404, str(exc))
        except ValueError as exc:
            self._error(400, str(exc))
        except Exception as exc:
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _route_post(self, path: str) -> None:
        # These three touch the filesystem or a background job rather than the
        # database, so they are handled before a connection is opened.
        if path == "/api/inbox/upload":
            return self._upload()
        if path == "/api/inbox/open":
            return self._open_inbox()
        if path == "/api/process":
            return self._start_process()

        body = self._body()
        conn = self._conn()
        try:
            if path == "/api/items/decide":
                return self._decide(conn, body)
            if path.startswith("/api/item/") and path.endswith("/quad"):
                return self._set_quad(conn, path.split("/")[3], body)
            if path.startswith("/api/item/") and path.endswith("/rotate"):
                return self._rotate(conn, path.split("/")[3], body)
            if path.startswith("/api/item/") and path.endswith("/delete"):
                return self._delete_item(conn, path.split("/")[3])
            if path.startswith("/api/item/") and path.endswith("/fields"):
                return self._edit_fields(conn, path.split("/")[3], body)
            if path.startswith("/api/item/") and path.endswith("/group"):
                return self._set_group(conn, path.split("/")[3], body)
            if path.startswith("/api/item/") and path.endswith("/reanalyze"):
                return self._reanalyze(conn, path.split("/")[3])
            if path.startswith("/api/photo/") and path.endswith("/item"):
                return self._add_item(conn, path.split("/")[3], body)
            raise FileNotFoundError(path)
        finally:
            conn.close()

    # ------------------------------------------------------------- actions --

    # ------------------------------------------------------------- inbox --

    def _upload(self) -> None:
        """Write one uploaded photo into the inbox.

        One file per request with the name in the query string, rather than a
        multipart form. That avoids hand-rolling a multipart parser, keeps
        memory to a single photo, and gives the page per-file progress for
        free when it uploads a batch.
        """
        from urllib.parse import parse_qs

        query = parse_qs(urlparse(self.path).query)
        raw_name = (query.get("name") or [""])[0]
        try:
            name = safe_upload_name(raw_name)
        except ValueError as exc:
            raise ValueError(str(exc))

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError(f"{name}: empty upload")
        if length > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"{name}: {length // (1024*1024)}MB exceeds the "
                f"{MAX_UPLOAD_BYTES // (1024*1024)}MB limit"
            )

        self.cfg.paths.inbox.mkdir(parents=True, exist_ok=True)
        target = self.cfg.paths.inbox / name

        # Never overwrite: two photos can legitimately share a camera filename,
        # and losing one silently would be worse than an odd name.
        if target.exists():
            stem, suffix = target.stem, target.suffix
            n = 2
            while target.exists():
                target = self.cfg.paths.inbox / f"{stem}-{n}{suffix}"
                n += 1

        # Streamed in chunks so a 50MB HEIC never sits in memory twice.
        remaining = length
        with target.open("wb") as fh:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)

        if remaining > 0:
            target.unlink(missing_ok=True)
            raise ValueError(f"{name}: upload ended early")

        self._json({"ok": True, "name": target.name, "bytes": length})

    def _open_inbox(self) -> None:
        """Reveal the inbox in the desktop file manager."""
        self.cfg.paths.inbox.mkdir(parents=True, exist_ok=True)
        opener = (
            ["open", str(self.cfg.paths.inbox)] if sys.platform == "darwin"
            else ["explorer", str(self.cfg.paths.inbox)] if sys.platform == "win32"
            else ["xdg-open", str(self.cfg.paths.inbox)]
        )
        if shutil.which(opener[0]) is None and sys.platform != "win32":
            # No file manager to call; the path is shown in the UI regardless.
            return self._json({"ok": False, "path": str(self.cfg.paths.inbox)})
        subprocess.Popen(
            opener, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._json({"ok": True, "path": str(self.cfg.paths.inbox)})

    def _start_process(self) -> None:
        if not _job.start(self.cfg):
            return self._error(409, "A run is already in progress.")
        self._json({"ok": True, **_job.snapshot()})

    # ------------------------------------------------------------ actions --

    def _decide(self, conn: sqlite3.Connection, body: dict) -> None:
        ids = body.get("ids") or []
        decision = body.get("decision")
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be 'approved' or 'rejected'")
        if not isinstance(ids, list) or not ids:
            raise ValueError("ids must be a non-empty list")

        placeholders = ",".join("?" * len(ids))
        with db.transaction(conn):
            conn.execute(
                f"UPDATE items SET status = ?, needs_human_review = 0, "
                f"updated_at = ? WHERE id IN ({placeholders})",
                [decision, db.utcnow(), *ids],
            )
            for item_id in ids:
                db.log_review(
                    conn,
                    item_id=item_id,
                    action="approve" if decision == "approved" else "reject",
                    detail=body.get("note"),
                    actor=body.get("actor", "review-ui"),
                )
        self._json({"ok": True, "count": len(ids), "decision": decision})

    def _rerectify(self, conn: sqlite3.Connection, item_id: str) -> dict:
        from ..rectify import rectify_item

        item = db.get_item(conn, item_id)
        if item is None:
            raise FileNotFoundError(item_id)
        rectify_item(conn, self.cfg.paths, item, self.cfg.rectify)
        refreshed = db.get_item(conn, item_id)
        return _item_payload(conn, self.cfg, refreshed)  # type: ignore[arg-type]

    def _set_quad(self, conn: sqlite3.Connection, item_id: str, body: dict) -> None:
        quad = body.get("quad")
        if not isinstance(quad, list) or len(quad) != 4:
            raise ValueError("quad must be four [x, y] points")
        for point in quad:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("quad must be four [x, y] points")

        with db.transaction(conn):
            db.update_item_quad(conn, item_id, quad)
            db.log_review(
                conn, item_id=item_id, action="recrop", actor="review-ui"
            )
        self._json({"ok": True, "item": self._rerectify(conn, item_id)})

    def _rotate(self, conn: sqlite3.Connection, item_id: str, body: dict) -> None:
        try:
            delta = int(body.get("degrees", 90))
        except (TypeError, ValueError):
            raise ValueError("degrees must be an integer")
        if delta % 90 != 0:
            raise ValueError("degrees must be a multiple of 90")

        item = db.get_item(conn, item_id)
        if item is None:
            raise FileNotFoundError(item_id)

        total = (int(item["rotation_applied"] or 0) + delta) % 360
        with db.transaction(conn):
            conn.execute(
                "UPDATE items SET rotation_applied = ?, updated_at = ? WHERE id = ?",
                (total, db.utcnow(), item_id),
            )
            db.log_review(
                conn, item_id=item_id, action="rotate",
                detail=f"{total} deg", actor="review-ui",
            )
        self._json({"ok": True, "item": self._rerectify(conn, item_id)})

    def _delete_item(self, conn: sqlite3.Connection, item_id: str) -> None:
        with db.transaction(conn):
            db.log_review(
                conn, item_id=item_id, action="delete", actor="review-ui"
            )
            conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        self._json({"ok": True, "deleted": item_id})

    def _set_group(self, conn: sqlite3.Connection, item_id: str, body: dict) -> None:
        """Link this item to the one it is part of, or cut the link loose.

        `part_of: null` breaks the grouping. The model makes this call from the
        text - a continued headline, a matching byline - and it is the kind of
        judgement that is right most of the time and confidently wrong the
        rest, so a human has to be able to undo it in one click.
        """
        parent = body.get("part_of")
        if parent is not None:
            parent = str(parent)
            if parent == item_id:
                raise ValueError("an item cannot be part of itself")
            if db.get_item(conn, parent) is None:
                raise ValueError(f"no such item: {parent}")

        with db.transaction(conn):
            db.set_item_part_of(conn, item_id, parent, body.get("reason") or None)
            db.log_review(
                conn,
                item_id=item_id,
                action="group" if parent else "ungroup",
                detail=parent,
                actor="review-ui",
            )
        self._json({"ok": True, "id": item_id, "part_of": parent})

    def _edit_fields(self, conn: sqlite3.Connection, item_id: str, body: dict) -> None:
        """Apply a human's corrections to the live analysis.

        Writes a *new* analysis row rather than updating in place, so the
        model's original reading survives alongside the correction and the
        provenance stays honest about which is which.
        """
        fields = body.get("fields")
        if not isinstance(fields, dict) or not fields:
            raise ValueError("fields must be a non-empty object")

        unknown = set(fields) - EDITABLE_FIELDS
        if unknown:
            raise ValueError(f"not editable: {', '.join(sorted(unknown))}")

        current = db.current_analysis(conn, item_id)
        if current is None:
            raise FileNotFoundError(f"{item_id} has no analysis to edit")

        merged = {field: current[field] for field in ANALYSIS_FIELDS}
        merged.update({k: v for k, v in fields.items()})

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
                ) VALUES (?, 'human', ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, NULL)
                """,
                (
                    item_id, current["model"], db.utcnow(),
                    merged["item_type"], merged["title"], merged["summary"],
                    merged["full_text"], merged["date_value"],
                    merged["date_precision"], merged["date_source"],
                    merged["date_note"], merged["presentation"],
                    merged["legibility"], merged["condition_notes"],
                    merged["alt_text"], merged["rotary_context"],
                    merged["orientation_hint"], merged["confidence"],
                    json.dumps(merged, ensure_ascii=False),
                ),
            )
            conn.execute(
                "INSERT INTO item_overrides (item_id, fields_json, updated_at, "
                "updated_by) VALUES (?, ?, ?, 'review-ui') "
                "ON CONFLICT(item_id) DO UPDATE SET fields_json = excluded.fields_json, "
                "updated_at = excluded.updated_at",
                (item_id, json.dumps(fields, ensure_ascii=False), db.utcnow()),
            )
            db.log_review(
                conn, item_id=item_id, action="edit",
                detail=", ".join(sorted(fields)), actor="review-ui",
            )

        item = db.get_item(conn, item_id)
        self._json({"ok": True, "item": _item_payload(conn, self.cfg, item)})

    def _reanalyze(self, conn: sqlite3.Connection, item_id: str) -> None:
        from ..analyze import analyze_items
        from ..providers import ProviderError, build_provider

        item = db.get_item(conn, item_id)
        if item is None:
            raise FileNotFoundError(item_id)

        try:
            provider = build_provider(self.cfg.llm)
        except ProviderError as exc:
            return self._error(400, str(exc))

        # Scoped explicitly to this one item. Without item_ids, analyze_items
        # would pick up every other unanalysed row in the archive and spend
        # real money on a single button click.
        try:
            summary = analyze_items(
                conn, self.cfg.paths, provider, self.cfg.llm, item_ids=[item_id],
            )
        except ProviderError as exc:
            return self._error(400, str(exc))
        if summary.failed and not summary.succeeded:
            reason = summary.errors[0][1] if summary.errors else "analysis failed"
            return self._error(502, reason)

        refreshed = db.get_item(conn, item_id)
        self._json({"ok": True, "item": _item_payload(conn, self.cfg, refreshed)})

    def _add_item(self, conn: sqlite3.Connection, sha: str, body: dict) -> None:
        """Register an item a human drew on a photo the detector missed."""
        quad = body.get("quad")
        if not isinstance(quad, list) or len(quad) != 4:
            raise ValueError("quad must be four [x, y] points")

        photo = db.get_photo(conn, sha)
        if photo is None:
            raise FileNotFoundError(sha)

        seq = db.next_item_seq(conn, sha)
        item_id = db.make_item_id(sha, seq)
        with db.transaction(conn):
            db.insert_item(
                conn,
                item_id=item_id,
                photo_sha256=sha,
                seq=seq,
                quad=quad,
                detection_confidence=1.0,   # a human drew it; trust it
                detection_method="manual",
                needs_human_review=False,
            )
            db.log_review(conn, item_id=item_id, action="add", actor="review-ui")
        self._json({"ok": True, "item": self._rerectify(conn, item_id)})


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8765) -> None:
    handler = type("BoundHandler", (Handler,), {"cfg": cfg})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
