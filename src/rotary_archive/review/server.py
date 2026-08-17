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
"""

from __future__ import annotations

import json
import sqlite3
import threading
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


def _item_payload(conn: sqlite3.Connection, cfg: Config, item: sqlite3.Row) -> dict:
    analysis = db.current_analysis(conn, item["id"])
    return {
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
        "title": analysis["title"] if analysis else None,
        "summary": analysis["summary"] if analysis else None,
    }


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
            if path.startswith("/api/photo/") and path.endswith("/item"):
                return self._add_item(conn, path.split("/")[3], body)
            raise FileNotFoundError(path)
        finally:
            conn.close()

    # ------------------------------------------------------------- actions --

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
