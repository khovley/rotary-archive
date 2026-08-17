"""Command line interface.

    rotary status              what is in the archive right now
    rotary ingest              pull photos out of the inbox
    rotary segment             find items in each photo
    rotary rectify             crop, deskew, and write derivatives
    rotary review              open the batch approval UI
    rotary run                 ingest -> segment -> rectify, then review

Every stage is independently re-runnable and skips completed work, so `run`
after adding photos to the inbox does only the new work.

Each command is a thin wrapper around a `_do_*` helper taking plain arguments,
so `run` can chain them without going through Typer's option machinery.
"""

from __future__ import annotations

import shutil
import sqlite3
import webbrowser

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import db
from .config import Config, ConfigError, load_config

app = typer.Typer(
    add_completion=False,
    help="Turn photos of analogue memorabilia into a searchable archive.",
    no_args_is_help=True,
)
console = Console()


def _load() -> tuple[Config, sqlite3.Connection]:
    try:
        cfg = load_config()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2)
    cfg.paths.ensure()
    return cfg, db.connect(cfg.paths.database)


# ----------------------------------------------------------------- status ---


def _do_status(cfg: Config, conn: sqlite3.Connection) -> None:
    stats = db.counts(conn)

    table = Table(title="Rotary Archive", show_header=True, header_style="bold")
    table.add_column("Stage")
    table.add_column("Count", justify="right")
    table.add_row("Photos ingested", str(stats["photos"]))
    for name, count in sorted(stats["photos_by_status"].items()):
        table.add_row(f"  photos: {name}", str(count))
    table.add_row("Items detected", str(stats["items"]))
    for name, count in sorted(stats["items_by_status"].items()):
        table.add_row(f"  items: {name}", str(count))
    table.add_row("Flagged for review", str(stats["flagged"]))
    table.add_row("Analyses on file", str(stats["analyses"]))
    console.print(table)

    from .ingest import find_photos, sha256_file

    present = find_photos(cfg.paths.inbox)
    if present:
        # Count what is genuinely new rather than what is merely sitting in the
        # inbox - without --move, already-ingested files stay there, and
        # reporting those as "waiting" would nag forever.
        fresh = sum(1 for p in present if not db.photo_exists(conn, sha256_file(p)))
        if fresh:
            console.print(
                f"[yellow]{fresh} new file(s) in {cfg.paths.inbox}[/yellow] "
                "- run [bold]rotary ingest[/bold]"
            )
        else:
            console.print(
                f"[dim]{len(present)} file(s) in the inbox, all already "
                f"ingested.[/dim]"
            )


@app.command()
def status() -> None:
    """Show what is in the archive and what stage each thing is at."""
    cfg, conn = _load()
    _do_status(cfg, conn)


# ----------------------------------------------------------------- ingest ---


def _do_ingest(cfg: Config, conn: sqlite3.Connection, move: bool) -> int:
    from .ingest import find_photos, ingest_inbox

    pending = find_photos(cfg.paths.inbox)
    if not pending:
        console.print(f"Nothing to ingest in {cfg.paths.inbox}")
        return 0

    with console.status(f"Ingesting {len(pending)} file(s)...") as ui:
        result = ingest_inbox(
            conn, cfg.paths, move=move,
            progress=lambda p: ui.update(f"Ingesting {p.name}"),
        )

    console.print(f"[green]Ingested {len(result.ingested)}[/green] new photo(s)")
    if result.skipped_duplicate:
        console.print(
            f"[dim]Skipped {len(result.skipped_duplicate)} already in the archive[/dim]"
        )
    for path, reason in result.unreadable:
        console.print(f"[red]Could not read {path.name}: {reason}[/red]")
    return len(result.ingested)


@app.command()
def ingest(
    move: bool = typer.Option(
        False, "--move", help="Delete inbox copies after archiving them."
    ),
) -> None:
    """Copy photos from the inbox into the archive, skipping duplicates."""
    cfg, conn = _load()
    _do_ingest(cfg, conn, move)


# ---------------------------------------------------------------- segment ---


def _do_segment(cfg: Config, conn: sqlite3.Connection, force: bool) -> int:
    from .segment import segment_pending

    flag_below = float(cfg.review.get("flag_below_confidence", 0.80))
    todo = len(
        db.all_photos(conn) if force else db.photos_with_status(conn, "ingested")
    )
    if not todo:
        console.print("No photos waiting to be segmented.")
        return 0

    if force:
        console.print(
            "[yellow]--force re-runs detection and discards any manual crop "
            "corrections on these photos.[/yellow]"
        )

    with console.status(f"Segmenting {todo} photo(s)...") as ui:
        results = segment_pending(
            conn, cfg.paths, cfg.segment, flag_below=flag_below, force=force,
            progress=lambda p: ui.update(f"Segmenting {p['original_name']}"),
        )

    found = sum(len(r.candidates) for r in results)
    flagged = sum(
        1 for r in results for c in r.candidates if c.confidence < flag_below
    )
    console.print(
        f"[green]Found {found} item(s)[/green] across {len(results)} photo(s)"
    )
    if flagged:
        console.print(f"[yellow]{flagged} need a look in review[/yellow]")
    for r in results:
        if r.note:
            console.print(f"[dim]{r.photo_sha256[:12]}: {r.note}[/dim]")
    return found


@app.command()
def segment(
    force: bool = typer.Option(
        False, "--force", help="Re-segment photos that were already processed."
    ),
) -> None:
    """Find the individual items in each ingested photo."""
    cfg, conn = _load()
    _do_segment(cfg, conn, force)


# ---------------------------------------------------------------- rectify ---


def _do_rectify(cfg: Config, conn: sqlite3.Connection, force: bool) -> int:
    from .rectify import rectify_pending

    statuses = (
        ["detected", "rectified", "analyzed", "approved"] if force else ["detected"]
    )
    todo = len(db.items_with_status(conn, statuses))
    if not todo:
        console.print("No items waiting to be rectified.")
        return 0

    with console.status(f"Rectifying {todo} item(s)...") as ui:
        results = rectify_pending(
            conn, cfg.paths, cfg.rectify, force=force,
            progress=lambda i: ui.update(f"Rectifying {i['id']}"),
        )

    straightened = [r for r in results if abs(r.fine_skew_deg) > 0]
    console.print(f"[green]Rectified {len(results)} item(s)[/green]")
    if straightened:
        worst = max(abs(r.fine_skew_deg) for r in straightened)
        console.print(
            f"[dim]Fine deskew applied to {len(straightened)}; "
            f"largest correction {worst:.2f} deg[/dim]"
        )
    return len(results)


@app.command()
def rectify(
    force: bool = typer.Option(
        False, "--force", help="Re-render items that already have masters."
    ),
) -> None:
    """Crop, deskew, and write archival masters plus web derivatives."""
    cfg, conn = _load()
    _do_rectify(cfg, conn, force)


# ----------------------------------------------------------------- review ---


def _do_review(cfg: Config, port: int | None, open_browser: bool) -> None:
    from .review.server import serve

    host = cfg.review.get("host", "127.0.0.1")
    resolved = int(port or cfg.review.get("port", 8765))
    url = f"http://{host}:{resolved}/"

    console.print(f"Review UI at [bold cyan]{url}[/bold cyan]  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        serve(cfg, host=host, port=resolved)
    except KeyboardInterrupt:
        console.print("\nReview server stopped.")


@app.command()
def review(
    port: Optional[int] = typer.Option(None, help="Port to serve the review UI on."),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Do not open a browser window."
    ),
) -> None:
    """Open the batch approval UI."""
    cfg, conn = _load()
    conn.close()  # the server opens its own connection per request
    _do_review(cfg, port, open_browser=not no_browser)


# -------------------------------------------------------------------- run ---


@app.command()
def run(
    move: bool = typer.Option(False, "--move", help="Delete inbox copies after archiving."),
    no_review: bool = typer.Option(
        False, "--no-review", help="Stop after rectify instead of opening review."
    ),
) -> None:
    """Ingest, segment, and rectify everything new, then open review."""
    cfg, conn = _load()
    _do_ingest(cfg, conn, move)
    _do_segment(cfg, conn, force=False)
    _do_rectify(cfg, conn, force=False)
    console.print()
    _do_status(cfg, conn)
    if no_review:
        return
    conn.close()
    console.print()
    _do_review(cfg, port=None, open_browser=True)


# ------------------------------------------------------------------ reset ---


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
    photos: bool = typer.Option(
        False,
        "--photos",
        help="Also unregister ingested photos (files on disk are still kept).",
    ),
) -> None:
    """Discard detected items, crops, and analyses so the pipeline can re-run.

    Ingested photos stay registered by default, so `rotary segment` rebuilds
    everything from the originals already on disk without re-ingesting.
    """
    cfg, conn = _load()
    stats = db.counts(conn)

    console.print(
        f"[red]This discards {stats['items']} detected item(s), their crops "
        f"and derivatives, and every analysis and review decision.[/red]"
    )
    if photos:
        console.print(
            f"[red]--photos also unregisters {stats['photos']} photo(s); "
            f"the files themselves stay in {cfg.paths.originals}.[/red]"
        )
    else:
        console.print(
            f"[green]{stats['photos']} ingested photo(s) stay registered - "
            f"run `rotary segment` afterwards to rebuild.[/green]"
        )

    if not yes and not typer.confirm("Proceed?"):
        console.print("Cancelled.")
        raise typer.Exit()

    with db.transaction(conn):
        # items cascades to derivatives, analyses, item_entities, and
        # item_overrides; review_log rows go with their item.
        conn.execute("DELETE FROM items")
        conn.execute("DELETE FROM entities")
        if photos:
            conn.execute("DELETE FROM photos")
        else:
            conn.execute(
                "UPDATE photos SET status = 'ingested', segment_note = NULL"
            )
    conn.close()

    for directory in (cfg.paths.items, cfg.paths.derivatives, cfg.paths.exports):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)

    console.print("[green]Reset complete.[/green]")


if __name__ == "__main__":
    app()
