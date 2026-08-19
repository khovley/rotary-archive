"""Site builder: export shape, publish filtering, and what must never leak."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from fake_provider import GOOD_RESPONSE, FakeProvider
from rotary_archive import db
from rotary_archive.analyze import analyze_items
from rotary_archive.build.site import (
    fold_groups,
    build_entity_index,
    build_timeline,
    build_site,
    decade_of,
    display_date,
    publishable_items,
    year_of,
)
from rotary_archive.ingest import ingest_inbox
from rotary_archive.rectify import rectify_pending
from rotary_archive.segment import segment_pending
from synthetic import make_table_shot

TESTS_DIR = Path(__file__).parent


@pytest.fixture
def analysed(project, conn):
    """Four items, ingested through analysis, with varied dates and types."""
    make_table_shot(project.paths.inbox / "table.jpg", n_items=4, seed=3)
    ingest_inbox(conn, project.paths)
    segment_pending(conn, project.paths, project.segment, flag_below=0.80)
    rectify_pending(conn, project.paths, project.rectify)

    variants = [
        {"date_value": "1962-07-14", "date_source": "printed", "date_precision": "day",
         "item_type": "newspaper_clipping", "presentation": "text"},
        {"date_value": "1975", "date_source": "inferred", "date_precision": "year",
         "date_note": "Deduced from the anniversary reference.",
         "item_type": "certificate", "presentation": "both"},
        {"date_value": "", "date_source": "unknown", "date_precision": "unknown",
         "item_type": "photograph", "presentation": "image",
         "people": [], "full_text": ""},
        {"date_value": "1988-03", "date_source": "printed", "date_precision": "month",
         "item_type": "program", "presentation": "both"},
    ]
    order = iter(variants)
    analyze_items(
        conn, project.paths,
        FakeProvider(responder=lambda job: {**GOOD_RESPONSE, **next(order)}),
        project.llm,
    )
    return project, conn


def load_payload(site_dir: Path) -> dict:
    """Parse the window.ARCHIVE assignment back into Python."""
    text = (site_dir / "data" / "archive.js").read_text()
    assert text.startswith("window.ARCHIVE = ")
    return json.loads(text[len("window.ARCHIVE = ") :].rstrip().rstrip(";"))


# ------------------------------------------------------------------ dates ---


@pytest.mark.parametrize(
    "value,expected",
    [("1962-07-14", "1960s"), ("1955", "1950s"), ("2003-01", "2000s"),
     ("", ""), ("undated", ""), ("19", "")],
)
def test_decade_of(value, expected):
    assert decade_of(value) == expected


@pytest.mark.parametrize("value,expected", [("1962-07-14", "1962"), ("", "")])
def test_year_of(value, expected):
    assert year_of(value) == expected


@pytest.mark.parametrize(
    "value,precision,expected",
    [
        ("1962-07-14", "day", "14 July 1962"),
        ("1962-07", "month", "July 1962"),
        ("1962", "year", "1962"),
        ("1962-07-14", "year", "1962"),      # precision wins over the value
        ("1960", "decade", "1960s"),
        ("", "day", ""),
    ],
)
def test_display_date_never_exceeds_its_precision(value, precision, expected):
    """Rendering '1 January 1962' for something only known to be from 1962
    would invent a specificity the archive does not have."""
    assert display_date(value, precision) == expected


# -------------------------------------------------------------- filtering ---


def test_publishable_excludes_items_without_an_analysis(analysed):
    """An unanalysed item would appear on the site as an untitled blank."""
    project, conn = analysed
    photo = db.all_photos(conn)[0]["sha256"]
    seq = db.next_item_seq(conn, photo)
    with db.transaction(conn):
        db.insert_item(
            conn, item_id=db.make_item_id(photo, seq), photo_sha256=photo, seq=seq,
            quad=[[0, 0], [10, 0], [10, 10], [0, 10]],
            detection_confidence=1.0, detection_method="manual",
        )

    published = {row["id"] for row in publishable_items(conn, approved_only=False)}
    assert db.make_item_id(photo, seq) not in published


def test_publishable_excludes_rejected_items(analysed):
    project, conn = analysed
    victim = db.items_with_status(conn, "analyzed")[0]["id"]
    with db.transaction(conn):
        db.set_item_status(conn, victim, "rejected")

    published = {row["id"] for row in publishable_items(conn, approved_only=False)}
    assert victim not in published


def test_approved_only_publishes_nothing_until_items_are_approved(analysed):
    project, conn = analysed
    assert publishable_items(conn, approved_only=True) == []

    approved = db.items_with_status(conn, "analyzed")[0]["id"]
    with db.transaction(conn):
        db.set_item_status(conn, approved, "approved")

    published = [row["id"] for row in publishable_items(conn, approved_only=True)]
    assert published == [approved]


# ----------------------------------------------------------------- export ---


def test_build_writes_the_expected_files(analysed):
    project, conn = analysed
    summary = build_site(conn, project.paths, project.site)

    site = project.paths.site
    for relative in ("index.html", "embed.html", "assets/app.js",
                     "assets/style.css", "data/archive.js"):
        assert (site / relative).exists(), relative
    assert summary.items == 4


def test_payload_carries_date_provenance(analysed):
    """The site must be able to show a deduced date differently from a printed
    one, which it cannot do if the exporter flattens them."""
    project, conn = analysed
    build_site(conn, project.paths, project.site)
    payload = load_payload(project.paths.site)

    sources = {item["date_source"] for item in payload["items"]}
    assert {"printed", "inferred", "unknown"} <= sources

    inferred = next(i for i in payload["items"] if i["date_source"] == "inferred")
    assert inferred["date_note"], "an inferred date must carry its reasoning"


def test_payload_is_valid_javascript(analysed):
    project, conn = analysed
    build_site(conn, project.paths, project.site)

    result = subprocess.run(
        ["node", "--check", str(project.paths.site / "data" / "archive.js")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_data_is_a_js_assignment_not_json(analysed):
    """A .json file fetched over file:// is blocked by the browser, so the
    archive would work when served and silently fail when opened from disk."""
    project, conn = analysed
    build_site(conn, project.paths, project.site)

    text = (project.paths.site / "data" / "archive.js").read_text()
    assert text.startswith("window.ARCHIVE = ")
    assert not (project.paths.site / "data" / "archive.json").exists()


def test_timeline_groups_by_decade_and_puts_undated_last(analysed):
    project, conn = analysed
    build_site(conn, project.paths, project.site)
    payload = load_payload(project.paths.site)

    decades = [block["decade"] for block in payload["timeline"]]
    assert decades == ["1960s", "1970s", "1980s", "Undated"]


def test_undated_items_are_kept_not_hidden(analysed):
    """A club archive has plenty of undated material; dropping it would make
    the timeline look more complete than it is."""
    project, conn = analysed
    build_site(conn, project.paths, project.site)
    payload = load_payload(project.paths.site)

    undated = next(b for b in payload["timeline"] if b["decade"] == "Undated")
    assert undated["count"] == 1
    assert len(payload["items"]) == 4


def test_entity_index_counts_items_per_entity(analysed):
    project, conn = analysed
    build_site(conn, project.paths, project.site)
    payload = load_payload(project.paths.site)

    people = payload["entities"]["person"]
    assert people, "expected people in the index"
    # The photograph variant has no people, so a name appears on three items.
    assert max(len(entry["items"]) for entry in people) == 3


def test_entity_index_is_sorted_by_frequency():
    records = [
        {"id": "a", "people": [{"name": "Common", "slug": "common"}],
         "orgs": [], "places": [], "topics": []},
        {"id": "b", "people": [{"name": "Common", "slug": "common"},
                               {"name": "Rare", "slug": "rare"}],
         "orgs": [], "places": [], "topics": []},
    ]
    index = build_entity_index(records)
    assert [e["slug"] for e in index["person"]] == ["common", "rare"]


def test_timeline_of_an_empty_archive_is_empty():
    assert build_timeline([]) == []


# ------------------------------------------------------------------ media ---


def test_only_web_derivatives_are_published(analysed):
    """Masters are the irreplaceable copy and have no business on a web
    server. Publishing only derivatives is the whole point of the split."""
    project, conn = analysed
    build_site(conn, project.paths, project.site)

    media = list((project.paths.site / "media").iterdir())
    assert media, "expected derivative images"
    assert all(path.suffix == ".webp" for path in media)
    assert not list(project.paths.site.rglob("*.jpg"))


def test_every_published_item_has_its_images(analysed):
    project, conn = analysed
    build_site(conn, project.paths, project.site)
    payload = load_payload(project.paths.site)

    for item in payload["items"]:
        for size in item["sizes"]:
            expected = project.paths.site / "media" / f"{item['id']}-{size}.webp"
            assert expected.exists(), expected.name


def test_rebuilding_does_not_accumulate_stale_media(analysed):
    """A rejected item's images must disappear from the site on the next
    build, or the archive quietly keeps publishing what was withdrawn."""
    project, conn = analysed
    build_site(conn, project.paths, project.site)
    before = len(list((project.paths.site / "media").iterdir()))

    victim = db.items_with_status(conn, "analyzed")[0]["id"]
    with db.transaction(conn):
        db.set_item_status(conn, victim, "rejected")
    build_site(conn, project.paths, project.site)

    after = list((project.paths.site / "media").iterdir())
    assert len(after) < before
    assert not any(path.name.startswith(victim) for path in after)


def test_build_leaves_unrelated_files_in_the_site_directory_alone(analysed):
    """`clean` must only remove what this builder owns - a site directory
    pointing somewhere unexpected should not lose its neighbours."""
    project, conn = analysed
    project.paths.site.mkdir(parents=True, exist_ok=True)
    keeper = project.paths.site / "CNAME"
    keeper.write_text("history.example.org")

    build_site(conn, project.paths, project.site, clean=True)
    assert keeper.exists()


# ------------------------------------------------------------- front end ---


def test_every_route_renders(analysed):
    """Runs the real app.js over the real payload in Node.

    Catches what a Python test cannot see: a template literal referencing a
    field the exporter stopped emitting would render 'undefined' into the page
    rather than raising anywhere Python could observe it.
    """
    project, conn = analysed
    build_site(conn, project.paths, project.site)

    result = subprocess.run(
        ["node", str(TESTS_DIR / "render_site.mjs"), str(project.paths.site)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout


# ------------------------------------------------------ grouped items ------


def page(item_id, *, part_of="", text="", reason=""):
    """A minimal record, shaped like build_records output."""
    return {
        "id": item_id,
        "title": item_id,
        "text": text,
        "alt": f"scan of {item_id}",
        "sizes": [800],
        "w": 800,
        "h": 1200,
        "part_of": part_of,
        "part_reason": reason,
        "pages": [],
    }


def test_a_continued_article_publishes_as_one_entry():
    """Two clippings of one story are one thing to a reader.

    The second strip becomes a further page of the first, and its text is
    folded into the body so a phrase from the continuation still finds the
    article in search.
    """
    folded = fold_groups([
        page("a", text="The club voted to"),
        page("b", part_of="a", text="fund the new wing.", reason="continues the story"),
    ])

    assert [r["id"] for r in folded] == ["a"]
    assert [p["id"] for p in folded[0]["pages"]] == ["b"]
    assert folded[0]["pages"][0]["note"] == "continues the story"
    assert "fund the new wing." in folded[0]["text"]


def test_an_orphaned_child_is_promoted_not_dropped():
    """Parent and child are approved separately, so a build can easily contain
    one and not the other. Hiding the child would silently lose a clipping."""
    folded = fold_groups([page("b", part_of="never-approved", text="orphan")])

    assert [r["id"] for r in folded] == ["b"]
    assert folded[0]["pages"] == []


def test_an_item_claiming_to_be_part_of_itself_stands_alone():
    folded = fold_groups([page("a", part_of="a")])
    assert [r["id"] for r in folded] == ["a"]


def test_several_pages_attach_to_the_same_parent_in_order():
    folded = fold_groups([
        page("a"), page("b", part_of="a"), page("c", part_of="a"),
    ])
    assert [p["id"] for p in folded[0]["pages"]] == ["b", "c"]


def test_folded_pages_keep_their_images(analysed):
    """A page's <img> points at a derivative that copy_media must still ship,
    even though the page is no longer a top-level record."""
    project, conn = analysed
    items = publishable_items(conn, approved_only=False)
    assert len(items) >= 2

    with db.transaction(conn):
        db.set_item_part_of(conn, items[1]["id"], items[0]["id"], "continues")

    build_site(conn, project.paths, project.site)
    site_dir = project.paths.site

    payload = load_payload(site_dir)
    parent = next(r for r in payload["items"] if r["id"] == items[0]["id"])
    assert [p["id"] for p in parent["pages"]] == [items[1]["id"]]
    assert not any(r["id"] == items[1]["id"] for r in payload["items"])

    for size in parent["pages"][0]["sizes"]:
        name = f"{items[1]['id']}-{size}.webp"
        assert (site_dir / "media" / name).exists(), f"{name} was not published"
