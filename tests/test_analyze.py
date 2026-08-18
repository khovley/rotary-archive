"""Analyse stage: normalisation, flagging, entities, exports, versioning."""

from __future__ import annotations

import json

import pytest

from fake_provider import GOOD_RESPONSE, FakeBatchProvider, FakeProvider
from rotary_archive import db
from rotary_archive.analyze import (
    analyze_items,
    consistency_flags,
    normalise,
    slugify,
    write_export,
)
from rotary_archive.ingest import ingest_inbox
from rotary_archive.providers.base import AnalysisResult, extract_json
from rotary_archive.rectify import rectify_pending
from rotary_archive.segment import segment_pending
from synthetic import make_table_shot


@pytest.fixture
def rectified(project, conn):
    make_table_shot(project.paths.inbox / "table.jpg", n_items=4, seed=3)
    ingest_inbox(conn, project.paths)
    segment_pending(conn, project.paths, project.segment, flag_below=0.80)
    rectify_pending(conn, project.paths, project.rectify)
    return project, conn


def run(project, conn, provider, **kwargs):
    return analyze_items(conn, project.paths, provider, project.llm, **kwargs)


# ------------------------------------------------------------ normalisation --


def test_normalise_passes_through_a_good_response():
    clean = normalise(dict(GOOD_RESPONSE))
    assert clean["item_type"] == "newspaper_clipping"
    assert clean["date_value"] == "1962-07-14"
    assert clean["people"] == ["Harold Pratt", "Eleanor Voss"]
    assert clean["confidence"] == pytest.approx(0.93)


def test_normalise_coerces_a_comma_string_into_a_list():
    """Some providers answer with prose where a list was requested; salvaging
    it beats discarding the names."""
    clean = normalise({**GOOD_RESPONSE, "people": "Harold Pratt, Eleanor Voss"})
    assert clean["people"] == ["Harold Pratt", "Eleanor Voss"]


def test_normalise_deduplicates_case_insensitively():
    clean = normalise({**GOOD_RESPONSE, "topics": ["Polio", "polio", "POLIO "]})
    assert clean["topics"] == ["Polio"]


def test_normalise_rescales_a_percentage_confidence():
    """A model answering 85 when asked for 0-1 means 85%, not 'impossibly
    confident'."""
    assert normalise({**GOOD_RESPONSE, "confidence": 85})["confidence"] == pytest.approx(
        0.85
    )


@pytest.mark.parametrize("value,expected", [(-1, 0.0), (0.5, 0.5), (2000, 1.0)])
def test_normalise_clamps_confidence(value, expected):
    assert normalise({**GOOD_RESPONSE, "confidence": value})["confidence"] == expected


@pytest.mark.parametrize("value", [0, 9, "high", None])
def test_normalise_clamps_legibility(value):
    assert 1 <= normalise({**GOOD_RESPONSE, "legibility": value})["legibility"] <= 5


def test_normalise_falls_back_on_an_invented_enum():
    clean = normalise({**GOOD_RESPONSE, "item_type": "papyrus_scroll"})
    assert clean["item_type"] == "other"


def test_normalise_survives_a_completely_empty_response():
    clean = normalise({})
    assert clean["item_type"] == "other"
    assert clean["people"] == []
    assert clean["date_source"] == "unknown"


def test_empty_date_can_never_claim_to_be_printed():
    """The rule matters more than the model's answer: a date with no value is
    not a printed date, whatever was returned."""
    clean = normalise(
        {**GOOD_RESPONSE, "date_value": "", "date_source": "printed",
         "date_precision": "day"}
    )
    assert clean["date_source"] == "unknown"
    assert clean["date_precision"] == "unknown"


# ---------------------------------------------------------------- flagging --


def test_no_flags_on_a_clean_confident_reading():
    assert consistency_flags(normalise(dict(GOOD_RESPONSE))) == []


@pytest.mark.parametrize(
    "override,fragment",
    [
        ({"legibility": 1}, "legibility"),
        ({"confidence": 0.2}, "confidence"),
        ({"date_source": "inferred"}, "inferred"),
        ({"orientation_hint": "rotate_90_cw"}, "rotating"),
        ({"title": ""}, "no title"),
    ],
)
def test_conditions_that_flag_for_review(override, fragment):
    reasons = consistency_flags(normalise({**GOOD_RESPONSE, **override}))
    assert any(fragment in reason for reason in reasons), reasons


def test_text_item_without_a_transcription_is_flagged():
    """A contradiction the model cannot catch about itself: it called the item
    text-bearing and then returned no text."""
    reasons = consistency_flags(
        normalise({**GOOD_RESPONSE, "presentation": "text", "full_text": ""})
    )
    assert any("no transcription" in r for r in reasons)


def test_printed_date_without_any_text_is_flagged():
    reasons = consistency_flags(
        normalise({**GOOD_RESPONSE, "full_text": "", "presentation": "image"})
    )
    assert any("printed but nothing was transcribed" in r for r in reasons)


# ----------------------------------------------------------------- slugify --


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Harold Pratt", "harold-pratt"),
        ("O'Brien", "obrien"),
        ("  Rotary   Club  ", "rotary-club"),
        ("Zoë Müller", "zoe-muller"),
        ("St. John's", "st-johns"),
    ],
)
def test_slugify(raw, expected):
    assert slugify(raw) == expected


def test_slug_collapses_spelling_variants_of_one_person():
    assert slugify("O'Brien") == slugify("OBrien") == slugify("o'brien")


# ------------------------------------------------------------- end-to-end ---


def test_analyze_writes_one_analysis_per_item(rectified):
    project, conn = rectified
    provider = FakeProvider()

    summary = run(project, conn, provider)

    assert summary.attempted == 4
    assert summary.succeeded == 4
    assert summary.failed == 0
    for item in db.items_with_status(conn, "analyzed"):
        analysis = db.current_analysis(conn, item["id"])
        assert analysis["title"] == GOOD_RESPONSE["title"]
        assert analysis["provider"] == "fake"


def test_analyze_extracts_and_deduplicates_entities(rectified):
    project, conn = rectified
    run(project, conn, FakeProvider())

    people = conn.execute(
        "SELECT name FROM entities WHERE kind = 'person' ORDER BY name"
    ).fetchall()
    # Two names shared by all four items - deduplicated to two rows, not eight.
    assert [r["name"] for r in people] == ["Eleanor Voss", "Harold Pratt"]

    links = conn.execute("SELECT COUNT(*) FROM item_entities").fetchone()[0]
    assert links == 4 * 8  # 2 people + 2 orgs + 1 place + 3 topics per item


def test_reanalysis_supersedes_without_discarding_history(rectified):
    project, conn = rectified
    run(project, conn, FakeProvider())

    revised = {**GOOD_RESPONSE, "title": "A Better Reading"}
    run(project, conn, FakeProvider(responder=lambda job: dict(revised)), force=True)

    item_id = db.items_with_status(conn, "analyzed")[0]["id"]
    assert db.current_analysis(conn, item_id)["title"] == "A Better Reading"

    total = conn.execute(
        "SELECT COUNT(*) FROM analyses WHERE item_id = ?", (item_id,)
    ).fetchone()[0]
    assert total == 2, "the earlier reading must remain for comparison"


def test_reanalysis_preserves_human_added_entities(rectified):
    """Correcting the archive by hand has to survive a re-run, or there is no
    point correcting it."""
    project, conn = rectified
    run(project, conn, FakeProvider())
    item_id = db.items_with_status(conn, "analyzed")[0]["id"]

    with db.transaction(conn):
        conn.execute(
            "INSERT INTO entities (kind, name, slug, created_at) "
            "VALUES ('person', 'Named By A Member', 'named-by-a-member', ?)",
            (db.utcnow(),),
        )
        entity_id = conn.execute(
            "SELECT id FROM entities WHERE slug = 'named-by-a-member'"
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO item_entities (item_id, entity_id, source) "
            "VALUES (?, ?, 'human')",
            (item_id, entity_id),
        )

    run(project, conn, FakeProvider(), force=True)

    surviving = conn.execute(
        "SELECT COUNT(*) FROM item_entities WHERE item_id = ? AND source = 'human'",
        (item_id,),
    ).fetchone()[0]
    assert surviving == 1


def test_analyze_skips_items_already_analysed(rectified):
    project, conn = rectified
    run(project, conn, FakeProvider())

    second = FakeProvider()
    summary = run(project, conn, second)

    assert summary.attempted == 0
    assert second.calls == []


def test_limit_caps_the_number_analysed(rectified):
    project, conn = rectified
    summary = run(project, conn, FakeProvider(), limit=2)
    assert summary.attempted == 2
    assert summary.succeeded == 2


def test_one_item_failing_does_not_abort_the_run(rectified):
    project, conn = rectified
    item_ids = sorted(i["id"] for i in db.items_with_status(conn, "rectified"))
    doomed = item_ids[1]

    def responder(job):
        if job.item_id == doomed:
            raise RuntimeError("provider exploded")
        return dict(GOOD_RESPONSE)

    summary = run(project, conn, FakeProvider(responder=responder))

    assert summary.succeeded == 3
    assert summary.failed == 1
    assert summary.errors[0][0] == doomed
    assert db.get_item(conn, doomed)["status"] == "rectified"  # untouched


def test_refusal_is_recorded_as_a_failure_not_an_analysis(rectified):
    project, conn = rectified

    def responder(job):
        return AnalysisResult(
            item_id=job.item_id, ok=False, provider="fake", model="fake-1",
            error="model declined to analyse this item (category: privacy)",
        )

    summary = run(project, conn, FakeProvider(responder=responder))

    assert summary.succeeded == 0
    assert summary.failed == 4
    assert db.counts(conn)["analyses"] == 0


def test_batch_results_are_matched_by_id_not_position(rectified):
    project, conn = rectified
    ids = sorted(i["id"] for i in db.items_with_status(conn, "rectified"))

    # Title each response after its own item, then have the provider return
    # them reversed. Matching by position would mislabel every item.
    provider = FakeBatchProvider(
        responder=lambda job: {**GOOD_RESPONSE, "title": f"Item {job.item_id}"}
    )
    run(project, conn, provider)

    for item_id in ids:
        assert db.current_analysis(conn, item_id)["title"] == f"Item {item_id}"


def test_items_dropped_from_a_batch_are_reported(rectified):
    project, conn = rectified
    ids = sorted(i["id"] for i in db.items_with_status(conn, "rectified"))

    summary = run(project, conn, FakeBatchProvider(drop={ids[0]}))

    assert summary.succeeded == 3
    assert summary.failed == 1
    assert db.current_analysis(conn, ids[0]) is None


# ------------------------------------------------------------------ export --


def test_text_items_get_a_markdown_export(rectified):
    project, conn = rectified
    run(project, conn, FakeProvider())

    exports = sorted(project.paths.exports.glob("*.md"))
    assert len(exports) == 4

    body = exports[0].read_text()
    assert GOOD_RESPONSE["title"] in body
    assert "ROTARY CLUB FUNDS NEW LIBRARY WING" in body
    assert "date_source: printed" in body


def test_image_only_items_get_no_export(project):
    """The keep-as-photo decision, made concrete: nothing to read means
    nothing to write."""
    clean = normalise({**GOOD_RESPONSE, "presentation": "image"})
    assert write_export(project.paths, "x-00", clean) is None


def test_text_item_with_no_transcription_gets_no_export(project):
    clean = normalise({**GOOD_RESPONSE, "presentation": "text", "full_text": ""})
    assert write_export(project.paths, "x-01", clean) is None


# ------------------------------------------------------------ JSON salvage --


def test_extract_json_handles_a_bare_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_handles_a_fenced_block():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_handles_surrounding_prose():
    assert extract_json('Here you go:\n{"a": 1}\nHope that helps!') == {"a": 1}


def test_extract_json_survives_braces_inside_a_transcription():
    """A regex would stop at the first closing brace inside the text; brace
    balancing is what makes this work on real transcriptions."""
    payload = '{"full_text": "the minutes {sic} recorded", "n": 2}'
    assert extract_json(f"prose {payload} more prose")["n"] == 2


def test_extract_json_rejects_a_response_with_no_object():
    with pytest.raises(ValueError):
        extract_json("I could not read this image.")


# --------------------------------------------------------- scoped re-runs --


def test_item_ids_scopes_analysis_to_named_items(rectified):
    """The property that keeps the review UI's per-item 'Re-read' button from
    analysing the whole archive - and spending real money - on one click."""
    project, conn = rectified
    ids = sorted(i["id"] for i in db.items_with_status(conn, "rectified"))

    provider = FakeProvider()
    summary = analyze_items(
        conn, project.paths, provider, project.llm, item_ids=[ids[0]]
    )

    assert summary.attempted == 1
    assert [job.item_id for job in provider.calls] == [ids[0]]
    assert db.current_analysis(conn, ids[1]) is None


def test_item_ids_re_reads_an_already_analysed_item(rectified):
    project, conn = rectified
    run(project, conn, FakeProvider())
    item_id = sorted(i["id"] for i in db.items_with_status(conn, "analyzed"))[0]

    revised = FakeProvider(responder=lambda job: {**GOOD_RESPONSE, "title": "Second look"})
    summary = analyze_items(
        conn, project.paths, revised, project.llm, item_ids=[item_id]
    )

    assert summary.attempted == 1
    assert db.current_analysis(conn, item_id)["title"] == "Second look"


def test_empty_item_ids_analyses_nothing(rectified):
    project, conn = rectified
    provider = FakeProvider()
    summary = analyze_items(conn, project.paths, provider, project.llm, item_ids=[])
    assert summary.attempted == 0
    assert provider.calls == []


def test_rejected_items_are_never_analysed(rectified):
    """No reason to pay to read something already thrown away."""
    project, conn = rectified
    victim = sorted(i["id"] for i in db.items_with_status(conn, "rectified"))[0]
    with db.transaction(conn):
        db.set_item_status(conn, victim, "rejected")

    provider = FakeProvider()
    run(project, conn, provider)

    assert victim not in [job.item_id for job in provider.calls]


def test_provider_error_stops_the_run(rectified):
    """A whole-run condition such as bad credentials must surface once, not be
    reported per item."""
    from rotary_archive.providers.base import ProviderError

    project, conn = rectified
    provider = FakeProvider(responder=lambda job: ProviderError("no credentials"))

    with pytest.raises(ProviderError):
        run(project, conn, provider)


def test_context_warns_against_using_the_capture_date():
    """The photograph's EXIF date says when the item was digitised, not how
    old the item is. Offered unqualified it would leak straight into
    date_value and date every clipping to the week it was photographed."""
    from rotary_archive.schema_item import build_context

    context = build_context(captured_at="2026-08-17T14:03:00")
    assert "2026-08-17" in context
    assert "not when the item itself is from" in context


def test_context_omits_the_capture_date_when_there_is_none():
    from rotary_archive.schema_item import build_context

    assert build_context() == "Catalogue this item."


def test_context_mentions_neighbouring_items(rectified):
    """Knowing an item was photographed alongside others is a genuine hint
    that related material exists."""
    project, conn = rectified
    provider = FakeProvider()
    run(project, conn, provider)

    assert "photographed alongside 3 other items" in provider.calls[0].context
