"""The analysis contract: what we ask the model for, and how we ask.

Kept apart from `analyze.py` because it is the part most likely to be tuned
by hand, and because the provider layer needs the schema without needing the
pipeline.

Schema design notes:

  * Every field is required and `additionalProperties` is false. Strict schema
    modes need this, and it means downstream code never guards for a missing
    key.
  * Nothing is nullable. "Unknown" is an empty string or an empty list, which
    removes a whole class of `None` handling from the pipeline and the site.
  * No numeric ranges or string lengths - several providers reject those in
    strict mode. Bounded values are expressed as enums, and `confidence` is
    clamped on the way in instead.
"""

from __future__ import annotations

from typing import Any

ITEM_TYPES = [
    "newspaper_clipping",
    "photograph",
    "document",
    "letter",
    "certificate",
    "program",
    "newsletter",
    "ephemera",
    "object",
    "other",
]

PRESENTATIONS = ["image", "text", "both"]
DATE_PRECISIONS = ["day", "month", "year", "decade", "unknown"]
DATE_SOURCES = ["printed", "inferred", "unknown"]
ORIENTATIONS = ["upright", "rotate_90_cw", "rotate_90_ccw", "rotate_180"]


def _string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _string_list(description: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "description": description}


ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "item_type": {
            "type": "string",
            "enum": ITEM_TYPES,
            "description": "What kind of object this is.",
        },
        "title": _string(
            "A short human-readable title. For a clipping, its headline. "
            "For a photograph, a plain description of what it shows."
        ),
        "summary": _string(
            "One paragraph describing the item and why it matters to the "
            "club's history. Empty string if there is nothing to say."
        ),
        "full_text": _string(
            "Verbatim transcription of every legible word, preserving line "
            "and paragraph breaks. Empty string if the item has no text. "
            "Use [illegible] for words you cannot read."
        ),
        "date_value": _string(
            "ISO 8601, as precise as the evidence supports: 1962-07-14, "
            "1962-07, or 1962. Empty string if unknown."
        ),
        "date_precision": {
            "type": "string",
            "enum": DATE_PRECISIONS,
            "description": "How precise date_value is.",
        },
        "date_source": {
            "type": "string",
            "enum": DATE_SOURCES,
            "description": (
                "'printed' only when a date is actually written on the item. "
                "'inferred' when deduced from context. 'unknown' otherwise."
            ),
        },
        "date_note": _string(
            "If date_source is 'inferred', the reasoning. Empty otherwise."
        ),
        "people": _string_list(
            "Names of people, taken only from text visible on the item."
        ),
        "organizations": _string_list("Organisations, companies, clubs named."),
        "places": _string_list("Towns, venues, streets, buildings named."),
        "topics": _string_list(
            "Three to eight subject tags, lowercase, e.g. 'fundraising', "
            "'polio eradication', 'youth exchange'."
        ),
        "rotary_context": _string(
            "Rotary-specific significance: offices held, recognitions such as "
            "Paul Harris Fellow, service projects, district or international "
            "themes. Empty if none apply."
        ),
        "presentation": {
            "type": "string",
            "enum": PRESENTATIONS,
            "description": (
                "How this should appear in the archive. 'text' when the value "
                "is in the words and a transcription serves the reader better "
                "than the scan. 'image' when the object itself is the point. "
                "'both' when the words matter and the artefact does too."
            ),
        },
        "legibility": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "5 = perfectly clear, 1 = mostly unreadable.",
        },
        "condition_notes": _string(
            "Visible damage: tears, foxing, fading, water staining. Empty if "
            "the item is in good condition."
        ),
        "alt_text": _string(
            "One sentence describing the image for a screen reader."
        ),
        "visual_description": _string(
            "What is actually visible in the picture, for a reader who cannot "
            "see it and for search to match on: who and how many people, what "
            "they are doing, the setting, banners, insignia, signage, objects "
            "on show, and anything that dates the scene such as clothing or "
            "vehicles. Describe only what is there. Empty string for an item "
            "that is purely text."
        ),
        "orientation_hint": {
            "type": "string",
            "enum": ORIENTATIONS,
            "description": (
                "'upright' if the content reads correctly. Otherwise the "
                "rotation needed to make it upright."
            ),
        },
        "confidence": {
            "type": "number",
            "description": "0.0 to 1.0. Your confidence in this analysis overall.",
        },
        "needs_human_review": {
            "type": "boolean",
            "description": (
                "True if a person should check this: poor legibility, an "
                "uncertain date, an ambiguous subject, or anything you had to "
                "guess at."
            ),
        },
        "review_reason": _string(
            "If needs_human_review is true, what specifically to check."
        ),
    },
    "required": [
        "item_type", "title", "summary", "full_text",
        "date_value", "date_precision", "date_source", "date_note",
        "people", "organizations", "places", "topics", "rotary_context",
        "presentation", "legibility", "condition_notes", "alt_text",
        "visual_description",
        "orientation_hint", "confidence", "needs_human_review", "review_reason",
    ],
}


SYSTEM_PROMPT = """\
You are cataloguing a Rotary Club's historical archive. Each image is one item \
of memorabilia - a newspaper clipping, photograph, certificate, program, or \
similar - that has been photographed, cropped, and straightened. Your job is to \
read it and record what it is, so club members can search and browse decades of \
their own history.

Return a single JSON object matching the schema. Nothing else.

Two rules override everything else in this prompt.

**Never name a person you cannot read.** Take names only from text that is \
actually on the item: a printed caption, a byline, a certificate, writing on the \
back. Never infer a name from a face, a uniform, a setting, or a resemblance to \
anyone else. An unidentified group photograph gets an empty `people` list. That \
is a correct answer, not a failure - the club will identify these themselves, \
and a confident wrong name is far worse than an honest gap, because it will be \
copied and believed for decades.

**Never invent a date.** If a date is printed on the item, use it and set \
`date_source` to "printed". If you can genuinely deduce one - a named event, a \
mentioned anniversary, a photographic process, a car in the background - set \
`date_source` to "inferred" and explain your reasoning in `date_note`, and set \
`date_precision` no finer than the evidence supports. If you cannot tell, leave \
`date_value` empty and set `date_source` to "unknown". Do not guess a decade \
because the item merely looks old.

On transcription: `full_text` should be a faithful transcription, not a summary. \
Preserve the original wording, spelling, and line breaks. Mark unreadable \
passages `[illegible]` rather than guessing at them. This text is what the \
archive's search will index, so it matters more than any other field.

On describing pictures: `alt_text` is one sentence for a screen reader. \
`visual_description` is the fuller account, and for a photograph it is the \
most important field in the record - there is no transcription to fall back \
on, so it is the only thing the archive's search can match. Say what is \
visible: how many people, what they are doing, the room, the banner behind \
them, the insignia on it, what is on the table. Do not name anyone whose name \
you cannot read on the item, and do not infer the occasion from the setting.

On `presentation`: decide how the item best serves a reader. A dense column of \
newsprint is easier to read as transcribed text, so choose "text". A portrait, \
a banner, or a hand-lettered certificate is the point in itself, so choose \
"image". Choose "both" when the words carry information and the object carries \
character - an illustrated program, a letter on headed paper.

Set `needs_human_review` generously. A flagged item costs someone a few seconds; \
a wrong one that slips through becomes part of the club's record.
"""


def build_context(
    *,
    captured_at: str | None = None,
    source_photo: str | None = None,
    neighbours: int = 0,
) -> str:
    """Per-item user-turn text.

    Deliberately thin, and placed after the image so it never disturbs the
    cached system prefix. Capture date is offered only as a bound on when the
    photograph was taken - it says nothing about the item's own age, and the
    prompt says so explicitly to stop it leaking into `date_value`.
    """
    lines = ["Catalogue this item."]
    if captured_at:
        lines.append(
            f"(The photograph of this item was taken on {captured_at[:10]}. "
            "That is when it was digitised, not when the item itself is from - "
            "do not use it as the item's date.)"
        )
    if source_photo and neighbours:
        lines.append(
            f"(It was photographed alongside {neighbours} other item"
            f"{'s' if neighbours != 1 else ''}, so related material may exist.)"
        )
    return "\n".join(lines)
