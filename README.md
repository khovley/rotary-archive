# Rotary Archive

Turns iPhone photos of analogue club memorabilia — newspaper clippings,
photographs, certificates, programs — into a searchable, interactive history.

You lay several items out on a table, take one photo, and the pipeline finds
each item, crops and straightens it, and (from Phase 2) reads and catalogues it
with an LLM. You approve the results in batches, and the archive is published as
a static site that drops into the club's WordPress site.

**Status: Phases 1–2 complete.** Ingest, segmentation, rectification, LLM
analysis, and the batch review UI all work. The site builder and publishing are
still to come — see *Roadmap* below.

---

## Quick start

```bash
# One-time setup
python3 -m venv .venv
.venv/bin/pip install -e .

# Every time
cp ~/Pictures/table-shots/*.HEIC inbox/
.venv/bin/rotary run
```

`run` ingests, segments, rectifies, and opens the review page at
<http://127.0.0.1:8765>. Read [SHOOTING.md](SHOOTING.md) **before** photographing
the bulk of the collection — background contrast is the single biggest factor
in how well the automatic cropping works.

---

## Commands

| Command | What it does |
|---|---|
| `rotary status` | What's in the archive and what stage everything is at |
| `rotary ingest` | Copy photos from `inbox/` into the archive, skipping duplicates |
| `rotary segment` | Find the individual items in each photo |
| `rotary rectify` | Crop, deskew, and write masters plus web derivatives |
| `rotary analyze` | Read and catalogue each item with a vision model |
| `rotary review` | Open the batch approval UI |
| `rotary run` | ingest → segment → rectify, then review |
| `rotary reset` | Discard items and crops so the pipeline can re-run |

Every stage skips work that's already done, so `rotary run` after adding new
photos only processes the new ones. An interrupted run can simply be restarted.

`--force` on `segment`, `rectify`, or `analyze` re-does completed work. Note
that `segment --force` discards manual crop corrections.

**`analyze` is the only command that costs money**, so it is deliberately not
part of `rotary run` unless you ask for it with `rotary run --analyze`. It
prints a cost estimate and asks for confirmation before sending anything:

```bash
rotary analyze --dry-run      # what would be sent, and what it would cost
rotary analyze --limit 10     # a trial run before committing to the whole lot
rotary analyze                # the real thing
```

---

## The review page

Built for getting through hundreds of items quickly.

- **Flagged** (the default view) shows only what the software was unsure about.
  Everything else can be approved in one click via *Approve all visible*.
- Each photo is shown with its detected crops outlined, so a **missed item is
  visible as a gap** rather than something you have to notice was absent.
- **Add missed item** draws a new crop by hand; **Crop** adjusts an existing
  one by dragging its corners. Both re-crop from the original full-resolution
  photo, not the preview.

Once items are analysed, each card also shows the catalogue reading: title,
type, date, entity chips, and the transcription. A date the model **deduced**
is marked `~` and coloured differently from one **printed** on the item —
collapsing those two would let a guess harden into a fact the club believes.

**Edit** corrects any field; the correction is saved as a new revision with
provider `human`, so the model's original reading survives beside it.
**Re-read** sends that one item back to the model.

| Key | Action |
|---|---|
| `J` / `K` | next / previous item |
| `A` / `X` | approve / reject |
| `R` | rotate 90° |
| `C` | open the crop editor |
| `E` | edit catalogue fields |
| `P` | approve the rest of this photo |
| `Shift`+`A` | approve everything visible |

Re-cropping an approved item returns it to pending — the approval applied to
the image that was approved, and a new crop is a different image.

---

## How it works

```
inbox/ ─► ingest ─► segment ─► rectify ─► analyze ─► review ─► build ─► publish
                                                                (Phase 3) (Phase 4)
```

**Ingest** content-addresses each photo by SHA-256, so re-dropping the same file
is a no-op and the inbox can stay messy. EXIF orientation is applied on read;
GPS is dropped via an allowlist, not a blocklist.

**Segment** runs three independent detection passes over a downscaled copy —
edge, adaptive-threshold, and background-subtraction — and merges the results.
The passes fail in opposite directions: edge and threshold crop inside the paper
to the printed content, while background subtraction can over-grow into a
shadow. When they disagree the widest quad wins, because cropping wide only adds
a margin a human can trim, while cropping inside destroys content. A separate
guard drops any detection that has engulfed two smaller, non-overlapping ones —
a merged pair is the only error that cannot be fixed in review, since the lost
item never appears at all.

**Confidence** is measured by sampling the pixels just outside each quad and
asking whether they look like the table or still look like the item. If they
look like the item, the crop is inside the object's real edge. This replaced an
earlier pass-agreement heuristic that was actively misleading — the passes
disagree precisely when the widest one is correct, so agreement measured
consensus rather than accuracy, and the two come apart exactly in the
low-contrast case where the score matters most.

**Rectify** perspective-warps the quad flat, then measures residual tilt with a
projection profile — rotating the binarised ink through a coarse-to-fine search
and taking the angle where the row-sum profile is sharpest. A Hough transform
was tried first and rejected: it resolves lines well but quantises angle by the
accumulator's theta step, and it reported exactly 0.0° for visibly crooked
scans. Post-warp corrections are fractions of a degree, which is the difference
between "scanned" and "photographed" for newsprint.

**Analyze** sends one image per item — the 1600px derivative, not the master,
which halves the image tokens with no legibility loss on an already-rectified
crop. The system prompt is byte-identical for every item and sits ahead of the
image, so it caches and bills at roughly a tenth of the input rate after the
first call; the whole pass goes through the Batch API at half price. Responses
are constrained to a JSON schema, then normalised — a model that answers `85`
for a 0–1 confidence, or a comma-joined string where a list was asked for, gets
coerced rather than discarded.

**SQLite is the source of truth.** Analyses are versioned rather than
overwritten, so re-running with a better model later keeps the history and never
requires re-shooting or re-cropping anything. Human corrections are written as
new revisions with provider `human`, and human-added entity links survive a
re-analysis — otherwise correcting the archive would be pointless.

## Swapping the model

Everything above the provider layer is model-agnostic. Change one line in
`config.toml`:

```toml
[llm]
provider = "anthropic"   # anthropic | openai | gemini | ollama | claude_cli
model    = "claude-opus-5"
```

| Provider | Notes |
|---|---|
| `anthropic` | Default. Batch API, prompt caching, schema-constrained output. |
| `openai` | Needs `OPENAI_API_KEY` and the `[openai]` extra. |
| `gemini` | Needs `GEMINI_API_KEY` and the `[gemini]` extra. |
| `ollama` | **Local and free.** No dependency, no data leaves the machine. Slower and less accurate — good for a first pass a human then corrects. |
| `claude_cli` | Runs through an existing Claude subscription rather than per-token API billing. Serial and slower; fine for tens of items, not thousands. |

Providers without a batch endpoint get a bounded thread pool instead, so the
CLI behaves identically either way. Providers that can't be schema-constrained
get the schema in the prompt and their output validated on return.

## What it costs

Estimated before every run and shown for confirmation. For 500 items:

| Model | Batch | Sync |
|---|---|---|
| `claude-opus-5` | ~$8.60 | ~$17.20 |
| `claude-sonnet-5` | ~$5.20 | ~$10.30 |
| `claude-haiku-4-5` | ~$1.70 | ~$3.40 |

`ollama` is free; `claude_cli` bills to your subscription rather than per token.

---

## Measured behaviour

From the synthetic benchmark in `tests/` (8 scenes, 47 items, ground-truth
corners), run via `pytest`:

| Metric | Result |
|---|---|
| Items found at IoU ≥ 0.8 | 46 / 47 (97.9%) |
| Scenes with the exact item count | 8 / 8 |
| Loose crops correctly flagged for review | 1 / 1 |
| Deskew accuracy against known angles | ±0.08° |
| Residual skew across 29 rectified items | median 0.06°, max 0.38° |
| Throughput | 29 items from 5 photos in ~5s |

The single loose crop is on a pale-background scene — the setup
[SHOOTING.md](SHOOTING.md) tells you to avoid — and it is flagged, so it surfaces
in review rather than passing silently.

---

## Layout

```
config.toml           thresholds, paths, LLM provider settings
archive.db            SQLite - source of truth
inbox/                drop photos here
masters/originals/    untouched source photos, content-addressed
masters/items/        full-resolution rectified items (never published)
derivatives/          web-optimised WebP at 1600 / 800 / 320px
site/                 generated static site (Phase 3)
src/rotary_archive/   the code
tests/                test suite + synthetic table-shot generator
```

**Back up `masters/` separately.** It is deliberately excluded from git — these
are the irreplaceable files, and the rest can always be regenerated from them.

---

## Two rules the cataloguing prompt enforces

**Never name a person you cannot read.** Names come only from text on the item —
a caption, a byline, a certificate. Never from a face, a uniform, or a
resemblance. An unidentified group photograph gets an empty `people` list and a
flag asking a club member to help. A confident wrong name is far worse than an
honest gap, because it gets copied and believed for decades.

**Never invent a date.** `date_source` records `printed`, `inferred`, or
`unknown`; an inferred date must carry its reasoning in `date_note`; and a date
with no value can never claim to be printed, whatever the model returned. The
review UI and the site display the three differently.

Both were verified against a real model on an unlabelled group photograph with
no text of any kind: it returned no names, `date_source: unknown`, and flagged
the item for a human.

## Roadmap

- **Phase 3** — Static site: timeline, client-side search, people and topic
  indexes, item detail pages.
- **Phase 4** — Publishing and the WordPress/Elementor embed.
