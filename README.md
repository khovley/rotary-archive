# Rotary Archive

Turns iPhone photos of analogue club memorabilia — newspaper clippings,
photographs, certificates, programs — into a searchable, interactive history.

You lay several items out on a table, take one photo, and the pipeline finds
each item, crops and straightens it, and (from Phase 2) reads and catalogues it
with an LLM. You approve the results in batches, and the archive is published as
a static site that drops into the club's WordPress site.

**Status: Phase 1 complete.** Ingest, segmentation, rectification, and the
batch review UI all work. LLM analysis, the site builder, and publishing are
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
| `rotary review` | Open the batch approval UI |
| `rotary run` | ingest → segment → rectify, then review |
| `rotary reset` | Discard items and crops so the pipeline can re-run |

Every stage skips work that's already done, so `rotary run` after adding new
photos only processes the new ones. An interrupted run can simply be restarted.

`--force` on `segment` or `rectify` re-does completed work. Note that
`segment --force` discards manual crop corrections.

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

| Key | Action |
|---|---|
| `J` / `K` | next / previous item |
| `A` / `X` | approve / reject |
| `R` | rotate 90° |
| `C` | open the crop editor |
| `P` | approve the rest of this photo |
| `Shift`+`A` | approve everything visible |

Re-cropping an approved item returns it to pending — the approval applied to
the image that was approved, and a new crop is a different image.

---

## How it works

```
inbox/ ──► ingest ──► segment ──► rectify ──► review ──► build ──► publish
                                                          (Phase 3)  (Phase 4)
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

**SQLite is the source of truth.** Analyses are versioned rather than
overwritten, so re-running with a better model later keeps the history and never
requires re-shooting or re-cropping anything.

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

## Roadmap

- **Phase 2** — LLM analysis. Transcription, dating, entity extraction, and the
  keep-as-image vs convert-to-text decision. Provider-agnostic: Claude by
  default, with OpenAI, Gemini, and local Ollama adapters behind one interface.
- **Phase 3** — Static site: timeline, client-side search, people and topic
  indexes, item detail pages.
- **Phase 4** — Publishing and the WordPress/Elementor embed.

Two rules already designed into the schema for Phase 2: never guess who is in an
unlabelled photograph (names come only from captions and printed text), and never
fabricate a date — `date_source` records whether a date was printed, inferred, or
is unknown, and the site will display those differently.
