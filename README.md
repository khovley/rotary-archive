# Rotary Archive

Turns iPhone photos of analogue club memorabilia — newspaper clippings,
photographs, certificates, programs — into a searchable, interactive history.

You lay several items out on a table and take one photo. The pipeline finds each
item, crops and straightens it, then reads and catalogues it with a vision
model. You approve the results in batches, and the archive is published as a
static site that drops into the club's WordPress page.

**Status: complete.** Ingest, segmentation, rectification, LLM analysis, the
batch review UI, the published static site, and uploading to the club's host all
work end to end.

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
| `rotary build` | Generate the published static site |
| `rotary serve` | Serve the built site locally to check it |
| `rotary publish` | Upload the site to the club's host (dry run by default) |
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

### Which model, measured

Both stages were run against the same three real table shots and the same three
cropped clippings, so this is comparison rather than assumption.

**Segmentation needs Sonnet.** On the same photographs Haiku 4.5 invented source
publications that do not exist - "Littleton Record", "Sample Magazine" - and put
dates out by as much as twenty-five years. It also lost the reasoning the stage
exists for: it found no merges and no duplicate on one photo, no relationships
at all on another, and on the third it linked all six objects to each other,
which collapses three separate events into one and is no more useful than
linking nothing. It did read a folded programme correctly as one object with two
panels.

**Cataloguing is closer.** On three cropped clippings both models returned the
same titles, the same dates with the same `printed`/`unknown` judgements, and
nearly the same people. Two differences matter:

* Haiku transcribed 5-27% less text per item, dropping the masthead line that
  carries the paper and the date. `full_text` is the search index, so text not
  transcribed is history not findable.
* Haiku spelled the same byline "Jennifer" on two items and "Jenniffer" on a
  third. Entities are deduplicated by name, so one reporter becomes two people
  and neither page shows her whole body of work.

**So: Sonnet for both.** The saving from downgrading the cataloguing stage is
around £3-4 per 500 items. That is not a sensible trade against a permanent
record - but the per-stage overrides are there if the club's circumstances
differ, and segmentation is the cheap stage either way, running once per
photograph rather than once per item.


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
config.toml           thresholds, paths, LLM provider, site metadata
archive.db            SQLite - source of truth
inbox/                drop photos here
masters/originals/    untouched source photos, content-addressed
masters/items/        full-resolution rectified items (never published)
derivatives/          web-optimised WebP at 1600 / 800 / 320px
site/                 generated static site (the deliverable)
src/rotary_archive/   the code
tests/                test suite + synthetic table-shot generator
```

**Back up `masters/` separately.** It is deliberately excluded from git — these
are the irreplaceable files, and the rest can always be regenerated from them.

---

## The published site

```bash
rotary build              # everything analysed, for previewing
rotary build --approved-only --serve   # the real publish, opened locally
```

Output lands in `site/` — a self-contained static folder with no server, no
database, and no external requests:

```
site/
  index.html      the archive
  embed.html      WordPress/Elementor snippet, with instructions
  assets/         one CSS file, one JS file
  data/archive.js the whole catalogue
  media/          WebP derivatives only — masters never leave your disk
```

Timeline by decade and year, gallery filtered by item type, live search across
transcriptions and names, and indexes of people, organisations, places, and
topics. Item pages carry the image, the transcription, entity links, and
related items found through shared entities — weighted so a shared *person*
counts for more than a shared topic, because two items naming the same person
are far more likely to be genuinely connected than two both tagged
"fundraising".

**Dates are never flattened.** A date printed on the item renders plainly; one
the model deduced renders as *about 1967* in italic gold, with the reasoning
beside it. The footer explains the distinction. Letting a guess look like a
fact is the failure mode this whole design is arranged against.

An unidentified photograph says so, and invites the club to help rather than
quietly leaving a blank — set `contact` under `[site]` in `config.toml` to
include a way to reach you.

Three things make it robust to where it ends up hosted: **hash routing**, so
deep links work from any subdirectory and inside an iframe with no server
rewrites; **data as a JS assignment rather than a JSON file**, because browsers
block `fetch()` on `file://` and a JSON file would work when served and
silently fail when opened from disk; and **no external requests at all**, so
nothing breaks if a CDN goes away in ten years.

### Putting it on the club's WordPress site

1. Upload `site/` to the host, e.g. `https://yourclub.org/history/`
2. Add an Elementor **HTML** widget and paste the snippet from `site/embed.html`

The snippet includes a `postMessage` height sync so the iframe grows with its
content instead of scrolling inside a short box. If the host can't serve static
subfolders, put the site on any static host and point the iframe there —
nothing else changes.

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

## How items relate to one another

A table of memorabilia is rarely a set of unrelated objects, and the ways they
relate are not all the same. Collapsing them into one idea damages the record,
so there are three, and the vision pass decides which applies.

**`part_of` — merges.** The two pieces are *the same document*: a story carried
onto a second strip, a column that spilled over, a photograph cut from the
article it illustrated. They publish as one entry, one title, with the extra
scans as further pages and their text folded into the searchable body — so a
phrase from the continuation still finds the story.

**`related_to` — links without merging.** The two items document the same
subject or occasion but are separate things in their own right: a ticket and
the programme for that night, a charity's own brochure lying beside a newspaper
story about that charity. Each keeps its own entry, kind and date, and each
shows a **See also** link to the other.

The test is authorship, not subject. Would the same person have printed both,
as parts of one thing? Then it is a merge. Different origins that happen to be
about the same event? Then it is a link. Merging those would attribute one
publisher's work to another.

**`duplicate_of` — hides one copy.** The same page cut out twice, which is
ordinary in a scrapbook. Both stay in the archive — they may be cropped or lit
differently — but only one reaches the site, so the timeline, the search index
and the entity counts agree with each other.

Every one of these is a judgement, so each carries a `link_confidence` separate
from the confidence in the crop itself. An uncaptioned photograph lying under an
article is *probably* that article's photograph, but position is weak evidence —
especially on a table where every item is about a similar subject. A low number
there sends it to review rather than into the record. All three can be corrected
in one click on the review page.

Separately from all of this, the site computes a **Related items** band from
shared people, organisations, places and topics. That is a good guess; the links
above are assertions. They are shown apart so a guess is never laundered into a
claim.

## What makes the archive searchable

Search runs client-side over the whole corpus — no server, works offline,
instant at this scale.

For anything with text, `full_text` is a verbatim transcription of every legible
word, with `[illegible]` where the model could not read rather than a guess.
Items whose content is really the words also get a Markdown rendering in
`exports/`, so a clipping becomes a readable article with the scan alongside.

For a photograph there is no transcription, so the description *is* the record.
`visual_description` says what is actually visible — how many people, what they
are doing, the room, the banner behind them, the insignia on it, anything that
dates the scene. `alt_text` stays a single sentence for screen readers. Both are
indexed, along with `rotary_context`, the summary, the title, extracted entities
and the date. A photograph with neither transcription nor visual description is
flagged: it would be in the archive and unfindable.

## Publishing

See **[DEPLOY.md](DEPLOY.md)** for the full walkthrough. In short:

```toml
[publish]
method      = "rsync"        # rsync | sftp | local
host        = "yourclub.org"
user        = "your-ssh-username"
remote_path = "/home/you/public_html/history"
```

```bash
rotary publish              # preview: what would change, nothing sent
rotary publish --execute    # upload, after one more confirmation
```

Three safety properties, because this is the one stage that reaches outside the
machine and cannot be undone from here:

- **Dry run is the default.** Uploading takes `--execute`, and then asks again.
- **Deletion is opt-in and needs both flags.** `--delete` alone is still a
  preview. The target directory may hold files this tool did not put there.
- **No password is ever handled here.** Transfers shell out to the system
  `ssh`, so keys and agents work as usual and nothing secret sits in
  `config.toml` — or in your backups and git history.

A build older than the last change to the archive is flagged before anything is
sent, and `archive.db` is on a hard exclusion list that config cannot override.

## Possible later

If the club wants it: a public "help us identify" form feeding corrections back
into the archive, and OCR-assisted search across handwriting.
