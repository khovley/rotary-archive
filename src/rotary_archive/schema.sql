-- Rotary Archive schema.
--
-- Design notes:
--   * `photos` are the source table shots you took. Immutable once ingested.
--   * `items` are individual pieces of memorabilia cropped out of a photo.
--     One photo yields many items. An item's quad can be corrected by a human
--     without losing the original detection (see quad vs quad_detected).
--   * `analyses` are versioned. Re-running with a better model appends a row
--     rather than overwriting, so nothing is ever lost and results are
--     comparable across models.
--   * `entities` are deduplicated people/places/orgs/topics, which is what
--     makes cross-referencing ("everything mentioning Harold Pratt") possible.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---------------------------------------------------------------- photos ---

CREATE TABLE IF NOT EXISTS photos (
    sha256          TEXT PRIMARY KEY,
    original_name   TEXT NOT NULL,
    stored_path     TEXT NOT NULL,      -- relative to project root
    width           INTEGER,
    height          INTEGER,
    captured_at     TEXT,               -- ISO 8601, from EXIF; NULL if absent
    exif_json       TEXT,               -- retained subset, GPS stripped
    bytes           INTEGER,
    ingested_at     TEXT NOT NULL,
    -- ingested -> segmented -> rectified
    status          TEXT NOT NULL DEFAULT 'ingested',
    segment_note    TEXT                -- why detection was unusual, if it was
);

CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);
CREATE INDEX IF NOT EXISTS idx_photos_captured ON photos(captured_at);

-- ----------------------------------------------------------------- items ---

CREATE TABLE IF NOT EXISTS items (
    id                   TEXT PRIMARY KEY,   -- <photo_sha[:12]>-<nn>
    photo_sha256         TEXT NOT NULL REFERENCES photos(sha256) ON DELETE CASCADE,
    seq                  INTEGER NOT NULL,   -- order within the source photo

    -- Corner quads in FULL-RESOLUTION source coordinates, ordered TL,TR,BR,BL.
    -- Stored as JSON [[x,y],[x,y],[x,y],[x,y]].
    quad                 TEXT NOT NULL,      -- current (possibly human-corrected)
    quad_detected        TEXT NOT NULL,      -- what segmentation originally proposed

    detection_confidence REAL NOT NULL DEFAULT 0.0,
    detection_method     TEXT,               -- edge | threshold | merged | whole_frame | manual

    -- Populated by rectify.
    master_path          TEXT,
    master_width         INTEGER,
    master_height        INTEGER,
    fine_skew_deg        REAL,
    rotation_applied     INTEGER NOT NULL DEFAULT 0,   -- extra 0/90/180/270 from review

    -- detected -> rectified -> analyzed -> approved | rejected
    status               TEXT NOT NULL DEFAULT 'detected',
    needs_human_review   INTEGER NOT NULL DEFAULT 0,
    review_reason        TEXT,

    -- Set when a vision pass judged this item to be a continuation of, or a
    -- piece belonging to, another item: a story carried onto a second strip, a
    -- photograph cut out alongside the article it illustrates.
    part_of_item_id      TEXT REFERENCES items(id) ON DELETE SET NULL,
    part_reason          TEXT,
    headline             TEXT,   -- what the model read, for identification

    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,

    UNIQUE (photo_sha256, seq)
);

CREATE INDEX IF NOT EXISTS idx_items_photo ON items(photo_sha256);
CREATE INDEX IF NOT EXISTS idx_items_status ON items(status);
CREATE INDEX IF NOT EXISTS idx_items_flagged ON items(needs_human_review, status);
-- idx_items_part_of is created by the migration in db.py: on a database that
-- predates the column, this file's CREATE TABLE IF NOT EXISTS is a no-op and
-- the column does not exist yet when the script runs.

-- ------------------------------------------------------------ derivatives ---

CREATE TABLE IF NOT EXISTS derivatives (
    item_id   TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    long_edge INTEGER NOT NULL,
    path      TEXT NOT NULL,
    width     INTEGER NOT NULL,
    height    INTEGER NOT NULL,
    bytes     INTEGER,
    PRIMARY KEY (item_id, long_edge)
);

-- -------------------------------------------------------------- analyses ---

-- Versioned. The newest row per item with superseded=0 is the live analysis.
CREATE TABLE IF NOT EXISTS analyses (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id            TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    provider           TEXT NOT NULL,
    model              TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    superseded         INTEGER NOT NULL DEFAULT 0,

    item_type          TEXT,
    title              TEXT,
    summary            TEXT,
    full_text          TEXT,

    date_value         TEXT,   -- ISO 8601, may be partial (1962, 1962-07)
    date_precision     TEXT,   -- day | month | year | decade | unknown
    date_source        TEXT,   -- printed | inferred | unknown
    date_note          TEXT,   -- reasoning when inferred

    presentation       TEXT,   -- image | text | both
    legibility         INTEGER,
    condition_notes    TEXT,
    alt_text           TEXT,
    rotary_context     TEXT,

    orientation_hint   TEXT,   -- upright | rotate_90_cw | rotate_90_ccw | rotate_180
    confidence         REAL,
    needs_human_review INTEGER NOT NULL DEFAULT 0,
    review_reason      TEXT,

    raw_json           TEXT NOT NULL,   -- full model response, for reprocessing
    usage_json         TEXT             -- token counts / cost accounting
);

CREATE INDEX IF NOT EXISTS idx_analyses_item ON analyses(item_id, superseded);

-- Human edits live separately from model output so both survive.
CREATE TABLE IF NOT EXISTS item_overrides (
    item_id    TEXT PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    fields_json TEXT NOT NULL,     -- {"title": "...", "date_value": "..."}
    updated_at TEXT NOT NULL,
    updated_by TEXT
);

-- ------------------------------------------------------------- entities ---

CREATE TABLE IF NOT EXISTS entities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,   -- person | organization | place | topic
    name       TEXT NOT NULL,
    slug       TEXT NOT NULL,   -- normalised for matching and URLs
    created_at TEXT NOT NULL,
    UNIQUE (kind, slug)
);

CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind);

CREATE TABLE IF NOT EXISTS item_entities (
    item_id   TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    -- Provenance matters: a name read off a printed caption is trustworthy,
    -- a name a human supplied later is trustworthy for different reasons.
    source    TEXT NOT NULL DEFAULT 'model',   -- model | human
    PRIMARY KEY (item_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_item_entities_entity ON item_entities(entity_id);

-- ----------------------------------------------------------- review log ---

CREATE TABLE IF NOT EXISTS review_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id    TEXT REFERENCES items(id) ON DELETE CASCADE,
    action     TEXT NOT NULL,   -- approve | reject | recrop | rotate | edit | add | delete
    detail     TEXT,
    actor      TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_log_item ON review_log(item_id);

-- ---------------------------------------------------------- convenience ---

-- The live analysis for each item: newest non-superseded row.
CREATE VIEW IF NOT EXISTS current_analyses AS
SELECT a.*
FROM analyses a
WHERE a.superseded = 0
  AND a.id = (
      SELECT MAX(a2.id) FROM analyses a2
      WHERE a2.item_id = a.item_id AND a2.superseded = 0
  );
