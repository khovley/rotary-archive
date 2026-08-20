/* Review UI.
 *
 * Built for throughput: the default view shows only what needs a human, bulk
 * approval is one keystroke, and every action is reachable without the mouse.
 * Quads are always stored in full-resolution source coordinates; the preview
 * is only ever a display scale, so a corner dragged here maps back to the
 * original pixels.
 */
(() => {
  "use strict";

  const state = {
    photos: [],
    filter: "flagged",
    flagBelow: 0.8,
    selected: null,      // item id
    editing: null,       // crop editor: { item, photo, quad }
    fieldEditing: null,  // field editor: { id, original }
  };

  const $ = (sel) => document.querySelector(sel);
  const app = $("#app");

  // ------------------------------------------------------------- helpers --

  async function api(path, options) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(payload.error || `${res.status} ${res.statusText}`);
    return payload;
  }

  let toastTimer;
  function toast(message, ms = 2200) {
    const el = $("#toast");
    el.textContent = message;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, ms);
  }

  function matchesFilter(item) {
    if (state.filter === "all") return true;
    const undecided = item.status !== "approved" && item.status !== "rejected";
    if (state.filter === "pending") return undecided;
    return undecided && item.flagged;           // "flagged"
  }

  function visibleItems() {
    return state.photos.flatMap((p) => p.items.filter(matchesFilter));
  }

  // --------------------------------------------------------------- render --

  function render() {
    const photos = state.photos
      .map((p) => ({ photo: p, items: p.items.filter(matchesFilter) }))
      .filter((g) => g.items.length > 0);

    if (!photos.length) {
      const msg = state.filter === "flagged"
        ? "Nothing flagged. Every detection looked clean — switch to Pending to review the rest."
        : "Nothing to show for this filter.";
      app.innerHTML = `<div class="empty">${msg}</div>`;
      updateSummary();
      return;
    }

    app.innerHTML = photos.map(renderPhoto).join("");
    wireEvents();
    updateSummary();
  }

  function renderPhoto({ photo, items }) {
    const shown = new Set(items.map((i) => i.id));
    const sw = photo.width || 1;
    const sh = photo.height || 1;

    // Overlays cover every item on the photo, so a missed item is obvious by
    // the gap, not just the ones passing the current filter.
    const shapes = photo.items.map((item, idx) => {
      const pts = item.quad.map(([x, y]) => `${x},${y}`).join(" ");
      const cls = [
        item.flagged ? "flagged" : "",
        item.id === state.selected ? "sel" : "",
      ].join(" ").trim();
      const dim = shown.has(item.id) ? "" : ' opacity="0.35"';
      const [lx, ly] = item.quad[0];
      return `<polygon points="${pts}" class="${cls}"${dim}></polygon>
              <text x="${lx + 30}" y="${ly + 70}"${dim}>${idx + 1}</text>`;
    }).join("");

    return `
      <section class="photo" data-photo="${photo.sha256}">
        <div class="photo-head">
          <div>
            <span class="photo-title">${escapeHtml(photo.name)}</span>
            <span class="muted"> · ${photo.items.length} item(s)${
              photo.captured_at ? " · " + escapeHtml(photo.captured_at.slice(0, 10)) : ""
            }</span>
            ${photo.note ? `<div class="muted">${escapeHtml(photo.note)}</div>` : ""}
          </div>
          <div>
            <button data-act="approve-photo" data-photo="${photo.sha256}">
              Approve these ${items.length}
            </button>
            <button data-act="add-item" data-photo="${photo.sha256}" class="ghost">
              Add missed item
            </button>
            <button data-act="source" data-photo="${photo.sha256}" class="ghost"
                    title="Full-size source photo with every crop drawn on it">
              Inspect crops
            </button>
          </div>
        </div>
        <div class="photo-body">
          <div class="source">
            <img src="/media/photo/${photo.sha256}" alt="Source photo ${escapeHtml(photo.name)}">
            <svg viewBox="0 0 ${sw} ${sh}" preserveAspectRatio="none">${shapes}</svg>
          </div>
          <div class="items">${items.map(renderCard).join("")}</div>
        </div>
      </section>`;
  }

  // A date the model deduced rather than read is shown differently from one
  // printed on the item. Collapsing the two would let a guess harden into a
  // fact the club then believes.
  function dateLabel(a) {
    if (!a.date_value) return '<span class="badge">no date</span>';
    const cls = a.date_source === "printed" ? "ok" : "warn";
    const mark = a.date_source === "printed" ? "" : "~";
    return `<span class="badge ${cls}" title="date ${escapeHtml(a.date_source)}">${
      mark
    }${escapeHtml(a.date_value)}</span>`;
  }

  function renderAnalysis(item) {
    const a = item.analysis;
    if (!a) {
      return '<div class="analysis none muted">Not yet analysed</div>';
    }
    const ents = a.entities || {};
    const chips = ["person", "organization", "place", "topic"]
      .flatMap((kind) => (ents[kind] || []).map(
        (name) => `<span class="chip ${kind}">${escapeHtml(name)}</span>`
      ))
      .join("");

    const text = (a.full_text || "").trim();
    return `
      <div class="analysis">
        <div class="a-title">${escapeHtml(a.title || "(untitled)")}</div>
        <div class="badges">
          ${dateLabel(a)}
          <span class="badge">${escapeHtml(a.item_type || "")}</span>
          <span class="badge">${escapeHtml(a.presentation || "")}</span>
          <span class="badge" title="legibility">leg ${a.legibility ?? "?"}</span>
        </div>
        ${a.summary ? `<p class="a-summary">${escapeHtml(a.summary)}</p>` : ""}
        ${chips ? `<div class="chips">${chips}</div>` : ""}
        ${text ? `<details class="a-text"><summary>Transcription (${
          text.length} chars)</summary><pre>${escapeHtml(text)}</pre></details>` : ""}
        <div class="muted a-provenance">${escapeHtml(a.provider)} · ${escapeHtml(a.model)}</div>
      </div>`;
  }

  function renderCard(item) {
    const conf = (item.confidence ?? 0).toFixed(2);
    const confClass = item.confidence >= state.flagBelow ? "ok" : "warn";
    const statusBadge = item.status === "approved"
      ? '<span class="badge ok">approved</span>'
      : item.status === "rejected"
      ? '<span class="badge bad">rejected</span>'
      : "";

    return `
      <article class="card ${item.id === state.selected ? "sel" : ""} ${
        item.flagged ? "flagged" : ""
      } ${item.status}" data-item="${item.id}" tabindex="0">
        <img class="thumb" src="/media/item/${item.id}" loading="lazy"
             alt="${escapeHtml((item.analysis && item.analysis.alt_text) || "Cropped item " + item.id)}"
             onerror="this.style.opacity=.25">
        <div class="card-meta">
          <span class="card-id">${item.id}</span>
          <div class="badges">
            <span class="badge ${confClass}">crop ${conf}</span>
            <span class="badge">${escapeHtml(item.method || "?")}</span>
            ${item.rotation ? `<span class="badge">${item.rotation}°</span>` : ""}
            ${item.part_of ? '<span class="badge part">part of a set</span>' : ""}
            ${item.duplicate_of ? '<span class="badge dup">duplicate</span>' : ""}
            ${(item.related || []).length
              ? `<span class="badge rel">${item.related.length} linked</span>` : ""}
            ${statusBadge}
          </div>
          ${item.headline
            ? `<span class="headline" title="What the model read off this item">${escapeHtml(item.headline)}</span>`
            : ""}
          ${renderAnalysis(item)}
          ${renderGrouping(item)}
          ${renderLinks(item)}
          ${item.reason ? `<span class="muted reason">${escapeHtml(item.reason)}</span>` : ""}
        </div>
        <div class="card-actions">
          <button data-act="approve" data-item="${item.id}">Approve</button>
          <button data-act="reject" data-item="${item.id}" class="ghost">Reject</button>
          <button data-act="crop" data-item="${item.id}" class="ghost">Crop</button>
          <button data-act="rotate-ccw" data-item="${item.id}" class="ghost rot"
                  title="Rotate left 90° (Shift+R)" aria-label="Rotate left">↺</button>
          <button data-act="rotate" data-item="${item.id}" class="ghost rot"
                  title="Rotate right 90° (R)" aria-label="Rotate right">Rotate ↻</button>
          ${item.analysis
            ? `<button data-act="edit" data-item="${item.id}" class="ghost">Edit</button>`
            : ""}
          <button data-act="source" data-item="${item.id}" class="ghost"
                  title="See this crop on the full source photo (s)">In photo</button>
          <button data-act="reanalyze" data-item="${item.id}" class="ghost"
                  title="Send this item to the model again">Re-read</button>
        </div>
      </article>`;
  }

  /* A clipping the model judged to be part of another one - a story carried
     onto a second strip, a photograph cut out alongside its article. Shown
     with the reason, because this is a judgement call that has to be easy to
     overrule: a wrong link would bury one item inside another on the site. */
  function renderGrouping(item) {
    if (!item.part_of) return "";
    const parent = state.photos
      .flatMap((p) => p.items)
      .find((i) => i.id === item.part_of);
    const name = (parent && parent.headline) || item.part_of;
    return `
      <div class="grouping">
        <span>Part of <a href="#" data-act="goto" data-item="${item.part_of}">${escapeHtml(name)}</a></span>
        ${item.part_reason ? `<span class="muted">${escapeHtml(item.part_reason)}</span>` : ""}
        <button data-act="ungroup" data-item="${item.id}" class="ghost tiny">Not part of it</button>
      </div>`;
  }

  const byItemId = (id) =>
    state.photos.flatMap((p) => p.items).find((i) => i.id === id);

  const itemLabel = (id) => {
    const found = byItemId(id);
    return (found && found.headline) || id;
  };

  /* Two relationships the model can assert that do NOT merge items, shown
     together because both are corrections a human makes in one glance:
     a duplicate hides this item from the site, a link only cross-references
     it. Neither should be silently trusted. */
  function renderLinks(item) {
    const links = item.related || [];
    if (!item.duplicate_of && !links.length) return "";
    return `
      <div class="grouping links">
        ${item.duplicate_of ? `
          <span>Second copy of
            <a href="#" data-act="goto" data-item="${item.duplicate_of}">${
              escapeHtml(itemLabel(item.duplicate_of))}</a>
            — will not be published</span>
          <button data-act="unduplicate" data-item="${item.id}" class="ghost tiny">
            Not a duplicate</button>` : ""}
        ${links.map((link) => `
          <span>See also
            <a href="#" data-act="goto" data-item="${link.id}">${
              escapeHtml(itemLabel(link.id))}</a>${
            link.reason ? ` — ${escapeHtml(link.reason)}` : ""}</span>
          <button data-act="unlink" data-item="${item.id}"
                  data-other="${link.id}" class="ghost tiny">Unlink</button>`).join("")}
      </div>`;
  }

  async function unduplicate(itemId) {
    await api(`/api/item/${itemId}/duplicate`, {
      method: "POST", body: JSON.stringify({ duplicate_of: null }),
    });
    const item = byItemId(itemId);
    if (item) item.duplicate_of = null;
    render();
    toast("Marked as its own item — it will publish");
  }

  async function unlink(itemId, otherId) {
    await api(`/api/item/${itemId}/link`, {
      method: "POST",
      body: JSON.stringify({ related_to: otherId, remove: true }),
    });
    for (const id of [itemId, otherId]) {
      const item = byItemId(id);
      if (item) item.related = (item.related || []).filter(
        (l) => l.id !== (id === itemId ? otherId : itemId)
      );
    }
    render();
    toast("Link removed");
  }

  async function ungroup(itemId) {
    await api(`/api/item/${itemId}/group`, {
      method: "POST",
      body: JSON.stringify({ part_of: null }),
    });
    const item = state.photos.flatMap((p) => p.items).find((i) => i.id === itemId);
    if (item) { item.part_of = null; item.part_reason = null; }
    render();
    toast("Ungrouped — it will publish as its own item");
  }

  function updateSummary() {
    const all = state.photos.flatMap((p) => p.items);
    const undecided = all.filter(
      (i) => i.status !== "approved" && i.status !== "rejected"
    );
    const flagged = undecided.filter((i) => i.flagged);
    $("#summary").textContent =
      `${all.length} items · ${undecided.length} undecided · ${flagged.length} flagged · ` +
      `${visibleItems().length} shown`;
    $("#approve-visible").disabled = visibleItems().length === 0;
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  // -------------------------------------------------------------- actions --

  async function decide(ids, decision) {
    if (!ids.length) return;
    await api("/api/items/decide", {
      method: "POST",
      body: JSON.stringify({ ids, decision }),
    });
    for (const photo of state.photos) {
      for (const item of photo.items) {
        if (ids.includes(item.id)) {
          item.status = decision;
          item.flagged = false;
        }
      }
    }
    toast(`${ids.length} ${decision}`);
    render();
  }

  /* Rotation is stored as a total on the item and the master is re-rectified
     from the original photograph each time, so turning it four times returns
     the exact original pixels rather than accumulating resampling loss. */
  async function rotate(id, degrees = 90) {
    const updated = await api(`/api/item/${id}/rotate`, {
      method: "POST",
      body: JSON.stringify({ degrees }),
    });
    replaceItem(updated.item);
    bustThumb(id);
    render();
  }

  function replaceItem(updated) {
    for (const photo of state.photos) {
      const idx = photo.items.findIndex((i) => i.id === updated.id);
      if (idx >= 0) photo.items[idx] = updated;
    }
  }

  // The derivative filename does not change when a crop does, so the browser
  // would keep showing the stale cached image. Force a re-fetch.
  function bustThumb(id) {
    document.querySelectorAll(`.card[data-item="${id}"] .thumb`).forEach((img) => {
      img.src = `/media/item/${id}?v=${Date.now()}`;
    });
  }

  function selectItem(id) {
    state.selected = id;
    render();
    const card = document.querySelector(`.card[data-item="${id}"]`);
    if (card) card.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function step(delta) {
    const items = visibleItems();
    if (!items.length) return;
    const at = items.findIndex((i) => i.id === state.selected);
    const next = at < 0 ? 0 : Math.min(items.length - 1, Math.max(0, at + delta));
    selectItem(items[next].id);
  }

  function wireEvents() {
    app.querySelectorAll("[data-act]").forEach((btn) => {
      btn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const { act, item, photo } = btn.dataset;
        try {
          if (act === "approve") await decide([item], "approved");
          else if (act === "reject") await decide([item], "rejected");
          else if (act === "rotate") await rotate(item, 90);
          else if (act === "rotate-ccw") await rotate(item, -90);
          else if (act === "crop") openEditor(item);
          else if (act === "edit") openFieldEditor(item);
          else if (act === "reanalyze") await reanalyze(item, btn);
          else if (act === "goto") { ev.preventDefault(); selectItem(item); }
          else if (act === "ungroup") await ungroup(item);
          else if (act === "unduplicate") await unduplicate(item);
          else if (act === "unlink") await unlink(item, btn.dataset.other);
          else if (act === "source") {
            const sha = photo || (state.photos.find((p) =>
              p.items.some((i) => i.id === item)) || {}).sha256;
            if (sha) openSourcePhoto(sha, item);
          }
          else if (act === "approve-photo") {
            const group = state.photos.find((p) => p.sha256 === photo);
            await decide(group.items.filter(matchesFilter).map((i) => i.id), "approved");
          } else if (act === "add-item") startAddItem(photo);
        } catch (err) {
          toast(err.message, 4000);
        }
      });
    });

    app.querySelectorAll(".card").forEach((card) => {
      card.addEventListener("click", () => selectItem(card.dataset.item));
    });

    // The thumbnail opens the full-resolution master, not the 800px
    // derivative - the point is to see whether the crop clipped anything.
    app.querySelectorAll(".card .thumb").forEach((img) => {
      img.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const id = img.closest(".card").dataset.item;
        selectItem(id);
        openLightbox(`/media/item/${id}?full=1`, id);
      });
    });

    app.querySelectorAll(".source").forEach((source) => {
      source.addEventListener("click", () => {
        openSourcePhoto(source.closest(".photo").dataset.photo);
      });
    });
  }

  /* The source photo at full resolution with every crop drawn on it. This is
     the view that answers "did it miss one, and did it cut that headline in
     half" - questions no amount of looking at the individual crops can settle,
     because a crop cannot show you what fell outside it. */
  function openSourcePhoto(sha, highlight) {
    const photo = state.photos.find((p) => p.sha256 === sha);
    if (!photo) return;
    openLightbox(
      `/media/photo/${sha}?full=1`,
      `${photo.name} · ${photo.items.length} crop(s)`,
      photo.items.map((item) => ({
        quad: item.quad,
        label: String(item.seq),
        selected: item.id === (highlight || state.selected),
      }))
    );
  }

  async function reanalyze(id, button) {
    if (button) { button.disabled = true; button.textContent = "reading…"; }
    try {
      const updated = await api(`/api/item/${id}/reanalyze`, { method: "POST" });
      replaceItem(updated.item);
      toast("Re-read by the model");
      render();
    } finally {
      if (button) { button.disabled = false; button.textContent = "Re-read"; }
    }
  }

  // -------------------------------------------------------- field editor --

  // Mirrors EDITABLE_FIELDS on the server. Anything else is the model's
  // reading and is displayed but not editable here.
  const FIELD_SPECS = [
    { key: "title", label: "Title", type: "text" },
    { key: "item_type", label: "Type", type: "select", options: [
      "newspaper_clipping", "photograph", "document", "letter", "certificate",
      "program", "newsletter", "ephemera", "object", "other"] },
    { key: "date_value", label: "Date", type: "text",
      hint: "ISO 8601: 1962-07-14, 1962-07, or 1962. Leave empty if unknown." },
    { key: "date_precision", label: "Date precision", type: "select",
      options: ["day", "month", "year", "decade", "unknown"] },
    { key: "date_source", label: "Date source", type: "select",
      options: ["printed", "inferred", "unknown"],
      hint: "Use 'printed' only if the date is actually written on the item." },
    { key: "presentation", label: "Presentation", type: "select",
      options: ["image", "text", "both"] },
    { key: "summary", label: "Summary", type: "textarea" },
    { key: "full_text", label: "Transcription", type: "textarea", big: true },
    { key: "visual_description", label: "What the picture shows", type: "textarea",
      hint: "For a photograph this is what search matches on - there is no transcription." },
    { key: "rotary_context", label: "Rotary context", type: "textarea" },
    { key: "condition_notes", label: "Condition", type: "text" },
    { key: "alt_text", label: "Alt text", type: "text" },
  ];

  function openFieldEditor(id) {
    const found = findItem(id);
    if (!found || !found.item.analysis) return;
    state.fieldEditing = { id, original: { ...found.item.analysis } };

    const a = found.item.analysis;
    const rows = FIELD_SPECS.map((spec) => {
      const value = a[spec.key] ?? "";
      let control;
      if (spec.type === "select") {
        control = `<select data-field="${spec.key}">${spec.options
          .map((o) => `<option value="${o}"${o === value ? " selected" : ""}>${o}</option>`)
          .join("")}</select>`;
      } else if (spec.type === "textarea") {
        control = `<textarea data-field="${spec.key}" rows="${
          spec.big ? 14 : 3}">${escapeHtml(value)}</textarea>`;
      } else {
        control = `<input type="text" data-field="${spec.key}" value="${escapeHtml(value)}">`;
      }
      return `<label class="field${spec.big ? " wide" : ""}">
        <span>${spec.label}</span>${control}
        ${spec.hint ? `<small class="muted">${spec.hint}</small>` : ""}
      </label>`;
    }).join("");

    $("#fields-title").textContent = `Edit — ${id}`;
    $("#fields-form").innerHTML = rows;
    $("#fields").hidden = false;
  }

  function closeFieldEditor() {
    state.fieldEditing = null;
    $("#fields").hidden = true;
    $("#fields-form").innerHTML = "";
  }

  async function saveFields() {
    if (!state.fieldEditing) return;
    const { id, original } = state.fieldEditing;

    // Send only what actually changed, so the stored override is a record of
    // the human's corrections rather than a copy of the whole analysis.
    const changed = {};
    $("#fields-form").querySelectorAll("[data-field]").forEach((el) => {
      const key = el.dataset.field;
      const value = el.value;
      if (value !== (original[key] ?? "")) changed[key] = value;
    });

    if (!Object.keys(changed).length) { closeFieldEditor(); return; }

    try {
      const updated = await api(`/api/item/${id}/fields`, {
        method: "POST", body: JSON.stringify({ fields: changed }),
      });
      replaceItem(updated.item);
      toast(`Saved ${Object.keys(changed).length} field(s)`);
      closeFieldEditor();
      render();
    } catch (err) {
      toast(err.message, 4000);
    }
  }

  // --------------------------------------------------------- crop editor --

  function findItem(id) {
    for (const photo of state.photos) {
      const item = photo.items.find((i) => i.id === id);
      if (item) return { item, photo };
    }
    return null;
  }

  function openEditor(id, seedQuad) {
    const found = findItem(id);
    if (!found) return;
    state.editing = {
      item: found.item,
      photo: found.photo,
      quad: (seedQuad || found.item.quad).map(([x, y]) => [x, y]),
      isNew: Boolean(seedQuad),
    };
    $("#editor-title").textContent = seedQuad
      ? `New item on ${found.photo.name}`
      : `Adjust crop — ${id}`;
    $("#editor").hidden = false;
    drawEditor();
  }

  function startAddItem(sha) {
    const photo = state.photos.find((p) => p.sha256 === sha);
    if (!photo) return;
    // Seed with a centred rectangle covering a quarter of the frame; the user
    // drags it onto the item they want.
    const w = photo.width, h = photo.height;
    const quad = [
      [w * 0.3, h * 0.3], [w * 0.7, h * 0.3],
      [w * 0.7, h * 0.7], [w * 0.3, h * 0.7],
    ];
    state.editing = { item: null, photo, quad, isNew: true };
    $("#editor-title").textContent = `New item on ${photo.name}`;
    $("#editor").hidden = false;
    drawEditor();
  }

  function drawEditor() {
    const { photo, quad } = state.editing;
    const stage = $("#editor-stage");
    const sw = photo.width || 1, sh = photo.height || 1;
    const radius = Math.max(sw, sh) * 0.012;

    stage.innerHTML = `
      <img src="/media/photo/${photo.sha256}" alt="">
      <svg viewBox="0 0 ${sw} ${sh}" preserveAspectRatio="none">
        <polygon points="${quad.map(([x, y]) => `${x},${y}`).join(" ")}"></polygon>
        ${quad.map(([x, y], i) =>
          `<circle class="handle" data-i="${i}" cx="${x}" cy="${y}" r="${radius}"></circle>`
        ).join("")}
      </svg>`;

    const svg = stage.querySelector("svg");
    const polygon = svg.querySelector("polygon");
    const handles = [...svg.querySelectorAll(".handle")];

    /* Corner positions are written onto the existing nodes rather than
       re-rendering the stage.

       Rebuilding the SVG on every pointermove detached the very element the
       drag closure measures against, and a detached node reports a zero-sized
       rect - so the next event divided by zero, the corner shot to the
       bottom-right of the image, and the handle could not be picked up again.
       It moved correctly for exactly one frame first, which made it look like
       a snapping feature rather than a crash. */
    const paint = () => {
      const current = state.editing.quad;
      polygon.setAttribute(
        "points", current.map(([x, y]) => `${x},${y}`).join(" ")
      );
      handles.forEach((handle, i) => {
        handle.setAttribute("cx", current[i][0]);
        handle.setAttribute("cy", current[i][1]);
      });
    };

    handles.forEach((handle) => {
      handle.addEventListener("pointerdown", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        handle.setPointerCapture(ev.pointerId);
        const index = Number(handle.dataset.i);

        const move = (e) => {
          const rect = svg.getBoundingClientRect();
          // A zero-sized rect means the stage is not laid out. Dividing by it
          // is what produced the jump, so refuse rather than guess.
          if (!rect.width || !rect.height) return;
          const x = ((e.clientX - rect.left) / rect.width) * sw;
          const y = ((e.clientY - rect.top) / rect.height) * sh;
          state.editing.quad[index] = [
            Math.max(0, Math.min(sw, x)),
            Math.max(0, Math.min(sh, y)),
          ];
          paint();
        };
        const up = (e) => {
          if (handle.hasPointerCapture?.(e.pointerId)) {
            handle.releasePointerCapture(e.pointerId);
          }
          window.removeEventListener("pointermove", move);
          window.removeEventListener("pointerup", up);
          window.removeEventListener("pointercancel", up);
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
        window.addEventListener("pointercancel", up);
      });
    });
  }

  function closeEditor() {
    state.editing = null;
    $("#editor").hidden = true;
    $("#editor-stage").innerHTML = "";
  }

  async function saveEditor() {
    const { item, photo, quad, isNew } = state.editing;
    try {
      const result = isNew
        ? await api(`/api/photo/${photo.sha256}/item`, {
            method: "POST", body: JSON.stringify({ quad }),
          })
        : await api(`/api/item/${item.id}/quad`, {
            method: "POST", body: JSON.stringify({ quad }),
          });

      if (isNew) {
        photo.items.push(result.item);
      } else {
        replaceItem(result.item);
        bustThumb(item.id);
      }
      toast(isNew ? "Item added and cropped" : "Re-cropped");
      closeEditor();
      render();
    } catch (err) {
      toast(err.message, 4000);
    }
  }

  // ------------------------------------------------------------ keyboard --

  document.addEventListener("keydown", async (ev) => {
    if (ev.target.matches("input, textarea")) return;

    if (!$("#lightbox").hidden) {
      // The lightbox owns the keyboard while it is open.
      const key = ev.key;
      if (key === "Escape") { ev.preventDefault(); closeLightbox(); }
      else if (key === "+" || key === "=") { ev.preventDefault(); zoomLightbox(1.25); }
      else if (key === "-" || key === "_") { ev.preventDefault(); zoomLightbox(1 / 1.25); }
      else if (key === "0") { ev.preventDefault(); fitLightbox(); }
      else if (key === "1") { ev.preventDefault(); $("#lb-full").click(); }
      else if (key === "b" && lb.boxes.length) {
        ev.preventDefault(); toggleLightboxBoxes();
      }
      return;
    }

    if (ev.key === "Escape") {
      if (!$("#intake").hidden) { closeIntake(); return; }
      if (state.fieldEditing) { closeFieldEditor(); return; }
      if (state.editing) { closeEditor(); return; }
    }
    if (state.editing || state.fieldEditing || !$("#intake").hidden) return;

    const key = ev.key.toLowerCase();
    try {
      if (key === "j") { ev.preventDefault(); step(1); }
      else if (key === "k") { ev.preventDefault(); step(-1); }
      else if (key === "a" && ev.shiftKey) {
        ev.preventDefault();
        await decide(visibleItems().map((i) => i.id), "approved");
      } else if (key === "a" && state.selected) {
        ev.preventDefault();
        const next = nextAfter(state.selected);
        await decide([state.selected], "approved");
        if (next) selectItem(next);
      } else if (key === "x" && state.selected) {
        ev.preventDefault();
        const next = nextAfter(state.selected);
        await decide([state.selected], "rejected");
        if (next) selectItem(next);
      } else if (key === "r" && state.selected) {
        ev.preventDefault(); await rotate(state.selected, 90);
      } else if (key === "R" && state.selected) {
        ev.preventDefault(); await rotate(state.selected, -90);
      } else if (key === "c" && state.selected) {
        ev.preventDefault(); openEditor(state.selected);
      } else if (key === "e" && state.selected) {
        ev.preventDefault(); openFieldEditor(state.selected);
      } else if (key === "z" && state.selected) {
        ev.preventDefault();
        openLightbox(`/media/item/${state.selected}?full=1`, state.selected);
      } else if (key === "s" && state.selected) {
        ev.preventDefault();
        const found = findItem(state.selected);
        if (found) openSourcePhoto(found.photo.sha256, state.selected);
      } else if (key === "p" && state.selected) {
        ev.preventDefault();
        const found = findItem(state.selected);
        if (found) {
          await decide(
            found.photo.items.filter(matchesFilter).map((i) => i.id), "approved"
          );
        }
      } else if (key === "?") {
        $("#help").hidden = !$("#help").hidden;
      }
    } catch (err) {
      toast(err.message, 4000);
    }
  });

  // Remember where we were so approving does not bounce focus to the top.
  function nextAfter(id) {
    const items = visibleItems();
    const at = items.findIndex((i) => i.id === id);
    if (at < 0) return null;
    const following = items[at + 1] || items[at - 1];
    return following ? following.id : null;
  }

  // ------------------------------------------------------------ lightbox --

  /* Judging whether a crop clipped a headline needs real pixels. Thumbnails
     are 172px wide; this opens the full-resolution master. */

  const lb = {
    scale: 1, x: 0, y: 0, natural: [0, 0], panning: false, from: null,
    boxes: [], showBoxes: true,
  };

  function stageBox() {
    return $("#lb-stage").getBoundingClientRect();
  }

  /* `boxes` are quads in the natural pixel coordinates of the image being
     shown, so they survive zooming without any recalculation - the SVG shares
     the transformed layer with the image. */
  function openLightbox(src, title, boxes) {
    const img = $("#lb-img");
    $("#lb-title").textContent = title || "";
    $("#lightbox").hidden = false;

    lb.boxes = boxes || [];
    $("#lb-boxes").hidden = lb.boxes.length === 0;

    img.onload = () => {
      lb.natural = [img.naturalWidth, img.naturalHeight];
      drawLightboxBoxes();
      fitLightbox();
    };
    // Cache-bust so a re-cropped item never shows the previous version.
    img.src = src.includes("?") ? src : `${src}?v=${Date.now()}`;
  }

  function closeLightbox() {
    $("#lightbox").hidden = true;
    $("#lb-img").src = "";
    lb.boxes = [];
  }

  function drawLightboxBoxes() {
    const svg = $("#lb-boxes-svg");
    const [w, h] = lb.natural;
    svg.setAttribute("viewBox", `0 0 ${w || 1} ${h || 1}`);
    svg.style.display = lb.showBoxes && lb.boxes.length ? "" : "none";
    if (!lb.showBoxes || !lb.boxes.length) { svg.innerHTML = ""; return; }

    svg.innerHTML = lb.boxes.map((box) => {
      const points = box.quad.map((p) => `${p[0]},${p[1]}`).join(" ");
      const [x, y] = box.quad[0];
      // The label is drawn at the image's own scale, then counter-scaled at
      // render time by the font size, so it stays legible when zoomed out.
      const size = Math.max(w, h) / 45;
      return `
        <polygon points="${points}" class="${box.selected ? "sel" : ""}"></polygon>
        <text x="${x + size * 0.3}" y="${y + size * 0.3}"
              style="font-size:${size}px">${escapeHtml(box.label)}</text>`;
    }).join("");
  }

  function toggleLightboxBoxes() {
    lb.showBoxes = !lb.showBoxes;
    $("#lb-boxes").classList.toggle("off", !lb.showBoxes);
    drawLightboxBoxes();
  }

  function applyLightbox() {
    const canvas = $("#lb-canvas");
    canvas.style.width = `${lb.natural[0]}px`;
    canvas.style.height = `${lb.natural[1]}px`;
    canvas.style.transform = `translate(${lb.x}px, ${lb.y}px) scale(${lb.scale})`;
    $("#lb-zoom").textContent = `${Math.round(lb.scale * 100)}%`;
  }

  /* Panning must not be able to lose the image. Without this you can flick it
     past the edge of the stage and be left looking at an empty backdrop with
     no way back except Fit - which reads as the viewer being broken. A margin
     of the image always stays on screen. */
  function clampLightbox() {
    const stage = stageBox();
    const w = lb.natural[0] * lb.scale;
    const h = lb.natural[1] * lb.scale;
    const keep = 80;

    lb.x = w <= stage.width
      ? Math.min(Math.max(lb.x, 0), stage.width - w)
      : Math.min(Math.max(lb.x, stage.width - w - keep), keep);
    lb.y = h <= stage.height
      ? Math.min(Math.max(lb.y, 0), stage.height - h)
      : Math.min(Math.max(lb.y, stage.height - h - keep), keep);
  }

  function fitLightbox() {
    const stage = stageBox();
    const [w, h] = lb.natural;
    if (!w || !h) return;
    // Never scale a small crop above 1:1 on "fit" - blowing up a 300px
    // thumbnail to fill a 4K screen just shows interpolation, not detail.
    lb.scale = Math.min(stage.width / w, stage.height / h, 1);
    lb.x = (stage.width - w * lb.scale) / 2;
    lb.y = (stage.height - h * lb.scale) / 2;
    applyLightbox();
  }

  function zoomLightbox(factor, originX, originY) {
    const stage = stageBox();
    const cx = originX ?? stage.width / 2;
    const cy = originY ?? stage.height / 2;
    const next = Math.min(8, Math.max(0.05, lb.scale * factor));
    // Keep the point under the cursor fixed while zooming.
    lb.x = cx - (cx - lb.x) * (next / lb.scale);
    lb.y = cy - (cy - lb.y) * (next / lb.scale);
    lb.scale = next;
    clampLightbox();
    applyLightbox();
  }

  function wireLightbox() {
    const stage = $("#lb-stage");

    $("#lb-close").addEventListener("click", closeLightbox);
    $("#lb-fit").addEventListener("click", fitLightbox);
    $("#lb-in").addEventListener("click", () => zoomLightbox(1.25));
    $("#lb-out").addEventListener("click", () => zoomLightbox(1 / 1.25));
    $("#lb-boxes").addEventListener("click", toggleLightboxBoxes);
    $("#lb-full").addEventListener("click", () => {
      // Centre 1:1 on whatever is currently in the middle of the stage, so
      // zooming to actual size keeps you where you were looking.
      const rect = stageBox();
      const cx = (rect.width / 2 - lb.x) / lb.scale;
      const cy = (rect.height / 2 - lb.y) / lb.scale;
      lb.scale = 1;
      lb.x = rect.width / 2 - cx;
      lb.y = rect.height / 2 - cy;
      clampLightbox();
      applyLightbox();
    });

    stage.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      const rect = stageBox();
      zoomLightbox(
        ev.deltaY < 0 ? 1.15 : 1 / 1.15,
        ev.clientX - rect.left, ev.clientY - rect.top
      );
    }, { passive: false });

    stage.addEventListener("pointerdown", (ev) => {
      lb.panning = true;
      lb.from = [ev.clientX - lb.x, ev.clientY - lb.y];
      stage.classList.add("panning");
      stage.setPointerCapture(ev.pointerId);
    });
    stage.addEventListener("pointermove", (ev) => {
      if (!lb.panning) return;
      lb.x = ev.clientX - lb.from[0];
      lb.y = ev.clientY - lb.from[1];
      clampLightbox();
      applyLightbox();
    });
    ["pointerup", "pointercancel"].forEach((type) =>
      stage.addEventListener(type, () => {
        lb.panning = false;
        stage.classList.remove("panning");
      })
    );

    // Clicking the backdrop closes; clicking the image does not, or panning
    // would dismiss the thing you are trying to inspect.
    stage.addEventListener("click", (ev) => {
      if (ev.target === stage) closeLightbox();
    });

    window.addEventListener("resize", () => {
      if (!$("#lightbox").hidden) fitLightbox();
    });
  }

  // ----------------------------------------------------------- add photos --

  /* Two ways in, because the two situations are genuinely different. A whole
     box of material is hundreds of multi-megabyte HEICs: Finder copies those
     far faster than a browser can, and survives being interrupted. A handful
     someone brings to a meeting is easier to just drop on the page. */

  let inboxPoll;

  async function refreshInbox() {
    try {
      const state = await api("/api/inbox");
      $("#inbox-path").textContent = state.path;

      const n = state.waiting;
      const pill = $("#waiting-pill");
      pill.hidden = n === 0;
      pill.textContent = state.exact ? n : `${n}+`;

      const line = $("#waiting-line");
      const run = $("#run-pipeline");
      if (n === 0) {
        line.textContent = state.present
          ? `Nothing new. All ${state.present} file(s) in the inbox are already in the archive.`
          : "The inbox is empty.";
        run.disabled = true;
        run.textContent = "Process photos";
      } else {
        const count = state.exact ? `${n}` : `${n}+`;
        line.innerHTML = `<strong>${count} new photo${n === 1 ? "" : "s"}</strong> waiting.`;
        run.disabled = false;
        run.textContent = `Process ${count} photo${n === 1 ? "" : "s"}`;
      }
      return state;
    } catch (err) {
      $("#waiting-line").textContent = `Could not read the inbox: ${err.message}`;
      return null;
    }
  }

  function openIntake() {
    $("#intake").hidden = false;
    refreshInbox();
  }

  function closeIntake() {
    $("#intake").hidden = true;
    clearInterval(inboxPoll);
  }

  async function uploadFiles(files) {
    const list = [...files];
    if (!list.length) return;

    const status = $("#upload-status");
    status.hidden = false;

    let done = 0;
    const failures = [];
    for (const file of list) {
      status.textContent = `Uploading ${done + 1} of ${list.length}: ${file.name}`;
      try {
        // One request per file: the server writes it straight to disk, so a
        // 50MB HEIC never sits in memory twice, and progress is per-file.
        await api(`/api/inbox/upload?name=${encodeURIComponent(file.name)}`, {
          method: "POST",
          headers: {},           // let the browser set Content-Type
          body: file,
        });
        done++;
      } catch (err) {
        failures.push(`${file.name}: ${err.message}`);
      }
    }

    status.textContent = failures.length
      ? `Uploaded ${done} of ${list.length}. ${failures[0]}`
      : `Uploaded ${done} file${done === 1 ? "" : "s"}.`;
    if (failures.length) toast(failures[0], 5000);

    await refreshInbox();
  }

  async function runPipeline() {
    const run = $("#run-pipeline");
    run.disabled = true;
    $("#run-progress").hidden = false;

    try {
      await api("/api/process", { method: "POST" });
    } catch (err) {
      toast(err.message, 5000);
      run.disabled = false;
      return;
    }

    // The run is a background thread on the server, so closing this tab does
    // not abandon it - we are only watching.
    const STAGES = { starting: 5, ingest: 25, segment: 65, rectify: 90, done: 100 };
    clearInterval(inboxPoll);
    inboxPoll = setInterval(async () => {
      let job;
      try {
        job = await api("/api/process");
      } catch (err) {
        return;
      }

      $("#run-bar-fill").style.width = `${STAGES[job.stage] ?? 5}%`;
      $("#run-message").textContent = job.message || job.stage;

      const c = job.counts || {};
      const parts = [];
      if (c.ingested != null) parts.push(`${c.ingested} photo(s) added`);
      if (c.duplicates) parts.push(`${c.duplicates} already had`);
      if (c.items != null) parts.push(`${c.items} item(s) found`);
      if (c.flagged != null) parts.push(`${c.flagged} flagged`);
      if (job.elapsed != null) parts.push(`${job.elapsed}s`);
      $("#run-counts").textContent = parts.join(" · ");

      if (job.state === "running") return;

      clearInterval(inboxPoll);
      $("#run-pipeline").disabled = false;

      if (job.state === "error") {
        toast(job.error || "The run stopped early.", 6000);
      } else {
        if (job.error) toast(job.error, 6000);
        toast(`Added ${c.ingested || 0} photo(s), found ${c.items || 0} item(s)`);
        await reload();
        await refreshInbox();
      }
    }, 700);
  }

  async function reload() {
    const data = await api("/api/photos");
    state.photos = data.photos;
    state.flagBelow = data.flag_below ?? 0.8;
    render();
  }

  // ---------------------------------------------------------------- boot --

  wireLightbox();

  document.querySelectorAll(".segmented button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".segmented button").forEach((b) => b.classList.remove("on"));
      btn.classList.add("on");
      state.filter = btn.dataset.filter;
      state.selected = null;
      render();
    });
  });

  $("#approve-visible").addEventListener("click", async () => {
    const ids = visibleItems().map((i) => i.id);
    if (!ids.length) return;
    if (ids.length > 20 && !confirm(`Approve all ${ids.length} visible items?`)) return;
    try { await decide(ids, "approved"); } catch (err) { toast(err.message, 4000); }
  });

  $("#help-toggle").addEventListener("click", () => {
    $("#help").hidden = !$("#help").hidden;
  });

  $("#add-photos").addEventListener("click", openIntake);
  $("#intake-close").addEventListener("click", closeIntake);
  $("#run-pipeline").addEventListener("click", runPipeline);

  $("#open-inbox").addEventListener("click", async () => {
    try {
      const res = await api("/api/inbox/open", { method: "POST" });
      if (!res.ok) toast(`Open it yourself: ${res.path}`, 6000);
    } catch (err) {
      toast(err.message, 5000);
    }
  });

  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => {
    uploadFiles(fileInput.files);
    fileInput.value = "";
  });
  ["dragenter", "dragover"].forEach((type) =>
    dropzone.addEventListener(type, (ev) => {
      ev.preventDefault();
      dropzone.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    dropzone.addEventListener(type, (ev) => {
      ev.preventDefault();
      dropzone.classList.remove("over");
    })
  );
  dropzone.addEventListener("drop", (ev) => {
    if (ev.dataTransfer?.files?.length) uploadFiles(ev.dataTransfer.files);
  });

  // The page itself must swallow drops too, or the browser navigates away from
  // the app and shows the dropped image on its own.
  ["dragover", "drop"].forEach((type) =>
    window.addEventListener(type, (ev) => {
      if (!ev.target.closest?.("#dropzone")) ev.preventDefault();
    })
  );
  $("#editor-cancel").addEventListener("click", closeEditor);
  $("#editor-save").addEventListener("click", saveEditor);
  $("#fields-cancel").addEventListener("click", closeFieldEditor);
  $("#fields-save").addEventListener("click", saveFields);
  $("#editor-reset").addEventListener("click", () => {
    if (!state.editing || !state.editing.item) return;
    state.editing.quad = state.editing.item.quad_detected.map(([x, y]) => [x, y]);
    drawEditor();
  });

  (async function load() {
    try {
      const data = await api("/api/photos");
      state.photos = data.photos;
      state.flagBelow = data.flag_below ?? 0.8;
      render();
      // Surface waiting photos in the toolbar so the inbox is discoverable
      // without having to know the panel exists.
      refreshInbox();
    } catch (err) {
      app.innerHTML = `<div class="empty">Could not load the archive: ${escapeHtml(err.message)}</div>`;
    }
  })();
})();
