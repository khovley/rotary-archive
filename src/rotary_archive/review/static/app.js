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
            ${statusBadge}
          </div>
          ${renderAnalysis(item)}
          ${item.reason ? `<span class="muted reason">${escapeHtml(item.reason)}</span>` : ""}
        </div>
        <div class="card-actions">
          <button data-act="approve" data-item="${item.id}">Approve</button>
          <button data-act="reject" data-item="${item.id}" class="ghost">Reject</button>
          <button data-act="crop" data-item="${item.id}" class="ghost">Crop</button>
          <button data-act="rotate" data-item="${item.id}" class="ghost">⟳</button>
          ${item.analysis
            ? `<button data-act="edit" data-item="${item.id}" class="ghost">Edit</button>`
            : ""}
          <button data-act="reanalyze" data-item="${item.id}" class="ghost"
                  title="Send this item to the model again">Re-read</button>
        </div>
      </article>`;
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

  async function rotate(id) {
    const updated = await api(`/api/item/${id}/rotate`, {
      method: "POST",
      body: JSON.stringify({ degrees: 90 }),
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
          else if (act === "rotate") await rotate(item);
          else if (act === "crop") openEditor(item);
          else if (act === "edit") openFieldEditor(item);
          else if (act === "reanalyze") await reanalyze(item, btn);
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
    const handles = quad.map(([x, y], i) =>
      `<circle class="handle" data-i="${i}" cx="${x}" cy="${y}" r="${Math.max(sw, sh) * 0.012}"></circle>`
    ).join("");

    stage.innerHTML = `
      <img src="/media/photo/${photo.sha256}" alt="">
      <svg viewBox="0 0 ${sw} ${sh}" preserveAspectRatio="none">
        <polygon points="${quad.map(([x, y]) => `${x},${y}`).join(" ")}"></polygon>
        ${handles}
      </svg>`;

    const svg = stage.querySelector("svg");
    svg.querySelectorAll(".handle").forEach((handle) => {
      handle.addEventListener("pointerdown", (ev) => {
        ev.preventDefault();
        handle.setPointerCapture(ev.pointerId);
        const index = Number(handle.dataset.i);

        const move = (e) => {
          const rect = svg.getBoundingClientRect();
          // Map pointer position back into full-resolution source coordinates.
          const x = ((e.clientX - rect.left) / rect.width) * sw;
          const y = ((e.clientY - rect.top) / rect.height) * sh;
          state.editing.quad[index] = [
            Math.max(0, Math.min(sw, x)),
            Math.max(0, Math.min(sh, y)),
          ];
          drawEditor();
        };
        const up = () => {
          window.removeEventListener("pointermove", move);
          window.removeEventListener("pointerup", up);
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
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

    if (ev.key === "Escape") {
      if (state.fieldEditing) { closeFieldEditor(); return; }
      if (state.editing) { closeEditor(); return; }
    }
    if (state.editing || state.fieldEditing) return;

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
        ev.preventDefault(); await rotate(state.selected);
      } else if (key === "c" && state.selected) {
        ev.preventDefault(); openEditor(state.selected);
      } else if (key === "e" && state.selected) {
        ev.preventDefault(); openFieldEditor(state.selected);
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

  // ---------------------------------------------------------------- boot --

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
    } catch (err) {
      app.innerHTML = `<div class="empty">Could not load the archive: ${escapeHtml(err.message)}</div>`;
    }
  })();
})();
