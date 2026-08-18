/* Rotary Archive - published site.
 *
 * Hash routing throughout, for three reasons that all matter here: the site
 * has to work from any subdirectory on the club's host, it has to work inside
 * an iframe embedded in a WordPress page, and it has to work opened straight
 * from disk. Path-based routing would need server rewrites and would break all
 * three.
 *
 * No build step, no dependencies, no external requests.
 */
(() => {
  "use strict";

  const A = window.ARCHIVE || { items: [], entities: {}, timeline: [], counts: {} };
  const byId = new Map(A.items.map((item) => [item.id, item]));
  const main = document.getElementById("main");

  const KIND_LABEL = {
    person: "People", organization: "Organisations",
    place: "Places", topic: "Topics",
  };
  const KIND_ROUTE = {
    person: "person", organization: "org", place: "place", topic: "topic",
  };
  const ROUTE_KIND = Object.fromEntries(
    Object.entries(KIND_ROUTE).map(([k, v]) => [v, k])
  );

  // ------------------------------------------------------------- helpers --

  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

  function media(item, size) {
    const sizes = item.sizes || [];
    if (!sizes.length) return "";
    // Nearest available size at or above the request, else the largest there is.
    const atLeast = sizes.filter((s) => s >= size);
    const chosen = atLeast.length ? Math.min(...atLeast) : Math.max(...sizes);
    return `media/${item.id}-${chosen}.webp`;
  }

  function srcset(item) {
    return (item.sizes || [])
      .map((s) => `media/${item.id}-${s}.webp ${s}w`)
      .join(", ");
  }

  /* A date the model deduced is never rendered as though it were printed on
     the item. Collapsing the two would let a guess harden into a fact. */
  function dateHtml(item, { long = false } = {}) {
    if (!item.date_display) return '<span class="date none">Date unknown</span>';
    const approx = item.date_source !== "printed";
    const label = approx ? `about ${item.date_display}` : item.date_display;
    // The tooltip says only *how* the date was arrived at. When the reasoning
    // is shown inline below, repeating it here would stutter.
    const title = approx
      ? "Deduced from the item, not printed on it"
      : "Printed on the item";
    return `<span class="date${approx ? " approx" : ""}" title="${title}">${
      esc(label)}</span>${long && approx && item.date_note
        ? `<span class="date-note muted"> — ${esc(item.date_note)}</span>` : ""}`;
  }

  function typeLabel(type) {
    return String(type || "other").replace(/_/g, " ");
  }

  function setTitle(part) {
    document.title = part ? `${part} — ${A.club} Archive` : `${A.club} Archive`;
  }

  // -------------------------------------------------------------- search --

  /* A linear scan rather than a prebuilt index. At a few hundred to a few
     thousand items the whole corpus is a couple of megabytes, which scans in
     well under a frame; an index would add build complexity and another file
     to load for no perceptible gain at this size. */
  function search(query) {
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return [];

    const results = [];
    for (const item of A.items) {
      const title = item.title.toLowerCase();
      const names = [...item.people, ...item.orgs, ...item.places, ...item.topics]
        .map((e) => e.name.toLowerCase()).join(" ");
      const body = (item.summary + " " + item.text).toLowerCase();

      let score = 0;
      let matchedAll = true;
      for (const term of terms) {
        let hit = 0;
        if (title.includes(term)) hit += 10;
        if (names.includes(term)) hit += 6;
        if (body.includes(term)) hit += 2;
        if (item.date.includes(term)) hit += 4;
        if (!hit) { matchedAll = false; break; }
        score += hit;
      }
      if (matchedAll) results.push({ item, score });
    }
    return results.sort((a, b) => b.score - a.score).map((r) => r.item);
  }

  function snippet(item, query) {
    const text = item.text || item.summary || "";
    if (!text) return "";
    const term = query.toLowerCase().split(/\s+/).filter(Boolean)[0] || "";
    const at = text.toLowerCase().indexOf(term);
    if (at < 0) return esc(text.slice(0, 180)) + (text.length > 180 ? "…" : "");
    const start = Math.max(0, at - 70);
    const raw = text.slice(start, at + 130);
    return (start ? "…" : "") + esc(raw).replace(
      new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "ig"),
      "<mark>$1</mark>"
    ) + "…";
  }

  // --------------------------------------------------------------- cards --

  function card(item) {
    const img = media(item, A.sizes.card);
    return `
      <a class="card" href="#/item/${encodeURIComponent(item.id)}">
        <div class="card-img">
          ${img ? `<img src="${img}" srcset="${srcset(item)}"
             sizes="(max-width: 600px) 45vw, 260px"
             alt="${esc(item.alt)}" loading="lazy" decoding="async">`
            : '<div class="noimg"></div>'}
        </div>
        <div class="card-body">
          <h3>${esc(item.title)}</h3>
          <div class="card-meta">
            ${dateHtml(item)}
            <span class="tag">${esc(typeLabel(item.type))}</span>
          </div>
        </div>
      </a>`;
  }

  function grid(items, emptyMessage) {
    if (!items.length) return `<p class="empty">${esc(emptyMessage)}</p>`;
    return `<div class="grid">${items.map(card).join("")}</div>`;
  }

  // --------------------------------------------------------------- views --

  function viewHome() {
    setTitle("");
    const c = A.counts;
    const withPhotos = A.items.filter((i) => i.type === "photograph").slice(0, 8);
    const decades = A.timeline.filter((d) => d.decade !== "Undated");

    return `
      <section class="hero">
        <h1>${esc(A.club)}</h1>
        <p class="lede">${esc(A.tagline)}</p>
        <dl class="stats">
          <div><dt>Items</dt><dd>${c.items}</dd></div>
          <div><dt>People</dt><dd>${c.people}</dd></div>
          <div><dt>Places</dt><dd>${c.places}</dd></div>
          <div><dt>Topics</dt><dd>${c.topics}</dd></div>
        </dl>
      </section>

      ${decades.length ? `
      <section class="band">
        <h2>Jump to a decade</h2>
        <div class="decades">
          ${decades.map((d) => `
            <a class="decade" href="#/decade/${encodeURIComponent(d.decade)}">
              <span class="decade-name">${esc(d.decade)}</span>
              <span class="decade-count">${d.count}</span>
            </a>`).join("")}
        </div>
      </section>` : ""}

      ${withPhotos.length ? `
      <section class="band">
        <h2>Photographs</h2>
        ${grid(withPhotos, "")}
        <p><a class="more" href="#/gallery">See everything →</a></p>
      </section>` : ""}
    `;
  }

  function viewTimeline() {
    setTitle("Timeline");
    if (!A.timeline.length) return '<p class="empty">Nothing in the archive yet.</p>';

    return `
      <h1>Timeline</h1>
      ${A.timeline.map((block) => `
        <section class="decade-block" id="decade-${esc(block.decade)}">
          <h2>${esc(block.decade)} <span class="muted">${block.count}</span></h2>
          ${block.years.map((y) => `
            <div class="year-row">
              <h3>${esc(y.year)}</h3>
              ${grid(y.items.map((id) => byId.get(id)).filter(Boolean), "")}
            </div>`).join("")}
        </section>`).join("")}
    `;
  }

  function viewDecade(decade) {
    setTitle(decade);
    const block = A.timeline.find((b) => b.decade === decade);
    if (!block) return '<p class="empty">No items from that decade.</p>';
    const items = block.years.flatMap((y) => y.items)
      .map((id) => byId.get(id)).filter(Boolean);
    return `
      <h1>${esc(decade)}</h1>
      <p class="muted">${items.length} item${items.length === 1 ? "" : "s"}</p>
      ${grid(items, "Nothing here.")}`;
  }

  function viewGallery(type) {
    setTitle("Gallery");
    const types = [...new Set(A.items.map((i) => i.type))].sort();
    const items = type ? A.items.filter((i) => i.type === type) : A.items;

    return `
      <h1>Gallery</h1>
      <div class="filters">
        <a class="chip${!type ? " on" : ""}" href="#/gallery">All ${A.items.length}</a>
        ${types.map((t) => `
          <a class="chip${t === type ? " on" : ""}"
             href="#/gallery/${encodeURIComponent(t)}">${esc(typeLabel(t))}</a>`
        ).join("")}
      </div>
      ${grid(items, "Nothing here.")}`;
  }

  function viewEntityIndex(kind) {
    setTitle(KIND_LABEL[kind]);
    const entries = A.entities[kind] || [];
    if (!entries.length) {
      return `<h1>${KIND_LABEL[kind]}</h1>
              <p class="empty">Nothing recorded yet.</p>`;
    }
    return `
      <h1>${KIND_LABEL[kind]}</h1>
      <p class="muted">${entries.length} in the archive, most mentioned first.</p>
      <ul class="entity-list">
        ${entries.map((e) => `
          <li>
            <a href="#/${KIND_ROUTE[kind]}/${encodeURIComponent(e.slug)}">
              ${esc(e.name)}
            </a>
            <span class="muted">${e.items.length}</span>
          </li>`).join("")}
      </ul>`;
  }

  function viewEntity(kind, slug) {
    const entry = (A.entities[kind] || []).find((e) => e.slug === slug);
    if (!entry) return '<p class="empty">Not found in the archive.</p>';
    setTitle(entry.name);

    const items = entry.items.map((id) => byId.get(id)).filter(Boolean);
    return `
      <p class="crumb"><a href="#/${KIND_ROUTE[kind]}s">${KIND_LABEL[kind]}</a></p>
      <h1>${esc(entry.name)}</h1>
      <p class="muted">${items.length} item${items.length === 1 ? "" : "s"}</p>
      ${grid(items, "Nothing here.")}`;
  }

  function viewItem(id) {
    const item = byId.get(id);
    if (!item) return '<p class="empty">That item is not in the archive.</p>';
    setTitle(item.title);

    const chips = (list, route) => list.map((e) =>
      `<a class="chip" href="#/${route}/${encodeURIComponent(e.slug)}">${esc(e.name)}</a>`
    ).join("");

    const related = relatedTo(item).slice(0, 6);
    const unidentified = item.type === "photograph" && !item.people.length;

    return `
      <article class="item">
        <div class="item-image">
          ${item.sizes.length ? `
            <a href="${media(item, 99999)}" target="_blank" rel="noopener"
               title="Open the full-size image">
              <img src="${media(item, A.sizes.detail)}" srcset="${srcset(item)}"
                   sizes="(max-width: 900px) 100vw, 640px"
                   alt="${esc(item.alt)}">
            </a>` : '<div class="noimg big"></div>'}
        </div>

        <div class="item-info">
          <h1>${esc(item.title)}</h1>
          <p class="item-date">${dateHtml(item, { long: true })}</p>
          <p><span class="tag">${esc(typeLabel(item.type))}</span></p>

          ${item.summary ? `<p class="summary">${esc(item.summary)}</p>` : ""}
          ${item.rotary ? `<p class="rotary">${esc(item.rotary)}</p>` : ""}

          ${item.people.length ? `<div class="facet"><h2>People</h2>
             <div class="chips">${chips(item.people, "person")}</div></div>` : ""}
          ${item.orgs.length ? `<div class="facet"><h2>Organisations</h2>
             <div class="chips">${chips(item.orgs, "org")}</div></div>` : ""}
          ${item.places.length ? `<div class="facet"><h2>Places</h2>
             <div class="chips">${chips(item.places, "place")}</div></div>` : ""}
          ${item.topics.length ? `<div class="facet"><h2>Topics</h2>
             <div class="chips">${chips(item.topics, "topic")}</div></div>` : ""}

          ${item.condition ? `<p class="muted small">Condition: ${
            esc(item.condition)}</p>` : ""}

          ${unidentified ? `
            <div class="identify">
              <strong>Do you recognise anyone here?</strong>
              <p>Nobody is named on this photograph, so the archive does not
                 name anyone. If you can identify these people${
                   A.contact ? `, please get in touch: ${esc(A.contact)}` : ""}.</p>
            </div>` : ""}
        </div>
      </article>

      ${item.text ? `
        <section class="transcript">
          <h2>Transcription</h2>
          <pre>${esc(item.text)}</pre>
        </section>` : ""}

      ${related.length ? `
        <section class="band">
          <h2>Related items</h2>
          ${grid(related, "")}
        </section>` : ""}
    `;
  }

  /* Relatedness is shared entities, weighted so a shared person counts for
     more than a shared topic - two items naming the same person are far more
     likely to be genuinely connected than two both tagged "fundraising". */
  function relatedTo(item) {
    const weights = { people: 5, orgs: 3, places: 2, topics: 1 };
    const scores = new Map();

    for (const [field, weight] of Object.entries(weights)) {
      for (const entity of item[field]) {
        for (const other of A.items) {
          if (other.id === item.id) continue;
          if (other[field].some((e) => e.slug === entity.slug)) {
            scores.set(other.id, (scores.get(other.id) || 0) + weight);
          }
        }
      }
    }

    return [...scores.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([id]) => byId.get(id))
      .filter(Boolean);
  }

  function viewSearch(query) {
    setTitle(`Search: ${query}`);
    const results = search(query);
    if (!results.length) {
      return `<h1>Search</h1>
              <p class="empty">Nothing matches “${esc(query)}”.</p>`;
    }
    return `
      <h1>Search</h1>
      <p class="muted">${results.length} result${
        results.length === 1 ? "" : "s"} for “${esc(query)}”</p>
      <ul class="results">
        ${results.slice(0, 100).map((item) => `
          <li>
            <a class="result" href="#/item/${encodeURIComponent(item.id)}">
              ${item.sizes.length ? `<img src="${media(item, A.sizes.thumb)}"
                    alt="" loading="lazy" decoding="async">` : ""}
              <div>
                <strong>${esc(item.title)}</strong>
                <div class="card-meta">${dateHtml(item)}
                  <span class="tag">${esc(typeLabel(item.type))}</span></div>
                <p class="snippet">${snippet(item, query)}</p>
              </div>
            </a>
          </li>`).join("")}
      </ul>`;
  }

  // -------------------------------------------------------------- router --

  function render() {
    const hash = location.hash.replace(/^#\/?/, "");
    const [route, ...rest] = hash.split("/").map(decodeURIComponent);
    const arg = rest.join("/");

    let html;
    switch (route) {
      case "": case "home":  html = viewHome(); break;
      case "timeline":       html = viewTimeline(); break;
      case "decade":         html = viewDecade(arg); break;
      case "gallery":        html = viewGallery(arg || null); break;
      case "item":           html = viewItem(arg); break;
      case "search":         html = viewSearch(arg); break;
      case "people":         html = viewEntityIndex("person"); break;
      case "orgs":           html = viewEntityIndex("organization"); break;
      case "places":         html = viewEntityIndex("place"); break;
      case "topics":         html = viewEntityIndex("topic"); break;
      default:
        if (ROUTE_KIND[route]) html = viewEntity(ROUTE_KIND[route], arg);
        else html = '<p class="empty">Page not found.</p>';
    }

    main.innerHTML = html;
    document.querySelectorAll(".nav a").forEach((a) => {
      a.classList.toggle("on", a.getAttribute("href") === `#/${route}`);
    });

    // Scroll to top on navigation, but not when only the query changed - that
    // would yank the page away while someone is typing.
    if (route !== "search") window.scrollTo({ top: 0 });
    notifyHeight();
  }

  // ------------------------------------------------------- embed support --

  /* Marks how the archive is being viewed so CSS can differ where it must.
     The only current use is the header: standalone it sticks, which helps on
     the long timeline page; embedded it must not, because the iframe is
     auto-resized to full content height and never scrolls, so a sticky bar
     would sit at the top of a very tall document and never actually stick. */
  try {
    document.documentElement.classList.add(
      window.parent === window ? "standalone" : "embedded"
    );
  } catch (err) {
    // A cross-origin parent can throw on access; treat that as embedded.
    document.documentElement.classList.add("embedded");
  }

  /* Tells a WordPress host page how tall to make the iframe. Without this the
     embed either scrolls inside a fixed box or gets cut off. */
  function notifyHeight() {
    if (window.parent === window) return;
    requestAnimationFrame(() => {
      const height = document.documentElement.scrollHeight;
      window.parent.postMessage(
        { type: "rotary-archive-height", height }, "*"
      );
    });
  }

  // ---------------------------------------------------------------- boot --

  const input = document.getElementById("q");
  let debounce;
  input.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => {
      const value = input.value.trim();
      location.hash = value ? `#/search/${encodeURIComponent(value)}` : "#/";
    }, 180);
  });

  // "/" focuses search, the convention people already expect.
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "/" && document.activeElement !== input) {
      ev.preventDefault();
      input.focus();
      input.select();
    } else if (ev.key === "Escape" && document.activeElement === input) {
      input.blur();
    }
  });

  window.addEventListener("hashchange", render);
  window.addEventListener("resize", notifyHeight);

  const counts = A.counts || {};
  document.getElementById("foot-counts").textContent =
    `${counts.items || 0} items · ${counts.dated || 0} dated · ` +
    `${counts.people || 0} people named. `;
  if (A.generated_at) {
    document.getElementById("foot-generated").textContent =
      `Built ${A.generated_at.slice(0, 10)}.`;
  }

  // Restore the query when arriving on a search URL directly.
  const initial = location.hash.match(/^#\/search\/(.+)$/);
  if (initial) input.value = decodeURIComponent(initial[1]);

  render();
})();
