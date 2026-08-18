/* Headless render check for the published site.
 *
 * Runs the real app.js against a minimal DOM shim and walks every route,
 * asserting each one produces markup and raises nothing. This catches the
 * failure a Python test cannot see - a template literal referencing a field
 * the exporter stopped emitting - without pulling in a browser engine.
 *
 * Usage: node render_site.mjs <site-dir>
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import vm from "node:vm";

const siteDir = process.argv[2];
if (!siteDir) {
  console.error("usage: node render_site.mjs <site-dir>");
  process.exit(2);
}

// --- minimal DOM ----------------------------------------------------------

function makeElement(id = "") {
  return {
    id,
    innerHTML: "",
    textContent: "",
    value: "",
    classList: { toggle() {}, add() {}, remove() {}, contains: () => false },
    style: {},
    addEventListener() {},
    getAttribute: () => "",
    focus() {},
    select() {},
    blur() {},
    querySelectorAll: () => [],
  };
}

const elements = new Map();
const listeners = {};

const document = {
  title: "",
  activeElement: null,
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  },
  querySelectorAll: () => [],
  addEventListener(type, fn) {
    (listeners[type] ||= []).push(fn);
  },
  documentElement: { scrollHeight: 800 },
};

const location = { hash: "" };

const sandbox = {
  window: null,
  document,
  location,
  console,
  setTimeout,
  clearTimeout,
  requestAnimationFrame: (fn) => fn(),
  encodeURIComponent,
  decodeURIComponent,
};
sandbox.window = sandbox;
sandbox.window.addEventListener = (type, fn) => {
  (listeners[type] ||= []).push(fn);
};
sandbox.window.scrollTo = () => {};
sandbox.window.parent = sandbox.window;

const context = vm.createContext(sandbox);

// --- run the real bundle ---------------------------------------------------

vm.runInContext(readFileSync(join(siteDir, "data/archive.js"), "utf8"), context);
vm.runInContext(readFileSync(join(siteDir, "assets/app.js"), "utf8"), context);

const A = sandbox.window.ARCHIVE;
const main = document.getElementById("main");

function fire(hash) {
  location.hash = hash;
  for (const fn of listeners.hashchange || []) fn();
  return main.innerHTML;
}

// --- walk every route ------------------------------------------------------

const routes = [
  ["#/", "home"],
  ["#/timeline", "timeline"],
  ["#/gallery", "gallery"],
  ["#/people", "people index"],
  ["#/places", "places index"],
  ["#/topics", "topics index"],
  ["#/orgs", "organisations index"],
];

for (const block of A.timeline) {
  routes.push([`#/decade/${encodeURIComponent(block.decade)}`, `decade ${block.decade}`]);
}
for (const kind of ["person", "place", "topic", "organization"]) {
  const first = (A.entities[kind] || [])[0];
  if (first) {
    const route = { person: "person", place: "place", topic: "topic", organization: "org" }[kind];
    routes.push([`#/${route}/${encodeURIComponent(first.slug)}`, `${kind} ${first.name}`]);
  }
}
for (const item of A.items.slice(0, 8)) {
  routes.push([`#/item/${encodeURIComponent(item.id)}`, `item ${item.id}`]);
}
routes.push(["#/search/rotary", "search 'rotary'"]);
routes.push(["#/search/zzzznothing", "search with no results"]);
routes.push(["#/item/does-not-exist", "missing item"]);
routes.push(["#/nonsense", "unknown route"]);

let failures = 0;
for (const [hash, label] of routes) {
  let html;
  try {
    html = fire(hash);
  } catch (err) {
    console.log(`FAIL  ${label.padEnd(34)} threw ${err.message}`);
    failures++;
    continue;
  }
  if (!html || html.length < 20) {
    console.log(`FAIL  ${label.padEnd(34)} rendered nothing`);
    failures++;
    continue;
  }
  if (/undefined|\[object Object\]|NaN/.test(html)) {
    const m = html.match(/.{0,60}(undefined|\[object Object\]|NaN).{0,60}/);
    console.log(`FAIL  ${label.padEnd(34)} leaked placeholder: ...${m[0]}...`);
    failures++;
    continue;
  }
  console.log(`ok    ${label.padEnd(34)} ${String(html.length).padStart(6)} chars`);
}

// --- targeted assertions ---------------------------------------------------

function assert(cond, message) {
  if (cond) {
    console.log(`ok    ${message}`);
  } else {
    console.log(`FAIL  ${message}`);
    failures++;
  }
}

console.log("");

const inferred = A.items.find((i) => i.date_source === "inferred");
if (inferred) {
  const html = fire(`#/item/${encodeURIComponent(inferred.id)}`);
  assert(html.includes("approx"), "deduced date is marked visually distinct");
  assert(html.includes("about "), "deduced date is prefixed 'about'");
}

const printed = A.items.find((i) => i.date_source === "printed" && i.date_display);
if (printed) {
  const html = fire(`#/item/${encodeURIComponent(printed.id)}`);
  assert(!html.includes("about " + printed.date_display),
    "printed date is NOT hedged with 'about'");
}

const unidentified = A.items.find((i) => i.type === "photograph" && !i.people.length);
if (unidentified) {
  const html = fire(`#/item/${encodeURIComponent(unidentified.id)}`);
  assert(html.includes("Do you recognise anyone"),
    "unidentified photograph invites the club to help");
}

const searchHtml = fire("#/search/rotary");
assert(searchHtml.includes("<mark>") || searchHtml.includes("result"),
  "search highlights or lists matches");

assert(fire("#/item/does-not-exist").includes("not in the archive"),
  "missing item shows a real message rather than a blank page");

console.log("");
if (failures) {
  console.log(`${failures} FAILURE(S)`);
  process.exit(1);
}
console.log(`All ${routes.length} routes + assertions passed.`);
