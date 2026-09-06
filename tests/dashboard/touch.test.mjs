// Real event handlers on minimal targets; no DOM/WebGPU or mobile-browser claim.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
const root = process.env.GFFX_DASHBOARD_STATIC;
assert.ok(root, "test the assets selected by pytest, not a guessed source tree");
const source = (name) => readFile(join(root, name), "utf8");
const moduleFrom = (name) => import(pathToFileURL(join(root, name)).href);
const { MeshView } = await moduleFrom("mesh.js");
class Target {
  constructor() { this.listeners = new Map(); this.captured = new Set(); this.style = {}; }
  addEventListener(name, fn) { const list = this.listeners.get(name) || []; list.push(fn); this.listeners.set(name, list); }
  removeEventListener(name, fn) { this.listeners.set(name, (this.listeners.get(name) || []).filter((f) => f !== fn)); }
  setPointerCapture(id) { this.captured.add(id); }
  hasPointerCapture(id) { return this.captured.has(id); }
  releasePointerCapture(id) { this.captured.delete(id); }
  closest() { return null; }
  emit(type, attrs = {}) {
    const e = { type, target: this, pointerId: 1, pointerType: "touch", button: 0, clientX: 100, clientY: 100,
      preventDefault() { this.prevented = true; }, stopPropagation() {}, ...attrs };
    for (const fn of this.listeners.get(type) || []) fn(e);
    return e;
  }
}
function mesh() {
  const canvas = new Target(), view = new MeshView(canvas, null);
  view.requestRender = () => {};
  view._matrices = () => ({ view: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] });
  const picks = []; view._pickAt = (...args) => picks.push(args);
  return { canvas, view, picks };
}
test("one finger orbits, captures, and a drag never picks", () => {
  const { canvas: c, view: v, picks } = mesh(), yaw = v.orbit.yaw;
  c.emit("pointerdown"); assert.ok(c.captured.has(1));
  c.emit("pointermove", { clientX: 125 }); c.emit("pointerup", { clientX: 125 });
  assert.ok(v.orbit.yaw < yaw); assert.equal(picks.length, 0); assert.equal(c.captured.size, 0);
});
test("foreign pointer moves and releases cannot steal a gesture", () => {
  const { canvas: c, view: v } = mesh(), yaw = v.orbit.yaw;
  c.emit("pointerdown"); c.emit("pointermove", { pointerId: 77, clientX: 800 });
  c.emit("pointerup", { pointerId: 77 }); assert.equal(v.orbit.yaw, yaw);
  c.emit("pointermove", { clientX: 130 }); assert.ok(v.orbit.yaw < yaw);
});
test("pinch changes distance, not orbit; release never picks", () => {
  const { canvas: c, view: v, picks } = mesh(), before = { ...v.orbit };
  c.emit("pointerdown"); c.emit("pointerdown", { pointerId: 2, clientX: 200 });
  c.emit("pointermove", { pointerId: 2, clientX: 250 });
  assert.ok(v.orbit.distance < before.distance); assert.equal(v.orbit.yaw, before.yaw);
  c.emit("pointerup", { pointerId: 2, clientX: 250 }); c.emit("pointerup"); assert.equal(picks.length, 0);
});
test("two fingers pan and leave no jump when one lifts", () => {
  const { canvas: c, view: v } = mesh(), yaw = v.orbit.yaw;
  c.emit("pointerdown"); c.emit("pointerdown", { pointerId: 2, clientX: 200 });
  c.emit("pointermove", { clientY: 130 }); c.emit("pointermove", { pointerId: 2, clientX: 200, clientY: 130 });
  assert.ok(v.orbit.target[1] > 0); assert.equal(v.orbit.yaw, yaw);
  c.emit("pointerup", { pointerId: 2 }); c.emit("pointermove", { clientX: 101, clientY: 130 });
  assert.ok(Math.abs(v.orbit.yaw - yaw) < 0.02);
});
test("tap tolerates finger jitter; cancel or lost capture never picks", () => {
  const { canvas: c, view: v, picks } = mesh();
  c.emit("pointerdown"); c.emit("pointermove", { clientX: 104 }); c.emit("pointerup", { clientX: 104 });
  assert.equal(picks.length, 1);
  for (const event of ["pointercancel", "lostpointercapture"]) {
    c.emit("pointerdown"); c.emit(event); const yaw = v.orbit.yaw;
    c.emit("pointermove", { clientX: 160 }); c.emit("pointerup");
    assert.equal(v.orbit.yaw, yaw); assert.equal(picks.length, 1);
  }
});
test("desktop right/shift drag pans and wheel zoom is clamped", () => {
  const { canvas: c, view: v, picks } = mesh(), yaw = v.orbit.yaw;
  c.emit("pointerdown", { pointerType: "mouse", button: 2 }); c.emit("pointermove", { clientX: 120 });
  c.emit("pointerup"); assert.equal(v.orbit.yaw, yaw); assert.notEqual(v.orbit.target[0], 0);
  c.emit("pointerdown", { pointerType: "mouse", shiftKey: true }); c.emit("pointerup"); assert.equal(picks.length, 0);
  c.emit("wheel", { deltaY: 1e6 }); assert.equal(v.orbit.distance, v.orbit.radius * 50);
  c.emit("wheel", { deltaY: -1e6 }); assert.equal(v.orbit.distance, v.orbit.radius * 0.05);
});
test("empty viewport reset is safe", () => { const { canvas } = mesh(); assert.doesNotThrow(() => canvas.emit("dblclick")); });
test("destroy unbinds mesh input", () => {
  const { canvas: c, view: v } = mesh(); v.destroy(); const yaw = v.orbit.yaw;
  c.emit("pointerdown"); c.emit("pointermove", { clientX: 160 }); assert.equal(v.orbit.yaw, yaw);
});
test("layout drag ignores controls, tracks one ID, and cancels on capture loss", async () => {
  const { bindDrag } = await moduleFrom("interaction.js");
  const target = new Target(), calls = []; let enabled = true;
  const dispose = bindDrag(target, { enabled: () => enabled, start: () => calls.push("start"),
    move: (_e, dx, dy) => calls.push([dx, dy]), end: () => calls.push("end"), cancel: () => calls.push("cancel") });
  target.emit("pointerdown", { target: { closest: () => ({}) } }); assert.deepEqual(calls, []);
  target.emit("pointerdown"); target.emit("pointermove", { pointerId: 5, clientX: 180 });
  target.emit("pointerup", { pointerId: 5 }); target.emit("pointermove", { clientX: 120, clientY: 115 });
  target.emit("lostpointercapture"); target.emit("pointermove", { clientX: 190 });
  assert.deepEqual(calls, ["start", [20, 15], "cancel"]);
  enabled = false; target.emit("pointerdown"); assert.equal(calls.length, 3);
  dispose(); enabled = true; target.emit("pointerdown"); assert.equal(calls.length, 3);
});
test("mobile reorder uses options and preserves every desktop cell", async () => {
  const { mobileOrder, moveMobilePanel } = await moduleFrom("interaction.js");
  const a = { col: 0, row: 0, w: 6, h: 7, options: { wire: true } }, b = { col: 6, row: 0, w: 6, h: 4 }, c = { col: 0, row: 7, w: 12, h: 3 };
  const specs = [c, a, b], cells = specs.map(({ col, row, w, h }) => [col, row, w, h]);
  assert.deepEqual(mobileOrder(specs), [a, b, c]); moveMobilePanel(specs, c, 0);
  assert.deepEqual(mobileOrder(specs), [c, a, b]);
  assert.deepEqual(specs.map(({ col, row, w, h }) => [col, row, w, h]), cells);
  assert.equal(a.options.wire, true);
  assert.deepEqual(mobileOrder(JSON.parse(JSON.stringify(specs))).map((p) => p.row), [7, 0, 0]);
});
test("CSS reserves only interactive touch surfaces and sizes mobile targets", async () => {
  const css = (await source("app.css")).replace(/\/\*[\s\S]*?\*\//g, "");
  for (const selector of [".mesh-canvas", ".panel.editing > .head", ".panel.editing .resize"]) {
    const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].filter((m) => m[1].split(",").map((s) => s.trim()).includes(selector));
    assert.ok(rules.some((m) => /touch-action:\s*none/.test(m[2])), selector);
  }
  assert.doesNotMatch(css, /(?:body|html|\*)\s*\{[^}]*touch-action:\s*none/);
  assert.match(css, /@media\s*\(max-width:\s*760px\)/); assert.match(css, /@media[^{}]*pointer:\s*coarse/);
  assert.match(css, /min-height:\s*44px/);
  assert.doesNotMatch(await source("index.html"), /user-scalable=no|maximum-scale=1/);
});
test("frontend wires touch handles, saved mobile options, and touch inspection", async () => {
  const js = await source("app.js");
  assert.match(js, /bindDrag\(panel.head/); assert.match(js, /bindDrag\(handle/);
  assert.match(js, /mobileOrder/); assert.match(js, /moveMobilePanel/); assert.match(js, /options:.*p.options/);
  assert.match(js, /pointerdown.*this.hover/); assert.match(js, /pointerdown.*this.probe/);
  assert.match(js, /aria-label.*Scrub/); assert.match(js, /gesture-help/);
});

test("live updates do not move a held scrubber; history replies cannot overwrite a newer choice", async () => {
  const { SteppedPanel } = await moduleFrom("app.js");
  const p = Object.create(SteppedPanel.prototype);
  const entry = { steps: [10, 11, 12], cache: new Map() }, pending = new Map(), shown = [];
  p.spec = { path: "demo/sphere" }; p.entry = () => entry;
  p.app = { connection: { history: (_path, step) => new Promise((resolve) => pending.set(step, resolve)) } };
  p.range = { max: 2, value: 1 }; p.stepLabel = {};
  p.liveButton = { classList: { toggle() {}, add() {} } };
  p.following = true; p.shown = null; p.show = (value) => shown.push(value.step);
  p.beginScrub(); entry.steps = [11, 12, 13]; p.onUpdate();
  assert.equal(p.range.value, 1); assert.equal(p.following, false);
  p.scrub(); assert.ok(pending.has(11), "held slider maps against its captured steps");
  p.range.value = 2; p.scrub(); assert.ok(pending.has(12));
  pending.get(12)({ step: 12 }); await Promise.resolve();
  pending.get(11)({ step: 11 }); await Promise.resolve();
  assert.deepEqual(shown, [12]); assert.equal(p.stepLabel.textContent, "12");
  p.endScrub(); assert.equal(p.scrubSteps, null); assert.equal(p.following, false);
});

test("actual mobile panel handlers reorder and resize without rewriting desktop cells", async () => {
  const { App } = await moduleFrom("app.js");
  const { mobileOrder } = await moduleFrom("interaction.js");
  const app = Object.create(App.prototype); app.editing = true; app.isMobile = () => true;
  const a = { col: 0, row: 0, w: 6, h: 7 }, b = { col: 6, row: 0, w: 6, h: 4 };
  const makePanel = (spec, top) => {
    const handle = new Target();
    return { spec, head: new Target(), handle, place() {},
      el: { style: {}, classList: { add() {}, remove() {} }, querySelector: () => handle,
        getBoundingClientRect: () => ({ top, height: 400 }) } };
  };
  app.panels = [makePanel(a, 0), makePanel(b, 400)]; app.layout = { panels: [a, b] };
  app.grid = { getBoundingClientRect: () => ({ width: 390 }) };
  const first = app.panels[0]; app.decorate(first);
  first.head.emit("pointerdown"); first.head.emit("pointermove", { clientY: 650 });
  first.head.emit("pointerup", { clientY: 650 });
  assert.deepEqual(mobileOrder(app.layout.panels), [b, a]);
  first.handle.emit("pointerdown"); first.handle.emit("pointermove", { clientY: 164 }); first.handle.emit("pointerup", { clientY: 164 });
  assert.equal(a.options.mobileHeight, 8); assert.deepEqual([a.col, a.row, a.w, a.h], [0, 0, 6, 7]);
  first.handle.emit("pointerdown"); first.handle.emit("pointermove", { clientY: 400 }); first.handle.emit("pointercancel");
  assert.equal(a.options.mobileHeight, 8);
  app.resizeMobile(first, -100); assert.equal(a.options.mobileHeight, 5);
  app.resizeMobile(first, 100); assert.equal(a.options.mobileHeight, 20);
});

test("actual save keeps mobile layout options and unrelated renderer options", async () => {
  const { App } = await moduleFrom("app.js");
  const app = Object.create(App.prototype); let saved;
  app.run = "demo"; app.layout = { name: "phone", panels: [{ kind: "mesh", path: "demo/sphere", col: 0, row: 0, w: 6, h: 7,
    options: { mobileOrder: 2, mobileHeight: 8, wireframe: true } }] };
  app.connection = { request: async (message) => { saved = message; } };
  app.layoutSelect = { options: [{ value: "phone" }] }; app.status = () => {};
  globalThis.prompt = () => "phone"; globalThis.localStorage = { setItem() {} };
  try { await app.saveLayout(); } finally { delete globalThis.prompt; delete globalThis.localStorage; }
  assert.equal(saved.t, "layout_save");
  assert.deepEqual(saved.spec.panels[0], app.layout.panels[0]);
});

test("mesh references reconstruct exact arrays and reject a missing base", async () => {
  const { MeshReceiver } = await moduleFrom("resources.js");
  const receiver = new MeshReceiver(1024);
  const faces = new Uint32Array([0, 1, 2]), colors = new Uint8Array([1, 2, 3]);
  receiver.decode({ path: "mesh", kind: "mesh", step: 1, arrays: { vertices: new Float32Array(9), faces, colors } });
  const vertices = new Float32Array(9).fill(2);
  const frame = receiver.decode({ path: "mesh", kind: "mesh", step: 2, mesh_delta: true, base_step: 1,
    mesh_fields: ["vertices", "faces"], arrays: { vertices } });
  assert.equal(frame.arrays.faces, faces); assert.equal(frame.arrays.vertices, vertices);
  assert.equal(frame.arrays.colors, undefined);
  assert.throws(() => receiver.decode({ ...frame, step: 3, mesh_delta: true, base_step: 0 }), /base/);
  receiver.clear(); assert.equal(receiver.bytes, 0);
});
test("plot reduction is bounded and preserves spikes and endpoints", async () => {
  const { plotSamples } = await moduleFrom("resources.js");
  const values = Array.from({ length: 2048 }, (_, i) => [i, i === 501 ? 1000 : 0]);
  const reduced = plotSamples(values, 512);
  assert.ok(reduced.length <= 512); assert.ok(reduced.some((p) => p[1] === 1000));
  assert.deepEqual(reduced[0], values[0]); assert.deepEqual(reduced.at(-1), values.at(-1));
});
test("page negotiates low bandwidth and pauses when hidden", async () => {
  const js = await source("app.js");
  assert.match(js, /visibilitychange/); assert.match(js, /view_config/);
  assert.match(js, /document.hidden/); assert.match(js, /mesh_deltas:\s*true/);
  assert.match(js, /history_limit/); assert.match(js, /trimCaches/);
});

test("recent server window includes frames coalesced out of page delivery and policy changes", async () => {
  const { Store } = await moduleFrom("app.js");
  const store = new Store();
  store.applyUpdate({ path: "mesh", kind: "mesh", queue: true, step: 8, recent_window: 4,
    history_limit: 4, snapshot_every: 1000, recent_steps: [5, 6, 7, 8] });
  assert.deepEqual(store.paths.get("mesh").steps, [5, 6, 7, 8]);
  store.applyUpdate({ path: "mesh", kind: "mesh", queue: true, step: 10, recent_window: 2,
    history_limit: 2, snapshot_every: 2000, recent_steps: [9, 10] });
  assert.deepEqual(store.paths.get("mesh").steps, [9, 10]);
  assert.equal(store.paths.get("mesh").snapshot_every, 2000);
});

test("checkpoint browsing is separate and a stale listing cannot replace a newer one", async () => {
  const { SteppedPanel } = await moduleFrom("app.js");
  const panel = Object.create(SteppedPanel.prototype), pending = [], shown = [];
  panel.spec = { path: "mesh" }; panel.liveButton = { classList: { remove() {} } };
  panel.range = {}; panel.olderButton = {}; panel.policyLabel = {}; panel.stepLabel = {};
  panel.entry = () => ({ steps: [9998, 9999], cache: new Map() });
  panel.showStep = async (step) => shown.push(step);
  panel.app = { connection: { request: (message) => new Promise((resolve) => pending.push({ message, resolve })) } };
  const first = panel.loadCheckpoints(), second = panel.loadCheckpoints(5000);
  assert.equal(pending[1].message.before, 5000);
  pending[1].resolve({ steps: [0, 1000, 2000], has_more: false }); await second;
  pending[0].resolve({ steps: [8000, 9000], has_more: true }); await first;
  assert.deepEqual(shown, [2000]); assert.deepEqual(panel.visibleSteps(), [0, 1000, 2000]);
  panel.onUpdate(); assert.deepEqual(panel.visibleSteps(), [0, 1000, 2000]);
  assert.equal(panel.olderButton.hidden, true);
});

test("evidence references round-trip page, run, artifact and exact snapshot", async () => {
  const { evidenceReference, parseReference } = await moduleFrom("references.js");
  const ref = { run: "run one", run_id: "run-uuid", page_id: "page-uuid", path: "camera/mesh", step: 1000,
    provenance: { revision: "abc123", dirty: true, artifact_sha256: "def456" } };
  const result = evidenceReference("https://example.test/?unrelated=secret", ref);
  const parsed = parseReference(result.url);
  for (const key of ["run", "run_id", "page_id", "path", "step"]) assert.equal(parsed[key], ref[key]);
  assert.match(result.text, /abc123/); assert.match(result.text, /dirty/); assert.match(result.text, /def456/);
  assert.ok(!result.url.includes("secret"));
  assert.throws(() => parseReference("https://example.test/?run=x&step=-1"), /step/);
});

test("unknown provenance and live references never claim an exact recorded result", async () => {
  const { evidenceReference } = await moduleFrom("references.js");
  const result = evidenceReference("http://localhost/", { run: "x", run_id: "r", page_id: "p", provenance: {} });
  assert.match(result.text, /unknown/); assert.match(result.text, /live/);
  assert.ok(!result.url.includes("step="));
});

test("deep links select their page before browser preferences and reject wrong run identity", async () => {
  const { App } = await moduleFrom("app.js");
  const app = Object.create(App.prototype), selected = [], statuses = [];
  app.reference = { page_id: "page-1", run_id: "run-1" };
  app.referenceEl = {}; app.status = (s) => statuses.push(s);
  // Fail closed before any DOM/layout work on a mismatching run.
  app.onSnapshot({ run_id: "wrong", pages: [] });
  assert.match(statuses[0], /identity/);
  app.layoutSelect = { replaceChildren() {} };
  app.chooseLayout = (...args) => selected.push(args);
  const oldDocument = globalThis.document;
  globalThis.document = { createElement: () => ({ setAttribute() {}, append() {} }) };
  try {
    app.onSnapshot({ run_id: "run-1", layouts: ["camera"], pages: [{ name: "camera", page_id: "page-1" }] });
    assert.deepEqual(selected, [["camera", true]]);
    app.reference.page_id = "missing";
    app.onSnapshot({ run_id: "run-1", layouts: [], pages: [] });
    assert.match(statuses.at(-1), /unavailable/);
  } finally { globalThis.document = oldDocument; }
});

test("clipboard refusal exposes the exact reference for manual copy", async () => {
  const { App } = await moduleFrom("app.js");
  const app = Object.create(App.prototype);
  app.run = "test"; app.activePageId = "page-1"; app.metadata = { run_id: "run-1", provenance: {} };
  app.status = () => {};
  app.referenceFallback = { hidden: true, focus() {}, select() {} };
  const oldLocation = globalThis.location;
  globalThis.location = { href: "https://example.test/" };
  try {
    await app.copyReference("fit/mesh", 1000);
    assert.equal(app.referenceFallback.hidden, false);
    assert.match(app.referenceFallback.value, /Step: 1000/);
    assert.match(app.referenceFallback.value, /page=page-1/);
  } finally { globalThis.location = oldLocation; }
});

test("unsaved edits cannot be copied as if they were the saved page", async () => {
  const { App } = await moduleFrom("app.js");
  const app = Object.create(App.prototype), messages = [];
  app.unsavedPage = true; app.status = (message) => messages.push(message);
  await app.copyReference();
  assert.match(messages[0], /Save the edited page/);
});

test("a linked mesh is pinned to the requested frame rather than following latest", async () => {
  const { App } = await moduleFrom("app.js");
  const app = Object.create(App.prototype), requested = [];
  app.panels = []; app.reference = { path: "mesh", step: 1000 };
  app.updateReferenceLabel = () => {}; app.status = () => {};
  app.mount = (spec) => app.panels.push({ spec, el: {}, following: true,
    liveButton: { classList: { remove() {} } }, showStep: (step) => requested.push(step) });
  app.applyLayout({ name: "camera", page_id: "page-1", panels: [{ kind: "mesh", path: "mesh" }] });
  assert.deepEqual(requested, [1000]); assert.equal(app.panels[0].following, false);
});
