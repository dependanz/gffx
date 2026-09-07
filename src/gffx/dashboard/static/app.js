// The dashboard page: one WebSocket to the server, a store of paths, a grid of panels.
// No library. The mesh panel renders with WebGPU through mesh.js.

import { MeshView, webgpuAvailable } from "./mesh.js";
import { bindDrag, mobileOrder, moveMobilePanel } from "./interaction.js";
import { MeshReceiver, plotSamples } from "./resources.js";
import { evidenceReference, parseReference } from "./references.js";

const COLUMNS = 12;
const DTYPES = { float32: Float32Array, float64: Float64Array, int32: Int32Array, uint32: Uint32Array, uint8: Uint8Array, uint16: Uint16Array, int16: Int16Array, int64: BigInt64Array };
const BINARY_KINDS = new Set(["mesh", "points", "image"]);

function decodeBinary(buffer) {
  const view = new DataView(buffer);
  const headerLength = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
  let offset = 4 + headerLength;
  const arrays = {};
  for (const entry of header.arrays) {
    const T = DTYPES[entry.dtype];
    arrays[entry.name] = new T(buffer.slice(offset, offset + entry.nbytes));
    offset += entry.nbytes;
  }
  header.arrays = arrays;
  return header;
}

const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
};
const fmt = (v) => (typeof v === "number" ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(3) : +v.toPrecision(5) + "") : String(v));

// ---- store ----------------------------------------------------------------------------------

export class Store {
  constructor() { this.paths = new Map(); this.controls = {}; this.currentStep = 0; this.listeners = new Set(); }
  subscribe(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  notify(path) { for (const fn of this.listeners) fn(path); }
  loadSnapshot(snapshot) {
    this.paths.clear();
    for (const [path, entry] of Object.entries(snapshot.paths)) {
      this.paths.set(path, { kind: entry.kind, steps: entry.steps.slice(-2048), bound: entry.bound, snapshot_every: entry.snapshot_every, history_limit: entry.history_limit || 2048, latest: entry.latest || null, values: (entry.values || []).slice(-2048), cache: new Map() });
    }
    this.controls = snapshot.controls || {};
    this.currentStep = snapshot.current_step || 0;
    this.notify(null);
  }
  applyUpdate(update) {
    const path = update.path;
    if (update.deleted) { this.paths.delete(path); this.notify(path); return; }
    let entry = this.paths.get(path);
    if (!entry) { entry = { kind: update.kind, steps: [], latest: null, values: [], cache: new Map() }; this.paths.set(path, entry); }
    entry.history_limit = update.history_limit || entry.history_limit || 2048;
    if (update.queue) {
      entry.bound = update.recent_window || entry.bound || 100;
      entry.snapshot_every = update.snapshot_every;
      if (Array.isArray(update.recent_steps)) {
        entry.steps = update.recent_steps.slice(-Math.min(entry.history_limit, 2048));
        for (const key of entry.cache.keys()) if (!entry.steps.includes(key)) entry.cache.delete(key);
      }
    }
    if (!entry.steps.includes(update.step)) {
      entry.steps.push(update.step);
      if (entry.steps.length > 1 && update.step < entry.steps[entry.steps.length - 2]) entry.steps.sort((a, b) => a - b);
      const bound = Math.min(entry.bound || 2048, entry.history_limit, 2048);
      while (entry.steps.length > bound || (entry.steps.length && update.retained_from != null && entry.steps[0] < update.retained_from)) { const gone = entry.steps.shift(); entry.cache.delete(gone); }
    }
    if (update.step > this.currentStep) this.currentStep = update.step;
    if (update.arrays) entry.cache.set(update.step, update);
    if (entry.kind === "scalar" || entry.kind === "control") {
      const i = entry.values.findIndex(([s]) => s === update.step);
      if (i >= 0) entry.values[i] = [update.step, update.value]; else entry.values.push([update.step, update.value]);
      entry.values = entry.values.filter(([step]) => entry.steps.includes(step));
    }
    if (update.control) this.controls[path] = update.control;
    if (!entry.latest || update.step >= entry.latest.step) entry.latest = { step: update.step, value: update.value };
    this.trimCaches();
    this.notify(path);
  }
  matching(binding) {
    const out = [];
    for (const [path, entry] of this.paths) if (matches(binding, path)) out.push([path, entry]);
    return out;
  }
  trimCaches() {
    const size = (value) => Object.values(value.arrays || {}).reduce((sum, array) => sum + array.byteLength, 0);
    let total = 0;
    for (const entry of this.paths.values()) {
      let bytes = [...entry.cache.values()].reduce((sum, value) => sum + size(value), 0);
      while (entry.cache.size && (bytes > 16 * 1024 * 1024 || entry.cache.size > 64)) {
        const key = entry.cache.keys().next().value; bytes -= size(entry.cache.get(key)); entry.cache.delete(key);
      }
      total += bytes;
    }
    while (total > 32 * 1024 * 1024) {
      const entry = [...this.paths.values()].find((value) => value.cache.size);
      if (!entry) break;
      const key = entry.cache.keys().next().value; total -= size(entry.cache.get(key)); entry.cache.delete(key);
    }
  }
}
const matches = (binding, path) => (binding.endsWith("/") ? path.startsWith(binding) : path === binding);

// ---- connection -----------------------------------------------------------------------------

export class Connection {
  constructor(run, store, status) {
    this.run = run; this.store = store; this.status = status;
    this.pending = new Map(); this.ids = 1; this.ws = null; this.closed = false; this.snapshot = null;
    this.onSnapshot = null;
    this.meshReceiver = new MeshReceiver();
    this.viewOptions = null;
  }
  connect() {
    if (this.closed) return; // a pending reconnect timer must not resurrect a disposed page
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => { this.meshReceiver.clear(); this.send({ t: "open", run: this.run, view: this.viewOptions }); this.status("open"); };
    ws.onmessage = (event) => {
      let message = typeof event.data === "string" ? JSON.parse(event.data) : decodeBinary(event.data);
      if (message.t === "snapshot") { this.snapshot = message; this.store.loadSnapshot(message); if (this.onSnapshot) this.onSnapshot(message); return; }
      if (message.t === "update") {
        try { message = this.meshReceiver.decode(message); }
        catch (error) { this.status(error.message, true); this.request({ t: "view_config", ...this.viewOptions }).catch(() => {}); return; }
        this.store.applyUpdate(message); return;
      }
      if (message.id !== undefined && this.pending.has(message.id)) {
        const { resolve, reject } = this.pending.get(message.id); this.pending.delete(message.id);
        if (message.t === "error") reject(new Error(message.message)); else resolve(message);
        return;
      }
      if (message.t === "layout_proposed" && this.onProposed) this.onProposed(message.spec);
    };
    ws.onclose = () => { for (const { reject } of this.pending.values()) reject(new Error("disconnected")); this.pending.clear(); this.status("disconnected", true); if (!this.closed) setTimeout(() => this.connect(), 2000); };
    ws.onerror = () => {};
  }
  send(envelope) { envelope.id = this.ids++; this.ws.send(JSON.stringify(envelope)); return envelope.id; }
  request(envelope) {
    return new Promise((resolve, reject) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) { reject(new Error("not connected")); return; }
      if (this.pending.size >= 32) { reject(new Error("too many pending page requests")); return; }
      const id = this.send(envelope);
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error("page request timed out")); }, 10000);
      this.pending.set(id, { resolve: (value) => { clearTimeout(timer); resolve(value); }, reject: (error) => { clearTimeout(timer); reject(error); } });
    });
  }
  history(path, step) { const env = { t: "history", path }; if (step !== undefined && step !== null) env.step = step; return this.request(env); }
  close() { this.closed = true; if (this.ws) this.ws.close(); }
}

// ---- panels ---------------------------------------------------------------------------------

class Panel {
  constructor(app, spec) {
    this.app = app; this.spec = spec; this.store = app.store;
    this.el = el("div", { class: "panel" });
    this.head = el("div", { class: "head" }, el("span", { class: "kind" }, spec.kind), el("span", { class: "path", title: spec.path }, spec.path || "(unbound)"));
    this.tools = el("div", { class: "tools" });
    this.head.append(this.tools);
    this.tool("ref", "Copy this artifact and selected step reference", () => this.app.copyReference(this.spec.path,
      this.shown ?? null));
    this.body = el("div", { class: "body" });
    this.el.append(this.head, this.body, el("div", { class: "resize" }));
    this.place();
    this.unsubscribe = this.store.subscribe((path) => { if (path === null || this.interested(path)) this.onUpdate(path); });
  }
  place() {
    const s = this.spec;
    this.el.style.gridColumn = `${s.col + 1} / span ${s.w}`; this.el.style.gridRow = `${s.row + 1} / span ${s.h}`;
    this.el.style.order = mobileOrder(this.app.layout?.panels || [s]).indexOf(s);
    this.el.style.setProperty("--mobile-height", `${(s.options?.mobileHeight || Math.max(5, s.h)) * 64}px`);
  }
  interested(path) { return matches(this.spec.path, path); }
  onUpdate() {}
  destroy() { for (const dispose of this.dragDisposers || []) dispose(); this.resizeObserver?.disconnect(); this.unsubscribe(); this.el.remove(); }
  tool(label, title, onclick) { const b = el("button", { title, onclick }, label); this.tools.append(b); return b; }
}

class TextPanel extends Panel {
  constructor(app, spec) { super(app, spec); this.view = el("div", { class: "text-body" }); this.body.append(this.view); this.onUpdate(); }
  onUpdate() {
    const entry = this.store.paths.get(this.spec.path);
    if (!entry || !entry.latest) { this.view.textContent = ""; return; }
    const v = entry.latest.value;
    if (entry.kind === "record") { this.view.className = "text-body"; this.view.textContent = JSON.stringify(v, null, 1); return; }
    if (v && typeof v === "object" && v.markdown) { this.view.className = "text-body md"; this.view.innerHTML = markdown(v.text); }
    else { this.view.className = "text-body"; this.view.textContent = typeof v === "string" ? v : JSON.stringify(v); }
  }
}

function markdown(text) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/\*([^*]+)\*/g, "<i>$1</i>");
  const out = []; let list = false, code = false, buffer = [];
  for (const line of text.split("\n")) {
    if (line.startsWith("```")) { if (code) { out.push("<pre>" + esc(buffer.join("\n")) + "</pre>"); buffer = []; } code = !code; continue; }
    if (code) { buffer.push(line); continue; }
    const h = /^(#{1,3})\s+(.*)$/.exec(line);
    if (list && !/^[-*]\s/.test(line)) { out.push("</ul>"); list = false; }
    if (h) out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`);
    else if (/^[-*]\s/.test(line)) { if (!list) { out.push("<ul>"); list = true; } out.push("<li>" + inline(line.slice(2)) + "</li>"); }
    else if (line.trim() === "") out.push("");
    else out.push("<p>" + inline(line) + "</p>");
  }
  if (list) out.push("</ul>");
  return out.join("\n");
}

class TestsPanel extends Panel {
  constructor(app, spec) {
    super(app, spec);
    this.summary = el("div", { class: "summary" });
    this.table = el("table", { class: "tests" }, el("thead", {}, el("tr", {}, el("th", {}, "test"), el("th", {}, "outcome"), el("th", {}, "seconds"))));
    this.tbody = el("tbody"); this.table.append(this.tbody);
    this.body.append(this.summary, this.table);
    this.onUpdate();
  }
  onUpdate() {
    const rows = this.store.matching(this.spec.path).filter(([, e]) => e.kind === "record" && e.latest && e.latest.value && e.latest.value.outcome);
    rows.sort((a, b) => (b[1].latest.step - a[1].latest.step));
    const counts = {};
    for (const [, e] of rows) counts[e.latest.value.outcome] = (counts[e.latest.value.outcome] || 0) + 1;
    this.summary.textContent = rows.length ? Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(" · ") : "no test outcomes yet";
    this.tbody.replaceChildren(...rows.map(([path, e]) => {
      const v = e.latest.value, name = path.slice(this.spec.path.length).replace(/\//g, " › ");
      return el("tr", {}, el("td", { class: "name", title: v.message || "" }, name), el("td", {}, el("span", { class: "pill " + v.outcome }, v.outcome)), el("td", { class: "dur" }, v.duration.toFixed(3)));
    }));
  }
}

class PlotPanel extends Panel {
  constructor(app, spec) {
    super(app, spec);
    this.canvas = el("canvas", { class: "fill" }); this.overlay = el("div", { class: "overlay" });
    this.body.append(this.canvas, this.overlay);
    this.log = false;
    this.logButton = this.tool("log", "logarithmic value axis", () => { this.log = !this.log; this.logButton.classList.toggle("on", this.log); this.draw(); });
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(this.canvas);
    this.canvas.addEventListener("pointerdown", (e) => this.hover(e));
    this.canvas.addEventListener("pointermove", (e) => this.hover(e));
    this.canvas.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse") this.overlay.textContent = ""; });
    this.draw();
  }
  onUpdate() { this.draw(); }
  series() { return this.store.matching(this.spec.path).filter(([, e]) => e.kind === "scalar" && e.values.length).map(([path, e]) => [path, { ...e, values: plotSamples(e.values) }]); }
  draw() {
    const ctx = this.canvas.getContext("2d"), dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.canvas.width = w * dpr; this.canvas.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const style = getComputedStyle(this.el), fg = style.getPropertyValue("--fg").trim(), mute = style.getPropertyValue("--mute").trim(), line = style.getPropertyValue("--line").trim();
    ctx.clearRect(0, 0, w, h);
    const series = this.series();
    if (!series.length) { ctx.fillStyle = mute; ctx.font = "12px system-ui"; ctx.fillText("no scalar values yet", 10, 20); return; }
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [, e] of series) for (const [s, v] of e.values) {
      if (this.log && !(v > 0)) continue;
      const y = this.log ? Math.log10(v) : v;
      if (s < x0) x0 = s; if (s > x1) x1 = s; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    if (!isFinite(y0)) return;
    if (y1 === y0) { y0 -= 1; y1 += 1; } if (x1 === x0) x1 = x0 + 1;
    const pad = { l: 52, r: 12, t: 10, b: 24 };
    const px = (s) => pad.l + ((s - x0) / (x1 - x0)) * (w - pad.l - pad.r);
    const py = (y) => pad.t + (1 - (y - y0) / (y1 - y0)) * (h - pad.t - pad.b);
    this.map = { px, py, x0, x1, y0, y1, pad, w, h };
    ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.fillStyle = mute; ctx.font = "10.5px system-ui"; ctx.textAlign = "right";
    const short = (v) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(1) : (+v.toPrecision(3)).toString());
    const yTicks = Math.max(2, Math.min(5, Math.floor((h - pad.t - pad.b) / 28)));
    for (let i = 0; i <= yTicks; i++) {
      const y = y0 + (i / yTicks) * (y1 - y0), yy = py(y);
      ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
      ctx.fillText(short(this.log ? Math.pow(10, y) : y), pad.l - 6, yy + 3.5);
    }
    ctx.textAlign = "center";
    const xTicks = Math.max(1, Math.min(6, Math.floor((w - pad.l - pad.r) / 64)));
    for (let i = 0; i <= xTicks; i++) { const s = x0 + (i / xTicks) * (x1 - x0); ctx.fillText(Math.round(s), px(s), h - 8); }
    const colors = ["#e07a3a", "#3a7ee0", "#3aa66a", "#c94c7a", "#8a6de0", "#c9a23a"];
    series.forEach(([path, e], i) => {
      ctx.strokeStyle = colors[i % colors.length]; ctx.lineWidth = 1.6; ctx.beginPath(); let started = false;
      for (const [s, v] of e.values) {
        if (this.log && !(v > 0)) { started = false; continue; }
        const x = px(s), y = py(this.log ? Math.log10(v) : v);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
      if (series.length > 1) { ctx.fillStyle = colors[i % colors.length]; ctx.textAlign = "left"; ctx.fillText(path.slice(this.spec.path.length), pad.l + 4, pad.t + 12 + i * 13); }
    });
    if (series.length === 1) {
      const [, e] = series[0], last = e.values[e.values.length - 1];
      ctx.fillStyle = fg; ctx.textAlign = "left"; ctx.font = "11px ui-monospace, monospace";
      ctx.fillText(`${fmt(last[1])} @ ${last[0]}`, pad.l + 4, pad.t + 12);
    }
  }
  hover(e) {
    if (!this.map) return;
    const rect = this.canvas.getBoundingClientRect(), x = e.clientX - rect.left;
    const s = this.map.x0 + ((x - this.map.pad.l) / (this.map.w - this.map.pad.l - this.map.pad.r)) * (this.map.x1 - this.map.x0);
    const lines = [];
    for (const [path, entry] of this.series()) {
      let best = null;
      for (const pair of entry.values) if (!best || Math.abs(pair[0] - s) < Math.abs(best[0] - s)) best = pair;
      if (best) lines.push(`${path.slice(this.spec.path.length) || path}: ${fmt(best[1])} @ ${best[0]}`);
    }
    this.overlay.textContent = lines.join("\n");
  }
}

// A panel that shows one step of a binary path, following the newest until the user scrubs back.
export class SteppedPanel extends Panel {
  constructor(app, spec) {
    super(app, spec);
    this.following = !(app.reference?.path === spec.path && app.reference.step !== null); this.shown = null;
    this._selectedStep = this.following ? null : app.reference.step;
    this.scrubSteps = null; this._showRequest = 0;
    this.range = el("input", { type: "range", min: 0, max: 0, value: 0, "aria-label": "Scrub " + spec.path, oninput: () => this.scrub() });
    this.stepLabel = el("span", {}, "–");
    this.range.addEventListener("pointerdown", () => this.beginScrub());
    this.range.addEventListener("keydown", (e) => { if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"].includes(e.key)) this.beginScrub(); });
    for (const event of ["change", "pointerup", "pointercancel", "lostpointercapture", "keyup", "blur"]) this.range.addEventListener(event, () => this.endScrub());
    this.liveButton = el("button", { onclick: () => { this.scrubSteps = null; this.following = true; this.liveButton.classList.add("on"); this.onUpdate(); } }, "live");
    this.liveButton.classList.add("on");
    this.checkpointButton = el("button", { onclick: () => this.loadCheckpoints(), title: "Browse saved snapshots, separate from the recent live window" }, "snapshots");
    this.olderButton = el("button", { onclick: () => this.loadCheckpoints(this.archiveSteps?.[0]), hidden: true }, "older");
    this.policyLabel = el("span", { class: "muted" });
    this.scrubBar = el("div", { class: "scrub" }, el("span", {}, "step"), this.range, this.stepLabel, this.liveButton,
      this.checkpointButton, this.olderButton, this.policyLabel);
    this.liveButton.addEventListener("click", () => { this.archiveSteps = null; this._archiveRequest = (this._archiveRequest || 0) + 1; this.olderButton.hidden = true; this.onUpdate(); });
    this.el.insertBefore(this.scrubBar, this.el.querySelector(".resize"));
  }
  entry() { return this.store.paths.get(this.spec.path); }
  visibleSteps() { return this.archiveSteps || this.entry()?.steps || []; }
  async loadCheckpoints(before = null) {
    const request = this._archiveRequest = (this._archiveRequest || 0) + 1;
    this.following = false; this.scrubSteps = null;
    this._showRequest = (this._showRequest || 0) + 1;
    this.liveButton.classList.remove("on");
    try {
      const result = await this.app.connection.request({ t: "checkpoints", path: this.spec.path, before, limit: 100 });
      if (request !== this._archiveRequest) return;
      this.archiveSteps = result.steps;
      this.olderButton.hidden = !result.has_more;
      this.range.max = Math.max(0, result.steps.length - 1); this.range.value = this.range.max;
      this._selectedStep = result.steps.at(-1);
      this.policyLabel.textContent = "saved snapshots";
      if (result.steps.length) await this.showStep(this._selectedStep);
      else this.stepLabel.textContent = "no saved snapshots";
    } catch (error) { if (request === this._archiveRequest) this.stepLabel.textContent = error.message; }
  }
  onUpdate() {
    const entry = this.entry();
    if (!entry || this.scrubSteps) return;
    if(this.scrubBar)this.scrubBar.hidden=Boolean(this.app.staticEvidence && entry.steps.length<=1 && !entry.bound);
    if (this.archiveSteps) return;
    if (this.policyLabel) this.policyLabel.textContent = entry.bound ? `recent ${entry.history_limit || entry.bound} · save ${entry.snapshot_every ? "every " + entry.snapshot_every + " steps" : "off"}` : "";
    this.range.max = Math.max(0, entry.steps.length - 1);
    if (this.following) { this.range.value = this.range.max; this.showLatest(); }
    else this.range.value = Math.max(0, entry.steps.indexOf(this._selectedStep));
  }
  beginScrub() {
    if (this.scrubSteps) return;
    const entry = this.entry(); if (!entry) return;
    this.scrubSteps = [...this.visibleSteps()]; this.following = false;
    this._showRequest = (this._showRequest || 0) + 1; // invalidate an in-flight live frame
    this._selectedStep = this.scrubSteps[+this.range.value];
    this.liveButton.classList.toggle("on", false);
  }
  endScrub() {
    if (!this.scrubSteps) return;
    this.scrubSteps = null;
    this.onUpdate(); // stay paused until the explicit live action
  }
  scrub() {
    const entry = this.entry(); if (!entry) return;
    const index = +this.range.value, step = (this.scrubSteps || this.visibleSteps())[index];
    this.following = false;
    this.liveButton.classList.toggle("on", this.following);
    this.showStep(step);
  }
  showLatest() { const entry = this.entry(); if (entry && entry.steps.length) this.showStep(entry.steps[entry.steps.length - 1]); }
  async showStep(step) {
    const entry = this.entry(); if (!entry) return;
    if (step === undefined || step === null) return;
    const request = this._showRequest = (this._showRequest || 0) + 1;
    this._selectedStep = step;
    this.stepLabel.textContent = String(step);
    let value = entry.cache.get(step);
    if (!value) {
      try { value = await this.app.connection.history(this.spec.path, step); } catch (error) {
        if (request === this._showRequest) this.stepLabel.textContent = `${step} unavailable`;
        return;
      }
      entry.cache.set(step, value);
      if (entry.cache.size > 64) entry.cache.delete(entry.cache.keys().next().value);
      this.store?.trimCaches();
    }
    if (request !== this._showRequest) return;
    try {
      if(this.ready && !(await this.ready))return;
      if(request!==this._showRequest)return;
      if (this.shown !== step || this.dirty) {
        this.show(value);this.shown=step;this.dirty=false;
        this.displayError?.remove();this.displayError=null;
      }
    } catch(error) {
      this.shown=null;
      if(!this.displayError){this.displayError=el('div',{class:'note display-error',role:'alert'});this.body.append(this.displayError);}
      this.displayError.textContent='Unable to display artifact: '+error.message;
    }
  }
  show() {}
}

class ImagePanel extends SteppedPanel {
  constructor(app, spec) {
    super(app, spec);
    this.canvas = el("canvas", { class: "fill" }); this.overlay = el("div", { class: "overlay" });
    this.canvas.style.imageRendering = "pixelated"; this.canvas.style.objectFit = "contain";
    this.body.append(this.canvas, this.overlay);
    this.exposure = 0;
    this.darker=this.tool("ev−", "darker (float images)", () => { this.exposure -= 0.5; this.dirty = true; this.showStep(this.shown); });
    this.brighter=this.tool("ev+", "brighter (float images)", () => { this.exposure += 0.5; this.dirty = true; this.showStep(this.shown); });
    this.canvas.addEventListener("pointerdown", (e) => this.probe(e));
    this.canvas.addEventListener("pointermove", (e) => this.probe(e));
    this.canvas.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse") this.overlay.textContent = this.caption || ""; });
    this.onUpdate();
  }
  show(value) {
    const { height, width, channels, dtype } = value, data = value.arrays.data;
    if(this.darker)this.darker.hidden=this.brighter.hidden=dtype==='uint8';
    this.image = value;
    this.canvas.width = width; this.canvas.height = height;
    const ctx = this.canvas.getContext("2d"), out = ctx.createImageData(width, height), gain = Math.pow(2, this.exposure);
    const encode = (x) => { x = Math.max(0, Math.min(1, x * gain)); return 255 * (x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055); };
    for (let i = 0; i < width * height; i++) {
      for (let c = 0; c < 3; c++) {
        const src = data[i * channels + (channels >= 3 ? c : 0)];
        out.data[i * 4 + c] = dtype === "uint8" ? src : encode(src);
      }
      out.data[i * 4 + 3] = channels === 4 ? (dtype === "uint8" ? data[i * 4 + 3] : Math.round(255 * Math.max(0, Math.min(1, data[i * 4 + 3])))) : 255;
    }
    ctx.putImageData(out, 0, 0);
    this.caption = `${width}×${height}×${channels} ${dtype}` + (dtype === "float32" ? ` ev ${this.exposure >= 0 ? "+" : ""}${this.exposure}` : "");
    this.overlay.textContent = this.caption;
  }
  probe(e) {
    if (!this.image) return;
    const rect = this.canvas.getBoundingClientRect();
    const { width, height, channels } = this.image, data = this.image.arrays.data;
    // object-fit: contain - find the drawn rectangle
    const s = Math.min(rect.width / width, rect.height / height), dw = width * s, dh = height * s;
    const ox = (rect.width - dw) / 2, oy = (rect.height - dh) / 2;
    const x = Math.floor((e.clientX - rect.left - ox) / s), y = Math.floor((e.clientY - rect.top - oy) / s);
    if (x < 0 || y < 0 || x >= width || y >= height) { this.overlay.textContent = this.caption; return; }
    const px = []; for (let c = 0; c < channels; c++) px.push(fmt(data[(y * width + x) * channels + c]));
    this.overlay.textContent = `(${x}, ${y}) ${px.join(" ")}`;
  }
}

class MeshPanel extends SteppedPanel {
  constructor(app, spec) {
    super(app, spec);
    this.canvas = el("canvas", { class: "fill mesh-canvas", "aria-label": "Mesh: drag to rotate, pinch to zoom, two fingers to pan, tap to pick" }); this.marker = el("canvas", { class: "fill" }); this.marker.style.pointerEvents = "none";
    this.overlay = el("div", { class: "overlay" });
    this.body.append(this.canvas, this.marker, this.overlay);
    this.el.insertBefore(el("div", { class: "gesture-help" }, "Drag: rotate · pinch: zoom · two fingers: pan · tap: pick. Scroll outside the mesh."), this.body);
    this.view = new MeshView(this.canvas, this.marker);
    this.view.onPick = (pick) => { this.pick = pick; this.caption(); };
    this.wireButton = this.tool("wire", "wireframe", () => { this.view.setOptions({ wireframe: !this.view.options.wireframe }); this.wireButton.classList.toggle("on", this.view.options.wireframe); });
    this.colorButton = this.tool("colour", "per-vertex colour", () => { this.view.setOptions({ vertexColors: !this.view.options.vertexColors }); this.colorButton.classList.toggle("on", this.view.options.vertexColors); });
    this.colorButton.classList.add("on");
    this.tool("frame", "frame geometry (double-click does too)", () => { if (this.mesh) this.view[this.mesh.positions?'setPoints':'setMesh'](this.mesh, { keepCamera: false }); });
    this.ready = this.view.init().then((ok) => {
      if (!ok) {
        const why = !window.isSecureContext
          ? `This page is not a secure context (${location.origin}), so the browser hides WebGPU. Open it over https - \`python -m gffx.dashboard --self-signed\`, or \`tailscale serve\` in front of --also-loopback - or on this machine as http://127.0.0.1:${location.port || 80}/.`
          : !navigator.gpu ? "This browser/device does not expose WebGPU. Try an updated browser with WebGPU support."
          : "This browser exposes WebGPU but returned no adapter; the GPU may be blocked or the page is in a restricted embedding.";
        this.body.replaceChildren(el("div", { class: "note" }, "The mesh panel renders with WebGPU. " + why));
      }
      else this.onUpdate();
      return ok;
    });
  }
  show(value) {
    const a = value.arrays;
    const points=Boolean(a.positions);
    this.head.querySelector('.kind').textContent=points?'points':'mesh';
    this.mesh = points ? {positions:a.positions,colors:a.colors||null,radii:a.radii||null} : { vertices: a.vertices, faces: a.faces, colors: a.colors || null, normals: a.normals || null };
    this.wireButton.hidden=points;
    if(points)this.view.setPoints(this.mesh);else this.view.setMesh(this.mesh);
    this.caption();
  }
  caption() {
    const m = this.mesh; if (!m) return;
    let text = m.positions ? `${m.positions.length/3} points` : `${m.vertices.length / 3} vertices · ${m.faces.length / 3} faces`;
    if (this.pick) text += `\n${m.positions?'point':'vertex'} ${this.pick.index} (${this.pick.position.map((v) => v.toFixed(4)).join(", ")})`+(m.positions?'':` face ${this.pick.face}`);
    this.overlay.textContent = text;
  }
  destroy() { super.destroy(); this.view.destroy(); }
}

class ControlPanel extends Panel {
  constructor(app, spec) { super(app, spec); this.list = el("div", { class: "controls" }); this.body.append(this.list); this.built = new Map(); this.onUpdate(); }
  interested(path) { return matches(this.spec.path, path); }
  onUpdate(path) {
    const defs = Object.entries(this.store.controls).filter(([p]) => matches(this.spec.path, p));
    for (const [p, def] of defs) {
      const entry = this.store.paths.get(p), value = entry && entry.latest ? entry.latest.value : def.default;
      let row = this.built.get(p);
      if (!row) { row = this.build(p, def); this.built.set(p, row); this.list.append(row.el); }
      if (!row.busy) row.set(value);
    }
    if (!defs.length && !this.list.children.length) this.list.append(el("div", { class: "note" }, "no controls declared under " + this.spec.path));
  }
  send(path, value) { return this.app.connection.request({ t: "control_set", path, value }).catch(() => {}); }
  build(path, def) {
    const label = el("div", { class: "label" }, def.label || path.slice(this.spec.path.length) || path);
    const wrap = el("div", { class: "control" }, label);
    const row = { el: wrap, busy: false, set: () => {} };
    if (def.kind === "slider") {
      const [lo, hi] = def.range || [0, 1], logScale = lo > 0 && hi / lo > 100;
      const toPos = (v) => logScale ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (v - lo) / (hi - lo);
      const fromPos = (p) => logScale ? Math.exp(Math.log(lo) + p * (Math.log(hi) - Math.log(lo))) : lo + p * (hi - lo);
      const val = el("span", { class: "val" });
      const input = el("input", { type: "range", min: 0, max: 1000, value: Math.round(toPos(def.default) * 1000), "aria-label": label.textContent });
      input.addEventListener("input", () => { row.busy = true; val.textContent = fmt(fromPos(input.value / 1000)); });
      input.addEventListener("change", () => { row.busy = false; this.send(path, fromPos(input.value / 1000)); });
      input.addEventListener("pointercancel", () => { row.busy = false; this.onUpdate(path); });
      wrap.append(el("div", { class: "row" }, input, val));
      row.set = (v) => { if (typeof v === "number") { input.value = Math.round(toPos(v) * 1000); val.textContent = fmt(v); } };
    } else if (def.kind === "toggle") {
      const input = el("input", { type: "checkbox", "aria-label": label.textContent });
      input.addEventListener("change", () => this.send(path, input.checked));
      wrap.append(el("label", { class: "row toggle-row" }, input, el("span", {}, "on")));
      row.set = (v) => { input.checked = !!v; };
    } else if (def.kind === "select") {
      const select = el("select", { "aria-label": label.textContent }, ...(def.options || []).map((o) => el("option", { value: o }, String(o))));
      select.addEventListener("change", () => this.send(path, select.value));
      wrap.append(el("div", { class: "row" }, select));
      row.set = (v) => { select.value = v; };
    } else {
      const count = el("span", { class: "val" }, "0");
      const button = el("button", { onclick: () => this.send(path, (+count.textContent || 0) + 1) }, "fire");
      wrap.append(el("div", { class: "row" }, button, count));
      row.set = (v) => { count.textContent = String(v || 0); };
    }
    return row;
  }
}

export const PANEL_KINDS = { mesh: MeshPanel, plot: PlotPanel, image: ImagePanel, text: TextPanel, record: TextPanel, tests: TestsPanel, control: ControlPanel };

// ---- layout ---------------------------------------------------------------------------------

function autoLayout(store) {
  const panels = [];
  const paths = [...store.paths.entries()];
  const has = (pred) => paths.some(([p, e]) => pred(p, e));
  for (const [path, entry] of paths) {
    if (path.startsWith("tests/") || path.startsWith("controls/")) continue;
    if (entry.kind === "mesh" || entry.kind === "points") panels.push({ kind: "mesh", path, w: 6, h: 7 });
    else if (entry.kind === "image") panels.push({ kind: "image", path, w: 4, h: 5 });
    else if (entry.kind === "scalar") panels.push({ kind: "plot", path, w: 4, h: 4 });
    else if (entry.kind === "text" || entry.kind === "record") panels.push({ kind: "text", path, w: 4, h: 3 });
  }
  if (has((p) => p.startsWith("controls/")) || Object.keys(store.controls).length) panels.push({ kind: "control", path: "controls/", w: 12, h: 3 });
  if (has((p) => p.startsWith("tests/"))) panels.push({ kind: "tests", path: "tests/", w: 12, h: 6 });
  // first-fit packing into the 12-column grid
  const occupied = [];
  const free = (c, r, w, h) => { for (let y = r; y < r + h; y++) for (let x = c; x < c + w; x++) if (occupied[y] && occupied[y][x]) return false; return true; };
  for (const p of panels) {
    outer: for (let r = 0; ; r++) for (let c = 0; c + p.w <= COLUMNS; c++) if (free(c, r, p.w, p.h)) {
      p.col = c; p.row = r;
      for (let y = r; y < r + p.h; y++) { occupied[y] = occupied[y] || []; for (let x = c; x < c + p.w; x++) occupied[y][x] = true; }
      break outer;
    }
  }
  return { name: "auto", columns: COLUMNS, panels };
}

// ---- the app --------------------------------------------------------------------------------

export class App {
  constructor() {
    this.store = new Store();
    try { this.reference = parseReference(location.href); }
    catch (error) { this.reference = {}; this.referenceError = error.message; }
    this.referenceEl = document.getElementById("page-reference");
    this.referenceFallback = document.getElementById("reference-fallback");
    document.getElementById("copy-reference").addEventListener("click", () => this.copyReference());
    document.getElementById("copy-link").addEventListener("click", () => this.copyReference(null, null, true));
    this.grid = document.getElementById("grid");
    this.statusEl = document.getElementById("status");
    this.runSelect = document.getElementById("run");
    this.layoutSelect = document.getElementById("layout");
    this.editButton = document.getElementById("edit");
    this.addButton = document.getElementById("add");
    this.delivery = document.getElementById("delivery");
    this.resourceEl = document.getElementById("resources");
    this.delivery.value = window.matchMedia("(pointer: coarse)").matches ? "low" : "standard";
    this.delivery.addEventListener("change", () => this.configureDelivery());
    document.addEventListener("visibilitychange", () => this.configureDelivery());
    this.panels = [];
    this.editing = false;
    this.layout = null;
    this.runSelect.addEventListener("change", () => this.openRun(this.runSelect.value));
    this.layoutSelect.addEventListener("change", () => this.chooseLayout(this.layoutSelect.value));
    document.getElementById("save").addEventListener("click", () => this.saveLayout());
    this.editButton.addEventListener("click", () => this.toggleEdit());
    this.addButton.addEventListener("click", () => this.addPanel());
    this.store.subscribe((path) => { if (path === null && !this.reference?.page_id && this.layout && this.layout.name === "auto") this.applyLayout(autoLayout(this.store)); });
    this.refreshRuns();
    setInterval(() => this.refreshRuns(), 5000);
  }
  status(text, bad = false) { this.statusEl.textContent = text; this.statusEl.classList.toggle("bad", bad); }
  viewOptions() {
    const low = this.delivery.value === "low";
    return { fps: low ? 2 : 10, bytes_per_second: low ? 128 * 1024 : 2 * 1024 * 1024,
      paused: document.hidden || this.delivery.value === "paused", mesh_deltas: true };
  }
  configureDelivery() {
    if (!this.connection) return;
    this.connection.viewOptions = this.viewOptions();
    this.connection.request({ t: "view_config", ...this.connection.viewOptions })
      .then(() => this.status(this.connection.viewOptions.paused ? "delivery paused" : `${this.delivery.value} bandwidth`))
      .catch((error) => this.status(`delivery limits unavailable: ${error.message}`, true));
  }
  async refreshRuns() {
    let runs = [];
    if (document.hidden) return;
    try { runs = (await (await fetch("/api/runs", { cache: "no-store" })).json()).runs; } catch { this.status("server unreachable", true); return; }
    const resources = runs.find((r) => r.run === this.run)?.resources;
    if (resources) {
      this.resourceEl.textContent = `history ${(resources.retained_bytes / 1048576).toFixed(1)} MiB · disk ${(resources.log_bytes / 1048576).toFixed(1)}/${(resources.log_byte_limit / 1048576).toFixed(0)} MiB · ${resources.evicted_entries} evicted` + (resources.recording_warning ? ` · ${resources.recording_warning}` : "");
      this.resourceEl.classList.toggle("bad", !!resources.recording_warning || resources.log_bytes > resources.log_byte_limit * .9);
    }
    const current = this.runSelect.value;
    this.runSelect.replaceChildren(...runs.map((r) => el("option", { value: r.run }, `${r.run}  (${r.names.join(", ") || "no process"}, step ${r.current_step})`)));
    const wanted = current || new URLSearchParams(location.search).get("run") || (runs.length ? runs[runs.length - 1].run : "");
    if (wanted) { this.runSelect.value = wanted; if (wanted !== this.run) this.openRun(wanted); }
  }
  openRun(run) {
    if (this.connection) this.connection.close();
    for (const panel of this.panels) panel.destroy();
    this.panels = []; this.layout = null;
    this.run = run;
    if (this.reference?.run && this.reference.run !== run) this.reference = {};
    this.metadata = null; this.activePageId = null;
    const address = new URL(location.href); address.searchParams.set("run", run);
    if (!this.reference?.page_id) { for (const key of ["page", "path", "step", "run_id"]) address.searchParams.delete(key); }
    history.replaceState(null, "", address);
    this.connection = new Connection(run, this.store, (t, bad) => this.status(`${run}: ${t}`, bad));
    this.connection.viewOptions = this.viewOptions();
    this.connection.onSnapshot = (snapshot) => this.onSnapshot(snapshot);
    this.connection.onProposed = (spec) => { if (!this.layout || this.layout.name === "auto") this.applyLayout(spec); };
    this.connection.connect();
  }
  onSnapshot(snapshot) {
    this.metadata = snapshot;
    if (this.referenceError || (this.reference?.run_id && this.reference.run_id !== snapshot.run_id)) {
      this.status(this.referenceError || "Referenced run identity does not match; refusing to substitute another run", true);
      this.referenceEl.textContent = "Reference unavailable";
      return;
    }
    this.layoutSelect.replaceChildren(el("option", { value: "" }, "auto"), ...snapshot.layouts.map((n) => el("option", { value: n }, n)));
    if (this.reference?.page_id) {
      const selected = snapshot.pages?.find((p) => p.page_id === this.reference.page_id);
      if (!selected) { this.status("Referenced page is unavailable; no fallback page selected", true); return; }
      if (selected.name === "auto") this.applyLayout({ ...autoLayout(this.store), page_id: selected.page_id });
      else if (selected.name === "@proposed") this.applyLayout({ ...snapshot.proposed_layout, page_id: selected.page_id });
      else { this.layoutSelect.value = selected.name; this.chooseLayout(selected.name, true); }
      return;
    }
    const remembered = localStorage.getItem(`gffx.dashboard.layout.${this.run}`);
    if (remembered && snapshot.layouts.includes(remembered)) { this.layoutSelect.value = remembered; this.chooseLayout(remembered); }
    else if (snapshot.proposed_layout) { this.applyLayout({ ...snapshot.proposed_layout, page_id: snapshot.pages?.find((p) => p.name === "@proposed")?.page_id }); }
    else this.applyLayout(autoLayout(this.store));
  }
  async chooseLayout(name, fromReference = false) {
    const request = this._layoutRequest = (this._layoutRequest || 0) + 1;
    if (!fromReference) this.reference = {};
    const connection = this.connection;
    localStorage.setItem(`gffx.dashboard.layout.${this.run}`, name);
    if (!name) { this.applyLayout(autoLayout(this.store)); return; }
    try { const result = await connection.request({ t: "layout_load", name }); if (connection === this.connection && request === this._layoutRequest) this.applyLayout(result.spec); } catch (error) { if (connection === this.connection && request === this._layoutRequest) this.status(`layout ${name}: ${error.message}`, true); }
  }
  applyLayout(spec) {
    for (const panel of this.panels) panel.destroy();
    this.panels = [];
    this.unsavedPage = false;
    this.layout = { name: spec.name || "auto", page_id: spec.page_id, columns: spec.columns || COLUMNS, panels: spec.panels.map((p) => ({ ...p })) };
    this.activePageId = spec.page_id || this.metadata?.pages?.find((p) => p.name === this.layout.name)?.page_id || null;
    for (const p of this.layout.panels) this.mount(p);
    this.updateReferenceLabel();
    if (this.reference?.path) {
      const panel = this.panels.find((p) => p.spec.path === this.reference.path);
      if (!panel) { this.status("Referenced artifact is not on this page; no substitute selected", true); return; }
      panel.el.scrollIntoView?.({ block: "nearest" });
      if (this.reference.step !== null && typeof panel.showStep === "function") {
        panel.following = false; panel._selectedStep = this.reference.step;
        panel.liveButton.classList.remove("on"); panel.showStep(this.reference.step);
      } else if (this.reference.step !== null) {
        panel.body.replaceChildren(el("p", {}, "This panel cannot display a pinned step. Use a mesh/image panel or the explicit history API; current values are not the referenced evidence."));
        panel.unsubscribe();
        this.status("Pinned step unsupported by this panel; not showing a substitute", true);
      }
    }
  }
  mount(spec) {
    const Kind = PANEL_KINDS[spec.kind] || TextPanel;
    const panel = new Kind(this, spec);
    this.panels.push(panel);
    this.grid.append(panel.el);
    this.decorate(panel);
    if (this.editing) panel.el.classList.add("editing");
  }
  async saveLayout() {
    const name = prompt("Save layout as", this.layout && this.layout.name !== "auto" ? this.layout.name : "");
    if (!name) return;
    const spec = { name, columns: COLUMNS, panels: this.layout.panels.map((p) => ({ ...p, options: { ...p.options } })) };
    try {
      const saved = await this.connection.request({ t: "layout_save", spec });
      this.layout.name = name;
      this.unsavedPage = false;
      if (saved?.page_id) { this.layout.page_id = saved.page_id; this.activePageId = saved.page_id; this.updateReferenceLabel(); }
      if (![...this.layoutSelect.options].some((o) => o.value === name)) this.layoutSelect.append(el("option", { value: name }, name));
      this.layoutSelect.value = name; localStorage.setItem(`gffx.dashboard.layout.${this.run}`, name);
      this.status(`${this.run}: layout ${name} saved`);
    } catch (error) { this.status(`save failed: ${error.message}`, true); }
  }
  toggleEdit() {
    this.editing = !this.editing;
    if (this.editing) this.unsavedPage = true;
    this.updateReferenceLabel();
    this.editButton.classList.toggle("on", this.editing);
    this.editButton.setAttribute("aria-pressed", String(this.editing));
    this.editButton.textContent = this.editing ? "done editing" : "edit";
    this.addButton.hidden = !this.editing;
    for (const panel of this.panels) { panel.el.classList.toggle("editing", this.editing); this.editForm(panel); }
  }
  updateReferenceLabel() {
    if (!this.referenceEl) return;
    const p = this.metadata?.provenance || {};
    this.referenceEl.textContent = `Page ${this.activePageId || "unsaved / unknown"}${this.unsavedPage ? " (unsaved edits)" : ""} · run ${this.run} (${this.metadata?.run_id || "unknown"}) · producer revision ${p.revision || "unknown"} · ${p.dirty === true ? "dirty" : p.dirty === false ? "clean" : "working-tree state unknown"}`;
  }
  async copyReference(path = null, step = null, linkOnly = false) {
    if (this.unsavedPage) { this.status("Save the edited page before copying its reference", true); return; }
    if (!this.activePageId || !this.metadata?.run_id) { this.status("Save this page before copying a stable reference", true); return; }
    const ref = evidenceReference(location.href, { run: this.run, run_id: this.metadata.run_id, page_id: this.activePageId,
      path, step, provenance: this.metadata.provenance });
    const text = linkOnly ? ref.url : ref.text;
    try { await navigator.clipboard.writeText(text); this.status("Reference copied; send it to your coding agent separately"); }
    catch { this.referenceFallback.hidden = false; this.referenceFallback.value = text; this.referenceFallback.focus(); this.referenceFallback.select(); this.status("Copy the selected reference manually"); }
  }
  addPanel() {
    const maxRow = Math.max(0, ...this.layout.panels.map((p) => p.row + p.h));
    const spec = { kind: "plot", path: "", col: 0, row: maxRow, w: 4, h: 4 };
    this.layout.panels.push(spec); this.mount(spec); this.editForm(this.panels[this.panels.length - 1]);
  }
  editForm(panel) {
    const existing = panel.el.querySelector(".edit-form");
    if (!this.editing) { if (existing) existing.remove(); return; }
    if (existing) return;
    const kind = el("select", { "aria-label": "Panel kind" }, ...Object.keys(PANEL_KINDS).map((k) => el("option", { value: k }, k)));
    kind.value = panel.spec.kind;
    const path = el("input", { type: "text", value: panel.spec.path, placeholder: "path or prefix/", "aria-label": "Panel path" });
    const apply = el("button", { onclick: () => this.rebind(panel, kind.value, path.value.trim()) }, "apply");
    const remove = el("button", { onclick: () => { this.layout.panels = this.layout.panels.filter((p) => p !== panel.spec); panel.destroy(); this.panels = this.panels.filter((p) => p !== panel); } }, "remove");
    const move = (delta) => {
      const index = mobileOrder(this.layout.panels).indexOf(panel.spec);
      moveMobilePanel(this.layout.panels, panel.spec, index + delta); this.placePanels();
    };
    const mobile = el("div", { class: "mobile-edit" },
      el("span", {}, "Phone layout:"),
      el("button", { onclick: () => move(-1), "aria-label": "Move panel up" }, "↑ up"),
      el("button", { onclick: () => move(1), "aria-label": "Move panel down" }, "↓ down"),
      el("button", { onclick: () => this.resizeMobile(panel, -1) }, "shorter"),
      el("button", { onclick: () => this.resizeMobile(panel, 1) }, "taller"));
    panel.el.insertBefore(el("div", { class: "edit-form" }, kind, path, apply, remove, mobile), panel.body);
  }
  rebind(panel, kind, path) {
    const spec = panel.spec;
    spec.kind = kind; spec.path = path;
    const index = this.panels.indexOf(panel);
    panel.destroy();
    const Kind = PANEL_KINDS[kind] || TextPanel;
    const fresh = new Kind(this, spec);
    this.panels[index] = fresh; this.grid.append(fresh.el); this.decorate(fresh);
    fresh.el.classList.add("editing"); this.editForm(fresh);
  }
  decorate(panel) {
    // Pointer capture owns only the header/handle; scrolling remains native everywhere else.
    const cell = () => { const rect = this.grid.getBoundingClientRect(); return { w: (rect.width - 16 - 8 * (COLUMNS - 1)) / COLUMNS + 8, h: 56 + 8 }; };
    let initial, mobile;
    const begin = () => { initial = { ...panel.spec, options: { ...panel.spec.options } }; mobile = this.isMobile(); };
    const clean = () => { panel.el.style.transform = ""; panel.el.classList.remove("dragging"); };
    const cancel = () => { Object.assign(panel.spec, initial); clean(); this.placePanels(); };
    const enabled = () => this.editing && (initial === undefined || mobile === this.isMobile());
    panel.dragDisposers = [bindDrag(panel.head, {
      enabled: () => this.editing, start: begin,
      move: (e, dx, dy) => {
        if (!enabled()) return;
        if (mobile) { panel.el.style.transform = `translateY(${dy}px)`; panel.el.classList.add("dragging"); }
        else {
          panel.spec.col = Math.max(0, Math.min(COLUMNS - panel.spec.w, initial.col + Math.round(dx / cell().w)));
          panel.spec.row = Math.max(0, initial.row + Math.round(dy / cell().h)); this.placePanels();
        }
      },
      end: (e, _dx, dy) => {
        if (!enabled()) { cancel(); return; }
        if (mobile && Math.abs(dy) >= 6) {
          const others = mobileOrder(this.layout.panels).filter((p) => p !== panel.spec);
          const index = others.filter((spec) => {
            const rect = this.panels.find((p) => p.spec === spec).el.getBoundingClientRect();
            return e.clientY > rect.top + rect.height / 2;
          }).length;
          moveMobilePanel(this.layout.panels, panel.spec, index); this.placePanels();
        }
        clean();
      }, cancel,
    })];
    const handle = panel.el.querySelector(".resize");
    panel.dragDisposers.push(bindDrag(handle, {
      enabled: () => this.editing, start: begin,
      move: (_e, dx, dy) => {
        if (!enabled()) return;
        if (mobile) this.resizeMobile(panel, Math.round(dy / 64), initial.options.mobileHeight || Math.max(5, initial.h));
        else {
          panel.spec.w = Math.max(1, Math.min(COLUMNS - panel.spec.col, initial.w + Math.round(dx / cell().w)));
          panel.spec.h = Math.max(1, initial.h + Math.round(dy / cell().h)); panel.place();
        }
      }, end: () => { if (!enabled()) cancel(); }, cancel,
    }));
  }
  isMobile() { return window.matchMedia("(max-width: 760px)").matches; }
  placePanels() { for (const panel of this.panels) panel.place(); }
  resizeMobile(panel, delta, base = panel.spec.options?.mobileHeight || Math.max(5, panel.spec.h)) {
    panel.spec.options = { ...panel.spec.options, mobileHeight: Math.max(5, Math.min(20, base + delta)) };
    panel.place();
  }
}

if (typeof window !== "undefined" && document.body?.dataset.gffxApp === "general") window.gffxDashboard = new App();
