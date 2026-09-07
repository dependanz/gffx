// The mesh viewport: WebGPU, the shaders in mesh.wgsl, and nothing else.
// It is a display. gffx's own rasterizer replaces it when the WebGPU provider exists.

const ORBIT_SENSITIVITY = 0.008;

export function webgpuAvailable() {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}

// ---- small matrix kit (column-major, as WebGPU reads mat4x4<f32>) ----------------------------

function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect; out[5] = f;
  out[10] = far / (near - far); out[11] = -1;
  out[14] = (near * far) / (near - far);
  return out;
}
function lookAt(eye, target, up) {
  const z = normalize(sub(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  const out = new Float32Array(16);
  out[0] = x[0]; out[1] = y[0]; out[2] = z[0];
  out[4] = x[1]; out[5] = y[1]; out[6] = z[1];
  out[8] = x[2]; out[9] = y[2]; out[10] = z[2];
  out[12] = -dot(x, eye); out[13] = -dot(y, eye); out[14] = -dot(z, eye); out[15] = 1;
  return out;
}
function multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  }
  return out;
}
function transformPoint(m, p) {
  const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
  const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
  const z = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
  return [x, y, z, w];
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function normalize(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }

// ---- mesh preparation -----------------------------------------------------------------------

function vertexNormals(positions, faces, count) {
  const normals = new Float32Array(count * 3);
  for (let f = 0; f < faces.length; f += 3) {
    const a = faces[f], b = faces[f + 1], c = faces[f + 2];
    const ax = positions[a * 3], ay = positions[a * 3 + 1], az = positions[a * 3 + 2];
    const e1 = [positions[b * 3] - ax, positions[b * 3 + 1] - ay, positions[b * 3 + 2] - az];
    const e2 = [positions[c * 3] - ax, positions[c * 3 + 1] - ay, positions[c * 3 + 2] - az];
    const n = cross(e1, e2);  // area-weighted: the cross product's length is twice the face area
    for (const i of [a, b, c]) { normals[i * 3] += n[0]; normals[i * 3 + 1] += n[1]; normals[i * 3 + 2] += n[2]; }
  }
  for (let i = 0; i < count; i++) {
    const l = Math.hypot(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]) || 1;
    normals[i * 3] /= l; normals[i * 3 + 1] /= l; normals[i * 3 + 2] /= l;
  }
  return normals;
}

function edgeIndices(faces) {
  const seen = new Set();
  const out = [];
  for (let f = 0; f < faces.length; f += 3) {
    const tri = [faces[f], faces[f + 1], faces[f + 2]];
    for (let k = 0; k < 3; k++) {
      const a = tri[k], b = tri[(k + 1) % 3];
      const key = a < b ? a * 4294967296 + b : b * 4294967296 + a;
      if (!seen.has(key)) { seen.add(key); out.push(a, b); }
    }
  }
  return new Uint32Array(out);
}

function toFloat32Colors(colors, count) {
  if (!colors) return null;
  if (colors instanceof Float32Array) return colors;
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count * 3; i++) out[i] = colors[i] / 255;
  return out;
}

// ---- the view -------------------------------------------------------------------------------

export class MeshView {
  constructor(canvas, overlay) {
    this.canvas = canvas;
    this.overlay = overlay;  // a 2D canvas above the viewport, for the pick marker
    this.device = null;
    this.context = null;
    this.mesh = null;
    this.buffers = null;
    this.options = { wireframe: false, vertexColors: true };
    this.picked = -1;
    this.orbit = { yaw: 0.6, pitch: 0.35, distance: 3, target: [0, 0, 0], radius: 1 };
    this.onPick = null;
    this._pending = false;
    this._pointers = new Map();
    this._bindEvents();
  }

  async init() {
    if (!webgpuAvailable()) return false;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;
    this.device = await adapter.requestDevice();
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context = this.canvas.getContext("webgpu");
    this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    const code = await (await fetch("/mesh.wgsl", { cache: "no-store" })).text();
    const module = this.device.createShaderModule({ code });
    const errors=(await module.getCompilationInfo()).messages.filter(message=>message.type==='error');
    if(errors.length)throw Error('Viewport shader: '+errors.map(message=>message.message).join('; '));
    this.uniformBuffer = this.device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bindGroupLayout = this.device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }] });
    this.bindGroup = this.device.createBindGroup({ layout: this.bindGroupLayout, entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }] });
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    const vertexBuffers = [
      { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] },
      { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }] },
      { arrayStride: 12, attributes: [{ shaderLocation: 2, offset: 0, format: "float32x3" }] },
    ];
    const depthStencil = { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" };
    this.surfacePipeline = this.device.createRenderPipeline({
      layout, vertex: { module, entryPoint: "vs_main", buffers: vertexBuffers },
      fragment: { module, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" }, depthStencil,
    });
    this.linePipeline = this.device.createRenderPipeline({
      layout, vertex: { module, entryPoint: "vs_main", buffers: vertexBuffers },
      fragment: { module, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "line-list" }, depthStencil,
    });
    this.pointPipeline = this.device.createRenderPipeline({
      layout, vertex: {module, entryPoint:"vs_point", buffers:[
        ...vertexBuffers.map(b=>({...b,stepMode:"instance"})),
        {arrayStride:4,stepMode:"instance",attributes:[{shaderLocation:3,offset:0,format:"float32"}]},
      ]},
      fragment:{module,entryPoint:"fs_point",targets:[{format:this.format}]},
      primitive:{topology:"triangle-list"},depthStencil,
    });
    this._resizeObserver = new ResizeObserver(() => this.requestRender());
    this._resizeObserver.observe(this.canvas);
    return true;
  }

  destroy() {
    this._unbindEvents();
    if (this._resizeObserver) this._resizeObserver.disconnect();
    if (this.device) this.device.destroy();
    this.device = null;
  }

  setPoints({positions, colors, radii}, options = {}) {
    if (!positions || positions.length%3 || !Array.from(positions).every(Number.isFinite)) throw Error("Invalid point positions");
    const count=positions.length/3;
    if(radii && (radii.length!==count || !Array.from(radii).every(r=>Number.isFinite(r)&&r>=0))) throw Error("Invalid point radii");
    this.setMesh({vertices:new Float32Array(positions),faces:new Uint32Array(),colors,kind:"points",radii:radii?new Float32Array(radii):undefined},options);
  }

  setMesh({ vertices, faces, colors, normals, kind="mesh", radii }, { keepCamera = true } = {}) {
    if(!vertices || !faces || vertices.length%3 || faces.length%3) throw Error("Invalid mesh vertices or faces");
    const count = vertices.length / 3;
    const previous = this.mesh;
    this.mesh = {
      positions: vertices, faces, count, kind, radii,
      normals: normals || vertexNormals(vertices, faces, count),
      colors: toFloat32Colors(colors, count),
      edges: null,
    };
    if (!this.mesh.colors) {
      this.mesh.colors = new Float32Array(count * 3).fill(0.78);
    }
    // frame the mesh on first arrival, or when its size changed materially
    let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < count; i++) for (let k = 0; k < 3; k++) {
      const v = vertices[i * 3 + k]; if (v < min[k]) min[k] = v; if (v > max[k]) max[k] = v;
    }
    const center = count ? scale(add(min, max), 0.5) : [0,0,0];
    const radius = count ? Math.max(Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2, 1e-6) : 1;
    if(kind==="points"&&!radii)this.mesh.radii=new Float32Array(count).fill(radius*.012);
    if (!keepCamera || !previous || Math.abs(radius - this.orbit.radius) / this.orbit.radius > 0.5) {
      this.orbit.target = center; this.orbit.radius = radius; this.orbit.distance = radius * 2.8;
    }
    if (this.picked >= count) this.picked = -1;
    this._upload();
    this.requestRender();
  }

  setOptions(options) { Object.assign(this.options, options); this.requestRender(); }

  _upload() {
    if (!this.device || !this.mesh) return;
    const m = this.mesh;
    const make = (data, usage) => {
      const buffer = this.device.createBuffer({ size: Math.max(4, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    if (this.buffers) for (const b of Object.values(this.buffers)) b.destroy();
    m.edges = edgeIndices(m.faces);
    this.buffers = {
      position: make(m.positions, GPUBufferUsage.VERTEX),
      normal: make(m.normals, GPUBufferUsage.VERTEX),
      color: make(m.colors, GPUBufferUsage.VERTEX),
      index: make(m.faces, GPUBufferUsage.INDEX),
      edges: make(m.edges, GPUBufferUsage.INDEX),
    };
    if(m.kind==="points")this.buffers.radius=make(m.radii,GPUBufferUsage.VERTEX);
  }

  _matrices() {
    const o = this.orbit;
    const eye = add(o.target, [
      o.distance * Math.cos(o.pitch) * Math.sin(o.yaw),
      o.distance * Math.sin(o.pitch),
      o.distance * Math.cos(o.pitch) * Math.cos(o.yaw),
    ]);
    const view = lookAt(eye, o.target, [0, 1, 0]);
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    const near = Math.max(o.distance - o.radius * 4, o.radius * 0.01), far = o.distance + o.radius * 4;
    const proj = perspective(Math.PI / 4, aspect, near, far);
    return { view, proj, mvp: multiply(proj, view), eye };
  }

  requestRender() {
    if (this._pending) return;
    this._pending = true;
    requestAnimationFrame(() => { this._pending = false; this.render(); });
  }

  render() {
    if (!this.device) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w; this.canvas.height = h;
      if (this.depth) this.depth.destroy();
      this.depth = this.device.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    if (!this.depth) this.depth = this.device.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: { r: 0.11, g: 0.11, b: 0.12, a: 1 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: this.depth.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    if (this.mesh && this.buffers) {
      const { view, mvp } = this._matrices();
      const uniforms = new Float32Array(40);
      uniforms.set(mvp, 0); uniforms.set(view, 16);
      uniforms.set([this.options.vertexColors ? 1 : 0, 0, this.picked, 0], 32);
      uniforms.set([0.78, 0.78, 0.78, 1], 36);
      this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);
      pass.setBindGroup(0, this.bindGroup);
      pass.setVertexBuffer(0, this.buffers.position);
      pass.setVertexBuffer(1, this.buffers.normal);
      pass.setVertexBuffer(2, this.buffers.color);
      if(this.mesh.kind==="points"){
        pass.setVertexBuffer(3,this.buffers.radius);pass.setPipeline(this.pointPipeline);
        pass.draw(6, this.mesh.count);
        pass.end();this.device.queue.submit([encoder.finish()]);this._drawOverlay(mvp);return;
      }
      pass.setPipeline(this.surfacePipeline);
      pass.setIndexBuffer(this.buffers.index, "uint32");
      pass.drawIndexed(this.mesh.faces.length);
      if (this.options.wireframe) {
        // a second uniform write would race the first inside one submission, so the wireframe
        // pass is its own submission
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        uniforms[33] = 1;
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);
        const encoder2 = this.device.createCommandEncoder();
        const pass2 = encoder2.beginRenderPass({
          colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: "load", storeOp: "store" }],
          depthStencilAttachment: { view: this.depth.createView(), depthLoadOp: "load", depthStoreOp: "store" },
        });
        pass2.setBindGroup(0, this.bindGroup);
        pass2.setVertexBuffer(0, this.buffers.position);
        pass2.setVertexBuffer(1, this.buffers.normal);
        pass2.setVertexBuffer(2, this.buffers.color);
        pass2.setPipeline(this.linePipeline);
        pass2.setIndexBuffer(this.buffers.edges, "uint32");
        pass2.drawIndexed(this.mesh.edges.length);
        pass2.end();
        this.device.queue.submit([encoder2.finish()]);
        this._drawOverlay(mvp);
        return;
      }
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      this._drawOverlay(mvp);
      return;
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this._drawOverlay(null);
  }

  _drawOverlay(mvp) {
    if (!this.overlay) return;
    const ctx = this.overlay.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (this.overlay.width !== Math.floor(w * dpr) || this.overlay.height !== Math.floor(h * dpr)) {
      this.overlay.width = Math.floor(w * dpr); this.overlay.height = Math.floor(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (mvp && this.mesh && this.picked >= 0) {
      const p = this.mesh.positions;
      const c = transformPoint(mvp, [p[this.picked * 3], p[this.picked * 3 + 1], p[this.picked * 3 + 2]]);
      if (c[3] > 0) {
        const x = (c[0] / c[3] * 0.5 + 0.5) * w, y = (1 - (c[1] / c[3] * 0.5 + 0.5)) * h;
        ctx.strokeStyle = "#ffb454"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x - 10, y); ctx.lineTo(x + 10, y); ctx.moveTo(x, y - 10); ctx.lineTo(x, y + 10); ctx.stroke();
      }
    }
  }

  // ---- interaction --------------------------------------------------------------------------

  _bindEvents() {
    const c = this.canvas;
    const points = this._pointers, listeners = [];
    const on = (name, fn, options) => { c.addEventListener(name, fn, options); listeners.push([name, fn, options]); };
    const pair = () => {
      const [a, b] = [...points.values()];
      return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, span: Math.hypot(a.x - b.x, a.y - b.y) };
    };
    const pan = (dx, dy) => {
      const { view } = this._matrices(), s = this.orbit.distance * 0.0015;
      this.orbit.target = add(this.orbit.target, add(scale([view[0], view[4], view[8]], -dx * s), scale([view[1], view[5], view[9]], dy * s)));
    };
    const zoom = (factor) => {
      const o = this.orbit;
      o.distance = Math.max(o.radius * 0.05, Math.min(o.radius * 50, o.distance * factor));
    };
    on("pointerdown", (e) => {
      if (e.button !== 0 && e.button !== 2 || points.size >= 2) return;
      e.preventDefault(); c.setPointerCapture(e.pointerId);
      points.set(e.pointerId, { x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY,
        button: e.button, shift: e.shiftKey, dragging: false, canPick: !e.shiftKey && e.button === 0 });
      if (points.size > 1) for (const p of points.values()) { p.canPick = false; p.dragging = true; }
    });
    on("pointermove", (e) => {
      const p = points.get(e.pointerId); if (!p) return;
      e.preventDefault();
      const before = points.size === 2 ? pair() : null;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      if (before) {
        const after = pair();
        pan(after.x - before.x, after.y - before.y);
        // Coincident fingers have no well-defined pinch scale.
        if (before.span > 2 && after.span > 2) zoom(before.span / after.span);
      } else {
        if (Math.hypot(p.x - p.startX, p.y - p.startY) >= 6) { p.dragging = true; p.canPick = false; }
        if (!p.dragging) return;
        if (p.button === 2 || p.shift) pan(dx, dy);
        else {
          this.orbit.yaw -= dx * ORBIT_SENSITIVITY;
          this.orbit.pitch = Math.max(-1.55, Math.min(1.55, this.orbit.pitch + dy * ORBIT_SENSITIVITY));
        }
      }
      this.requestRender();
    });
    const release = (e, cancelled = false) => {
      const p = points.get(e.pointerId); if (!p) return;
      points.delete(e.pointerId);
      if (c.hasPointerCapture(e.pointerId)) c.releasePointerCapture(e.pointerId);
      if (!cancelled && p.canPick && Math.hypot(e.clientX - p.startX, e.clientY - p.startY) < 6) this._pickAt(e.clientX, e.clientY);
    };
    on("pointerup", (e) => release(e));
    on("pointercancel", (e) => release(e, true));
    on("lostpointercapture", (e) => release(e, true));
    on("contextmenu", (e) => e.preventDefault());
    on("wheel", (e) => { e.preventDefault(); zoom(Math.exp(e.deltaY * 0.0012)); this.requestRender(); }, { passive: false });
    on("dblclick", () => {
      if (this.mesh) this.setMesh({ ...this.mesh, vertices: this.mesh.positions }, { keepCamera: false });
    });
    this._unbindEvents = () => {
      for (const [name, fn, options] of listeners) c.removeEventListener(name, fn, options);
      for (const id of points.keys()) if (c.hasPointerCapture(id)) c.releasePointerCapture(id);
      points.clear();
    };
  }

  _pickAt(clientX, clientY) {
    if (!this.mesh) return;
    const rect = this.canvas.getBoundingClientRect();
    if(this.mesh.kind==="points"){
      const {mvp}=this._matrices();let best=12*12,index=-1;
      for(let i=0;i<this.mesh.count;i++){
        const p=this.mesh.positions.slice(i*3,i*3+3),c=transformPoint(mvp,p);
        if(c[3]<=0 || c[2]<0 || c[2]>c[3])continue;
        const x=rect.left+(c[0]/c[3]*.5+.5)*rect.width,y=rect.top+(1-(c[1]/c[3]*.5+.5))*rect.height;
        const d=(x-clientX)**2+(y-clientY)**2;if(d<best){best=d;index=i;}
      }
      this.picked=index;this.requestRender();
      this.onPick?.(index<0?null:{index,face:null,position:Array.from(this.mesh.positions.slice(index*3,index*3+3))});return;
    }
    const nx = ((clientX - rect.left) / rect.width) * 2 - 1, ny = 1 - ((clientY - rect.top) / rect.height) * 2;
    const { view, eye } = this._matrices();
    const aspect = rect.width / rect.height, t = Math.tan(Math.PI / 8);
    // the ray in view space, then rotated into world space by the transpose of the view rotation
    const dv = normalize([nx * t * aspect, ny * t, -1]);
    const dir = normalize([
      view[0] * dv[0] + view[1] * dv[1] + view[2] * dv[2],
      view[4] * dv[0] + view[5] * dv[1] + view[6] * dv[2],
      view[8] * dv[0] + view[9] * dv[1] + view[10] * dv[2],
    ]);
    const p = this.mesh.positions, f = this.mesh.faces;
    let best = Infinity, bestFace = -1, bestUV = null;
    for (let i = 0; i < f.length; i += 3) {
      const a = f[i] * 3, b = f[i + 1] * 3, c = f[i + 2] * 3;
      const v0 = [p[a], p[a + 1], p[a + 2]];
      const e1 = [p[b] - v0[0], p[b + 1] - v0[1], p[b + 2] - v0[2]];
      const e2 = [p[c] - v0[0], p[c + 1] - v0[1], p[c + 2] - v0[2]];
      const h = cross(dir, e2), det = dot(e1, h);
      if (Math.abs(det) < 1e-12) continue;
      const inv = 1 / det, s = sub(eye, v0), u = dot(s, h) * inv;
      if (u < 0 || u > 1) continue;
      const q = cross(s, e1), v = dot(dir, q) * inv;
      if (v < 0 || u + v > 1) continue;
      const tt = dot(e2, q) * inv;
      if (tt > 1e-6 && tt < best) { best = tt; bestFace = i / 3; bestUV = [u, v]; }
    }
    if (bestFace < 0) { this.picked = -1; this.requestRender(); if (this.onPick) this.onPick(null); return; }
    const w = [1 - bestUV[0] - bestUV[1], bestUV[0], bestUV[1]];
    const corner = w.indexOf(Math.max(...w));
    this.picked = f[bestFace * 3 + corner];
    this.requestRender();
    if (this.onPick) this.onPick({ index: this.picked, face: bestFace, position: [p[this.picked * 3], p[this.picked * 3 + 1], p[this.picked * 3 + 2]] });
  }
}
