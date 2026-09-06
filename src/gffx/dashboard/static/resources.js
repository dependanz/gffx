// Display resource policy. No scientific/conformance claims and no external dependency.
export class MeshReceiver {
  constructor(maxBytes = 8 * 1024 * 1024) { this.maxBytes = maxBytes; this.paths = new Map(); this.bytes = 0; }
  clear() { this.paths.clear(); this.bytes = 0; }
  decode(message) {
    if (message.kind !== "mesh" || !message.arrays) return message;
    const old = this.paths.get(message.path);
    let arrays = message.arrays;
    if (message.mesh_delta) {
      if (!old || old.step !== message.base_step) throw new Error("mesh base missing; requesting full frame");
      arrays = Object.fromEntries(message.mesh_fields.map((name) => [name, message.arrays[name] || old.arrays[name]]));
      if (Object.values(arrays).some((value) => !value)) throw new Error("mesh base array missing");
    }
    if (old) { this.paths.delete(message.path); this.bytes -= old.bytes; }
    const saved = Object.fromEntries(Object.entries(arrays).filter(([name]) => name !== "vertices"));
    const bytes = Object.values(saved).reduce((total, value) => total + value.byteLength, 0);
    while (this.paths.size && this.bytes + bytes > this.maxBytes) {
      const key = this.paths.keys().next().value; this.bytes -= this.paths.get(key).bytes; this.paths.delete(key);
    }
    if (bytes <= this.maxBytes) { this.paths.set(message.path, { step: message.step, arrays: saved, bytes }); this.bytes += bytes; }
    return { ...message, arrays };
  }
}

export function plotSamples(values, limit = 512) {
  if (values.length <= limit) return values;
  const result = [values[0]], buckets = Math.floor((limit - 2) / 2);
  for (let i = 0; i < buckets; i++) {
    const start = 1 + Math.floor(i * (values.length - 2) / buckets);
    const end = 1 + Math.floor((i + 1) * (values.length - 2) / buckets);
    let low = start, high = start;
    for (let j = start + 1; j < end; j++) { if (values[j][1] < values[low][1]) low = j; if (values[j][1] > values[high][1]) high = j; }
    for (const index of [...new Set([low, high])].sort((a, b) => a - b)) result.push(values[index]);
  }
  result.push(values.at(-1)); return result;
}
