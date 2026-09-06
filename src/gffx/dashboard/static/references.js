// Evidence addressing only: never sends a message or executes an agent action.
export function parseReference(input) {
  const url = new URL(input), q = url.searchParams;
  const step = q.has("step") ? Number(q.get("step")) : null;
  if (step !== null && (!/^\d+$/.test(q.get("step")) || !Number.isSafeInteger(step))) throw new Error("Invalid reference step");
  return { run: q.get("run"), run_id: q.get("run_id"), page_id: q.get("page"), path: q.get("path"), step };
}

export function evidenceReference(base, ref) {
  const url = new URL(base); url.search = ""; url.hash = "";
  for (const [param, key] of [["run", "run"], ["run_id", "run_id"], ["page", "page_id"], ["path", "path"], ["step", "step"]]) {
    if (ref[key] !== null && ref[key] !== undefined) url.searchParams.set(param, String(ref[key]));
  }
  const p = ref.provenance || {};
  const dirty = p.dirty === true ? "dirty" : p.dirty === false ? "clean" : "working-tree state unknown";
  const text = [`Page: ${ref.page_id || "unknown (save page first)"}`, `Run: ${ref.run} (${ref.run_id || "unknown"})`,
    `Producer-reported revision: ${p.revision || "unknown"} (${dirty})`, `Artifact SHA-256: ${p.artifact_sha256 || "unknown"}`,
    `Artifact path: ${ref.path || "whole page"}`, `Step: ${ref.step ?? "live / not pinned"}`,
    "A reference does not guarantee that an ephemeral frame is still retained.", url.href].join("\n");
  return { url: url.href, text };
}
