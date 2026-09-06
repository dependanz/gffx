// Pointer ownership and a narrow-screen layout projection. No browser/rendering dependency.
export function bindDrag(target, { enabled, start, move, end = () => {}, cancel = () => {} }) {
  let drag = null;
  const finish = (e, cancelled) => {
    if (!drag || e.pointerId !== drag.id) return;
    const old = drag; drag = null;
    if (target.hasPointerCapture(old.id)) target.releasePointerCapture(old.id);
    (cancelled ? cancel : end)(e, e.clientX - old.x, e.clientY - old.y);
  };
  const handlers = {
    pointerdown(e) {
      if (drag || !enabled() || e.button !== 0 || e.target.closest("button, input, select, a")) return;
      e.preventDefault(); e.stopPropagation();
      drag = { id: e.pointerId, x: e.clientX, y: e.clientY };
      target.setPointerCapture(e.pointerId); start(e);
    },
    pointermove(e) {
      if (!drag || e.pointerId !== drag.id) return;
      if (!enabled()) { finish(e, true); return; }
      e.preventDefault(); move(e, e.clientX - drag.x, e.clientY - drag.y);
    },
    pointerup: (e) => finish(e, !enabled()),
    pointercancel: (e) => finish(e, true),
    lostpointercapture: (e) => finish(e, true),
  };
  for (const [name, fn] of Object.entries(handlers)) target.addEventListener(name, fn);
  return () => {
    if (drag) finish({ pointerId: drag.id }, true);
    for (const [name, fn] of Object.entries(handlers)) target.removeEventListener(name, fn);
  };
}

export function mobileOrder(specs) {
  const rank = (p) => Number.isFinite(p.options?.mobileOrder) ? p.options.mobileOrder : 1e6 + p.row * 12 + p.col;
  return [...specs].sort((a, b) => rank(a) - rank(b));
}

export function moveMobilePanel(specs, spec, index) {
  const ordered = mobileOrder(specs).filter((p) => p !== spec);
  ordered.splice(Math.max(0, Math.min(ordered.length, index)), 0, spec);
  ordered.forEach((p, i) => { p.options = { ...p.options, mobileOrder: i }; });
}
