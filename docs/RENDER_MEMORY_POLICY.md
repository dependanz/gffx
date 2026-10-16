# Renderer memory admission

The development renderer keeps its64MiB default workspace allowance. An explicit
CUDA workspace_limit_bytes may be larger, up to the selected device's physical
capacity. This value is a caller-supplied admission ceiling, not a reservation or
a guarantee that an allocation will succeed. CPU admission retains the existing
64MiB limit. Changing this argument does not change renderer arithmetic, default
outputs, device ownership or the current Principled point/light/resource limits.

For large CUDA experiments, inspect memory after creating inputs and resources.
The current development host policy reserves max(1GiB,20 percent of free VRAM),
ignores reusable allocator cache, and rounds its process tensor ceiling down to
whole MiB. Subtract the entire current allocated tensor baseline before passing
the remaining allowance into render. This conservatively counts borrowed input
storage twice and leaves room for tensors outside the scene inventory. The pure
calculation lives in tools/deferred_vram_budget.py; it is development tooling,
not a new public renderer API or a promise about other applications' allocations.

Prefer a whole frame when its complete forward/backward storage and time estimate
fits. Choose the largest qualified tiles only if needed; that tiled execution path
still needs qualification. Keep its layout fixed during forward/backward, preserve
global pixel coordinates and shared-parameter gradient accumulation, and include
all transfers/recomputation in measured work. CPU output streaming is a further
fallback requiring explicit ownership. Available VRAM never implies tile parity.

The current vram19 installed checkpoint passes39 focused resource/device controls,
including exact small colour/mask/albedo-gradient bytes with64MiB and256MiB
allowances in both precisions. Full128/256 Classic CUDA calibration frames pass.
The Principled calibration passes memory admission but larger frames exceed the
prospective time allowance. These are finite1986-vertex/3968-face float64 head
calibrations; they do not close FLAME-scale,soft/full-gradient,tiling or M1 gates.
Authoritative exact artifacts and retained failures are in devbrain NUMBERS.md.
