# Local log-loss interpolation experiment

This is offline scalar CPU float64 research tooling, not a new public operation
or a replacement for the existing energy table. The next question is whether a
practical bounded resource can approximate the repaired remaining-light reference.

`tools.material_loss_cell.generate` explicitly samples eight loss endpoints:
two roughness knots, two interior view-cosine knots, and F0=0/1. It uses at most
4194304 light evaluations at resolution 512. `LossCell.evaluate` performs only
interpolation, with no integration, cache generation or hidden fallback.

Roughness and cosine coordinates are logarithmic. Interpolate each endpoint's
logarithmic loss bilinearly and exponentiate; combine the Fresnel endpoints
affinely. Inside the explicit cell, nonnegative weights keep the result within
positive corner bounds. Return both `loss` and `energy=1-loss`. Never reconstruct
loss from energy: energy can round to one while a useful positive loss survives.
Only stored knot data are constant; queried coordinates retain their derivatives.
Stored tuples do not alias the caller's tensors. Unsupported shapes, dtypes,
invalid cells and out-of-cell queries fail explicitly.

The fixed first screen compares six centers, three F0 values, and two logarithmic
cell half-widths (.01/.005) with the repaired reference's other integration
coordinate. Limits are 0.1% relative loss error and refinement, 5e-4 absolute energy
error, and .01+.02*abs(reference) for each first derivative. Test records expose
all measurements before assertions. These are finite local screens, not public
operation tolerances or a global accuracy proof.

Still open: the full angular endpoints, practical grid sizes, a consistent
cosine-weighted mean, cell-boundary derivatives, full-range accuracy and furnace
tests, artifact construction/loading, dtype/device/compile/export support, and
measured cost. Tiny cells chosen around a query do not establish that an economical
global grid exists. The physical roughness domain remains (0,1]; a local cell's
explicit coverage does not change it.

Tester pages: `shading/loss-cell` and `shading/loss-cell-demo`. The finite demo
compares 17 samples in one cell and magnifies losses by one trillion and roughness
derivatives by one billion. It does not animate a new material or measure FPS.
