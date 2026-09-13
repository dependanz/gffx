# One curve, one average

`tools/material_loss_resource.py` is an experimental offline CPU float64 resource,
not the public material API. It extends the local log-loss experiment across the
whole view-cosine interval [0,1] and the explicitly stored roughness interval.
The experimental generator currently covers [.001,1]; this is **not** an adopted
public roughness floor. No extra dependency is added to `gffx.dashboard` or the
base package. These development tools use the existing PyTorch environment.

## Why the average belongs to the curve

Let L(c) be remaining light at view cosine c. The matte coupling uses both L(c)
and its cosine-weighted average, `2 integral_0^1 c L(c) dc`. Storing that average
in a second independent table risks disagreement. Here we integrate the exact
piecewise interpolant after roughness interpolation, rather than interpolating
precomputed averages.

Positive angular segments interpolate log L linearly in log c; hence each
segment is a power of c. For endpoints a,b with losses La,Lb its mean contribution
is `2 log(b/a) logarithmic_mean(a^2 La, b^2 Lb)`. The logarithmic mean's removable
singularity is handled by a symmetric series that preserves derivatives.
The first [0,b] segment is linear, with contribution `b^2 (Lh + 2 Lb)/3`.
There is no hidden quadrature in `evaluate`; `integrate_for_test` is an explicit
independent numerical comparator. Runtime cost is linear in angular knot count;
no fast-runtime or frame-rate acceptance has been measured.

The resource stores immutable tuples copied from caller tensors. Interior knots
select the right cell; the final endpoint selects the left. Slopes need not agree
between cells. No second-derivative or endpoint view-derivative reference claim
is made. Roughness derivatives remain connected through interpolation.

## Endpoint references

The normalized visible-normal density follows
[Heitz 2018, equation 3](https://jcgt.org/published/0007/04/01/paper.pdf).
The following limiting substitutions are derived for our height-correlated Smith
model, not formulas quoted from that paper.

At normal view, the valid slope radius ends at one. With alpha = roughness squared,
the rejected probability is `alpha^2/(1+alpha^2)` and the valid integral is one
dimensional. Roughness one and Fresnel one give loss exactly log(2).

At the horizon, unit-Fresnel loss tends to zero. For the zero-Fresnel endpoint,
substitute `y=exp(-asinh(t/alpha))` in the limiting visible-slope density, then
`y=alpha*sinh(u)` to resolve the sharp Fresnel transition. No tail is cut off.
Endpoint generation shares geometry between Fresnel values. It computes F0=1
loss directly; at F0=0 it integrates loss when loss is small, otherwise integrates
the small reflected energy and subtracts that from one. This conditioning choice
is explicit, not a clamp or density renormalization.

## Acceptance is a separate question

The predeclared coarse/fine grids are 33x129/65x257 at 64/128 quadrature nodes,
with a 1.2-billion shared-geometry-sample cap. Tests retain 48 directional triples,
12 mean comparisons and 108 independently integrated furnace cases before
asserting the unchanged accuracy limits. An exact integral of an inaccurate
curve is still inaccurate. Any failed screen blocks promotion; a pleasing image,
algebra pass or white-furnace pass cannot overrule it.

The original fast table, scalar references and their preserved negative controls
remain unchanged. Inspect `shading/loss-resource` in the testing dashboard;
finite demonstration measurements appear separately under
`shading/loss-resource-demo`.

The first grid failed all three physical screens while nine endpoint/algebra
cases passed. A denser 97x385/129x513 retry uses 192/32 and 280/32 radial/azimuth
orders. Equal-order94 and reallocated140/64 guards failed before large generation;
separate axis doubling identified radial error. The 280/32 versus512/64 guard
passes the unchanged relative 0.0001 limit at 48 endpoint values. Dense generation
still respects the 1.2-billion shared-sample cap. The reused development points
cannot substitute for later fresh off-grid validation.

The dense trial has 13 passing cases and one failing directional-screen case.
Its twelve mean comparisons and 108 furnace cases pass, but two directional
triples exceed the 0.1-percent loss/refinement limits; one also fails the view
derivative limit. At r=.173,c=.025,F0=1, relative loss error is about 0.12696
percent; at r=.733,c=.97,F0=1 it is about 0.12659 percent. These are explicit
non-admission results, not tolerances to relax. The next repair must improve the
representation/grid within the existing caps and then use fresh validation points.
