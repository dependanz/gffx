# Bounded cubic remaining-light curves

This is an experimental scalar CPU float64 resource in repository tooling, not
an installed/public material API. The original `LossResource`, its measurements
and its failed directional checks remain available as negative controls.

## What changed

The original grid's two remaining value errors came mainly from interpolation
across roughness. Its angular value error was small, but a piecewise-linear
log-space slope still failed one derivative check. Increasing the grid further
would exceed the adopted node/work limits.

The replacement interpolates **log loss versus log roughness** with a bounded
cubic. At each queried roughness it then interpolates **loss versus log cosine**
with another bounded cubic. The second choice is deliberate: a polynomial in
log cosine has an analytic cosine-weighted integral. Exponentiating an angular
cubic log loss would not have that same simple integral.

Slopes start as three-point quadratic estimates. Their sign and magnitude are
limited so each cubic's Bezier control polygon is monotone and stays between its
endpoint values. A common scale for paired Fresnel slopes keeps the two fields'
controls ordered. Convex Bezier evaluation therefore cannot invent a negative
loss or reverse Fresnel endpoint ordering. This constrains the representation;
it does not clamp a measured energy or renormalize a reflection model.

## One curve, one average

For an angular segment [a,b], let h=log(b/a), and let P0..P3 be its controls.
The mean contribution is

`2*b^2*h * sum_k Pk * integral_0^1 exp(-2*h*(1-u))*B_k^3(u) du`.

Fixed Bernstein weights are computed analytically using a positive 64-term
series for 2h<=8 and an exponential-moment recurrence above8. Independent Gauss
integration tests cover six widths from1e-9 to100. Runtime queries do not perform
quadrature; the separate test integrator does. Queried roughness/Fresnel/view
parameters remain connected to autodiff; only immutable construction constants
are precomputed.

The first horizon segment remains linear. Interior positive angular knots have
continuous first derivatives; the first linear/log junction remains C0 and uses
the right derivative. Limiters may be nonsmooth at active-set transitions; no
C2 or unrestricted higher-order-autodiff guarantee is made. Input and numerical
representability errors remain explicit.

## Evidence and scope

The initial cubic development screen reused the old97x385/129x513 knot arrays.
Its first fresh physical comparisons passed, but two nearly edge-on generation
values missed the unchanged0.01-percent convergence guard. Axis doubling showed
azimuth, not radial resolution, was responsible. Those cases are now development
data; the saved failure remains evidence.

The repaired generation policy starts at192/32 (coarse) and280/32 (fine)
radial/azimuth samples. Only when cosine<roughness squared/16, it halves radial
and doubles azimuth. The conservative sample product and node caps do not change.
Normal views retain their radial allocation; a globally halved radial rule had
already failed there. This threshold changes an integration rule, not the physical
domain, loss formula or public roughness range. Tests check both sides of it.

Default tests regenerate both tables. An explicit
`GFFX_LOSS_RESOURCE_FIXTURE_DIR` may reuse saved JSON only when both versioned
SHA-256 checks pass; reuse is not new generation. Original data and the original
power interpolation remain available for historical comparisons.

Development checks use the same48 directional triples,12 means and108 furnace
cases. Fresh validation was declared before execution: four new roughnesses,
five views and three Fresnel values give60 directional triples,12 means and144
furnace cases, plus56 endpoint-generation values. The exact results belong to
the saved executions, not this recipe. After its generation failure, a second
cohort was declared before execution: roughness .00473,.0813,.419,.963;
cosines .003,.337,.907,.995,.0015*r*r; Fresnel .29,.61,1. That adds60 directional,
12 mean,144 furnace and56 generation comparisons. Subsequent reruns of either
cohort are regression checks, not new unseen validation. Failures are never
excused by internal gradcheck or attractive visuals.

Roughness [.001,1] is still the experiment's stored coverage, not a public
physical floor. No runtime timing, native/GPU, compilation/export, complete
physical-domain or public-material acceptance follows automatically. No new
external dependency was introduced.

## Reporting the regression

The initial combined regression passed392 tests and skipped the existing Windows
symlink-permission check, but exceeded the dashboard's128 artifact paths per
execution. Test outcomes and complete visual-evidence capture are different
claims: that execution retains its reporting diagnostic. Report subsequent
regressions in disjoint candidate, loss-reference, earlier-reference, old-table,
input/contract and teaching/tester groups. Keep the cap and saved history intact;
do not silently discard artifacts or relabel a repeated cohort as unseen.
