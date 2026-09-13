# Coloured world-environment preflight

This development-only fixture extends uniform illumination to
`L_k(direction) = 1 + b_k dot direction`, with separate RGB world-direction
coefficients. Each coefficient vector has length below one, so the field is
strictly positive without clipping. It is not an HDR file loader or a new runtime
dependency. Public computation uses existing complete materials and `light.integrate`.

Eight predetermined points on the immutable head fixture are tested for both
classic and principled materials. Sampling uses a view-aligned local frame, but
the light is evaluated in world coordinates. Both azimuth halves are explicit:
doubling one half, valid for the old constant-light fixture, gives wrong colours here.

The independent principled reference integrates glossy reflection and separately
checks two scalar-curve moments: `M = 2 integral loss(mu)*mu dmu` and
`J = 2 integral loss(mu)*mu^2 dmu`. Its affine matte result is
`albedo * loss(N.V) * (1 + dot(b,N)*J/M)`. The public candidate does not use this
shortcut; it evaluates the complete BRDF at every sampled direction.

Moment quadrature uses physical cosine coordinates on each curve segment, orders
8/16/32 and two successful refinements. The initial log-coordinate implementation
failed sparse polynomial controls; that failed test evidence is retained. The
repair changes coordinates, not the integrand, thresholds or work budget.

Full-azimuth reference orders 8/16/32/64/128 retain two independent partitions and
two successful transitions, at `1e-11 + 1e-9*abs(reference)`. Each public estimate
must pass `1e-10 + 1e-8*abs(reference)`. Samples, histories, geometry, source/wheel
hashes and actual-worker time/memory survive strict saved replay.

The sixteen-point preflight has a separate 600-second launch budget, reserving a
180-second watchdog before each serial worker. Each worker retains the 120-second
cooperative deadline and 1 GiB sampled whole-process limit. No full-image sweep is
authorized by this preflight, and no unresolved estimate is accepted as a result.
See the devbrain shading contract section 8.26 and numbers ledger for exact status.

The testing demo shows an incoming-light diagram and measured point responses.
Diagram exposure/sRGB are display-only; it is not a shaded head image. Complete
environment images, varying albedo/roughness, image gradients, device/export
qualification and the gffit end-to-end milestone remain separate gates.
