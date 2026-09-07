# Sharp-material reference experiment

`tools.material_budget_sharp` is explicit offline CPU float64 quadrature, not an
installed public material or a fast-table fallback. It introduces no dependency
beyond the existing optional PyTorch development environment. The old experimental
table retains its roughness [0.2,1] range; no physical roughness floor is adopted.

## Change of variables

For view cosine c, perceptual roughness r, alpha=r², half-vector slope t and
azimuth phi, set b=sqrt(1-c²)cos(phi). Reflection above the surface requires
c(1-t²)+2bt>0. Its positive boundary is (b+sqrt(b²+c²))/c; for negative b,
use the cancellation-resistant equivalent c/(sqrt(b²+c²)-b).

The projected GGX measure is alpha²*t/[pi*(alpha²+t²)²] dt dphi.
Substituting t=alpha*tan(beta) gives sin(beta)cos(beta)/pi d_beta d_phi.
The remaining visibility/half-vector factor is
2*nL*(c+b*t)/(nL*sqrt(alpha²+(1-alpha²)c²)+c*sqrt(alpha²+(1-alpha²)nL²)).
Multiply by Schlick reflectance. Integrate beta from zero to atan(t_max/alpha),
and the symmetric half-azimuth split at pi/2. Both moving coordinates and the
upper bound retain their derivatives. No parameter cache, energy clamp, detached
boundary or implicit mirror substitution is used.

## Evidence and bounds

The directional call uses 2*n*n samples with n in [16,512]. The average is
2*integral E(c)c dc with 16..64 view nodes and at most 2097152 light samples.
Inputs are scalar CPU float64: 0<c<1, 0<r<=1, 0<=F0<=1. Arithmetic that cannot
be represented fails explicitly; this is not a promise to resolve all valid
mathematical inputs to the same accuracy. Keep heavy experiments serial and
one-thread, and never put these calls in a frame loop.

The finite sweep samples four roughnesses (.01,.05,.1,.199), five view cosines
(.025,.11,.43,.79,.97) and three F0 values (0,.04,1). Refinement limits are
5e-5 for values and .001+.002*abs(finer) for derivatives. Independent VNDF
values must also refine and agree within 5e-5. The latter's original uniform
radial rule failed three cases; explicit `radial_rule='tail-dense'` transforms
u=1-(1-x)² and weights by 2(1-x), with the same sample count. Uniform remains
the default for historical reproduction.

Analytic roughness-one controls, direct-quadrature overlap, gradcheck, affine F0,
repeatability, averaged gradients and error fixtures supplement the sweep.
A separate 108-case sharp furnace combines independent specular with numerically
integrated coupled matte: energy <=1.001 and unit-albedo error <.001. The residual
integral and its normalization use different view resolutions. This is finite
energy evidence, not a global theorem or compiled/device parity.

The four-sphere demonstration uses 512 deterministic VNDF environment samples per
pixel, gold-colored Fresnel endpoints and two smooth bright windows. It is a
single-scattering analytic-sphere preview, not the principled material, a mesh
renderer, a converged image or furnace acceptance. Display tone mapping is separate.
