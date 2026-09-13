# Direct remaining-light budget reference

This is offline scalar CPU float64 tooling, not a public material, lookup table,
or a hidden per-pixel fallback. Use `tools.material_loss_reference.directional`
for directional loss and `average` for its cosine-weighted mean. Both return
the **remaining** budget directly. Do not recover it by subtracting a rounded
near-one reflected-energy value.

## Why the previous estimates disagreed

The regular-view mean did not resolve the tiny remaining budget. Directly
integrating 1-E with redistributed views improved refinement, but still inherits
errors in each near-one E. The finite visible-normal sampler also misses rare
rejected reflections: at normal view and roughness .001, its 128-node tail-dense
grid reports zero rejected samples, while the exact rejected probability is
alpha²/(1+alpha²), about one in a trillion. A small absolute error is unacceptable
when it is comparable to the entire remainder.

The old implementations and failing executions are retained. A separate
roughness-one analytic example also shows the old energy gradient is unresolved
at view cosine one millionth. It is not used as derivative truth at those
extreme points.

## Loss identity and stable evaluation

[Heitz (2018), equation 3](https://jcgt.org/published/0007/04/01/paper.pdf)
gives the normalized visible-normal density. For the existing height-correlated
Smith model, let W=G2/G1 on valid reflected directions. Under that distribution:

```text
1-E = P(reflection below surface)
      + integral_valid [(1-F) + F*(1-W)] p_visible(h) dh
```

The loss split and following antiderivative are derived for this reference.
They do not change the BRDF or add multiple-scattering physics.

Set c=N.V, alpha=r², b=sqrt(1-c²)cos(phi), t=tan(theta_h).
The projected normal measure is 2*alpha²*t/(alpha²+t²)² dt dphi/(2*pi).
Visible-normal density adds the factor 2*(c+b*t)/(c+Rv), where
Rv=sqrt(alpha²+(1-alpha²)*c²). A valid reflection requires
0<t<tmax, with tmax=(b+sqrt(b²+c²))/c; use the cancellation-resistant equivalent
c/(sqrt(b²+c²)-b) when b is negative.

For valid samples, mu=N.L and Rl=sqrt(alpha²+(1-alpha²)*mu²):

```text
1-W = c*alpha²*(1-mu²) / [(Rl+mu)*(mu*Rv+c*Rl)]
1-F = (1-F0)*vh*(5-10*vh+10*vh²-5*vh³+vh⁴)
```

These avoid subtracting almost-equal roots or Fresnel weights. Integrate in either
t=alpha*sinh(u) or t=alpha*expm1(u), with explicit moving bounds and Jacobians.
The full graph is retained, including dependence of the sampling coordinates on
view and roughness.

For rejected reflections use inverse slope s=alpha/t, from
lo=max(-alpha*b/c,0) to hi=alpha/tmax. The azimuth-conditioned rejected mass is

```text
2/(c+Rv) * integral_lo^hi 2*(c*s+alpha*b)/(1+s²)² ds
```

Its antiderivative combines -c/(1+s²) and
alpha*b*(atan(s)+s/(1+s²)). The implementation evaluates factored differences,
including atan2(hi-lo,1+hi*lo), instead of subtracting two near-identical
antiderivative values. Direct numerical integration of the density and the exact
normal-view probability independently check this closed integral.

For the average, redistribute view nodes with
c=alpha*sinh(x*asinh(1/alpha)), x in [0,1], and integrate 2*c*loss(c)*dc.

## What the checks establish

The initial six-pair gate keeps its 0.1% relative residual tolerance and original
gradient and furnace limits. Added controls cover analytic roughness-one values
and means, independently integrated rejected density, retained missed-tail and
old-gradient failures, complement/gradient agreement in the resolved old overlap,
gradcheck, and relative finite differences for tiny roughness derivatives.

The sinh/log checks are two numerical parameterizations of the **same** physical
identity. They are not independent physical models or a proof over all positive
roughness. The old independent sampler is explicitly unresolved for the tiny tail;
its disagreement is retained as evidence, not erased or relabelled as agreement.

Directional calls use 2*n*n valid-reflection evaluations plus O(n) analytic
tail arithmetic, n=16..512. Means use 16..64 views with the same 2,097,152
light-evaluation cap. Unknown modes, invalid scalar inputs, underflowing roughness
and nonrepresentable results fail explicitly. No clamp, mirror substitution,
extra runtime dependency, work-cap increase, fast-table extension or compiled/
device-performance claim is introduced.

See testing pages `shading/loss-contract`, `shading/boundary-contract`, and the
finite `shading/residual-demo` for current results and preserved history.
