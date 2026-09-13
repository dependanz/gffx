"""Cancellation-resistant scalar GGX loss reference, not a public material.

Normalized VNDF identity: Heitz 2018 equation 3,
https://jcgt.org/published/0007/04/01/paper.pdf .
The slope-tail antiderivative and loss split are derived here for the existing
height-correlated Smith G2. Coordinate agreement is not an independent model.
"""
import math
import torch
from tools.material_budget_reference import _validate, _resolution
from tools.shading_energy_probe import _gauss_unit


def _geometry(c, r, resolution):
    a = r.square()
    nodes, weights = _gauss_unit(resolution)
    phi = torch.cat((nodes, nodes+1))*(math.pi/2)
    pw = torch.cat((weights, weights))/2
    b = torch.sqrt(1-c.square())*phi.cos()
    root = torch.hypot(b, c)
    pos = b >= 0
    bp = torch.where(pos, b, torch.zeros_like(b))
    bn = torch.where(pos, torch.zeros_like(b), b)
    limit = torch.where(pos, (bp+root)/c, c/(root-bn))
    rv = torch.sqrt(a.square()+(1-a.square())*c.square())
    return a, nodes, weights, pw, b, limit, rv


def _rejected(c, a, b, limit, rv):
    # Inverse slope s=a/t. Visible-normal support ends at s=max(-a*b/c,0).
    # Use differences of antiderivatives without subtracting near-equal values.
    hi = a/limit
    lo = torch.where(b < 0, -a*b/c, torch.zeros_like(b))
    delta = hi-lo
    denominator = (1+lo.square())*(1+hi.square())
    first = delta*(hi+lo)/denominator
    second = torch.atan2(delta, 1+hi*lo)+delta*(1-hi*lo)/denominator
    return 2*(c*first+b*a*second)/(c+rv)


def rejected_probability(cos_view, roughness, *, resolution=128):
    # Normal view is admitted for its exact scalar control, not endpoint AD.
    c = cos_view
    _validate(f0=c, roughness=roughness)
    if not bool(c > 0):
        raise ValueError('cos_view must lie in (0,1]')
    _resolution(resolution)
    a, _, _, pw, b, limit, rv = _geometry(c, roughness, resolution)
    result = (_rejected(c, a, b, limit, rv)*pw).sum()
    if not bool(torch.isfinite(result)):
        raise ValueError("rejected probability is not representable")
    return result


def directional(cos_view, roughness, f0, *, resolution=128, method="sinh"):
    """Return 1-E directly with 2*n*n valid-reflection evaluations."""
    _validate(cos_view=cos_view, roughness=roughness, f0=f0)
    _resolution(resolution)
    if method not in ("sinh", "log"):
        raise ValueError("unknown loss coordinate method")
    c = cos_view
    a, nodes, weights, pw, b, limit, rv = _geometry(c, roughness, resolution)
    maximum = limit/a
    if method == "sinh":
        bound = torch.asinh(maximum)
        u = nodes[:,None]*bound
        s = torch.sinh(u)
        jacobian = torch.cosh(u)*bound
    else:
        bound = torch.log1p(maximum)
        u = nodes[:,None]*bound
        s = torch.expm1(u)
        jacobian = torch.exp(u)*bound
    slope = a*s
    nh2 = 1/(1+slope.square())
    vh = (c+b*slope)*nh2.sqrt()
    nl = (c*(1-slope.square())+2*b*slope)*nh2
    if not bool((torch.isfinite(nl)&(nl > 0)).all()):
        raise ValueError("reflected cosine is not representable")
    rl = torch.sqrt(a.square()+(1-a.square())*nl.square())
    # 1-G2/G1, with rl-nl rationalized, never subtracting near-one weights.
    masking = c*a.square()*(1-nl.square())/((rl+nl)*(nl*rv+c*rl))
    absorption = (1-f0)*vh*(5+vh*(-10+vh*(10+vh*(-5+vh))))
    fresnel = f0+(1-f0)*(1-vh).pow(5)
    measure = 2*s/(1+s.square()).square()*jacobian
    density_ratio = 2*(c+b*slope)/(c+rv)
    valid = ((absorption+fresnel*masking)*density_ratio*measure*
             weights[:,None]*pw[None,:]).sum()
    rejected = (_rejected(c, a, b, limit, rv)*pw).sum()
    result = valid+rejected
    if not bool(torch.isfinite(result)):
        raise ValueError("loss is not representable")
    return result


def average(roughness, f0, *, resolution=128, view_resolution=64, method="sinh"):
    """Return cosine-weighted mean loss, retaining its small magnitude and AD."""
    _validate(roughness=roughness, f0=f0)
    _resolution(resolution)
    _resolution(view_resolution, maximum=64)
    if 2*resolution**2*view_resolution > 2**21:
        raise ValueError("requested quadrature work exceeds 2097152 light samples")
    if method not in ("sinh", "log"):
        raise ValueError("unknown loss coordinate method")
    nodes, weights = _gauss_unit(view_resolution)
    a = roughness.square()
    bound = torch.asinh(1/a)
    u = nodes*bound
    cosines = a*torch.sinh(u)
    jacobian = a*torch.cosh(u)*bound
    values = torch.stack([directional(c, roughness, f0, resolution=resolution,
                                     method=method) for c in cosines])
    return 2*(values*cosines*jacobian*weights).sum()
