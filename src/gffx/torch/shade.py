"""Elementary differentiable reflection factors, using only PyTorch compositions.

These are eager, validated float32/float64 operations, not complete materials.
The first tensor owns the batch/point layout. Parameters may share a scalar or
batch but never silently broadcast the point axis. Validation can synchronize a
device; compiled/export and device qualification are separate from eager support.
"""
import math

import torch
from ._reference_channels import multiply, expand
from ._reference_power import power_unit

__all__ = ['lambert', 'blinn_phong', 'ggx_distribution', 'smith_visibility', 'fresnel_schlick']


def _inputs(**fields):
    first = next(iter(fields.values()))
    for name, value in fields.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'{name} must be a tensor')
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f'{name} dtype must be float32 or float64')
        if value.dtype != first.dtype:
            raise TypeError(f'{name} must have the same dtype')
        if value.device != first.device:
            raise ValueError(f'{name} must have the same device')
    if first.device.type == 'cuda' and len(fields)>1:
        # All metadata checks precede value checks. One small flag readback
        # preserves their original first-error order without per-field stalls.
        flags=torch.stack([torch.isfinite(value).all() for value in fields.values()]).tolist()
        for name,valid in zip(fields,flags):
            if not valid:raise ValueError(f'{name} must be finite')
    else:
        for name, value in fields.items():
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'{name} must be finite')


def _domain(value, name, *, positive=False, upper=1.):
    bad = value <= 0 if positive else value < 0
    if upper is not None:
        bad = bad | (value > upper)
    if bool(bad.any()):
        raise ValueError(f'{name} outside its allowed domain')


def _query(value, name):
    if value.ndim not in (1, 2) or (value.ndim == 2 and value.shape[0] == 0):
        raise ValueError(f'{name} shape must be [P] or nonempty-batch [B,P]')
    _domain(value, name)
    return value[None] if value.ndim == 1 else value


def _scalar(value, query, name):
    b, p = query.shape
    if value.ndim == 0:
        return value.expand(b, p)
    if value.shape == (p,):
        value = value[None]
    if value.ndim != 2 or value.shape[1] != p or value.shape[0] not in (1, b):
        raise ValueError(f'{name} shape/batch must be scalar, [P], [1,P] or [B,P]')
    return value.expand(b, p)


def _roughness(value, query):
    _domain(value, 'roughness', positive=True)
    fourth = value.square().square()
    # Validate shared parameters before expansion, including empty point calls.
    if bool((fourth == 0).any()):
        raise ValueError('positive roughness fourth power is not representable')
    return _scalar(fourth, query, 'roughness')


def _result(value, first):
    if not bool(torch.isfinite(value).all()):
        raise ValueError('shading result is not representable in the input dtype')
    return value[0] if first.ndim == 1 else value


def lambert(albedo):
    """Return RGB albedo/pi for [P,3] or [B,P,3] reflectance in [0,1].

    This is a BRDF, not illuminated colour: apply light and cosine separately.
    """
    _inputs(albedo=albedo)
    if albedo.ndim not in (2, 3) or albedo.shape[-1] != 3 or (albedo.ndim == 3 and albedo.shape[0] == 0):
        raise ValueError('albedo shape must be [P,3] or nonempty-batch [B,P,3]')
    _domain(albedo, 'albedo')
    # Keep the denominator a same-dtype device operand. CUDA scalar lowering
    # otherwise rounds the reciprocal differently from CPU for float32 values
    # and their albedo VJPs (the matched-operand regression preserves the case).
    return albedo / albedo.new_tensor(math.pi)


def blinn_phong(n_dot_h, exponent):
    """Half-vector highlight (n+8)/(8*pi)*c**n; not energy-conserving.

    At c=0 return zero (1/pi for n=0), with zero first derivatives to both
    inputs by convention. Exponent is an explicit nonnegative tensor.
    """
    _inputs(n_dot_h=n_dot_h, exponent=exponent)
    q = _query(n_dot_h, 'n_dot_h')
    _domain(exponent, 'exponent', upper=None)
    n = _scalar(exponent, q, 'exponent')
    active = q > 0
    safe = torch.where(active, q, torch.ones_like(q))
    interior = (n / n.new_tensor(8*math.pi) + 1/math.pi) * power_unit(safe, n)
    boundary = torch.where(n == 0, torch.full_like(q, 1/math.pi), torch.zeros_like(q))
    return _result(torch.where(active, interior, boundary), n_dot_h)


def ggx_distribution(n_dot_h, roughness):
    """GGX normal distribution, with perceptual roughness in (0,1].

    No roughness floor or denominator epsilon; lost positive divisors raise.
    An NDF alone is not a complete BRDF.
    """
    _inputs(n_dot_h=n_dot_h, roughness=roughness)
    c = _query(n_dot_h, 'n_dot_h')
    a2 = _roughness(roughness, c)
    denominator = ((1-c)*(1+c) + a2*c.square()).square()
    if bool((denominator == 0).any()):
        raise ValueError('GGX denominator is not representable')
    return _result((a2 / denominator) / c.new_tensor(math.pi), n_dot_h)


def smith_visibility(n_dot_v, n_dot_l, roughness):
    """Height-correlated GGX visibility, including division by 4*nV*nL.

    Either cosine zero gives exact zero with zero first derivatives. Inactive
    inputs are masked before square roots and division, not only at the output.
    """
    _inputs(n_dot_v=n_dot_v, n_dot_l=n_dot_l, roughness=roughness)
    v = _query(n_dot_v, 'n_dot_v')
    _domain(n_dot_l, 'n_dot_l')
    l = _scalar(n_dot_l, v, 'n_dot_l')
    a2 = _roughness(roughness, v)
    active = (v > 0) & (l > 0)
    v = torch.where(active, v, torch.ones_like(v))
    l = torch.where(active, l, torch.ones_like(l))
    a2 = torch.where(active, a2, torch.ones_like(a2))
    denominator = l*torch.sqrt(v.square()*(1-a2)+a2) + v*torch.sqrt(l.square()*(1-a2)+a2)
    if bool((denominator == 0).any()):
        raise ValueError('Smith denominator is not representable')
    return _result(torch.where(active, .5/denominator, torch.zeros_like(denominator)), n_dot_v)


def fresnel_schlick(v_dot_h, f0):
    """RGB f0 + (1-f0)*(1-c)**5, without clamping endpoint derivatives.

    f0 has explicit RGB channels: [3], [P,3], [1,P,3] or [B,P,3].
    """
    _inputs(v_dot_h=v_dot_h, f0=f0)
    c = _query(v_dot_h, 'v_dot_h')
    _domain(f0, 'f0')
    b, p = c.shape
    if f0.shape == (3,):
        value = expand(f0, (b, p, 3))
    else:
        value = f0[None] if f0.shape == (p, 3) else f0
        if value.ndim != 3 or value.shape[1:] != (p, 3) or value.shape[0] not in (1, b):
            raise ValueError('f0 shape/batch must be [3], [P,3], [1,P,3] or [B,P,3]')
        value = expand(value, (b, p, 3))
    distance = 1-c[..., None]
    square = distance*distance
    fifth = (square*square)*distance
    return _result(value + multiply(1-value, fifth), v_dot_h)
