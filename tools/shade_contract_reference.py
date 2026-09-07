"""Executable Phase-0 elementary shading contract; not a public gffx API.

Torch-only recipes with eager validation, no hidden epsilon and no device transfer.
Compile/export, public adapters and full material acceptance are later work.
"""
import math

import torch


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
    for name, value in fields.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{name} must be finite')


def _range(value, name, lower=0., upper=1., *, open_lower=False):
    invalid = value <= lower if open_lower else value < lower
    if upper is not None:
        invalid = invalid | (value > upper)
    if bool(invalid.any()):
        raise ValueError(f'{name} outside its allowed domain')


def _query(value, name):
    if value.ndim not in (1, 2) or (value.ndim == 2 and value.shape[0] == 0):
        raise ValueError(f'{name} shape must be [P] or nonempty-batch [B,P]')
    _range(value, name)
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


def _result(value, first):
    if not bool(torch.isfinite(value).all()):
        raise ValueError('shading result is not representable in the input dtype')
    return value[0] if first.ndim == 1 else value


def _roughness(value, query):
    _range(value, 'roughness', open_lower=True)
    fourth = value.square().square()
    if bool((fourth == 0).any()):
        raise ValueError('positive roughness fourth power is not representable')
    return _scalar(fourth, query, 'roughness')


def lambert(albedo):
    """Diffuse BRDF, not outgoing radiance: no light or cosine multiplied here."""
    _inputs(albedo=albedo)
    if albedo.ndim not in (2, 3) or albedo.shape[-1] != 3 or (albedo.ndim == 3 and albedo.shape[0] == 0):
        raise ValueError('albedo shape must be [P,3] or nonempty-batch [B,P,3]')
    _range(albedo, 'albedo')
    return albedo/math.pi


def blinn_phong(n_dot_h, exponent):
    """Approximate classic highlight scaling; explicitly NOT energy-conserving."""
    _inputs(n_dot_h=n_dot_h, exponent=exponent)
    query = _query(n_dot_h, 'n_dot_h')
    _range(exponent, 'exponent', upper=None)
    power = _scalar(exponent, query, 'exponent')
    positive = query > 0
    safe = torch.where(positive, query, torch.ones_like(query))
    interior = (power/(8*math.pi)+1/math.pi)*safe.pow(power)
    # Define the c=0,n=0 corner without 0**0 or log(0) in the backward graph.
    boundary = torch.where(power == 0, torch.full_like(query, 1/math.pi), torch.zeros_like(query))
    return _result(torch.where(positive, interior, boundary), n_dot_h)


def ggx_distribution(n_dot_h, roughness):
    """Projected-area-normalized GGX NDF, not the full specular BRDF."""
    _inputs(n_dot_h=n_dot_h, roughness=roughness)
    c = _query(n_dot_h, 'n_dot_h')
    a2 = _roughness(roughness, c)
    denominator = ((1-c)*(1+c)+a2*c.square()).square()
    if bool((denominator == 0).any()):
        raise ValueError('GGX denominator is not representable')
    return _result((a2/denominator)/math.pi, n_dot_h)


def smith_visibility(n_dot_v, n_dot_l, roughness):
    """Height-correlated GGX visibility, already divided by 4*nV*nL."""
    _inputs(n_dot_v=n_dot_v, n_dot_l=n_dot_l, roughness=roughness)
    v = _query(n_dot_v, 'n_dot_v')
    _range(n_dot_l, 'n_dot_l')
    l = _scalar(n_dot_l, v, 'n_dot_l')
    a2 = _roughness(roughness, v)
    valid = (v > 0) & (l > 0)
    denominator = l*torch.sqrt(v.square()*(1-a2)+a2)+v*torch.sqrt(l.square()*(1-a2)+a2)
    if bool(((denominator == 0) & valid).any()):
        raise ValueError('Smith denominator is not representable')
    safe = torch.where(valid, denominator, torch.ones_like(denominator))
    return _result(torch.where(valid, .5/safe, torch.zeros_like(safe)), n_dot_v)


def fresnel_schlick(v_dot_h, f0):
    """RGB reflectance approximation, with unclamped polynomial derivatives."""
    _inputs(v_dot_h=v_dot_h, f0=f0)
    c = _query(v_dot_h, 'v_dot_h')
    _range(f0, 'f0')
    b, p = c.shape
    if f0.shape == (3,):
        value = f0.expand(b, p, 3)
    else:
        value = f0[None] if f0.shape == (p, 3) else f0
        if value.ndim != 3 or value.shape[1:] != (p, 3) or value.shape[0] not in (1, b):
            raise ValueError('f0 shape/batch must be [3], [P,3], [1,P,3] or [B,P,3]')
        value = value.expand(b, p, 3)
    return _result(value+(1-value)*(1-c[..., None]).pow(5), v_dot_h)
