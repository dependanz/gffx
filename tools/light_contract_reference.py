"""Executable M1 light contract recipes; not a public gffx module.

Torch-only diagnostic, intentionally separate from the installed package. Eager
validation may synchronize a device. No compile/export/native parity claim.
"""
import math

import torch


def _inputs(*values):
    for value in values:
        if not isinstance(value, torch.Tensor):
            raise TypeError('light inputs must be tensors')
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError('light dtype must be float32 or float64')
        if value.dtype != values[0].dtype:
            raise TypeError('light inputs must have the same dtype')
        if value.device != values[0].device:
            raise ValueError('light inputs must have the same device')
        if not bool(torch.isfinite(value).all()):
            raise ValueError('light inputs must be finite')


def _vectors(value, name):
    if value.ndim not in (2, 3) or value.shape[-1] != 3:
        raise ValueError(f'{name} shape must be [P,3] or [B,P,3]')
    if value.ndim == 3 and value.shape[0] == 0:
        raise ValueError(f'{name} batch must be nonempty')
    return value.unsqueeze(0) if value.ndim == 2 else value


def _shared(value, batch, tail, name):
    if value.shape == tail:
        value = value.unsqueeze(0)
    elif value.ndim != len(tail)+1 or value.shape[1:] != tail:
        raise ValueError(f'{name} shape must be {tail} or [B,{tail}]')
    if value.shape[0] not in (1, batch):
        raise ValueError(f'{name} batch must be one or match the query batch')
    return value.expand(batch, *tail)


def _unit(value):
    # Scale before length to keep large or tiny nonzero normals usable.
    scale = value.abs().amax(dim=-1, keepdim=True)
    if bool((scale == 0).any()):
        raise ValueError('zero direction has no unit direction')
    scaled = value/scale
    return scaled/torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)


def _finite_result(value):
    if not bool(torch.isfinite(value).all()):
        raise ValueError('light result is not representable in the input dtype')
    return value


def basis(direction):
    """Real band-2 basis in the contract's y,z,x first-order ordering."""
    _inputs(direction)
    if direction.ndim < 1 or direction.shape[-1] != 3:
        raise ValueError('direction shape must end in three')
    x, y, z = _unit(direction).unbind(-1)
    c0 = math.sqrt(1/(4*math.pi))
    c1 = math.sqrt(3/(4*math.pi))
    c2 = math.sqrt(15/(4*math.pi))
    c3 = math.sqrt(5/(16*math.pi))
    c4 = math.sqrt(15/(16*math.pi))
    return torch.stack((torch.ones_like(x)*c0, c1*y, c1*z, c1*x,
                        c2*x*y, c2*y*z, c3*(3*z*z-1), c2*x*z, c4*(x*x-y*y)), -1)


def spherical_harmonics(normals, coefficients):
    """Incident-radiance SH -> cosine-convolved irradiance, not shaded RGB."""
    _inputs(normals, coefficients)
    query = _vectors(normals, 'normals')
    coeff = _shared(coefficients, query.shape[0], (9, 3), 'coefficients')
    factors = normals.new_tensor([math.pi]+[2*math.pi/3]*3+[math.pi/4]*5)
    result = (basis(query)*factors) @ coeff
    result = _finite_result(result)
    return result[0] if normals.ndim == 2 else result


def directional(directions, irradiance):
    """Delta light weights: normal-incidence irradiance before surface cosine."""
    _inputs(directions, irradiance)
    query = _vectors(directions, 'directions')
    weights = _shared(irradiance, query.shape[0], (query.shape[1], 3), 'irradiance')
    if bool((weights < 0).any()):
        raise ValueError('irradiance must be nonnegative')
    result = dict(direction=_unit(query), irradiance=weights)
    return {k: v[0] for k, v in result.items()} if directions.ndim == 2 else result


def point(points, positions, intensity):
    """Inverse-square point light; coincident point/source pairs are masked.

Irradiance is the delta-light weight I/r^2, before multiplying by N dot L.
It is not an ordinary finite incident radiance or a solid-angle PDF.
"""
    _inputs(points, positions, intensity)
    query = _vectors(points, 'points')
    lights = _vectors(positions, 'positions')
    batch, count = query.shape[0], lights.shape[1]
    lights = _shared(positions, batch, (count, 3), 'positions')
    weights = _shared(intensity, batch, (count, 3), 'intensity')
    if bool((weights < 0).any()):
        raise ValueError('intensity must be nonnegative')
    offset = _finite_result(lights[:, None, :, :]-query[:, :, None, :])
    coincident = (offset == 0).all(dim=-1)
    distance_squared = _finite_result(offset.square().sum(dim=-1))
    if bool(((distance_squared == 0) & ~coincident).any()):
        raise ValueError('nonzero squared distance is not representable in the input dtype')
    valid = ~coincident
    # Mask BEFORE division/sqrt so invalid pairs have finite zero derivatives.
    denominator = torch.where(valid, distance_squared, torch.ones_like(distance_squared))
    direction = offset/denominator.sqrt()[..., None]
    weight = _finite_result(weights[:, None, :, :]/denominator[..., None])
    result = dict(direction=torch.where(valid[..., None], direction, 0.),
                  irradiance=torch.where(valid[..., None], weight, 0.),
                  distance_squared=distance_squared, valid=valid)
    return {k: v[0] for k, v in result.items()} if points.ndim == 2 else result
