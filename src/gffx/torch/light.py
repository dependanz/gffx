"""Eager differentiable SH, directional and point lighting compositions.

All inputs share float32/float64 dtype and device. No shadowing, albedo or
material evaluation occurs here. Light/point axes are explicit; only singleton
batch axes share. Custom callbacks use exact tagged packets, with no automatic physical claims.
"""
import math

import torch

from .shade import _inputs
from ._reference_channels import sum3, divide, sqrt_unit, multiply, expand, _sum_last
from ._registry import Registry, parameters as _parameters

__all__ = ['spherical_harmonics', 'directional', 'point', 'rectangle', 'integrate', 'custom', 'Registry']


def _packet(normals, samples):
    from collections.abc import Mapping
    if not isinstance(samples, Mapping):
        raise TypeError('samples must be a mapping')
    measure = samples.get('measure')
    if not isinstance(measure, str) or measure not in ('delta', 'solid_angle'):
        raise ValueError('samples measure must be delta or solid_angle')
    colour = 'irradiance' if measure == 'delta' else 'radiance'
    keys = {'measure', 'direction', colour, 'valid'}
    if measure == 'solid_angle': keys.add('solid_angle')
    if set(samples) != keys:
        raise ValueError('samples keys do not match the declared measure')
    fields = dict(normals=normals, direction=samples['direction'], **{colour: samples[colour]})
    if measure == 'solid_angle': fields['solid_angle'] = samples['solid_angle']
    _inputs(**fields)
    _vectors(normals, 'normals')
    direction, incoming, valid = samples['direction'], samples[colour], samples['valid']
    if direction.ndim != normals.ndim+1 or direction.shape[:-2] != normals.shape[:-1] or direction.shape[-1] != 3:
        raise ValueError('direction shape must match exact normal point/batch axes plus sample axis')
    if incoming.shape != direction.shape:
        raise ValueError('light RGB shape must match direction exactly')
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool:
        raise TypeError('valid must be a bool tensor')
    if valid.device != normals.device or valid.shape != direction.shape[:-1]:
        raise ValueError('valid shape/device must match samples')
    if measure == 'delta' and bool((incoming < 0).any()):
        raise ValueError('brdf and delta irradiance must be nonnegative')
    if normals.device.type=='cuda':
        # Elementwise masking retains every invalid-sample check, without
        # building two variable-sized index selections and synchronizing each.
        invalid=(~valid)[...,None]
        bad=((direction!=0)&invalid).any() | ((incoming!=0)&invalid).any()
        if bool(bad):raise ValueError('invalid sample placeholders must be zero')
    elif bool((direction[~valid] != 0).any()) or bool((incoming[~valid] != 0).any()):
        raise ValueError('invalid sample placeholders must be zero')
    omega = None
    if measure == 'solid_angle':
        omega = samples['solid_angle']
        if omega.shape != valid.shape or bool((omega < 0).any()):
            raise ValueError('solid_angle shape must match samples and values must be nonnegative')
        if bool((omega[~valid] != 0).any()):
            raise ValueError('invalid solid_angle placeholders must be zero')
    return direction, incoming, valid, omega


def integrate(normals, brdf, samples):
    """Sum tagged light samples with one positive cosine and one measure weight."""
    _inputs(normals=normals, brdf=brdf)
    direction, incoming, valid, omega = _packet(normals, samples)
    if brdf.shape != direction.shape or bool((brdf < 0).any()):
        raise ValueError('brdf must be nonnegative with exact packet RGB shape')
    from ._opaque_lighting import integrate as fixed_integrate
    fixed=fixed_integrate(normals,brdf,direction,incoming,valid,omega)
    if fixed is not None:return fixed
    n = _unit(normals, 'normals')
    fallback = direction.new_tensor([0., 0., 1.])
    d = _unit(torch.where(valid[..., None], direction, fallback), 'direction')
    cosine = sum3(expand(n[..., None, :], d.shape)*d)
    active = valid & (cosine > 0)
    cosine = torch.where(active, cosine, torch.zeros_like(cosine))
    values = multiply(torch.where(active[..., None], brdf, 0.) * torch.where(active[..., None], incoming, 0.), cosine[..., None])
    if omega is not None: values = multiply(values, torch.where(active, omega, 0.)[..., None])
    return _finite(_sum_last(values.transpose(-1, -2)))



def rectangle(points,*,center,u,v,radiance,samples=4):
    """Sample a one-sided rectangle using full-width oriented spans.

    Radiance is scene-linear RGB. This low-level packet has no visibility;
    the scene renderer owns shadow maps and visible emitter geometry.
    """
    from ._area import rectangle_packet
    return rectangle_packet(points,center=center,u=u,v=v,radiance=radiance,samples=samples)


def custom(function, points, *, parameters, measure):
    """Normalize and validate a trusted light callback's exact tagged packet."""
    from collections.abc import Mapping
    if not isinstance(measure, str) or measure not in ('delta', 'solid_angle'):
        raise ValueError('measure must be delta or solid_angle')
    if not callable(function):
        raise TypeError('function must be callable')
    _inputs(points=points)
    _vectors(points, 'points')
    _parameters(parameters, points)
    output = function(points, parameters=dict(parameters))
    if not isinstance(output, Mapping):
        raise TypeError('custom light output must be a mapping')
    if 'measure' in output:
        raise ValueError('callback keys must not include measure')
    result = dict(output, measure=measure)
    direction, incoming, valid, omega = _packet(points, result)
    safe = torch.where(valid[..., None], direction, direction.new_tensor([0., 0., 1.]))
    result['direction'] = torch.where(valid[..., None], _unit(safe, 'direction'), 0.)
    key = 'irradiance' if measure == 'delta' else 'radiance'
    result[key] = torch.where(valid[..., None], incoming, 0.)
    if omega is not None:
        result['solid_angle'] = torch.where(valid, omega, 0.)
    return result


def _vectors(value, name):
    if value.ndim not in (2, 3) or value.shape[-1] != 3:
        raise ValueError(f'{name} shape must be [P,3] or [B,P,3]')
    if value.ndim == 3 and value.shape[0] == 0:
        raise ValueError(f'{name} batch must be nonempty')
    return value[None] if value.ndim == 2 else value


def _shared(value, batch, tail, name):
    if value.shape == tail:
        value = value[None]
    elif value.ndim != len(tail)+1 or value.shape[1:] != tail:
        raise ValueError(f'{name} shape must have explicit trailing dimensions {tail}')
    if value.shape[0] not in (1, batch):
        raise ValueError(f'{name} batch must be one or match the first input')
    return value.expand(batch, *tail)


def _unit(value, name):
    from ._reference_channels import unit_forward
    fused=unit_forward(value)
    if fused is not None:return fused
    scale = value.abs().amax(dim=-1, keepdim=True)
    if bool((scale == 0).any()):
        raise ValueError(f'{name} zero direction has no unit direction')
    scaled = divide(value,scale)
    return divide(scaled,sqrt_unit(sum3(scaled.square())).unsqueeze(-1))


def _finite(value):
    if not bool(torch.isfinite(value).all()):
        raise ValueError('light arithmetic is not representable in the input dtype')
    return value


def _basis(unit):
    x, y, z = unit.unbind(-1)
    c0, c1 = math.sqrt(1/(4*math.pi)), math.sqrt(3/(4*math.pi))
    c2, c3, c4 = math.sqrt(15/(4*math.pi)), math.sqrt(5/(16*math.pi)), math.sqrt(15/(16*math.pi))
    return torch.stack((torch.ones_like(x)*c0, c1*y, c1*z, c1*x,
                        c2*x*y, c2*y*z, c3*(3*z*z-1), c2*x*z, c4*(x*x-y*y)), -1)


class _SHContraction(torch.autograd.Function):
    """Nine separately rounded products, with fixed first-adjoint reductions.

    Basis and RGB axes never use BLAS/FMA contraction. Coefficient adjoints
    reduce points by the existing pair tree, without a [B,P,9,3] product buffer.
    """
    @staticmethod
    def forward(ctx, basis, coefficients):
        ctx.save_for_backward(basis, coefficients)
        value = basis[..., 0, None] * coefficients[:, None, 0, :]
        for i in range(1, 9):
            value = value + basis[..., i, None] * coefficients[:, None, i, :]
        return value

    @staticmethod
    def backward(ctx, gradient):
        basis, coefficients = ctx.saved_tensors
        db = torch.stack([sum3(gradient * coefficients[:, None, i, :])
                          for i in range(9)], -1) if ctx.needs_input_grad[0] else None
        dc = torch.stack([_sum_last((gradient * basis[..., i, None]).transpose(-1, -2))
                          for i in range(9)], -2) if ctx.needs_input_grad[1] else None
        return db, dc


def spherical_harmonics(normals, coefficients):
    """Band-two incident-radiance SH to cosine-convolved RGB irradiance.

    Normalize nonzero [P,3]/[B,P,3] normals. Coefficients [9,3]/[1,9,3]/
    [B,9,3] use real y,z,x first-order basis ordering. Signed values are retained;
    this irradiance is for diffuse convolution, not full glossy environment light.
    """
    _inputs(normals=normals, coefficients=coefficients)
    q = _vectors(normals, 'normals')
    c = _shared(coefficients, q.shape[0], (9, 3), 'coefficients')
    factors = normals.new_tensor([math.pi]+[2*math.pi/3]*3+[math.pi/4]*5)
    c = expand(coefficients.reshape(-1, 9, 3), c.shape)
    result = _finite(_SHContraction.apply(_basis(_unit(q, 'normals'))*factors, c))
    return result[0] if normals.ndim == 2 else result


def directional(directions, irradiance):
    """Unit directions and unchanged nonnegative RGB delta-light weights.

    Directions are [L,3]/[B,L,3]; weights may additionally share [1,L,3].
    The caller applies the surface cosine once. There is no distance or PDF.
    """
    _inputs(directions=directions, irradiance=irradiance)
    q = _vectors(directions, 'directions')
    weights = _shared(irradiance, q.shape[0], (q.shape[1], 3), 'irradiance')
    if bool((irradiance < 0).any()):
        raise ValueError('irradiance must be nonnegative')
    result = dict(direction=_unit(q, 'directions'), irradiance=weights)
    return {k: v[0] for k, v in result.items()} if directions.ndim == 2 else result


def point(points, positions, intensity):
    """Inverse-square point lights, before multiplying BRDF or surface cosine.

    Points [P,3]/[B,P,3]; positions and intensity [L,3]/[1,L,3]/[B,L,3].
    Return direction/irradiance [...,P,L,3] and squared distance/bool valid
    [...,P,L]. Coincidence is invalid, with zero values and first derivatives.
    No epsilon distance or intensity cap changes the physical model.
    """
    _inputs(points=points, positions=positions, intensity=intensity)
    q = _vectors(points, 'points')
    positions_view = _vectors(positions, 'positions')
    b, count = q.shape[0], positions_view.shape[1]
    sources = _shared(positions, b, (count, 3), 'positions')
    weights = _shared(intensity, b, (count, 3), 'intensity')
    if bool((intensity < 0).any()):
        raise ValueError('intensity must be nonnegative')
    offset = _finite(sources[:, None]-q[:, :, None])
    valid = (offset != 0).any(dim=-1)
    distance_squared = _finite(offset.square().sum(dim=-1))
    if bool(((distance_squared == 0) & valid).any()):
        raise ValueError('nonzero squared distance is not representable in the input dtype')
    denominator = torch.where(valid, distance_squared, torch.ones_like(distance_squared))
    direction = offset/denominator.sqrt()[..., None]
    weight = _finite(weights[:, None]/denominator[..., None])
    result = dict(direction=torch.where(valid[..., None], direction, 0.),
                  irradiance=torch.where(valid[..., None], weight, 0.),
                  distance_squared=distance_squared, valid=valid)
    return {k: v[0] for k, v in result.items()} if points.ndim == 2 else result
