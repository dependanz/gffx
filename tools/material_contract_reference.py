"""Phase-0 material fields and environment integration, NOT public gffx APIs.

Material fields have already been sampled/interpolated to surface points.
Environment weights are solid-angle quadrature weights, never delta-light
intensities or unlabelled probability densities. All arithmetic is differentiable
in the declared interior; eager validation and layout choices are not.
"""
import torch

from tools import light_contract_reference as light
from tools import shade_contract_reference as shade


def _rgb(value, batch, points, name, *, achromatic=False):
    if achromatic and value.ndim == 0:
        return value.expand(batch, points, 3)
    if value.shape == (3,):
        return value.expand(batch, points, 3)
    if value.ndim == 2:
        value = value[None]
    channels = (1, 3) if achromatic else (3,)
    if (value.ndim != 3 or value.shape[0] not in (1, batch) or
            value.shape[1] != points or value.shape[2] not in channels):
        raise ValueError(f'{name} shape/batch/channels do not match material points')
    return value.expand(batch, points, 3)


def parameters(normals, *, model, albedo, specular=None, exponent=None, roughness=None, f0=None):
    """Expand pre-sampled fields to the normal query layout, never infer topology."""
    if model == 'classic':
        if specular is None or exponent is None or roughness is not None or f0 is not None:
            raise ValueError('classic requires specular/exponent, not roughness/f0')
        fields = dict(albedo=albedo, specular=specular, exponent=exponent)
    elif model == 'principled':
        if roughness is None or f0 is None or specular is not None or exponent is not None:
            raise ValueError('principled requires roughness/f0, not specular/exponent')
        fields = dict(albedo=albedo, roughness=roughness, f0=f0)
    else:
        raise ValueError('unknown material model')
    shade._inputs(normals=normals, **fields)
    query = light._vectors(normals, 'normals')
    light._unit(query)  # zero normals are invalid, including before material evaluation
    batch, points = query.shape[:2]
    result = {}
    for name, value in fields.items():
        shade._range(value, name, upper=None if name == 'exponent' else 1., open_lower=name == 'roughness')
        if name in ('roughness', 'exponent'):
            if name == 'roughness' and bool((value.pow(4) == 0).any()):
                raise ValueError('roughness fourth power is not representable')
            result[name] = shade._scalar(value, query[...,0], name)
        else:
            result[name] = _rgb(value, batch, points, name, achromatic=name != 'albedo')
    return {name: value[0] for name,value in result.items()} if normals.ndim == 2 else result


def environment_radiance(directions, coefficients):
    """Evaluate incident-radiance SH without Lambert band convolution factors."""
    shade._inputs(directions=directions, coefficients=coefficients)
    query = light._vectors(directions, 'directions')
    coeff = light._shared(coefficients, query.shape[0], (9,3), 'coefficients')
    value = light._finite_result(light.basis(query) @ coeff)
    return value[0] if directions.ndim == 2 else value


def integrate_environment(normals, directions, radiance, brdf, weights):
    """Sum f_r(N,V,L)*L_i(L)*max(N.L,0)*domega; BRDF is already evaluated.

N owns B/P. Directions own an independent Q axis shared across points.
No hidden normalization of weights, implicit visibility or sampling PDF.
Signed incident SH radiance is retained; built-in BRDF samples are nonnegative.
"""
    shade._inputs(normals=normals, directions=directions, radiance=radiance, brdf=brdf, weights=weights)
    query = light._vectors(normals, 'normals')
    samples = light._vectors(directions, 'directions')
    batch, points = query.shape[:2]
    count = samples.shape[1]
    samples = light._shared(directions, batch, (count,3), 'directions')
    incoming = light._shared(radiance, batch, (count,3), 'radiance')
    values = light._shared(brdf, batch, (points,count,3), 'brdf')
    omega = light._shared(weights, batch, (count,), 'weights')
    if bool((omega < 0).any()) or bool((values < 0).any()):
        raise ValueError('weights and brdf must be nonnegative')
    n, l = light._unit(query), light._unit(samples)
    cosine = (n[:,:,None,:]*l[:,None,:,:]).sum(-1)
    positive = torch.where(cosine > 0, cosine, torch.zeros_like(cosine))
    result = (values*incoming[:,None,:,:]*positive[...,None]*omega[:,None,:,None]).sum(-2)
    result = light._finite_result(result)
    return result[0] if normals.ndim == 2 else result
