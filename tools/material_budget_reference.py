"""Bounded differentiable CPU/float64 quadrature; NOT a public or fast shader.

Fixed hemisphere nodes do not depend on material/view parameters, so autodiff
does not have to account for a moving sample rejection boundary. Only constant
Gauss nodes are cached. No learned/differentiable input is detached or cached.
The scalar positive-interior view domain is intentional reference scope, not a
restriction adopted for the eventual public material API.
"""
import math

import torch

from tools.shading_energy_probe import _gauss_unit


def _validate(**fields):
    for name, value in fields.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
            raise TypeError(f'{name} requires a float64 tensor')
        if value.device.type != 'cpu' or value.ndim != 0:
            raise ValueError(f'{name} requires a scalar CPU tensor')
        if not bool(torch.isfinite(value)):
            raise ValueError(f'{name} must be finite')
        if name == 'cos_view':
            valid = (value > 0) & (value < 1)
        elif name == 'roughness':
            valid = (value > 0) & (value <= 1)
        else:
            valid = (value >= 0) & (value <= 1)
        if not bool(valid):
            raise ValueError(f'{name} outside reference domain')
        if name == 'roughness' and not bool(value.pow(4) > 0):
            raise ValueError('roughness fourth power is not representable')


def _resolution(value, maximum=512):
    if type(value) is not int or not 16 <= value <= maximum:
        raise ValueError(f'quadrature resolution must be an integer in [16,{maximum}]')


def _azimuth(rule):
    if rule not in ('gauss','midpoint'):
        raise ValueError("azimuth must be 'gauss' or 'midpoint'")


def _directional(cos_view, roughness, f0, resolution, azimuth='gauss'):
    mu, weights = _gauss_unit(resolution)
    if azimuth=='gauss':
        # Isotropy makes the integrand even in phi. Integrate [0,pi], double
        # the measure, and cluster fixed nodes near the narrow phi=pi peak.
        # No coordinate depends on the queried view or material parameters.
        phi=math.pi*mu
    else:
        phi=(torch.arange(2*resolution,dtype=torch.float64)+.5)*(math.pi/resolution)
    mu = mu[:, None]
    dot_vl = cos_view*mu + torch.sqrt(1-cos_view.square())*torch.sqrt(1-mu.square())*torch.cos(phi)
    half_length_squared = 2*(1+dot_vl)
    if not bool((half_length_squared > 0).all()):
        raise ValueError('half vector is not representable')
    half_length = torch.sqrt(half_length_squared)
    nh = (cos_view+mu)/half_length
    vh = half_length/2
    a2 = roughness.pow(4)
    denominator = ((1-nh)*(1+nh)+a2*nh.square()).square()
    if not bool((denominator > 0).all()):
        raise ValueError('GGX divisor is not representable')
    distribution = a2/(math.pi*denominator)
    rv = torch.sqrt(a2+(1-a2)*cos_view.square())
    rl = torch.sqrt(a2+(1-a2)*mu.square())
    visibility = .5/(mu*rv+cos_view*rl)
    fresnel = f0+(1-f0)*(1-vh).pow(5)
    integrand=distribution*visibility*fresnel*mu
    angular=(integrand*weights[None,:]).sum(dim=1) if azimuth=='gauss' else integrand.mean(dim=1)
    result=(angular*weights).sum()*(2*math.pi)
    if not bool(torch.isfinite(result)):
        raise ValueError('energy is not representable')
    return result


def directional(cos_view, roughness, f0, *, resolution=256, azimuth='gauss'):
    """E(c), scalar float64 CPU, 0<c<1; no hidden clamp or accuracy promise."""
    _validate(cos_view=cos_view, roughness=roughness, f0=f0)
    _resolution(resolution)
    _azimuth(azimuth)
    return _directional(cos_view, roughness, f0, resolution, azimuth)


def average(roughness, f0, *, resolution=128, view_resolution=32, azimuth='gauss'):
    """2 integral_0^1 E(c)c dc; graphs retain all roughness/F0 dependence."""
    _validate(roughness=roughness, f0=f0)
    _resolution(resolution)
    _resolution(view_resolution, maximum=64)
    _azimuth(azimuth)
    if 2*resolution*resolution*view_resolution > 2**21:
        raise ValueError('requested quadrature work exceeds 2097152 light samples')
    cosines, weights = _gauss_unit(view_resolution)
    values = torch.stack([_directional(c, roughness, f0, resolution, azimuth) for c in cosines])
    return 2*(values*cosines*weights).sum()
