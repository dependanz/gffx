"""Paired GGX loss integrals, including derived horizon/normal limits.

Experimental scalar CPU float64 reference. Based on the normalized visible-normal
density (Heitz 2018, https://jcgt.org/published/0007/04/01/paper.pdf, equation 3).
The endpoint substitutions and conditioning split are our derivations. This is
not a public shader, an endpoint view derivative, or a global accuracy guarantee.
"""
import math

import torch

from tools.material_budget_reference import _resolution, _validate
from tools.material_loss_reference import _geometry, _rejected
from tools.shading_energy_probe import _gauss_unit


def _absorption(vh):
    return vh*(5+vh*(-10+vh*(10+vh*(-5+vh))))


def _pair(vh, masking, weight, measure, rejected):
    fresnel = (1-vh).pow(5)
    direct = rejected + ((_absorption(vh)+fresnel*masking)*measure).sum()
    # Integrate the smaller side directly. Never clamp or renormalize the PDF.
    zero = torch.where(direct <= .5, direct, 1-(fresnel*weight*measure).sum())
    one = rejected + (masking*measure).sum()
    return torch.stack((zero, one))


def endpoints(cosine, roughness, *, resolution=128, azimuth_resolution=None):
    """Return losses at F0=0 and F0=1, sharing one geometry quadrature.

    Cosine is a fixed scalar in [0,1]; roughness retains autodiff. Interior views
    agree with material_loss_reference. Endpoints use their limiting densities.
    """
    c = torch.as_tensor(cosine, dtype=torch.float64)
    _validate(f0=c, roughness=roughness)
    _resolution(resolution)
    azimuth_resolution = resolution if azimuth_resolution is None else azimuth_resolution
    _resolution(azimuth_resolution)
    a = roughness.square()
    nodes, weights = _gauss_unit(resolution)
    phi_nodes, phi_weights = _gauss_unit(azimuth_resolution)
    if bool(c == 0):
        # y=exp(-asinh(t/a)), then y=a*sinh(u). The second substitution
        # resolves the sharp y~a Fresnel transition without truncating its tail.
        bound = torch.asinh(1/a)
        u = nodes*bound
        y = a*torch.sinh(u)
        jacobian = a*torch.cosh(u)*bound
        b = (phi_nodes*(math.pi/2)).cos()
        q = 1-y.square()
        vh = a*q[:, None]*b/torch.sqrt(4*y[:, None].square()+a.square()*q[:, None].square())
        density = (8/math.pi)*q.square()/(1+y.square()).pow(3)
        measure = (density*jacobian*weights)[:, None]*(b*phi_weights*(math.pi/2))
        result = torch.stack(((_absorption(vh)*measure).sum(), roughness*0))
    elif bool(c == 1):
        bound = torch.asinh(1/a)
        u = nodes*bound
        s = torch.sinh(u)
        slope = a*s
        nh2 = 1/(1+slope.square())
        vh = nh2.sqrt()
        nl = (1-slope.square())*nh2
        rl = torch.sqrt(a.square()+(1-a.square())*nl.square())
        masking = a.square()*(1-nl.square())/(rl+nl).square()
        weight = 2*nl/(nl+rl)
        measure = 2*s/(1+s.square()).square()*torch.cosh(u)*bound*weights
        result = _pair(vh, masking, weight, measure, a.square()/(1+a.square()))
    else:
        a, _, _, pw, b, limit, rv = _geometry(c, roughness, azimuth_resolution)
        bound = torch.asinh(limit/a)
        u = nodes[:, None]*bound
        s = torch.sinh(u)
        slope = a*s
        nh2 = 1/(1+slope.square())
        vh = (c+b*slope)*nh2.sqrt()
        nl = (c*(1-slope.square())+2*b*slope)*nh2
        if not bool((torch.isfinite(nl)&(nl > 0)).all()):
            raise ValueError('reflected cosine is not representable')
        rl = torch.sqrt(a.square()+(1-a.square())*nl.square())
        masking = c*a.square()*(1-nl.square())/((rl+nl)*(nl*rv+c*rl))
        weight = nl*(c+rv)/(nl*rv+c*rl)
        measure = (2*s/(1+s.square()).square()*torch.cosh(u)*bound
                   *2*(c+b*slope)/(c+rv)*weights[:, None]*pw)
        rejected = (_rejected(c, a, b, limit, rv)*pw).sum()
        result = _pair(vh, masking, weight, measure, rejected)
    if not bool((torch.isfinite(result)&(result >= 0)&(result <= 1)).all()):
        raise ValueError('endpoint loss is outside representable [0,1]')
    return result
