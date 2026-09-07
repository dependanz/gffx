"""Sharp GGX half-vector quadrature experiment, not a public or fast material.

Map only half vectors that reflect above the surface. The moving integration
boundary is part of the graph, not a detached mask. No roughness/energy clamp.
"""
import math

import torch

from tools.material_budget_reference import _validate,_resolution
from tools.shading_energy_probe import _gauss_unit


def directional(cos_view,roughness,f0,*,resolution=256):
    """Scalar float64 CPU reference; 2*n*n samples, positive interior view."""
    _validate(cos_view=cos_view,roughness=roughness,f0=f0)
    _resolution(resolution)
    c=cos_view;alpha=roughness.square();a2=alpha.square()
    nodes,weights=_gauss_unit(resolution)
    phi=torch.cat((nodes,nodes+1))*(math.pi/2)
    phi_weights=torch.cat((weights,weights))/2
    b=torch.sqrt(1-c.square())*phi.cos()
    root=torch.hypot(b,c)
    # Safe alternative expressions avoid cancellation for b<0. Both branches
    # remain finite before selection, so the unused branch cannot poison AD.
    positive=b>=0
    direct_b=torch.where(positive,b,torch.zeros_like(b))
    reciprocal_b=torch.where(positive,torch.zeros_like(b),b)
    tmax=torch.where(positive,(direct_b+root)/c,c/(root-reciprocal_b))
    if not bool(torch.isfinite(tmax).all()):
        raise ValueError('half-vector integration boundary is not representable')
    upper=torch.atan2(tmax,alpha)
    beta=nodes[:,None]*upper[None,:]
    slope=alpha*beta.tan()
    nh=torch.rsqrt(1+slope.square())
    vh=(c+b*slope)*nh
    nl=(c*(1-slope.square())+2*b*slope)*nh.square()
    if not bool((torch.isfinite(nl)&(nl>0)).all()):
        raise ValueError('positive reflected cosine is not representable')
    rv=torch.sqrt(a2+(1-a2)*c.square())
    rl=torch.sqrt(a2+(1-a2)*nl.square())
    ratio=2*nl*(c+b*slope)/(nl*rv+c*rl)
    fresnel=f0+(1-f0)*(1-vh).pow(5)
    measure=2*beta.sin()*beta.cos()*upper
    value=(ratio*fresnel*measure*weights[:,None]*phi_weights[None,:]).sum()
    if not bool(torch.isfinite(value)):
        raise ValueError('energy is not representable')
    return value


def average(roughness,f0,*,resolution=128,view_resolution=32):
    """Cosine-weighted mean; explicit work cap, no cached parameter outputs."""
    _validate(roughness=roughness,f0=f0)
    _resolution(resolution);_resolution(view_resolution,maximum=64)
    if 2*resolution*resolution*view_resolution>2**21:
        raise ValueError('requested quadrature work exceeds 2097152 light samples')
    cosines,weights=_gauss_unit(view_resolution)
    values=torch.stack([directional(c,roughness,f0,resolution=resolution) for c in cosines])
    return 2*(values*cosines*weights).sum()
