"""Direct isotropic GGX surfaces for the raster integrator.

This is a conventional single-scatter microfacet model with a metallic workflow
and an optional dielectric coat. It is separate from material.principled and its
named reciprocal-matte float64 resources; it makes no claim of their certificate.
"""
import math
import torch


def unit(x):
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)


def _microfacet(n, v, l, roughness, f0):
    h = unit(v[:, None, :] + l)
    nv = (n*v).sum(-1).clamp(0, 1)[:, None]
    nl = (n[:, None, :]*l).sum(-1).clamp(0, 1)
    nh = (n[:, None, :]*h).sum(-1).clamp(0, 1)
    vh = (v[:, None, :]*h).sum(-1).clamp(0, 1)
    a2 = roughness.square().square()
    d = a2 / (math.pi * ((nh.square()*(a2-1)+1).square()).clamp_min(1e-20))
    visibility = .5 / (nl*torch.sqrt(nv.square()*(1-a2)+a2) +
                       nv*torch.sqrt(nl.square()*(1-a2)+a2)).clamp_min(1e-12)
    fresnel = f0 + (1-f0)*(1-vh[..., None]).pow(5)
    return d[..., None]*visibility[..., None]*fresnel, fresnel, nl


def pbr(normals, view, directions, *, base_color, roughness=.3, metallic=0.,
        ior=1.5, coat=0., coat_roughness=.12, normal_map=None,_geometric_normals=None):
    """BRDF values [P,L,3], without light weights or the incident cosine.

    Inputs are linear RGB and perceptual roughness. Raster scene setup validates
    domains once; the arithmetic retains smooth material derivatives.
    """
    n, v, l = unit(normals), unit(view), unit(directions)
    count = len(n)
    def field(value, channels=False):
        value = torch.as_tensor(value, dtype=n.dtype, device=n.device)
        if channels:
            if value.ndim == 0: value=value.expand(3)
            if value.ndim == 1: value=value.expand(count,3)
            return value[:,None,:]
        if value.ndim == 0: value=value.expand(count)
        return value.reshape(count,1)
    base=field(base_color,True);r=field(roughness);m=field(metallic)
    eta=field(ior);c=field(coat);cr=field(coat_roughness)
    dielectric=((eta-1)/(eta+1)).square()[...,None]
    f0=dielectric*(1-m[...,None])+base*m[...,None]
    spec,fresnel,nl=_microfacet(n,v,l,r,f0)
    value=spec+(1-m[...,None])*(1-fresnel)*base/math.pi
    if isinstance(coat,torch.Tensor) or coat != 0:
        coat_spec,_,_=_microfacet(n,v,l,cr,torch.full_like(f0,.04))
        nv=(n*v).sum(-1).clamp(0,1)[:,None]
        fv=.04+.96*(1-nv).pow(5)
        fl=.04+.96*(1-nl).pow(5)
        value=value*(1-c*fv)[...,None]*(1-c*fl)[...,None]+c[...,None]*coat_spec
    facing=n if _geometric_normals is None else _geometric_normals
    return torch.where(((facing*v).sum(-1)>0)[:,None,None] & (nl>0)[...,None],value,0.)


def spot(points, *, positions, intensity, direction, angle, blend):
    from .light import point
    packet=point(points,positions,intensity)
    cosine=-(packet['direction']*unit(direction)).sum(-1)
    outer=math.cos(angle/2);inner=outer+(1-outer)*blend
    if blend == 0: weight=(cosine>=outer).to(points.dtype)
    else:
        t=((cosine-outer)/(inner-outer)).clamp(0,1)
        weight=t*t*(3-2*t)
    packet['irradiance']=packet['irradiance']*weight[...,None]
    return packet


def validate_material(parameters):
    from ._deferred import TextureSource
    ranges={'base_color':(0,1),'metallic':(0,1),'roughness':(.001,1),
            'ior':(1,4),'coat':(0,1),'coat_roughness':(.001,1),'subsurface':(0,1),
            'subsurface_radius':(1e-8,1e3),'subsurface_transmission':(0,1)}
    if 'subsurface' in parameters and bool(torch.as_tensor(parameters.get("metallic",0)).any()):raise ValueError("subsurface requires a dielectric material")
    for key,value in parameters.items():
        if key=='normal_space':
            if value not in ('tangent','world'):raise ValueError('normal_space must be tangent or world')
            continue
        if isinstance(value,TextureSource):continue
        if key=='normal_map':
            if value is not None:raise TypeError('normal_map must be a TextureSource')
            continue
        if key not in ranges:raise ValueError('unknown PBR field: '+key)
        low,high=ranges[key]
        if not isinstance(value,torch.Tensor):raise TypeError('PBR fields must be authored tensors')
        if not bool(torch.isfinite(value).all()) or not bool(((value>=low)&(value<=high)).all()):
            raise ValueError(key+' is outside its physical input domain')
