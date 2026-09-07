"""Experimental self-generated energy tables; not an admitted runtime fast path.

Only constant knot data are detached. Query interpolation retains derivatives
through cosines, roughness and Fresnel reflectance. No hidden out-of-range clamp.
"""
from dataclasses import dataclass
import math

import torch

from tools.shading_energy_probe import _gauss_unit
from tools.shade_contract_reference import _inputs


def _weights(axis,query,interpolation):
    high=torch.searchsorted(axis,query.contiguous(),right=True).clamp(1,len(axis)-1)
    if interpolation=='linear':
        indices=torch.stack((high-1,high),dim=-1)
        fraction=(query-axis[high-1])/(axis[high]-axis[high-1])
        return indices,torch.stack((1-fraction,fraction),dim=-1)
    start=(high-1).clamp(max=len(axis)-3)
    indices=start[...,None]+torch.arange(3,device=axis.device)
    x0,x1,x2=axis[indices].unbind(-1)
    weights=torch.stack(((query-x1)*(query-x2)/((x0-x1)*(x0-x2)),
                         (query-x0)*(query-x2)/((x1-x0)*(x1-x2)),
                         (query-x0)*(query-x1)/((x2-x0)*(x2-x1))),dim=-1)
    return indices,weights


@dataclass(frozen=True)
class BudgetTable:
    roughness: torch.Tensor
    cosines: torch.Tensor
    energy: torch.Tensor
    mean: torch.Tensor
    metadata: dict

    def evaluate(self,cosine,roughness,f0):
        _inputs(cosine=cosine,roughness=roughness,f0=f0)
        if cosine.device.type!='cpu':
            raise ValueError('experimental table supports CPU only')
        for value,name,low,high in ((cosine,'cosine',0.,1.),(roughness,'roughness',.2,1.),(f0,'f0',0.,1.)):
            if bool(((value<low)|(value>high)).any()):
                raise ValueError(f'{name} outside explicit table domain [{low},{high}]')
            if name!='cosine' and value.ndim!=0 and value.shape!=cosine.shape:
                raise ValueError(f'{name} must be scalar or exactly match query shape')
        roughness=roughness.expand_as(cosine);f0=f0.expand_as(cosine)
        rn=self.roughness.to(dtype=cosine.dtype);cn=self.cosines.to(dtype=cosine.dtype)
        data=self.energy.to(dtype=cosine.dtype);means=self.mean.to(dtype=cosine.dtype)
        # Clamps select the endpoint cell only, AFTER physical domain validation.
        interpolation=self.metadata['interpolation']
        ri,rw=_weights(rn,roughness,interpolation);ci,cw=_weights(cn,cosine,interpolation)
        samples=data[ri[..., :,None],ci[...,None,:]]
        pair=(samples*rw[..., :,None,None]*cw[...,None,:,None]).sum(dim=(-3,-2))
        avg=(means[ri]*rw[..., :,None]).sum(dim=-2)
        return dict(directional=pair[...,0]+f0*(pair[...,1]-pair[...,0]),
                    average=avg[...,0]+f0*(avg[...,1]-avg[...,0]))


def directional_albedo_endpoints(cos_view,roughness,*,resolution=256):
    """Two Fresnel sums over one VNDF geometry grid; no parameter-dependent cache.

    The original scalar oracle remains separate for equivalence checks. Only
    geometric square-root roundoff is guarded; endpoint energies are not clamped.
    """
    if not (0<=cos_view<=1 and 0<roughness<=1):
        raise ValueError('invalid diagnostic cosine or roughness')
    if type(resolution) is not int or not 16<=resolution<=512:
        raise ValueError('VNDF resolution must be an integer in [16,512]')
    alpha=roughness**2;vx=math.sqrt(1-cos_view*cos_view)
    length=math.hypot(alpha*vx,cos_view);hx,hz=alpha*vx/length,cos_view/length
    u,weights=_gauss_unit(resolution)
    phi=(torch.arange(2*resolution,dtype=torch.float64)+.5)*(math.pi/resolution)
    disk_x=u[:,None].sqrt()*phi.cos()[None,:]
    disk_y=u[:,None].sqrt()*phi.sin()[None,:]
    mix=(1+hz)/2
    disk_y=(1-mix)*(1-disk_x.square()).sqrt()+mix*disk_y
    projected_z=(1-disk_x.square()-disk_y.square()).clamp_min(0).sqrt()
    nx=alpha*(-hz*disk_y+hx*projected_z);ny=alpha*disk_x
    nz=hx*disk_y+hz*projected_z
    scale=(nx.square()+ny.square()+nz.square()).sqrt();nx,nz=nx/scale,nz/scale
    vh=vx*nx+cos_view*nz;nl=2*vh*nz-cos_view
    valid=nl>0;positive_nl=torch.where(valid,nl,torch.ones_like(nl))
    root_v=math.sqrt(alpha*alpha+(1-alpha*alpha)*cos_view*cos_view)
    root_l=(alpha*alpha+(1-alpha*alpha)*positive_nl.square()).sqrt()
    ratio=positive_nl*(cos_view+root_v)/(positive_nl*root_v+cos_view*root_l)
    zero=torch.where(valid,(1-vh).pow(5)*ratio,0.)
    one=torch.where(valid,ratio,0.)
    return float((zero.mean(dim=1)*weights).sum()),float((one.mean(dim=1)*weights).sum())


def generate(*,roughness_nodes=33,cosine_nodes=257,resolution=256,roughness_grid='uniform',interpolation='quadratic'):
    """Offline scalar CPU oracle integration; expensive, deliberately explicit."""
    for value,name,low,high in ((roughness_nodes,'roughness_nodes',3,65),
                               (cosine_nodes,'cosine_nodes',3,257),
                               (resolution,'resolution',16,512)):
        if type(value) is not int or not low<=value<=high:
            raise ValueError(f'{name} outside bounded generation range')
    sample_count=roughness_nodes*cosine_nodes*2*resolution*resolution
    if sample_count>1_200_000_000:
        raise ValueError('generation exceeds 1200000000 quadrature samples')
    if roughness_grid not in ('sharp-dense','uniform'):
        raise ValueError("roughness_grid must be 'sharp-dense' or 'uniform'")
    if interpolation not in ('quadratic','linear'):
        raise ValueError("interpolation must be 'quadratic' or 'linear'")
    if roughness_grid=='uniform':
        rn=torch.linspace(.2,1.,roughness_nodes,dtype=torch.float64)
    else:
        u=torch.linspace(0.,1.,roughness_nodes,dtype=torch.float64)
        rn=torch.where(u<=.25,.2+.4*u,.3+(u-.25)*(.7/.75))
    cn=torch.linspace(0.,1.,cosine_nodes,dtype=torch.float64).square()
    values=[]
    for r in rn:
        values.append([directional_albedo_endpoints(float(c),float(r),resolution=resolution) for c in cn])
    energy=torch.tensor(values,dtype=torch.float64)
    if not bool((torch.isfinite(energy)&(energy>=0)&(energy<=1)).all()):
        raise ValueError('unclamped generated energy is outside [0,1]')
    lo,hi=cn[:-1],cn[1:]
    half=(hi-lo)/2;mid=(hi+lo)/2
    nodes=mid[:,None]+half[:,None]*torch.tensor([-1/math.sqrt(3),1/math.sqrt(3)],dtype=torch.float64)
    indices,weights=_weights(cn,nodes,interpolation)
    curve=(energy[:,indices]*weights[None,...,None]).sum(dim=-2)
    mean=(curve*(2*nodes*half[:,None])[None,...,None]).sum(dim=(1,2))
    return BudgetTable(rn,cn,energy,mean,dict(roughness_min=.2,roughness_max=1.,
        roughness_nodes=roughness_nodes,cosine_nodes=cosine_nodes,resolution=resolution,roughness_grid=roughness_grid,
        interpolation=interpolation,
        sample_count=sample_count,sample_count_basis='shared geometric quadrature samples',
        endpoint_outputs_per_sample=2,model='correlated-Smith GGX, Schlick F0 endpoints',
        status='experimental candidate; not public material acceptance'))
