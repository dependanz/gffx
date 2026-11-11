"""Continuous rectangle integration using the Heitz et al. GGX LTC fit.

Data provenance/license: _ltc_LICENSE.txt beside _ltc.bin. Tensor fallback uses
the same fit but independent vectorized polygon clipping/integration. Visibility
uses cosine-weighted shadow-map quadrature, an explicitly approximate shared
visibility factor for the diffuse/specular lobes.
"""
from pathlib import Path
import math
import torch
from ._raster_material import unit

_tables={}
def table(like):
    key=(like.device,like.dtype)
    if key not in _tables:
        raw=bytearray(Path(__file__).with_name('_ltc.bin').read_bytes())
        values=torch.frombuffer(raw,dtype=torch.float32).reshape(64,64,8)
        _tables[key]=values.to(device=like.device,dtype=like.dtype)
    return _tables[key]

def lookup(rough,nv):
    data=table(nv);x=rough.clamp(0,1)*63;y=(1-nv.clamp(0,1)).sqrt()*63
    ix=x.floor().long();iy=y.floor().long();jx=(ix+1).clamp_max(63);jy=(iy+1).clamp_max(63)
    x=(x-ix)[:,None];y=(y-iy)[:,None]
    return (1-y)*((1-x)*data[iy,ix]+x*data[iy,jx])+y*((1-x)*data[jy,ix]+x*data[jy,jx])

def _clip(poly,count,normal):
    # At most six vertices after clipping a convex quad against two planes.
    batch=len(poly);capacity=poly.shape[1];ids=torch.arange(capacity,device=poly.device)
    following=(ids[None]+1)%count.clamp_min(1)[:,None]
    other=poly.gather(1,following[...,None].expand(-1,-1,3))
    da=(poly*normal[:,None]).sum(-1);db=(other*normal[:,None]).sum(-1)
    valid=ids[None]<count[:,None];keep=(da>=0)&valid;cross=((da>0)&(db<0)|(da<0)&(db>0))&valid
    den=da-db;t=da/torch.where(den!=0,den,1)
    intersections=poly+t[...,None]*(other-poly)
    candidates=torch.stack((poly,intersections),2).reshape(batch,-1,3)
    flags=torch.stack((keep,cross),2).reshape(batch,-1)
    index=flags.long().cumsum(1)-1;result=torch.zeros_like(poly)
    rows=torch.arange(batch,device=poly.device)[:,None].expand_as(flags)
    result[rows[flags],index[flags]]=candidates[flags]
    return result,flags.sum(1)

def polygon(points,n,geo,view,area,fit=None):
    corners=points.new_tensor([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])
    vertices=area['center']+corners[:,:1]*area['u']+corners[:,1:]*area['v']
    poly=points.new_zeros((len(points),8,3));poly[:,:4]=vertices[None]-points[:,None]
    count=torch.full((len(points),),4,dtype=torch.long,device=points.device)
    poly,count=_clip(poly,count,geo)
    nv=(n*view).sum(-1);t= view-nv[:,None]*n
    axis=torch.where((n[:,2].abs()<.9)[:,None],n.new_tensor([0,0,1]),n.new_tensor([0,1,0]))
    t=unit(torch.where((t.square().sum(-1)<1e-10)[:,None],torch.linalg.cross(axis,n),t));b=torch.linalg.cross(n,t)
    x=(poly*t[:,None]).sum(-1);y=(poly*b[:,None]).sum(-1);z=(poly*n[:,None]).sum(-1)
    poly=torch.stack((x if fit is None else fit[:,0,None]*x+fit[:,2,None]*z,y,z if fit is None else fit[:,1,None]*x+fit[:,3,None]*z),-1)
    poly,count=_clip(poly,count,n.new_tensor([0,0,1]).expand_as(n));poly=unit(poly)
    ids=torch.arange(8,device=points.device);following=(ids[None]+1)%count.clamp_min(1)[:,None]
    other=poly.gather(1,following[...,None].expand(-1,-1,3));cross=torch.linalg.cross(poly,other)
    length=torch.linalg.vector_norm(cross,dim=-1);dot=(poly*other).sum(-1).clamp(-1,1)
    edge=cross[...,2]*torch.atan2(length,dot)/length.clamp_min(1e-10)
    return ((edge*(ids[None]<count[:,None])).sum(1).abs()/(2*math.pi)).clamp(0,1)

def evaluate(points,n,geo,view,params,area,visibility=None):
    n=unit(n);view=unit(view);count=len(points);nv=(n*view).sum(-1).clamp(0,1)
    def field(key,default):
        v=torch.as_tensor(params.get(key,default),device=points.device,dtype=points.dtype)
        return v.expand(count) if v.ndim==0 else v
    base=field('base_color',0).expand(count,3);metal=field('metallic',0);ior=field('ior',1.5)
    fit=lookup(field('roughness',.3),nv);spec=polygon(points,n,geo,view,area,fit);diff=polygon(points,n,geo,view,area)
    dielectric=((ior-1)/(ior+1)).square();f0=dielectric[:,None]*(1-metal[:,None])+base*metal[:,None]
    energy=f0*fit[:,4,None]+(1-f0)*fit[:,5,None]
    value=energy*spec[:,None]+(1-metal[:,None])*(1-dielectric[:,None])*base*diff[:,None]
    coat=field('coat',0);cr=field('coat_roughness',.12)
    if bool((coat>0).any()):
        cf=lookup(cr,nv);cs=polygon(points,n,geo,view,area,cf);nl=(n*unit(area['center']-points)).sum(-1).clamp(0,1)
        attenuation=(1-coat*(.04+.96*(1-nv).pow(5)))*(1-coat*(.04+.96*(1-nl).pow(5)))
        value=value*attenuation[:,None]+(coat*(.04*cf[:,4]+.96*cf[:,5])*cs)[:,None]
    if visibility is not None:value=value*visibility[:,None]
    front=((points-area['center'])*unit(torch.linalg.cross(area['u'],area['v']))).sum(-1)>0
    front=front&((geo*view).sum(-1)>0)
    return torch.where(front[:,None],value*area['radiance'],0.)
