"""Explicit tapered tube geometry with a parallel-transport frame."""
import math
import torch
from ._raster_material import unit

def strands(cls,name,points,*,radius=.001,tip_radius=0.,sides=6,material=None,dtype=None,device=None):
    from ._scene_builders import resolve_device
    from .scene import Attribute
    p=torch.as_tensor(points,dtype=dtype or (points.dtype if isinstance(points,torch.Tensor)else torch.float32),device=resolve_device(device))
    if p.ndim!=3 or p.shape[-1]!=3 or p.shape[1]<2:raise ValueError('strand points must be [strands, samples>=2, 3]')
    if not bool(torch.isfinite(p).all()):raise ValueError('strand points must be finite')
    if type(sides)is not int or not 3<=sides<=32:raise ValueError('strand sides must be in [3,32]')
    if len(p)==0:return cls(name,p.reshape(0,3),torch.empty((0,3),dtype=torch.int32,device=p.device),material=material)
    edges=p[:,1:]-p[:,:-1]
    if not bool((edges.square().sum(-1)>1e-20).all()):raise ValueError('adjacent strand samples must be distinct')
    tangent=unit(torch.cat((edges[:,:1],edges[:,:-1]+edges[:,1:],edges[:,-1:]),1))
    if not bool((tangent.square().sum(-1)>.5).all()):raise ValueError('strand cannot reverse by180degrees at a sample')
    axis=torch.nn.functional.one_hot(tangent[:,0].abs().argmin(-1),3).to(p.dtype)
    frame=unit(torch.linalg.cross(tangent[:,0],axis));frames=[frame]
    for i in range(1,p.shape[1]):
        before,after=tangent[:,i-1],tangent[:,i];rotation=torch.linalg.cross(before,after);cos=(before*after).sum(-1,keepdim=True)
        if bool((cos<-.9999).any()):raise ValueError('strand tangent reversal is unsupported')
        frame=frame+torch.linalg.cross(rotation,frame)+torch.linalg.cross(rotation,torch.linalg.cross(rotation,frame))/(1+cos)
        frame=unit(frame-after*(frame*after).sum(-1,keepdim=True));frames.append(frame)
    a=torch.stack(frames,1);b=torch.linalg.cross(tangent,a)
    theta=torch.arange(sides,device=p.device,dtype=p.dtype)*2*math.pi/sides
    normals=a[:,:,None,:]*theta.cos()[None,None,:,None]+b[:,:,None,:]*theta.sin()[None,None,:,None]
    r=torch.as_tensor(radius,device=p.device,dtype=p.dtype);tip=torch.as_tensor(tip_radius,device=p.device,dtype=p.dtype)
    if not bool(torch.isfinite(r).all()and torch.isfinite(tip).all())or not bool((r>0).all()and(tip>=0).all()):raise ValueError('strand radii must be finite, root>0, tip>=0')
    if r.ndim>1 or(r.ndim==1 and len(r)!=len(p)):raise ValueError('radius must be scalar or one value per strand')
    t=torch.linspace(0,1,p.shape[1],device=p.device,dtype=p.dtype)
    radii=r.reshape(-1,1)*(1-t)+tip.reshape(-1,1)*t
    # A small finite terminal ring avoids collapsed/zero-area tip triangles.
    radii=radii.clamp_min(torch.finfo(p.dtype).eps*float(r.max()))
    vertices=(p[:,:,None,:]+normals*radii[:,:,None,None]).reshape(-1,3)
    start=torch.arange(len(p),device=p.device)[:,None,None]*p.shape[1]*sides+torch.arange(p.shape[1]-1,device=p.device)[None,:,None]*sides+torch.arange(sides,device=p.device)[None,None,:]
    nxt=(start//sides)*sides+(start+1)%sides
    faces=torch.stack((torch.stack((start,nxt,start+sides),-1),torch.stack((nxt,nxt+sides,start+sides),-1)),-2).reshape(-1,3).to(torch.int32)
    length=torch.cat((p.new_zeros((len(p),1)),edges.norm(dim=-1).cumsum(1)),1);v=length/length[:,-1:].clamp_min(1e-20)
    uv=torch.stack((torch.arange(sides,device=p.device,dtype=p.dtype).expand(len(p),p.shape[1],sides)/sides,v[:,:,None].expand(-1,-1,sides)),-1).reshape(-1,2)
    return cls(name,vertices,faces,{'normals':Attribute(normals.reshape(-1,3),'vertex'),'uv':Attribute(uv,'vertex')},material)
