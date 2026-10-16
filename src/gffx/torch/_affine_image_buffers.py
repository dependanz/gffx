"""Opt-in signed-area-offset image buffers using public raster/interpolation.

A caller supplies image-space visibility depth explicitly. Physical cameras and
all ordinary scene/rasterization defaults retain their existing contracts.
"""
import math
import torch
from . import render


def affine_image_buffers(obj,camera,vertices,faces,face_uv,*,denominator_offset,
                         visibility_depth,workspace_limit_bytes,cpu_workspace_allowance_bytes):
    if isinstance(denominator_offset,bool) or not isinstance(denominator_offset,(float,int)) or not math.isfinite(denominator_offset) or denominator_offset<=0:
        raise ValueError('explicit finite positive NDC denominator offset')
    if camera.batch_size!=1 or camera.projection_kind!='orthographic':raise ValueError('one parallel image camera required')
    if visibility_depth is None or visibility_depth.shape!=(len(vertices),) or visibility_depth.dtype!=vertices.dtype or visibility_depth.device!=vertices.device or not bool(torch.isfinite(visibility_depth).all() & (visibility_depth>=0).all()):
        raise ValueError('finite nonnegative per-vertex image visibility depth')
    from ._deferred import _plan_workload
    workload=_plan_workload(obj,camera,1,workspace_limit_bytes,('mask','face_index','barycentric','uv'),cpu_workspace_allowance_bytes=cpu_workspace_allowance_bytes)
    # Duplicated per-face geometry/depth is additional to the ordinary plan.
    layers=8
    extra=(len(faces)*3*3*4+camera.image_height*camera.image_width*layers*7*4)*vertices.element_size()
    if workload['estimated_bytes']+extra>workspace_limit_bytes:raise ValueError('affine image-buffer workspace exceeded')
    projected=camera.project(vertices)
    xy=torch.stack((2*projected.pixels[:,0]/camera.image_width-1,1-2*projected.pixels[:,1]/camera.image_height),-1)
    corners=xy[faces.long()];b=corners[:,1]-corners[:,0];d=corners[:,2]-corners[:,0]
    area=b[:,0]*d[:,1]-b[:,1]*d[:,0]
    keep=area.abs()>denominator_offset;original=keep.nonzero().flatten();area=area[keep]
    factor=area/(area+denominator_offset)
    # q*sum(lambda*z) is the offset-denominator depth; no physical-depth
    # remapping or clipping is silently inherited from the camera.
    depth=visibility_depth[faces[keep].long()]*factor[:,None]
    virtual=torch.cat((corners[keep],depth[...,None]),-1).reshape(-1,3)
    triangles=torch.arange(len(virtual),dtype=torch.int32,device=vertices.device).reshape(-1,3)
    owner,bary,_,_=render.rasterize(virtual.contiguous(),triangles,camera.image_height,camera.image_width,faces_per_pixel=layers,cull_mode=render.CULL_NONE,eps=0.)
    # The source admits STRICTLY positive barycentrics. Keep a bounded list so
    # an excluded foreground edge does not hide a valid face behind it. Refuse
    # an unresolved full edge stack rather than inventing background coverage.
    strict=(owner>=0)&(bary>0).all(-1)
    if bool(((owner>=0).all(-1)&~strict.any(-1)).any()):raise ValueError('strict edge stack exceeds explicit8fragment capacity')
    chosen=strict.to(torch.int64).argmax(-1,keepdim=True)
    owner=owner.gather(-1,chosen);bary=bary.gather(-2,chosen[...,None].expand(*chosen.shape,3))
    owner=torch.where(strict.any(-1,keepdim=True),owner,torch.full_like(owner,-1))
    valid=owner>=0;index=owner.clamp_min(0).long()
    if len(original):
        mapped=torch.where(valid,original[index],-1).to(torch.int32)
        adjusted=bary*torch.where(valid,factor[index],torch.zeros_like(bary[...,0]))[...,None]
    else:
        mapped=owner;adjusted=bary*0
    uv=render.interpolate(mapped,adjusted,(face_uv*2-1).contiguous())[...,0,:]
    uv=(uv+1)/2;uv=torch.where(valid[...,0,None],uv,torch.zeros_like(uv))
    return dict(mask=valid[...,0].to(vertices.dtype).detach(),face_index=mapped[...,0],
                barycentric=adjusted[...,0,:],uv=uv)
