"""Quality-oriented raster composition with bounded fragment shading.

Primary visibility and shadow maps use gffx's native rasterizer. Only a compact
fragment record is full-frame; positions, material fields and light packets are
evaluated in bounded blocks. This forward image path is separate from the
existing differentiable deferred integrator and does not change its semantics.
"""
import math
import time
from dataclasses import replace
import torch
from . import render as native, transforms, scene as types, light, material
from .camera import Camera
from ._deferred import TextureSource, _prepare
from ._raster_material import pbr, spot, unit, validate_material


def _notify(callback, **event):
    if callback is not None:callback(event)


def _camera(eye,target,size,*,fov=90.,extent=None,near=.001,far=20.):
    delta=target-eye
    up=(0,0,1) if abs(float(unit(delta)[2]))<.9 else (0,1,0)
    result=Camera.look_at(eye,target,up=up,size=(size,size),fov=fov,near=near,far=far,
                          dtype=eye.dtype,device=eye.device)
    if extent is not None:
        result=Camera.from_orthographic(size/(2*extent),size/(2*extent),size/2,size/2,
                                        size,size,near,far,pose=result.pose)
    return result


def _project(vertices,camera):
    offsets=torch.tensor([0,len(vertices)],dtype=torch.int32,device=vertices.device)
    clip=transforms.transform_points(vertices,camera.world_to_clip().contiguous(),offsets)
    ndc,valid=transforms.perspective_divide(clip)
    return clip,ndc,valid & (clip[:,3]>native.DEFAULT_EPS)


def _raster(vertices,faces,camera,*,cull_mode=native.CULL_BACK,dx=0.,dy=0.,inverse_depth=False):
    clip,ndc,valid=_project(vertices,camera)
    keep=torch.index_select(valid,0,faces.reshape(-1)).reshape(-1,3).all(-1)
    face_z=torch.index_select(ndc[:,2],0,faces.reshape(-1)).reshape(-1,3)
    keep=keep & ~((face_z < -1).all(-1)|(face_z>1).all(-1))
    del face_z
    selected=torch.nonzero(keep).flatten()
    if inverse_depth:ndc=torch.cat((ndc[:,:2],(-camera.near/clip[:,3:]).clamp(-2,0)),1)
    if dx or dy:
        ndc=ndc+ndc.new_tensor([2*dx/camera.image_width,-2*dy/camera.image_height,0])
    local=faces if bool(keep.all()) else faces[keep].contiguous()
    if not len(local):
        count=camera.image_height*camera.image_width
        return (torch.full((count,),-1,dtype=torch.int32,device=vertices.device),
                vertices.new_zeros((count,3)),vertices.new_full((count,),float('inf')),clip)
    from . import _raster_accel
    accelerated=_raster_accel.raster(ndc.contiguous(),local,camera.image_height,camera.image_width,cull_mode)
    if accelerated is None:
        idx,beta,depth,_=native.rasterize(ndc.contiguous(),local,camera.image_height,camera.image_width,cull_mode=cull_mode)
    else:idx,beta,depth=accelerated
    idx=idx.reshape(-1);beta=beta.reshape(-1,3);depth=depth.reshape(-1)
    active=(idx>=0)&(depth>=-1)&(depth<=1)
    index=torch.full_like(idx,-1)
    if len(selected):index[active]=selected[idx[active].long()].to(index.dtype)
    return index,beta,depth,clip


def _clip_shadow_near(vertices,faces,camera):
    """Clip crossing caster triangles instead of discarding an entire face."""
    if not len(faces):return vertices,faces
    view=camera.world_to_view()[0]
    vertex_distance=-(vertices@view[:3,:3].T+view[:3,3])[:,2]-camera.near*1.00001
    distance=torch.index_select(vertex_distance,0,faces.reshape(-1)).reshape(-1,3)
    inside=distance>=0;count=inside.sum(-1)
    crossing=(count>0)&(count<3)
    if not bool(crossing.any()):return vertices,faces if bool((count==3).all())else faces[count==3]
    generated=[]
    for n in [1,2]:
        chosen=count==n;tri=vertices[faces[chosen].long()];d=distance[chosen];good=inside[chosen]
        if not len(tri):continue
        i=(good if n==1 else ~good).long().argmax(-1);j=(i+1)%3;k=(i+2)%3;b=torch.arange(len(tri),device=vertices.device)
        a0,a1,a2=tri[b,i],tri[b,j],tri[b,k];d0,d1,d2=d[b,i],d[b,j],d[b,k]
        q1=a0+(d0/(d0-d1))[:,None]*(a1-a0);q2=a0+(d0/(d0-d2))[:,None]*(a2-a0)
        if n==1:generated.append(torch.stack((a0,q1,q2),1))
        else:generated.extend((torch.stack((a1,a2,q2),1),torch.stack((a1,q2,q1),1)))
    extra=torch.cat(generated).reshape(-1,3)
    extra_faces=torch.arange(len(extra),device=faces.device,dtype=faces.dtype).reshape(-1,3)+len(vertices)
    return torch.cat((vertices,extra)),torch.cat((faces[count==3],extra_faces))


def _shadow_geometry(vertices,faces,camera):
    if camera.projection_kind=='perspective' and len(faces):
        view=camera.world_to_view()[0]
        distance=-(vertices@view[:3,:3].T+view[:3,3])[:,2]
        face_distance=torch.index_select(distance,0,faces.reshape(-1)).reshape(-1,3)
        positive_min=torch.where(face_distance>camera.near,face_distance,float("inf")).amin()
        crosses=((face_distance<=0).any(-1)&(face_distance>0).any(-1)).any()
        if bool(torch.isfinite(positive_min)) and not bool(crosses):
            # Fit depth precision to the actual caster set. An unnecessarily
            # tiny near plane plus a few NDC ulps used to erase contact shadows.
            camera.near=max(camera.near,float(positive_min)*.5)
    vertices,faces=_clip_shadow_near(vertices,faces,camera)
    return vertices,faces


def _make_map(vertices,faces,camera,prepared=False):
    if not prepared:vertices,faces=_shadow_geometry(vertices,faces,camera)
    # Private immutable camera matrices live for this render only. Public camera
    # mutation/caching semantics remain unchanged.
    view=camera.world_to_view();projection=camera.projection();matrix=projection@view;inverse=torch.linalg.inv(matrix)
    class FrozenCamera:
        def __getattr__(self,name):return getattr(camera,name)
        def world_to_view(self):return view
        def world_to_clip(self):return matrix
        def projection(self):return projection
    frozen=FrozenCamera()
    from . import _raster_accel
    clip,ndc,valid=_project(vertices,frozen);keep=torch.index_select(valid,0,faces.reshape(-1)).reshape(-1,3).all(-1)
    z=torch.index_select(ndc[:,2],0,faces.reshape(-1)).reshape(-1,3)
    keep=keep & ~((z < -1).all(-1)|(z>1).all(-1))
    del z
    if camera.projection_kind=='perspective':ndc=torch.cat((ndc[:,:2],(-camera.near/clip[:,3:]).clamp(-2,0)),1)
    depth=_raster_accel.shadow(ndc,faces if bool(keep.all())else faces[keep],camera.image_height,camera.image_width)
    if depth is None:
        idx,_,z,_=_raster(vertices,faces,frozen,cull_mode=native.CULL_NONE,inverse_depth=camera.projection_kind=='perspective')
        depth=torch.where(idx>=0,z,float('inf')).reshape(camera.image_height,camera.image_width)
    else:depth=torch.where((depth>=-1)&(depth<=1),depth,float('inf'))
    return dict(camera=frozen,depth=depth,
                matrix=matrix[0],inverse=inverse[0],inverse_depth=camera.projection_kind=='perspective',
                view=view[0],projection=projection[0],origin=camera.pose[0,:3,3])


def _maps(vertices,faces,source,*,directional=False,size=512):
    low,high=vertices.amin(0),vertices.amax(0);centre=(low+high)/2
    radius=float(torch.linalg.vector_norm(high-low)/2)+.01
    far=radius*6+float(torch.linalg.vector_norm(source-centre)) if not directional else radius*6
    if directional:
        d=unit(source);eye=centre+d*radius*2
        cam=_camera(eye,centre,size,extent=radius*1.02,near=.001,far=far)
        return [_make_map(vertices,faces,cam)]
    # Six 90-degree views cover the whole sphere, including every seam.
    # One-pixel overlap prevents a PCF footprint at a seam becoming falsely lit.
    result=[]
    for axis in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        target=source+source.new_tensor(axis)
        cam=_camera(source,target,size,fov=90+math.degrees(4/size),near=.0001,far=far)
        result.append(_make_map(vertices,faces,cam))
    return result


def _lookup(entry,points,geometric,curvature=None):
    value=_shadow_group([entry],points,geometric,curvature)[:,0]
    homogeneous=torch.cat((points,torch.ones_like(points[:,:1])),-1)
    clip=homogeneous@entry['matrix'].T;ndc=clip[:,:3]/torch.where(clip[:,3:]>0,clip[:,3:],1.)
    inside=(clip[:,3]>0)&(ndc[:,:2].abs()<=1).all(-1)&(ndc[:,2]>=-1)&(ndc[:,2]<=1)
    return value,inside,ndc


def _shadow(entries,points,geometric,curvature=None):
    if len(entries)==1:return _lookup(entries[0],points,geometric,curvature)[0]
    # Choose the view nearest its forward axis, then evaluate only that subset.
    origin=entries[0]['camera'].pose[0,:3,3]
    delta=points-origin;axis=delta.abs().argmax(-1)
    choice=axis*2+(delta.gather(1,axis[:,None])[:,0]<0).long()
    value=torch.ones(len(points),dtype=points.dtype,device=points.device)
    for i,entry in enumerate(entries):
        selected=choice==i
        if bool(selected.any()):value[selected]=_lookup(entry,points[selected],geometric[selected],None if curvature is None else curvature[selected])[0]
    return value


def _shadow_group(entries,points,geometric,curvature=None):
    """Bilinear comparison filtering with receiver-plane depth correction.

    A larger fixed box invents occlusion across curved receiver triangles.
    Emitter extent is integrated by source samples, not by broadening PCF.
    """
    matrices=torch.stack([e['matrix'] for e in entries]);inverse=torch.stack([e['inverse'] for e in entries])
    depths=torch.stack([e['depth'] for e in entries]);size=depths.shape[-1]
    homogeneous=torch.cat((points,torch.ones_like(points[:,:1])),-1)
    clip=homogeneous[None]@matrices.transpose(-1,-2)
    w=clip[...,3];ndc=clip[...,:3]/torch.where(w>0,w,1.)[...,None]
    inside=(w>0)&(ndc[...,:2].abs()<=1).all(-1)&(ndc[...,2]>=-1)&(ndc[...,2]<=1)
    coordinate=torch.stack(((ndc[...,0]+1)*size/2,(1-ndc[...,1])*size/2),-1)-.5
    centre=coordinate.floor().long();fraction=coordinate-centre
    axis=torch.arange(0,2,device=points.device)
    ox,oy=torch.meshgrid(axis,axis,indexing='xy');offsets=torch.stack((ox.flatten(),oy.flatten()),-1)
    pixels=centre[:,:,None,:]+offsets
    wx=(1-(fraction[...,0,None]-offsets[:,0]).abs()).clamp_min(0)
    wy=(1-(fraction[...,1,None]-offsets[:,1]).abs()).clamp_min(0)
    px=pixels[...,0].clamp(0,size-1);py=pixels[...,1].clamp(0,size-1)
    sx=2*(px.to(points.dtype)+.5)/size-1;sy=1-2*(py.to(points.dtype)+.5)/size
    # Transform the receiver plane once. Perspective maps store scaled
    # reciprocal view depth near zero, retaining relative precision at contact.
    planes=torch.cat((geometric,-(geometric*points).sum(-1,keepdim=True)),-1)
    plane=planes[None]@inverse
    orthographic=-(plane[...,0,None]*sx+plane[...,1,None]*sy+plane[...,3,None])/torch.where(plane[...,2,None].abs()>1e-12,plane[...,2,None],1.)
    views=torch.stack([e['view'] for e in entries]);origins=torch.stack([e['origin'] for e in entries])
    n_view=geometric[None]@views[:,:3,:3].transpose(-1,-2)
    distance=((origins[:,None]-points[None])*geometric[None]).sum(-1)
    pxscale=torch.stack([e['projection'][0,0] for e in entries])[:,None,None]
    pyscale=torch.stack([e['projection'][1,1] for e in entries])[:,None,None]
    near=points.new_tensor([e['camera'].near for e in entries])[:,None,None]
    denominator=n_view[...,0,None]*sx/pxscale+n_view[...,1,None]*sy/pyscale-n_view[...,2,None]
    reciprocal=near*denominator/torch.where(distance.abs()>1e-12,distance,1.)[...,None]
    is_inverse=torch.tensor([e['inverse_depth'] for e in entries],device=points.device)[:,None,None]
    reference=torch.where(is_inverse,reciprocal,orthographic)
    bias=16*torch.finfo(points.dtype).eps*torch.where(is_inverse,reference.abs(),1+reference.abs())
    if curvature is not None:
        # Quadratic receiver uncertainty, estimated from geometric vertex
        # normals before any normal map. Flat surfaces retain precision bias.
        curvature=torch.as_tensor(curvature,device=points.device,dtype=points.dtype).expand(len(points))
        scale=(1+n_view[...,:2].square().sum(-1)/n_view[...,2].square().clamp_min(.01)).clamp_max(16)
        pixel=2*torch.where(is_inverse[...,0],w,torch.ones_like(w))/torch.minimum(pxscale,pyscale)[...,0]/size
        error=curvature[None]*pixel.square()*scale
        inv_extra=reference.abs()/distance.abs().clamp_min(1e-12)[...,None]*error[...,None]
        depth_scale=matrices[:,:3,:3][:,2].norm(dim=-1)[:,None]
        ortho_extra=error*depth_scale/n_view[...,2].abs().clamp_min(1e-3)
        gradient=torch.where(is_inverse[...,0],(near[...,0]/distance).abs()*(n_view[...,0].abs()/pxscale[...,0]+n_view[...,1].abs()/pyscale[...,0]),(plane[...,0].abs()+plane[...,1].abs())/plane[...,2].abs().clamp_min(1e-12))*2/size
        cap=.25*gradient[...,None]+.02*torch.where(is_inverse,reference.abs()*pixel[...,None]/w.abs().clamp_min(1e-12)[...,None],(pixel*depth_scale)[...,None])
        bias=bias+torch.minimum(torch.where(is_inverse,inv_extra,ortho_extra[...,None]),cap)
    batches=torch.arange(len(entries),device=points.device)[:,None,None]
    compare=(reference<=depths[batches,py,px]+bias).to(points.dtype)
    value=(compare*wx*wy).sum(-1).clamp(0,1)
    if any('stratum' in e for e in entries):
        zero_s=points.new_zeros(6)
        stratum=torch.stack([e.get('stratum',zero_s).to(points.dtype) for e in entries])
        has_s=is_inverse[...,0]&(stratum[:,:4].abs().sum(-1)>0)[:,None]
        if curvature is not None:has_s=has_s&(curvature[None]<=0)
        if bool(has_s.any()):
            ux,uy,vx,vy,ox_s,oy_s=[stratum[:,k][:,None] for k in range(6)]
            x0=coordinate[...,0];y0=coordinate[...,1]
            w_safe=w.clamp_min(1e-12);ref0=-near[...,0]/w_safe;eta_mid=1./w_safe
            safe_dist=torch.where(distance.abs()>1e-12,distance,1.)
            cx_s=torch.where(distance.abs()>1e-12,near[...,0]*n_view[...,0]/(pxscale[...,0]*safe_dist),0.)
            cy_s=torch.where(distance.abs()>1e-12,near[...,0]*n_view[...,1]/(pyscale[...,0]*safe_dist),0.)
            cz_s=torch.where(distance.abs()>1e-12,-near[...,0]*n_view[...,2]/safe_dist,0.)
            eta_sum=torch.zeros_like(x0);eta_w=torch.zeros_like(x0);map_idx=torch.arange(len(entries),device=points.device)[:,None]
            for iu in (-2,-1,0,1,2):
                for iv in (-2,-1,0,1,2):
                    su=iu*.625;sv=iv*.625
                    sx_i=torch.floor(x0+eta_mid*(ox_s+su*ux+sv*vx)+.5).long().clamp(0,size-1)
                    sy_i=torch.floor(y0+eta_mid*(oy_s+su*uy+sv*vy)+.5).long().clamp(0,size-1)
                    zs=depths[map_idx,sy_i,sx_i]
                    refs=cx_s*(2*(sx_i.to(points.dtype)+.5)/size-1)+cy_s*(1-2*(sy_i.to(points.dtype)+.5)/size)+cz_s
                    ec=(ref0-zs)/near[...,0]
                    # Relative reciprocal-depth separation is invariant to the
                    # shadow camera near plane and the scene length unit.
                    receiver_limit=torch.minimum(ref0,refs)
                    blocker_margin=2e-3*receiver_limit.abs()
                    hit=has_s&torch.isfinite(zs)&(zs<receiver_limit-blocker_margin)&(ec>0)
                    eta_sum=eta_sum+torch.where(hit,ec,0.);eta_w=eta_w+hit.to(points.dtype)
            active_s=eta_w>0
            if bool(active_s.any()):
                eta=torch.where(active_s,eta_sum/eta_w.clamp_min(1.),0.);vsum=torch.zeros_like(x0)
                for ku in range(6):
                    for kv in range(6):
                        xu=(ku+.5)/6.-.5;xv=(kv+.5)/6.-.5
                        tx=x0+eta*(ox_s+xu*ux+xv*vx);ty=y0+eta*(oy_s+xu*uy+xv*vy)
                        tix=torch.floor(tx);tiy=torch.floor(ty);tfx=tx-tix;tfy=ty-tiy
                        tcells=torch.stack((tix.long(),tiy.long()),-1)[:,:,None,:]+offsets
                        tpx=tcells[...,0].clamp(0,size-1);tpy=tcells[...,1].clamp(0,size-1)
                        twx=(1-(tfx[...,None]-offsets[:,0]).abs()).clamp_min(0)
                        twy=(1-(tfy[...,None]-offsets[:,1]).abs()).clamp_min(0)
                        tref=cx_s[...,None]*(2*(tpx.to(points.dtype)+.5)/size-1)+cy_s[...,None]*(1-2*(tpy.to(points.dtype)+.5)/size)+cz_s[...,None]
                        tbias=16*torch.finfo(points.dtype).eps*tref.abs()
                        tcmp=(tref<=depths[batches,tpy,tpx]+tbias).to(points.dtype)
                        vsum=vsum+(tcmp*twx*twy).sum(-1)*(1./36.)
                value=torch.where(active_s,vsum.clamp(0,1),value)
    return torch.where(inside,value,1.).T


def _receiver_curvature(corners,normals):
    """Frobenius bound of affine geometric-normal variation per world unit.

    Edgewise slopes can underestimate two-dimensional normal variation.
    The inverse tangent metric gives ||dN/dX||_F, bounding every direction.
    Use cross-product area for the determinant, stable for skinny triangles.
    This remains an estimate of geometric curvature, before normal textures.
    """
    e1=corners[:,1]-corners[:,0];e2=corners[:,2]-corners[:,0]
    d1=normals[:,1]-normals[:,0];d2=normals[:,2]-normals[:,0]
    a=e1.square().sum(-1);b=(e1*e2).sum(-1);c=e2.square().sum(-1)
    numerator=c*d1.square().sum(-1)+a*d2.square().sum(-1)-2*b*(d1*d2).sum(-1)
    determinant=torch.linalg.cross(e1,e2).square().sum(-1)
    return (numerator.clamp_min(0)/determinant.clamp_min(1e-30)).sqrt()


def _rectangle_samples(center,u,v,radiance,count):
    from ._area import _samples
    return _samples(center,u,v,radiance,count,stratified=True)


def _attach_stratum_meta(entry,area,index):
    if not entry.get('inverse_depth',False):return entry
    axis=int(area['samples']);size=int(entry['depth'].shape[-1]);half=.5*size
    u=area['u']/axis;v=area['v']/axis
    multiplier=next(k for k in (5,3,7,11,1)if math.gcd(k,axis)==1);offset=(multiplier+1)//2
    row=index//axis;col=index%axis
    du=(((multiplier*col+2*row+offset)%axis)+.5)/axis
    dv=(((multiplier*row+2*col+offset)%axis)+.5)/axis
    ou=.5-du;ov=.5-dv
    view=entry['view'];proj=entry['projection'];p00=proj[0,0];p11=proj[1,1]
    ux=(view[0,:3]*u).sum()*p00*half;uy=-(view[1,:3]*u).sum()*p11*half
    vx=(view[0,:3]*v).sum()*p00*half;vy=-(view[1,:3]*v).sum()*p11*half
    ox=ou*ux+ov*vx;oy=ou*uy+ov*vy
    entry['stratum']=torch.stack((ux,uy,vx,vy,ox,oy))
    return entry


def _light_sources(packed,vertices,faces,size,shadows,callback):
    result=[]
    # One slab prevents duplicating every depth map during native packet packing.
    map_count=0
    if shadows:
        for lamp in packed.lights:
            par=lamp.parameters
            if lamp.evaluator is light.rectangle:map_count+=par['samples']**2
            elif lamp.evaluator in (light.point,spot):map_count+=par['positions'].numel()//3*6
            elif lamp.evaluator is light.directional:map_count+=par['directions'].numel()//3*(16 if float(par.get('angle',0)) else 1)
    slab=vertices.new_empty((map_count,size,size));slot=0
    def retain(entries):
        nonlocal slot
        for entry in entries:
            slab[slot].copy_(entry['depth']);entry['depth']=slab[slot];slot+=1
        return entries
    from ._area import _samples
    for lamp in packed.lights:
        params=dict(lamp.parameters);world=packed.world_matrices[lamp.name][0]
        def move(p):return p@world[:3,:3].T+world[:3,3]
        if lamp.evaluator is light.rectangle:
            centre=move(params['center']);u=params['u']@world[:3,:3].T;v=params['v']@world[:3,:3].T
            positions,normal,intensity=_rectangle_samples(centre,u,v,params['radiance'],params['samples'])
            area=dict(name=lamp.name,center=centre,u=u,v=v,radiance=params['radiance'],samples=params['samples'])
            # Rectangles emit into one hemisphere. Cull other geometry only in
            # this light's shadow map; primary geometry remains unchanged.
            vertex_plane=((vertices-centre)*normal).sum(-1)
            selected=torch.index_select(vertex_plane,0,faces.reshape(-1)).reshape(-1,3).amax(-1)>-1e-5
            indices=faces if bool(selected.all())else faces[selected]
            low,high=vertices.amin(0),vertices.amax(0);radius=float(torch.linalg.vector_norm(high-low))
            for i,source in enumerate(positions):
                entries=[]
                if shadows:
                    cam=_camera(source,source+normal,size,fov=140.,near=.0001,far=radius*4+float(torch.linalg.vector_norm(source)))
                    if i==0:
                        caster_vertices,caster_faces=_shadow_geometry(vertices,indices,cam);fitted_near=cam.near
                    else:cam.near=fitted_near
                    entries=[_attach_stratum_meta(_make_map(caster_vertices,caster_faces,cam,prepared=True),area,i)]
                result.append(dict(kind='point',position=source,intensity=intensity,normal=normal,maps=retain(entries),area=area,sample_index=i))
                _notify(callback,stage='shadow maps',light=lamp.name,completed=i+1,total=len(positions))
        elif lamp.evaluator in (light.point,spot):
            positions=move(params['positions'].reshape(-1,3));power=params['intensity'].reshape(-1,3)
            for i,source in enumerate(positions):
                item=dict(kind='point',position=source,intensity=power[i],maps=retain(_maps(vertices,faces,source,size=size)) if shadows else [])
                if lamp.evaluator is spot:
                    item.update(kind='spot',direction=unit(params['direction']@world[:3,:3].T),angle=params['angle'],blend=params['blend'])
                result.append(item)
        elif lamp.evaluator is light.directional:
            directions=unit(params['directions'].reshape(-1,3)@world[:3,:3].T)
            for direction,irradiance in zip(directions,params['irradiance'].reshape(-1,3)):
                angle=float(params.get('angle',0));directions_to_trace=[direction]
                if angle:
                    up=direction.new_tensor([0,0,1] if abs(float(direction[2]))<.9 else [0,1,0]);tangent=unit(torch.linalg.cross(direction,up));bitangent=torch.linalg.cross(direction,tangent)
                    directions_to_trace=[unit(direction+math.tan(angle/2)*math.sqrt((i+.5)/16)*(math.cos(i*2.39996323)*tangent+math.sin(i*2.39996323)*bitangent)) for i in range(16)]
                for d in directions_to_trace:
                    result.append(dict(kind='sun',direction=d,irradiance=irradiance/len(directions_to_trace),maps=retain(_maps(vertices,faces,d,directional=True,size=size)) if shadows else []))
        else:raise NotImplementedError('raster integrator requires rectangle, point, spot or sun lights')
    return result


def _uv_setup(vertices,faces,clip,camera,uv):
    chosen=clip[faces.long()];xy=chosen[...,:2]/chosen[...,3:]
    xy=torch.stack(((xy[...,0]+1)*camera.image_width/2,(1-xy[...,1])*camera.image_height/2),-1)
    e1=xy[:,1]-xy[:,0];e2=xy[:,2]-xy[:,0];det=e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0]
    safe=torch.where(det.abs()>1e-12,det,1.)
    gx=torch.stack((e1[:,1]-e2[:,1],e2[:,1],-e1[:,1]),-1)/safe[:,None]
    gy=torch.stack((e2[:,0]-e1[:,0],-e2[:,0],e1[:,0]),-1)/safe[:,None]
    invw=1/clip[:,3][faces.long()]
    return gx,gy,invw


def _texture(source,uv,dx,dy):
    lod=source.lod
    if lod is None:
        scale=uv.new_tensor([source.width,source.height])
        rho=torch.maximum(torch.linalg.vector_norm(dx*scale,dim=-1),torch.linalg.vector_norm(dy*scale,dim=-1))
        lod=torch.log2(rho.clamp_min(1)).clamp_max(len(source.level_offsets)-2)
    if not isinstance(lod,torch.Tensor):lod=uv.new_tensor(lod)
    if lod.ndim==0:lod=lod.expand(len(uv))
    value=native.texture(source.pyramid,source.level_offsets,source.height,source.width,
                         uv.contiguous(),channel_count=source.channels,lod=lod.contiguous(),
                         filter=source.filter,mip_filter=native.MIP_LINEAR,
                         wrap_u=source.wrap_u,wrap_v=source.wrap_v,border=source.border)
    return value[:,0] if source.scalar else value


def render(scene,camera,*,supersample=1,shadows=False,shadow_map_size=512,
           background=None,outputs=('colour','mask'),fragment_block=262144,progress=None,environment=None,probes=(),irradiance=None,mirrors=(),
           caustics=None,_context=None,_clip_plane=None):
    """Render one frame, forward-only, using native gffx visibility.

    Gradients remain available in the existing deferred integrator. This explicit
    raster route does not silently discard a requested autograd graph.
    """
    # Opaque scenes retain the existing raster path exactly. A dielectric
    # requires geometric secondary visibility after each change of direction.
    from ._dielectric import dielectric, validate_parameters as validate_dielectric
    dielectric_names={prim.name for prim in scene.prims if isinstance(prim,types.Material) and prim.evaluator is dielectric}
    if any(isinstance(prim,types.Mesh) and prim.material in dielectric_names for prim in scene.prims):
        from ._dielectric_render import render as dielectric_render
        return dielectric_render(scene,camera,supersample=supersample,shadows=shadows,
            shadow_map_size=shadow_map_size,background=background,outputs=outputs,
            fragment_block=fragment_block,progress=progress,environment=environment,
            probes=probes,irradiance=irradiance,mirrors=mirrors,
            caustics=caustics,_context=_context,_clip_plane=_clip_plane)
    if caustics is not None:raise ValueError('plane caustics require an active dielectric mesh')
    if type(supersample) is not int or not 1<=supersample<=16:raise ValueError('supersample must be an integer in [1,16]')
    if type(fragment_block) is not int or fragment_block<1:raise ValueError('fragment_block must be positive')
    if type(shadow_map_size) is not int or not 16<=shadow_map_size<=4096:raise ValueError('shadow_map_size must be in [16,4096]')
    if set(outputs)-{'colour','mask','depth','face_index','timings'}:raise ValueError('unsupported raster output')
    profile='timings' in outputs
    def sync():
        if profile and torch.cuda.is_available() and resolved.pose.is_cuda:torch.cuda.synchronize(resolved.pose.device)
    from ._deferred import _resolve_camera
    resolved,_=_resolve_camera(scene,camera)
    sync();job_start=time.perf_counter()
    cache={}if _context is None else _context
    key=tuple(id(prim)for prim in scene.prims)
    if _context is not None and not isinstance(camera,str) and key in cache.get('packed',{}):packed=cache['packed'][key]
    else:
        packed,camera=_prepare(scene,camera)
        cache.setdefault('packed',{})[key]=packed
    if camera.batch_size!=1:raise ValueError('raster integrator accepts one frame at a time; reuse the scene for animation')
    tracked=[packed.vertices,camera.fx,camera.fy,camera.cx,camera.cy,camera.pose]
    tracked.extend(a.values for a in packed.attributes.values())
    tracked.extend(packed.world_matrices.values())
    for mat in packed.materials:
        tracked.extend(v for v in mat.parameters.values() if isinstance(v,torch.Tensor))
        for value in mat.parameters.values():
            if isinstance(value,TextureSource):
                tracked.append(value.pyramid)
                if isinstance(value.lod,torch.Tensor):tracked.append(value.lod)
        if mat.emission is not None:tracked.append(mat.emission)
        if mat.evaluator is pbr:validate_material(mat.parameters)
        elif mat.evaluator is dielectric:validate_dielectric(mat.parameters)  # unused material only
        elif mat.evaluator is not material.classic:raise NotImplementedError('raster surface must be Material.pbr, diffuse or glossy')
    for lamp in packed.lights:tracked.extend(v for v in lamp.parameters.values() if isinstance(v,torch.Tensor))
    for obj in [environment,*[p.environment for p in probes if p.environment is not None]]:
        if obj is None:continue
        for tensor in [*obj.levels,obj.irradiance]:
            if tensor.dtype!=packed.vertices.dtype or tensor.device!=packed.vertices.device:raise ValueError('lighting image dtype/device must match scene; move explicitly')
            tracked.append(tensor)
    if irradiance is not None:
        if irradiance.coefficients.dtype!=packed.vertices.dtype or irradiance.coefficients.device!=packed.vertices.device:raise ValueError('irradiance dtype/device must match scene')
        tracked.append(irradiance.coefficients)
    for probe in probes:
        if isinstance(probe.position,torch.Tensor):tracked.append(probe.position)
    if torch.is_grad_enabled() and any(t.requires_grad for t in tracked):
        raise ValueError('raster integrator is forward-only; use integrator=deferred for gradients, or torch.no_grad() explicitly')
    with torch.no_grad():
        start=time.perf_counter();context={}if _context is None else _context
        full=packed
        from . import _raster_accel
        # A capture may exclude geometry that remains visible in its parent.
        # Share maps across cameras only when caster/light inputs and shadow
        # configuration are identical; otherwise exclusions leave ghost shadows.
        lightkey=(key,shadows,shadow_map_size)
        lighting=context.setdefault('lighting',{})
        if lightkey not in lighting:
            # Parent render locals retain their active packet. Only the latest
            # child caster set needs a reusable cache across its capture faces.
            lighting.clear()
            sources=_light_sources(packed,packed.vertices,packed.faces,shadow_map_size,shadows,progress)
            accelerated=_raster_accel.available(packed.vertices.device) and packed.vertices.dtype==torch.float32
            data=_raster_accel.pack_sources(sources,packed.vertices,shadow_map_size)if accelerated else None
            lighting[lightkey]=(sources,data)
        sources,source_data=lighting[lightkey];accelerated=source_data is not None
        sync();preparation_end=time.perf_counter()
        captured=[]
        for probe in probes:
            captured.append(probe if probe.environment is not None else probe.capture(scene,environment=environment,irradiance=irradiance,shadows=shadows,shadow_map_size=shadow_map_size,_context=context))
            _notify(progress,stage='reflection captures',completed=len(captured),total=len(probes))
        probes=tuple(captured);planar=[]
        for mirror in mirrors:
            planar.append((mirror,mirror.capture(scene,camera,supersample=supersample,shadows=shadows,shadow_map_size=shadow_map_size,environment=environment,probes=probes,irradiance=irradiance,_context=context)))
            _notify(progress,stage='planar captures',completed=len(planar),total=len(mirrors))
        indirect_data=None
        if accelerated and (environment is not None or probes or irradiance is not None or planar):
            cachekey=(id(environment),tuple(id(p.environment)for p in probes),id(irradiance),tuple(id(c['image'])for _,c in planar))
            indirect_cache=context.setdefault('indirect',{})
            if cachekey not in indirect_cache:indirect_cache[cachekey]=_raster_accel.pack_indirect(environment,probes,irradiance,planar,packed.vertices)
            indirect_data=indirect_cache[cachekey]
        if captured or planar:
            # Child capture maps are no longer read once their images exist.
            # Keep the parent's active sources/data alive in local variables.
            context.pop('lighting',None)
        sync();capture_end=time.perf_counter()
        from ._raster_clip import clip as clip_scene
        packed,original_faces=clip_scene(packed,camera,_clip_plane)
        v,f=packed.vertices,packed.faces
        normals=packed.attributes.get('normals')
        normal_values=normals.values if normals is not None else None
        uv_attr=packed.attributes.get('uv');uv_values=uv_attr.values if uv_attr is not None else None
        if not accelerated:fragment_block=min(fragment_block,8192)
        h,w=camera.image_height,camera.image_width;count=h*w
        colour=v.new_zeros((count,3));mask=v.new_zeros(count);extra={}
        bg=v.new_zeros(3) if background is None else torch.as_tensor(background,dtype=v.dtype,device=v.device)
        if bg.shape!=(3,):raise ValueError('background must be RGB')
        has_subsurface=any('subsurface' in m.parameters and bool((packed.face_material==i).any()) for i,m in enumerate(packed.materials))
        if has_subsurface and (v.dtype!=torch.float32 or not accelerated):raise NotImplementedError('raster subsurface currently requires native float32 CPU/CUDA')
        if has_subsurface:
            from . import _subsurface
            diffuse_buffer=v.new_zeros((count,3));irradiance_buffer=v.new_zeros((count,3));base_buffer=v.new_zeros((count,3))
            sss_data=v.new_zeros((count,6));sss_ids=torch.full((count,),-1,dtype=torch.int32,device=v.device)
        total=supersample**2
        # Material compaction avoids a CPU/GPU predicate for every material in
        # every scanline block. Only textured surfaces need UV derivatives.
        for sample in range(total):
            row,column=divmod(sample,supersample);dx=.5-(column+.5)/supersample;dy=.5-(row+.5)/supersample
            index,beta,depth,clip=_raster(v,f,camera,dx=dx,dy=dy)
            valid=index>=0;mask+=valid.to(v.dtype)/total
            if environment is not None and background is None:
                from ._raster_lighting import _sample_cube
                sky=torch.nonzero(~valid).flatten()
                if len(sky):
                    xy=torch.stack(((sky%w+.5-dx-camera.cx[0])/camera.fx[0],-(sky//w+.5-dy-camera.cy[0])/camera.fy[0],-torch.ones_like(sky,dtype=v.dtype)),-1)
                    colour[sky]+=environment.sample(unit(xy@camera.pose[0,:3,:3].T))/total
            else:colour+=(~valid)[:,None]*bg/total
            material_ids=torch.where(valid,packed.face_material[index.clamp_min(0).long()],-1)if len(f)else torch.full_like(index,-1)
            for m,mat in enumerate(packed.materials):
                all_positions=torch.nonzero(material_ids==m).flatten()
                textured=any(isinstance(x,TextureSource) for x in mat.parameters.values())
                for begin in range(0,len(all_positions),fragment_block):
                    positions=all_positions[begin:begin+fragment_block]
                    face=index[positions].long();affine=beta[positions];local_faces=f[face]
                    corners=v[local_faces.long()];edge1=corners[:,1]-corners[:,0];edge2=corners[:,2]-corners[:,0]
                    geo=unit(torch.linalg.cross(edge1,edge2))
                    iw=1/clip[:,3][local_faces.long()]
                    den=(affine*iw).sum(-1,keepdim=True);weights=affine*iw/den
                    points=(corners*weights[...,None]).sum(1)
                    local_normals=normal_values[face]if normal_values is not None else geo[:,None,:].expand(-1,3,-1)
                    curvature=_receiver_curvature(corners,local_normals)
                    norm=unit((local_normals*weights[...,None]).sum(1))
                    vw=unit(camera.pose[0,:3,3]-points)
                    params=dict(mat.parameters)
                    if textured:
                        local_uv=uv_values[face]if uv_values is not None else v.new_zeros((len(face),3,2))
                        uv=(local_uv*weights[...,None]).sum(1)
                        gx,gy,_=_uv_setup(v,local_faces,clip,camera,None)
                        weighted_uv=iw[...,None]*local_uv;num=(affine[...,None]*weighted_uv).sum(1)
                        derivatives=[]
                        for grad in [gx,gy]:
                            dn=(grad[...,None]*weighted_uv).sum(1);dd=(grad*iw).sum(-1,keepdim=True)
                            derivatives.append((dn*den-num*dd)/den.square())
                        params={key:(_texture(value,uv,derivatives[0],derivatives[1]) if isinstance(value,TextureSource) else value) for key,value in params.items()}
                    scattering={key:params.pop(key)for key in ['subsurface','subsurface_radius','subsurface_transmission']if key in params}
                    normal_space=params.pop('normal_space','tangent')
                    normal_map=params.pop('normal_map',None)
                    if normal_map is not None:
                        u1=local_uv[:,1]-local_uv[:,0];u2=local_uv[:,2]-local_uv[:,0]
                        determinant=u1[:,0]*u2[:,1]-u1[:,1]*u2[:,0]
                        tangent_value=unit((edge1*u2[:,1:]-edge2*u1[:,1:])/torch.where(determinant.abs()>1e-12,determinant,1.)[:,None])
                        tangent_value=unit(tangent_value-norm*(norm*tangent_value).sum(-1,keepdim=True))
                        bitangent=torch.linalg.cross(norm,tangent_value)*torch.where(determinant<0,-1.,1.)[:,None]
                        local=normal_map*2-1
                        norm=unit(local)if normal_space=="world"else unit(tangent_value*local[:,:1]+bitangent*local[:,1:2]+norm*local[:,2:])
                        norm=torch.where(((norm*geo).sum(-1)>0)[:,None],norm,geo)
                    value=_raster_accel.shade(points,norm,geo,vw,params,source_data,shadow_map_size,classic=mat.evaluator is material.classic,curvature=curvature) if accelerated else None
                    if value is None:
                        value=torch.zeros_like(points)
                        evaluated=sources
                        if mat.evaluator is pbr:
                            from ._ltc import evaluate
                            evaluated=[s for s in sources if 'area' not in s]
                            for source in sources:
                                if 'area' not in source or source['sample_index']!=0:continue
                                area=source['area'];group=[s for s in sources if s.get('area') is area]
                                visibility=None
                                if shadows:
                                    numerator=points.new_zeros(len(points));denominator=numerator.clone()
                                    for item in group:
                                        delta=item['position']-points;d2=delta.square().sum(-1);direction=unit(delta)
                                        weight=(norm*direction).sum(-1).clamp_min(0)*(-(direction*item['normal']).sum(-1)).clamp_min(0)/d2.clamp_min(1e-12)
                                        denominator+=weight;numerator+=weight*_shadow(item['maps'],points,geo,curvature)
                                    visibility=numerator/denominator.clamp_min(1e-20)
                                value+=evaluate(points,norm,geo,vw,params,area,visibility)
                        for first_source in range(0,len(evaluated),8):
                            group=evaluated[first_source:first_source+8];directions=[];weights=[]
                            for source in group:
                                if source['kind']=='sun':
                                    direction=source['direction'].expand(len(points),3);incoming=source['irradiance'].expand(len(points),3)
                                else:
                                    offset=source['position']-points;distance=offset.square().sum(-1)
                                    direction=unit(offset);incoming=source['intensity']/distance.clamp_min(1e-12)[:,None]
                                    if 'normal' in source:incoming=incoming*(-(direction*source['normal']).sum(-1)).clamp_min(0)[:,None]
                                    if source['kind']=='spot':
                                        cosine=-(direction*source['direction']).sum(-1);outer=math.cos(source['angle']/2);inner=outer+(1-outer)*source['blend']
                                        t=((cosine-outer)/max(inner-outer,1e-12)).clamp(0,1) if source['blend'] else (cosine>=outer).to(points.dtype)
                                        incoming=incoming*(t*t*(3-2*t))[:,None]
                                directions.append(direction);weights.append(incoming)
                            direction=torch.stack(directions,1);incoming=torch.stack(weights,1)
                            if shadows:
                                if all(len(s['maps'])==1 for s in group):visibility=_shadow_group([s['maps'][0] for s in group],points,geo,curvature)
                                else:visibility=torch.stack([_shadow(s['maps'],points,geo,curvature) if s['maps'] else torch.ones(len(points),device=points.device) for s in group],1)
                                incoming=incoming*visibility[...,None]
                            brdf=(pbr(norm,vw,direction,_geometric_normals=geo,**params) if mat.evaluator is pbr else
                                  material.classic(norm,vw,direction,_geometric_normals=geo,**params))
                            value+=(brdf*incoming*(norm[:,None,:]*direction).sum(-1).clamp_min(0)[...,None]).sum(1)
                    if environment is not None or probes or irradiance is not None or planar:
                        from ._raster_lighting import surface
                        mirror_entries=[]
                        for descriptor,capture in planar:
                            if descriptor.mesh not in full.mesh_names:raise ValueError('unknown planar mirror mesh '+descriptor.mesh)
                            mi=full.mesh_names.index(descriptor.mesh);lo=full.object_face_offsets[0,mi];hi=full.object_face_offsets[0,mi+1]
                            selected=(original_faces[face]>=lo)&(original_faces[face]<hi)
                            if bool(selected.any()):mirror_entries.append(dict(selected=selected,capture=capture))
                        indirect_value=None
                        if indirect_data is not None:
                            ids=torch.full((len(points),),-1,device=points.device,dtype=torch.int32)
                            for mi,entry in enumerate(mirror_entries):
                                index_in_pack=next(i for i,(_,c)in enumerate(planar)if c is entry['capture']);ids[entry['selected']]=index_in_pack
                            indirect_value=_raster_accel.indirect(points,norm,vw,params,indirect_data,ids,classic=mat.evaluator is material.classic)
                        if indirect_value is None:indirect_value=surface(points,norm,vw,params,environment=environment,probes=probes,irradiance=irradiance,mirror=mirror_entries,classic=mat.evaluator is material.classic)
                        value+=indirect_value
                    if scattering:
                        from . import _subsurface
                        only_specular={**params,'base_color':torch.zeros_like(params['base_color'])}
                        specular=_raster_accel.shade(points,norm,geo,vw,only_specular,source_data,shadow_map_size,curvature=curvature)
                        if environment is not None or probes or irradiance is not None or planar:
                            if indirect_data is not None:specular+=_raster_accel.indirect(points,norm,vw,only_specular,indirect_data,ids)
                            else:specular+=surface(points,norm,vw,only_specular,environment=environment,probes=probes,irradiance=irradiance,mirror=mirror_entries)
                        diffuse_component=(value-specular).clamp_min(0)
                        base=torch.as_tensor(params['base_color'],device=v.device,dtype=v.dtype).expand(len(points),3)
                        diffuse_buffer[positions]+=diffuse_component/total
                        irradiance_buffer[positions]+=diffuse_component/base.clamp_min(1e-8)/total
                        base_buffer[positions]+=base/total
                        sss_data[positions,0]=-(points@camera.world_to_view()[0,:3,:3].T+camera.world_to_view()[0,:3,3])[:,2]
                        sss_data[positions,1:4]=_subsurface.diffusion_length(scattering['subsurface_radius'],base)
                        sss_data[positions,4]=scattering['subsurface']
                        sss_data[positions,5]+=1/total
                        owner=torch.bucketize(original_faces[face].contiguous(),full.object_face_offsets[0,1:],right=True).to(torch.int32)
                        sss_ids[positions]=owner
                        if shadows and bool(scattering.get('subsurface_transmission',1)>0):
                            value+=_subsurface.transmit(points,geo,{**params,**scattering},source_data,shadow_map_size)*scattering['subsurface']
                    if mat.emission is not None:value+=mat.emission
                    colour[positions]+=value/total
            _notify(progress,stage='shading',completed=sample+1,total=total,elapsed_seconds=time.perf_counter()-start)
        if has_subsurface:
            coverage=sss_data[:,5:6].clamp_min(1e-8)
            scattered=_subsurface.diffuse((irradiance_buffer/coverage).reshape(h,w,3),sss_data.reshape(h,w,6),sss_ids.reshape(h,w),camera.fx[0],camera.fy[0],perspective=camera.projection_kind=='perspective').reshape(-1,3)
            colour+=sss_data[:,4:5]*(scattered*base_buffer-diffuse_buffer)
        if set(outputs)&{'depth','face_index'}:
            index,_,depth,_=_raster(v,f,camera)
            if 'depth' in outputs:extra['depth']=depth.reshape(1,h,w)
            if 'face_index' in outputs:extra['face_index']=(torch.where(index>=0,original_faces[index.clamp_min(0).long()],-1)if len(original_faces)else index).reshape(1,h,w)
        sync();render_end=time.perf_counter()
        if profile:extra['timings']=dict(preparation_seconds=preparation_end-job_start,reflection_update_seconds=capture_end-preparation_end,primary_render_seconds=render_end-capture_end,total_seconds=render_end-job_start)
        result={'colour':colour.reshape(1,h,w,3),'mask':mask.reshape(1,h,w),**extra}
        if not bool(torch.isfinite(colour).all()):raise ValueError('raster colour is not finite')
        _notify(progress,stage='complete',elapsed_seconds=time.perf_counter()-start)
        return {key:result[key] for key in outputs if key in result}
