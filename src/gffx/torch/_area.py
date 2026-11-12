"""Generic sampled rectangle lighting and raster shadow-map orchestration.

Visibility is a discrete, detached shadow-map comparison. Geometry projection,
rasterization and camera inverses use existing gffx operations. Smooth light and
material paths remain differentiable; this module is not a path tracer.
"""
import torch


def _samples(center,u,v,radiance,count,*,stratified=False):
    if type(count) is not int or not 1<=count<=16:
        raise ValueError('rectangle samples must be an integer in [1,16] per axis')
    for name,value in [('center',center),('u',u),('v',v),('radiance',radiance)]:
        if not isinstance(value,torch.Tensor) or value.shape!=(3,):
            raise ValueError(name+' must be one xyz/RGB tensor')
        if value.dtype!=center.dtype or value.device!=center.device:
            raise ValueError('rectangle tensors must share dtype and device')
        if not bool(torch.isfinite(value).all()):raise ValueError('rectangle values must be finite')
    if not bool((radiance>=0).all()):raise ValueError('radiance must be nonnegative')
    cross=torch.linalg.cross(u,v);area=torch.linalg.vector_norm(cross)
    if not bool(area>0):raise ValueError('rectangle spans must not be parallel')
    normal=cross/area
    grid=(torch.arange(count,dtype=center.dtype,device=center.device)+.5)/count-.5
    a,b=torch.meshgrid(grid,grid,indexing='ij')
    if stratified:
        # Deterministic multi-jittered strata: one sample in every2D cell and
        # every1D sub-stratum. Complementary cells mirror around the emitter
        # centre. No per-frame randomness; camera movement cannot reseed it.
        import math
        multiplier=next(k for k in (5,3,7,11,1)if math.gcd(k,count)==1)
        offset=(multiplier+1)//2
        i,j=torch.meshgrid(torch.arange(count,device=center.device),torch.arange(count,device=center.device),indexing='ij')
        a=(i.to(center.dtype)+(((multiplier*j+2*i+offset)%count).to(center.dtype)+.5)/count)/count-.5
        b=(j.to(center.dtype)+(((multiplier*i+2*j+offset)%count).to(center.dtype)+.5)/count)/count-.5
    positions=center+a.reshape(-1,1)*u+b.reshape(-1,1)*v
    return positions,normal,radiance*(area/(count*count))


def rectangle_packet(points,*,center,u,v,radiance,samples=4):
    from . import light
    positions,normal,intensity=_samples(center,u,v,radiance,samples)
    packet=light.point(points,positions,intensity.expand(len(positions),3))
    cosine=(-(packet['direction']*normal).sum(-1)).clamp_min(0)
    return dict(measure='delta',direction=packet['direction'],valid=packet['valid'],
                irradiance=packet['irradiance']*cosine[...,None])


def expand_emitters(model):
    from . import scene,light
    prims=list(model.prims)
    for lamp in model.prims:
        if type(lamp) is not scene.Light or lamp.evaluator is not light.rectangle or not lamp.visible:continue
        p=lamp.parameters;mat_name=lamp.name+'::emission';mesh_name=lamp.name+'::surface'
        prims.append(scene.Material.emissive(mat_name,p['radiance']))
        prims.append(scene.Mesh.quad(mesh_name,center=p['center'],u=p['u'],v=p['v'],
                                    material=mat_name))
        from dataclasses import replace
        prims[-1]=replace(prims[-1],parent=lamp.name)
    return scene.Scene(prims) if len(prims)!=len(model.prims) else model


class _ShadowInverse:
    """Backend proxy for one private, immutable shadow camera in this render."""
    def __init__(self, backend, matrix, inverse):
        self.backend, self.matrix, self.value = backend, matrix, inverse

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def inverse(self, matrix):
        return self.value if matrix is self.matrix else self.backend.inverse(matrix)


def _map(packed,frame,source,size):
    from .camera import Camera
    from . import render
    # This camera is created from detached shadow geometry and never escapes
    # the render-local visibility cache. Reuse its fixed transforms across
    # materials and PCF taps; public/mutable cameras are never cached.
    class ShadowCamera(Camera):
        def world_to_view(self):
            return self._shadow_view

        def world_to_clip(self):
            return self._shadow_clip

        def freeze(self):
            self._shadow_view = super().world_to_view()
            self._shadow_clip = self.projection() @ self._shadow_view
            inverse = self._backend.inverse(self._shadow_clip)
            self._backend = _ShadowInverse(self._backend, self._shadow_clip, inverse)
            return self

    first,last=(int(x) for x in packed.vertex_offsets[frame:frame+2])
    f0,f1=(int(x) for x in packed.face_offsets[frame:frame+2])
    vertices=packed.vertices[first:last].detach()
    faces=(packed.faces[f0:f1]-first).contiguous()
    if not len(vertices):return None
    target=(vertices.amin(0)+vertices.amax(0))/2
    # A forward hemisphere projection is sufficient for one-sided rectangle
    # emitters. Point-light shadows use the same explicitly documented map cone.
    direction=target-source
    if not bool((direction!=0).any()):direction=source.new_tensor([0.,-1.,0.])
    up=(0,0,-1) if bool(direction[1].abs()>direction[[0,2]].abs().max()) else (0,1,0)
    scale=float((vertices.amax(0)-vertices.amin(0)).amax().clamp_min(.001))
    far=max(float((vertices-source).norm(dim=-1).max())*2,scale)
    cam=ShadowCamera.look_at(source.detach(),target,up=up,size=(size,size),fov=140.,
                             near=max(scale*.01,.0001),far=far).freeze()
    projected=cam.project(vertices,in_front_only=True)
    ndc=torch.stack((projected.pixels[:,0]*2/size-1,1-projected.pixels[:,1]*2/size,projected.ndc_depth),-1)
    keep=projected.valid[faces.long()].all(-1)
    indices,_,depth,_=render.rasterize(ndc.contiguous(),faces[keep].contiguous(),size,size,
                                      cull_mode=render.CULL_NONE)
    return cam,torch.where(indices[0,:,:,0]>=0,depth[0,:,:,0],float('inf')),source.detach()


@torch.no_grad()
def visibility(packed,frame,points,normals,source,size,cache,key):
    if key not in cache:cache[key]=_map(packed,frame,source.detach(),size)
    entry=cache[key]
    if entry is None:return points.new_ones(len(points))
    cam,depth,origin=entry
    points=points.detach();normals=normals.detach()
    projected=cam.project(points,in_front_only=True)
    # Bilinear PCF: a separable tent over 4x4 texel centres is the exact
    # bilinear interpolation of the previous 3x3 box filter. Comparison comes
    # before filtering; interpolation of raw depths would invent occluders.
    coordinate=projected.pixels-.5
    xy=coordinate.floor().to(torch.long)
    fraction=coordinate-xy
    axis=torch.arange(-1,3,device=points.device)
    ox,oy=torch.meshgrid(axis,axis,indexing='xy')
    offsets=torch.stack((ox.reshape(-1),oy.reshape(-1)),-1)
    pixels=xy[:,None,:]+offsets[None]
    # Sum of three adjacent bilinear footprints, each of width two.
    wx=sum((1-(fraction[:,0,None]+shift-offsets[None,:,0]).abs()).clamp_min(0) for shift in (-1,0,1))/3
    wy=sum((1-(fraction[:,1,None]+shift-offsets[None,:,1]).abs()).clamp_min(0) for shift in (-1,0,1))/3
    weights=wx*wy
    inside=(projected.valid[:,None]&(pixels[...,0]>=0)&(pixels[...,0]<size)
            &(pixels[...,1]>=0)&(pixels[...,1]<size))
    x=pixels[...,0].clamp(0,size-1);y=pixels[...,1].clamp(0,size-1)
    # Receiver-plane depth correction makes PCF compare each tap against the
    # same local surface, rather than mistaking surface slope for occlusion.
    # It is generic: no primitive names, sphere special cases or scene exceptions.
    grid=torch.stack((x.to(points.dtype)+.5,y.to(points.dtype)+.5),-1).reshape(-1,2)
    ray=cam.unproject_points(grid,points.new_ones(grid.shape[0]),kind='view').reshape(-1,16,3)-origin
    denominator=(normals[:,None,:]*ray).sum(-1)
    numerator=(normals*(points-origin)).sum(-1)[:,None]
    safe=denominator.abs()>1e-7
    distance=numerator/torch.where(safe,denominator,torch.ones_like(denominator))
    usable=safe&(distance>cam.near)&(distance<cam.far)
    reference=cam.view_depth_to_ndc(torch.where(usable,distance,torch.ones_like(distance)))
    reference=torch.where(usable,reference,projected.ndc_depth[:,None])
    # Receiver-plane correction already handles slope. A large constant NDC
    # bias erases real close-contact occluders (e.g. a1mm parallel sheet).
    # Cover rounding at the current precision instead of a fixed world gap.
    bias=16*torch.finfo(depth.dtype).eps*(1+reference.abs())
    lit=reference<=depth[y,x]+bias
    return (torch.where(inside,lit,True).to(points.dtype)*weights).sum(-1).clamp(0,1)


def packets(lamp,packed,frame,points,normals,*,shadows,size,cache,ordinary,sample_block=1):
    from . import light
    if lamp.evaluator is light.rectangle:
        p=lamp.parameters;world=packed.world_matrices[lamp.name][frame]
        center=p['center']@world[:3,:3].T+world[:3,3]
        u=p['u']@world[:3,:3].T;v=p['v']@world[:3,:3].T
        positions,normal,intensity= _samples(center,u,v,p['radiance'],p['samples'])
        for left in range(0,len(positions),sample_block):
            sources=positions[left:left+sample_block]
            data=light.point(points,sources,intensity.expand(len(sources),3))
            cosine=(-(data['direction']*normal).sum(-1)).clamp_min(0)
            incoming=data['irradiance']*cosine[...,None]
            if shadows:
                weights=[visibility(packed,frame,points,normals,source,size,cache,
                                    (frame,lamp.name,left+number))
                         for number,source in enumerate(sources)]
                incoming=incoming*torch.stack(weights,-1)[...,None]
            yield dict(measure='delta',direction=data['direction'],irradiance=incoming,valid=data['valid'])
        return
    packet=ordinary(lamp,packed,frame,points)
    if shadows:
        if lamp.evaluator is not light.point:
            raise NotImplementedError('shadow maps currently support point and rectangle lights')
        p=lamp.parameters;position=p['positions']
        if position.ndim==3:position=position[frame if len(position)>1 else 0]
        world=packed.world_matrices[lamp.name][frame]
        position=position@world[:3,:3].T+world[:3,3]
        weights=[]
        for number,source in enumerate(position):
            weights.append(visibility(packed,frame,points,normals,source,size,cache,(frame,lamp.name,number)))
        if weights:packet=dict(packet,irradiance=packet['irradiance']*torch.stack(weights,-1)[...,None])
    yield packet


def shadow_reserve(model,size):
    from . import scene,light
    count=0;first=None
    for p in model.prims:
        if type(p) is scene.Mesh:first=p.vertices if first is None else first
        if type(p) is scene.Light:
            if p.evaluator is light.rectangle:count+=p.parameters['samples']**2
            elif p.evaluator is light.point:count+=p.parameters['positions'].shape[-2]
            elif p.evaluator is not light.spherical_harmonics:
                raise NotImplementedError('shadow maps currently support point and rectangle lights')
    return count*size*size*(first.element_size() if first is not None else 4)
