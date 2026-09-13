"""Fixed test-head image inputs via installed CPU gffx, independent scalar lighting.

Development tooling only. No public shader, independent rasterizer or principled
resource admission is implied. No implicit disk writes or service changes.
"""
import math
import time
import torch

from tools import shading_independent_evaluator as scalar
from tools import shading_environment_reference as environment

PROFILE='classic-images-v1'
SIZE=128
ALBEDOS=('constant','vertex','gradient','checker')


def _float(value):
    return torch.tensor(value,dtype=torch.float64,device='cpu')


def build_buffers(scene):
    from gffx.torch.camera import Camera
    from gffx.torch import mesh,render
    camera=scene['scene']['camera']
    expected=dict(position=[0.,0.,3.],target=[0.,0.,0.],up=[0.,1.,0.],
                  vertical_fov_degrees=40.,width=128,height=128)
    if camera!=expected or scene.get('id')!='synthetic_face_m1' or scene.get('recipe_version')!=1:
        raise ValueError('scene outside fixed image profile')
    if len(scene['vertices'])!=1986 or len(scene['triangles'])!=3968:
        raise ValueError('geometry outside fixed image profile')
    vertices=_float(scene['vertices']); faces=torch.tensor(scene['triangles'],dtype=torch.int32)
    pose=torch.eye(4,dtype=torch.float64)[None];pose[0,2,3]=3.
    focal=64/math.tan(math.radians(20))
    cam=Camera.from_intrinsics(focal,focal,64.,64.,128,128,.1,10.,pose=pose)
    projection=cam.project(vertices,in_front_only=True)
    if not projection.valid.all() or not torch.isfinite(projection.pixels).all():
        raise ValueError('invalid/inactive vertex projection')
    ndc=torch.stack((projection.pixels[:,0]/64-1,1-projection.pixels[:,1]/64,projection.ndc_depth),dim=-1).contiguous()
    face,bary,depth,_=render.rasterize(ndc,faces,128,128,faces_per_pixel=1,blur_radius_px=0.,cull_mode=render.CULL_BACK)
    mask=face.reshape(128,128)>=0
    if not mask.any() or mask.all():raise ValueError('coverage must contain both head and background')
    indices=face.clamp_min(0).long()
    corner_depth=projection.view_depth[faces.long()][indices]
    numerator=bary/corner_depth
    denominator=numerator.sum(-1,keepdim=True)
    corrected=torch.where(face[...,None]>=0,numerator/torch.where(denominator>0,denominator,torch.ones_like(denominator)),torch.zeros_like(numerator)).contiguous()
    smooth=mesh.vertex_normals(vertices,faces,weighting=mesh.WEIGHTING_AREA)
    if not torch.isfinite(smooth).all() or (smooth.norm(dim=-1)==0).any():raise ValueError('invalid smooth normal')
    attrs=torch.cat((vertices[faces.long()],smooth[faces.long()],_float(scene['corner_uvs']),
                     _float(scene['vertex_rgb'])[faces.long()]),dim=-1).contiguous()
    interpolated=render.interpolate(face,corrected,attrs).reshape(128,128,11)
    world=interpolated[...,:3].contiguous()
    normal=interpolated[...,3:6]
    lengths=normal.norm(dim=-1,keepdim=True)
    normal=torch.where(mask[...,None],normal/torch.where(lengths>0,lengths,torch.ones_like(lengths)),torch.zeros_like(normal)).contiguous()
    view=_float([0.,0.,3.])-world;view/=view.norm(dim=-1,keepdim=True)
    view=torch.where(mask[...,None],view,torch.zeros_like(view)).contiguous()
    uv=interpolated[...,6:8].contiguous()
    albedos=dict(constant=torch.where(mask[...,None],_float([.2,.4,.6]),torch.zeros_like(world)),
                 vertex=interpolated[...,8:11].contiguous())
    constant_corner=_float([.2,.4,.6]).expand(len(faces),3,3).contiguous()
    corner_result=render.interpolate(face,corrected,constant_corner).reshape(128,128,3)
    constant_error=float((corner_result-albedos['constant']).abs().max())
    texture_error=None
    for name in ('constant','gradient','checker'):
        texture=_float(scene['textures'][name])
        pyramid,offsets=render.texture_pyramid(texture,levels=1)
        result=render.texture(pyramid,offsets,32,32,uv.reshape(-1,2).contiguous(),channel_count=3,
            filter=render.FILTER_BILINEAR,mip_filter=render.MIP_NEAREST,wrap_u=render.WRAP_REPEAT,wrap_v=render.WRAP_REPEAT).reshape(128,128,3)
        result=torch.where(mask[...,None],result,torch.zeros_like(result))
        if name=='constant':texture_error=float((result-albedos['constant']).abs().max())
        else:albedos[name]=result
    result=dict(covered=mask,face=face.reshape(128,128),affine=bary.reshape(128,128,3),
                barycentric=corrected.reshape(128,128,3),world=world,normal=normal,view=view,uv=uv,
                depth=torch.where(mask,depth.reshape(128,128),torch.zeros((128,128),dtype=torch.float64)),
                vertex_pixels=projection.pixels,albedos=albedos,
                camera=dict(focal=focal,near=.1,far=10.,pose=pose.tolist(),projection=cam.projection().tolist(),world_to_clip=cam.world_to_clip().tolist()),
                controls=dict(constant_vertex_max_error=constant_error,constant_texture_max_error=texture_error))
    for name in ('world','normal','view','uv','barycentric','affine','depth'):
        if not torch.isfinite(result[name]).all() or (result[name][~mask]!=0).any():
            raise ValueError('nonfinite or nonzero-background '+name)
    return result


def reference_images(buffers,scene):
    mask=buffers['covered']; n=buffers['normal'][mask].tolist(); v=buffers['view'][mask].tolist()
    light=scene['scene']['directional_light']
    direction=scalar._normal(light['direction']); weight=light['irradiance']
    if weight!=[1.,1.,1.]:raise ValueError('light outside fixed image profile')
    result={}
    for name in ALBEDOS:
        colours=buffers['albedos'][name][mask].tolist()
        for model,specular in (('diffuse',0.),('classic',.1)):
            values=[]
            for normal,view,colour in zip(n,v,colours):
                brdf=scalar.classic(normal,view,direction,albedo=colour,specular=specular,exponent=2.)
                cosine=max(0.,scalar._dot(scalar._normal(normal),direction))
                values.append([x*y*cosine for x,y in zip(brdf,weight)])
            image=torch.zeros((128,128,3),dtype=torch.float64);image[mask]=_float(values)
            result[model+'/'+name]=image
        active=mask & ((buffers['normal']*buffers['view']).sum(-1)>0)
        result['environment_diffuse/'+name]=torch.where(active[...,None],buffers['albedos'][name],torch.zeros_like(buffers['albedos'][name]))
    return result


def environment_controls(buffers,*,deadline=None):
    active=buffers['covered'] & ((buffers['normal']*buffers['view']).sum(-1)>0)
    indices=active.flatten().nonzero().flatten().tolist()
    if len(indices)<8:raise ValueError('insufficient covered environment controls')
    selected=[indices[i*(len(indices)-1)//7] for i in range(8)]
    coefficients=[[math.sqrt(4*math.pi)]*3]+[[0.,0.,0.] for _ in range(8)]
    rows=[]
    for index in selected:
        seconds=30. if deadline is None else min(30.,deadline-time.perf_counter())
        if seconds<=0:raise TimeoutError('image environment deadline')
        n=buffers['normal'].reshape(-1,3)[index].tolist();v=buffers['view'].reshape(-1,3)[index].tolist()
        albedo=buffers['albedos']['gradient'].reshape(-1,3)[index].tolist()
        report=environment.environment_reference(n,v,coefficients,deadline_seconds=seconds,
            brdf=lambda normal,view,light:scalar.classic(normal,view,light,albedo=albedo,specular=0.,exponent=2.))
        rows.append(dict(pixel_index=index,status=report['status'],evaluations=report['evaluations'],
                         max_error=max(abs(a-b) for a,b in zip(report['radiance'],albedo)),history=report['history']))
    return rows


def compare_images(candidate,reference,covered):
    if any(not isinstance(x,torch.Tensor) or x.device.type!='cpu' for x in (candidate,reference,covered)):
        raise TypeError('CPU tensors required')
    if candidate.dtype!=torch.float64 or reference.dtype!=torch.float64 or covered.dtype!=torch.bool:
        raise TypeError('float64 images and boolean mask required')
    if candidate.shape!=reference.shape or reference.ndim!=3 or reference.shape[-1]!=3 or covered.shape!=reference.shape[:-1] or reference.numel()>128*128*3:
        raise ValueError('image/mask shape or size mismatch')
    if not torch.isfinite(candidate).all() or not torch.isfinite(reference).all():raise ValueError('nonfinite comparison image')
    pixels=int(covered.sum())
    if pixels==0:raise ValueError('empty comparison mask')
    delta=candidate[covered]-reference[covered]
    if not torch.isfinite(delta).all():raise ValueError('nonrepresentable image difference')
    mse=float((delta*delta).mean())
    if not math.isfinite(mse):raise ValueError('nonrepresentable mean squared error')
    if mse==0 and (delta!=0).any():raise ValueError('mean squared error underflow')
    background=bool((candidate[~covered]==0).all() and (reference[~covered]==0).all())
    exact=bool((delta==0).all()) and background
    return dict(passed=bool((delta.abs()<=1e-10+1e-8*reference[covered].abs()).all()) and background,
                covered_pixels=pixels,covered_channels=3*pixels,background_equal=background,
                signed_mean_error=float(delta.mean()),mae=float(delta.abs().mean()),
                max_absolute_error=float(delta.abs().max()),mse=mse,peak=1.,
                psnr_db=None if mse==0 else -10*math.log10(mse),exact_match=exact,
                unavailable_metrics=['SSIM','FLIP'])
