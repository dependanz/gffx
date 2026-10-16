"""Eager deferred composition; see DEFERRED_ACCEPTANCE_V0_1.md.

No native algorithms, renderer cache, display conversion or implicit asset I/O.
Principled uses bounded first-order recomputation, not retained point graphs.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import math
from numbers import Real
import torch
from torch.autograd.function import once_differentiable
from . import render as primitives, transforms, mesh as geometry, scene as scene_types
from . import material as materials, light as lighting
from .camera import Camera
from ._reference_channels import sum3, divide, sqrt_unit
from ._reference_sigmoid import sigmoid

_OUTPUTS={'colour','mask','depth','world_position','normal','uv','uv_present',
          'vertex_colour','colour_present','face_index','material_index','barycentric','visibility'}


@dataclass(frozen=True)
class AttributeSource:
    name: str
    scalar: bool = False


@dataclass(frozen=True)
class TextureSource:
    pyramid: torch.Tensor
    level_offsets: torch.Tensor
    height: int
    width: int
    channels: int
    uv: str = 'uv'
    filter: int = primitives.FILTER_BILINEAR
    mip_filter: int = primitives.MIP_NEAREST
    wrap_u: int = primitives.WRAP_REPEAT
    wrap_v: int = primitives.WRAP_REPEAT
    lod: object = None
    border: object = None
    scalar: bool = False


def _finite(value,name):
    if not bool(torch.isfinite(value).all()):raise ValueError(name+' is not finite/representable')
    return value


def _unit(value,name):
    from ._reference_channels import unit_forward
    fused=unit_forward(value)
    if fused is not None:return fused
    _finite(value,name)
    scale=value.abs().amax(-1,keepdim=True)
    if bool((scale==0).any()):raise ValueError(name+' contains zero vectors')
    scaled=divide(value,scale)
    return _finite(divide(scaled,sqrt_unit(sum3(scaled.square())).unsqueeze(-1)),name)


def _real(value,name):
    if isinstance(value,bool) or not isinstance(value,Real):raise TypeError(name+' must be a static real')
    if not math.isfinite(value) or value<0:raise ValueError(name+' must be finite and nonnegative')
    return float(value)


def _camera(camera,batch,*,pose=None,factor=1):
    fields=[getattr(camera,k).expand(batch)*factor for k in ('fx','fy','cx','cy')]
    constructor=Camera.from_orthographic if camera.projection_kind=='orthographic' else Camera.from_intrinsics
    return constructor(*fields,camera.image_height*factor,camera.image_width*factor,
                                 camera.near,camera.far,pose=(camera.pose if pose is None else pose).expand(batch,4,4))


def _resolve_camera(scene,camera):
    if not isinstance(scene,scene_types.Scene):raise TypeError('scene must be Scene')
    binding=None
    if isinstance(camera,str):
        matches=[p for p in scene.prims if type(p)==scene_types.Camera and p.name==camera]
        if len(matches)!=1:raise ValueError('camera name must resolve exactly one Camera')
        binding=matches[0];camera=binding.camera
    if not isinstance(camera,Camera):raise TypeError('camera must be Camera or a scene camera name')
    return camera,binding


def _prepare(scene,camera):
    camera,binding=_resolve_camera(scene,camera)
    meshes=[p for p in scene.prims if type(p)==scene_types.Mesh]
    packed=scene.flatten(**({} if meshes else dict(dtype=camera.fx.dtype,device=camera.fx.device)))
    batch=max(len(packed.vertex_offsets)-1,camera.batch_size)
    if len(packed.vertex_offsets)-1 not in (1,batch) or camera.batch_size not in (1,batch):
        raise ValueError('scene and camera batches must singleton-broadcast')
    if len(packed.vertex_offsets)-1!=batch:
        packed=scene.flatten(batch_size=batch,dtype=packed.vertices.dtype,device=packed.vertices.device)
    if packed.vertices.device!=camera.fx.device:raise ValueError('camera must share scene device')
    pose=camera.pose.expand(batch,4,4)
    if binding is not None:pose=packed.world_matrices[binding.name] @ pose
    return packed,_camera(camera,batch,pose=pose)


def _plan_workload(scene,camera,factor,limit,outputs,background=None,soft=0.,cpu_workspace_allowance_bytes=67108864):
    """Conservative built-in estimate using shapes only, before scene packing.

    A trusted callback's unknown allocations are not certified by this estimate.
    Its public admission policy remains a separately reviewed contract boundary.
    """
    if type(limit)!=int:raise TypeError('workspace_limit_bytes must be integer bytes')
    if limit<=0:raise ValueError('workspace budget must be positive')
    camera,_=_resolve_camera(scene,camera)
    if type(cpu_workspace_allowance_bytes)!=int or not 0<cpu_workspace_allowance_bytes<=536870912:
        raise ValueError('explicit CPU workspace allowance must be in (0,512 MiB]')
    if camera.fx.device.type=='cpu':
        if limit>cpu_workspace_allowance_bytes:
            raise ValueError('CPU workspace budget exceeds explicit allowance (default64 MiB)')
    elif limit>67108864:
        if limit>torch.cuda.get_device_properties(camera.fx.device).total_memory:
            raise ValueError('workspace budget exceeds CUDA device capacity')
    # Larger CUDA limits are explicit host-admitted budgets. No ambient-memory
    # sampling or scheduling change occurs here; the complete estimate follows.
    meshes=[p for p in scene.prims if type(p)==scene_types.Mesh]
    mats=[p for p in scene.prims if type(p)==scene_types.Material]
    lamps=[p for p in scene.prims if type(p)==scene_types.Light]
    tensors=[];ordinary_tensors=[];sizes=[camera.batch_size];vertices=faces=channels=sources=0
    def inspect(value,*,immutable_budget=False):
        # An explicit stack avoids a recursive closure retaining tensor storage
        # until cyclic GC, which is unsafe for repeated bounded render calls.
        pending=[(value,immutable_budget)]
        while pending:
            value,immutable_budget=pending.pop()
            if isinstance(value,torch.Tensor):
                if value.layout!=torch.strided:raise TypeError('renderer inputs must be strided tensors')
                tensors.append(value)
                if not immutable_budget or value.requires_grad:ordinary_tensors.append(value)
            elif isinstance(value,TextureSource):
                pending.extend((field,False) for field in reversed((value.pyramid,value.level_offsets,value.lod,value.border)))
            elif isinstance(value,materials.PrincipledBudget):
                pending.extend((field,True) for field in reversed(tuple(value._data.values())))
            elif isinstance(value,Mapping):
                pending.extend((field,False) for field in reversed(tuple(value.values())))
    for field in (camera.fx,camera.fy,camera.cx,camera.cy,camera.pose,background):inspect(field)
    for prim in scene.prims:
        inspect(prim.transform)
        if isinstance(prim.transform,torch.Tensor) and prim.transform.ndim==3:sizes.append(prim.transform.shape[0])
        if type(prim)==scene_types.Mesh:
            v,f=prim.vertices,prim.faces
            if not isinstance(v,torch.Tensor):raise TypeError('vertices must be a tensor')
            if v.ndim not in (2,3) or v.shape[-1]!=3:
                raise ValueError('vertices must have shape [V,3] or [B,V,3]')
            if not isinstance(f,torch.Tensor):raise TypeError('faces must be a tensor')
            if f.ndim!=2 or f.shape[-1]!=3:
                raise ValueError('faces must have shape [F,3]')
            inspect(v);inspect(f);vertices+=v.shape[-2];faces+=len(f)
            if v.ndim==3:sizes.append(v.shape[0])
            for attribute in prim.attributes.values():
                if not isinstance(attribute,scene_types.Attribute):raise TypeError('attributes must be Attribute objects')
                inspect(attribute.values)
                if not isinstance(attribute.values,torch.Tensor) or attribute.values.ndim<1:raise TypeError('attribute values must be tensors')
                channels+=attribute.values.shape[-1]
                rank={'constant':1,'uniform':2,'vertex':2,'face_varying':3}.get(attribute.domain)
                if rank is not None and attribute.values.ndim==rank+1:sizes.append(attribute.values.shape[0])
        elif isinstance(prim,scene_types.Material):
            inspect(prim.parameters)
            sources+=sum(isinstance(v,(AttributeSource,TextureSource)) for v in prim.parameters.values())
    batch=max(sizes);element=meshes[0].vertices.element_size() if meshes else camera.fx.element_size()
    pixels=batch*camera.image_height*camera.image_width*factor*factor
    base_pixels=batch*camera.image_height*camera.image_width
    separate_base=bool(set(outputs)-{'colour','mask'}) and (factor>1 or soft>0)
    stores={}
    for value in tensors:
        storage=value.untyped_storage()
        stores[(value.device,storage.data_ptr())]=storage.nbytes()
    ordinary_stores={(value.device,value.untyped_storage().data_ptr()) for value in ordinary_tensors}
    backward=torch.is_grad_enabled() and any(value.requires_grad for value in tensors)
    # Include borrowed buffers, prospective geometry copies/transforms and their
    # derivatives. Overlapping aliases count once; expansion is counted below.
    # Budget tables are immutable borrowed constants with no table adjoints.
    # Their query/backward copies are charged separately in material_scratch.
    # A storage also supplied as an ordinary binding keeps the3x allowance.
    parts=dict(inputs=sum(size*(3 if backward and key in ordinary_stores else 1)
                          for key,size in stores.items()),
        packed=batch*(vertices*96*element+faces*(96+channels*24*element)+len(scene.prims)*64*element),
        buffers=(pixels+(base_pixels if separate_base else 0))*(256*element+32),
        sources=pixels*sources*32*element*batch,
        selection_masks=pixels*batch*len(mats)*(3 if backward else 1) if 'colour' in outputs else 0,
        lighting=0,material_scratch=0)
    qualified=True;principled_work=[]
    if 'colour' in outputs:
        sample_counts=[]
        for lamp in lamps:
            if lamp.evaluator in (lighting.directional,lighting.point):
                name='directions' if lamp.evaluator is lighting.directional else 'positions'
                value=lamp.parameters.get(name)
                if not isinstance(value,torch.Tensor):raise TypeError('light '+name+' must be a tensor')
                if value.ndim not in (2,3) or value.shape[-1]!=3:raise ValueError('light sample shape is invalid')
                sample_counts.append(value.shape[-2])
            elif lamp.evaluator is not lighting.spherical_harmonics:qualified=False
        # A pixel's single retained fragment selects exactly one material. The
        # disjoint selected arrays sum to at most P, even during backward. Full
        # per-frame/material boolean masks are separately counted above; they
        # must not disappear with this removal of duplicate point-light work.
        parts['lighting']=pixels*sum(sample_counts)*element*(512 if backward else 128)
        for mat in mats:
            if mat.evaluator is materials.principled:
                budget=mat.parameters.get('budget')
                if not isinstance(budget,materials.PrincipledBudget):raise TypeError('Principled requires a PrincipledBudget')
                for samples in sample_counts:
                    if samples>1024:raise ValueError('Principled light sample budget exceeded')
                    principled_work.append((budget,samples,isinstance(mat.parameters.get('roughness'),torch.Tensor) and mat.parameters['roughness'].ndim==0))
            elif mat.evaluator is not materials.classic:qualified=False
    # Only sequential forward shader scratch changes with this partition. All
    # retained inputs, packed geometry, images, sources and lighting keep their
    # original charges. Fitted workloads retain the qualified four-point order.
    for point_block_limit in ((4,) if backward else (4,2,1)):
        candidates=range(128,3,-4) if principled_work and point_block_limit==4 else (128,)
        for group_point_limit in candidates:
            parts['material_scratch']=0
            for budget,samples,shared_curve in principled_work:
                block=min(point_block_limit,pixels,1024//max(1,samples))
                if samples and pixels:
                    from ._principled_groups import capacity,workspace_bound
                    cohort=block
                    if point_block_limit==4 and pixels>4:
                        cohort=min(pixels,group_point_limit,block*capacity(budget,block,samples,backward,67108864,shared_curve=shared_curve))
                    # Native evaluation retains no eager curve graph. Charge the
                    # source-inventoried simultaneous storage of its actual cohort.
                    scratch=workspace_bound(budget,cohort,samples,backward,shared_curve=shared_curve)
                    # A sparsely covered window may shade <=4points through
                    # the unchanged dense direct path. Retain its full reserve.
                    if shared_curve:scratch=max(scratch,workspace_bound(budget,min(block,pixels),samples,backward))
                else:scratch=0
                parts['material_scratch']=max(parts['material_scratch'],scratch)
            estimate=sum(parts.values())
            if estimate<=limit:break
        if estimate<=limit:break
    if estimate>limit:raise ValueError(f'deferred workspace budget exceeded before scene allocation: {estimate} > {limit} bytes')
    return dict(estimated_bytes=estimate,components=parts,builtins_only=qualified,backward=backward,
                principled_point_block_limit=point_block_limit,principled_group_point_limit=group_point_limit)


def _validate_bindings(packed):
    """Validate structure without evaluating a shader on invented dummy points."""
    reference=packed.vertices
    def floating(value,name):
        if not isinstance(value,torch.Tensor) or value.dtype!=reference.dtype:
            raise TypeError(name+' must share scene floating dtype')
        if value.device!=reference.device:raise ValueError(name+' must share scene device')
        _finite(value,name)
    def check(value):
        if isinstance(value,(AttributeSource,TextureSource)):
            name=value.name if isinstance(value,AttributeSource) else value.uv
            if not isinstance(name,str) or not name:raise TypeError('source name must be a nonempty string')
            if type(value.scalar)!=bool:raise TypeError('source scalar flag must be bool')
        if isinstance(value,TextureSource):
            for name in ('height','width','channels'):
                size=getattr(value,name)
                if type(size)!=int:raise TypeError('texture '+name+' must be an integer')
                if size<=0:raise ValueError('texture '+name+' must be positive')
            if value.channels>4:raise ValueError('native texture channels must not exceed four')
            floating(value.pyramid,'texture pyramid')
            if value.pyramid.ndim!=1:raise ValueError('texture pyramid must be flat')
            offsets=value.level_offsets
            if not isinstance(offsets,torch.Tensor) or offsets.dtype!=torch.int32:
                raise TypeError('texture offsets must be int32')
            if offsets.device!=reference.device or offsets.ndim!=1 or len(offsets)<2:
                raise ValueError('texture offset device/shape mismatch')
            if int(offsets[0])!=0 or int(offsets[-1])!=value.pyramid.numel():raise ValueError('texture offsets must partition pyramid')
            h,w=value.height,value.width
            for left,right in zip(offsets[:-1],offsets[1:]):
                if int(right)-int(left)!=h*w*value.channels:raise ValueError('texture level extent mismatch')
                h,w=max(1,h//2),max(1,w//2)
            for name,allowed in (('filter',(1,2)),('mip_filter',(1,2)),('wrap_u',(1,2,3,4)),('wrap_v',(1,2,3,4))):
                if type(getattr(value,name))!=int or getattr(value,name) not in allowed:raise ValueError('invalid texture '+name)
            for name in ('lod','border'):
                if getattr(value,name) is not None:floating(getattr(value,name),'texture '+name)
            if value.scalar and value.channels!=1:raise ValueError('scalar texture requires one channel')
        elif isinstance(value,torch.Tensor):floating(value,'binding parameter')
        elif isinstance(value,Mapping):return tuple(value.values())
        return ()
    for binding in (*packed.materials,*packed.lights):
        pending=[binding.parameters]
        while pending:pending.extend(reversed(check(pending.pop())))
    # Built-in C/R fields retain their channel axis; a squeezed [P] field with
    # P=3 would otherwise be mistaken for one constant RGB value. Custom
    # evaluators retain their own explicitly declared parameter semantics.
    for binding in packed.materials:
        names=(('albedo','specular') if binding.evaluator is materials.classic else
               ('albedo','f0') if binding.evaluator is materials.principled else ())
        for name in names:
            value=binding.parameters.get(name)
            if isinstance(value,(AttributeSource,TextureSource)) and value.scalar:
                raise ValueError(name+' must retain its channel axis; use scalar=False')


def _buffers(packed,camera,soft,cull_mode,*,window=None):
    v,f=packed.vertices,packed.faces;batch=camera.batch_size
    h,w=camera.image_height,camera.image_width
    if window is not None:
        row,column,h,w=window
    clip=transforms.transform_points(v,camera.world_to_clip().to(dtype=v.dtype).contiguous(),packed.vertex_offsets)
    ndc,valid=transforms.perspective_divide(clip)
    valid=valid & (clip[:,3]>primitives.DEFAULT_EPS)
    if len(f):
        z=ndc[:,2][f.long()]
        keep=valid[f.long()].all(-1)&~((z < -1).all(-1)|(z > 1).all(-1))
    else:keep=torch.zeros(0,dtype=torch.bool,device=v.device)
    original=torch.nonzero(keep).flatten()
    faces=f[keep].contiguous()
    counts=[keep[int(packed.face_offsets[b]):int(packed.face_offsets[b+1])].sum() for b in range(batch)]
    offsets=torch.cat((packed.face_offsets.new_zeros(1),torch.stack(counts).cumsum(0).to(torch.int32)))
    if window is None:
        index,beta,depth,distance=primitives.rasterize(ndc.contiguous(),faces,h,w,
            blur_radius_px=soft,cull_mode=cull_mode,vertex_offsets=packed.vertex_offsets,face_offsets=offsets)
    else:
        if ndc.requires_grad:raise ValueError('window geometry backward is not qualified')
        index,beta,depth,distance=torch.ops.gffx.rasterize_window(ndc.contiguous(),faces,
            packed.vertex_offsets,offsets,camera.image_height,camera.image_width,row,column,h,w,
            1,soft,cull_mode,primitives.DEFAULT_EPS)
    active=(index>=0)&(depth>=-1)&(depth<=1)
    face=torch.where(active,original[index.clamp_min(0).long()] if len(original) else index.new_zeros(index.shape),-1).to(torch.int32)
    active=face>=0
    raw_beta=beta
    if soft:
        beta=beta.clamp_min(0)
        denominator=sum3(beta).unsqueeze(-1)
        beta=divide(beta,torch.where(denominator>0,denominator,torch.ones_like(denominator)))
    if len(f):
        corner_w=clip[:,3][f.long()][face.clamp_min(0).long()]
        corner_w=torch.where(active[...,None],corner_w,torch.ones_like(corner_w))
        numerator=beta/corner_w
        denominator=sum3(numerator).unsqueeze(-1)
        if bool((active & (denominator[...,0]<=0)).any()):raise ValueError('invalid perspective interpolation denominator')
        corrected=torch.where(active[...,None],divide(numerator,torch.where(active[...,None],denominator,torch.ones_like(denominator))),0.).contiguous()
    else:corrected=beta*0
    def interpolate(corners):return primitives.interpolate(face.contiguous(),corrected,corners.contiguous())[...,0,:]
    world=_finite(interpolate(v[f.long()]),'interpolated world position')
    if len(f):
        normal,_,_=geometry.face_geometry(v,f)
        normals=normal[:,None].expand(-1,3,-1)
    else:normals=v.new_zeros((0,3,3))
    if 'normals' in packed.attributes:
        a=packed.attributes['normals'];normals=torch.where(a.present[:,None,None],a.values,normals)
    n=interpolate(normals);covered=active[...,0]
    unit=n.new_zeros(n.shape)
    if bool(covered.any()):unit[covered]=_unit(n[covered],'interpolated normals')
    zero=v.reshape(-1)[:0].sum()+clip.reshape(-1)[:0].sum()
    if soft:
        safe=torch.where(active,distance,torch.zeros_like(distance))
        mask=torch.where(active,sigmoid(-safe/safe.new_tensor(soft*soft)),0.)[...,0]
    else:mask=covered.to(v.dtype)+zero
    _finite(mask,'coverage')
    result=dict(face_index=face[...,0],barycentric=corrected[...,0,:],world_position=world,
        normal=unit,mask=mask,depth=torch.where(covered,depth[...,0],torch.full_like(depth[...,0],float('inf'))),
        _face=face,_weights=corrected,_covered=covered,_projection=(ndc,valid,clip),_affine=raw_beta)
    ids=face[...,0].clamp_min(0).long()
    result['material_index']=torch.where(covered,packed.face_material[ids] if len(f) else face[...,0],-1)
    for name,key,presence,channels in [('uv','uv','uv_present',2),('colour','vertex_colour','colour_present',3)]:
        if name in packed.attributes:
            a=packed.attributes[name];result[key]=interpolate(a.values)
            result[presence]=covered & a.present[ids]
        else:
            result[key]=v.new_zeros((batch,h,w,channels));result[presence]=torch.zeros_like(covered)
    return result


def _source(value,packed,buffer,selected):
    if isinstance(value,AttributeSource):
        if not isinstance(value.name,str) or value.name not in packed.attributes:raise ValueError('missing material attribute source')
        a=packed.attributes[value.name];face=buffer['_face']
        if not bool(a.present[face[...,0][selected].long()].all()):raise ValueError('missing material attribute on selected face')
        values=primitives.interpolate(face.contiguous(),buffer['_weights'],a.values.contiguous())[...,0,:][selected]
    elif isinstance(value,TextureSource):
        if value.uv not in packed.attributes:raise ValueError('missing texture UV source')
        uv=_source(AttributeSource(value.uv),packed,buffer,selected)
        lod=value.lod
        if isinstance(lod,torch.Tensor) and lod.ndim==0:lod=lod.expand(len(uv)).contiguous()
        values=primitives.texture(value.pyramid,value.level_offsets,value.height,value.width,uv.contiguous(),
            channel_count=value.channels,filter=value.filter,mip_filter=value.mip_filter,
            wrap_u=value.wrap_u,wrap_v=value.wrap_v,lod=lod,border=value.border)
    else:return value
    if type(value.scalar)!=bool:raise TypeError('source scalar flag must be bool')
    if value.scalar:
        if values.shape[-1]!=1:raise ValueError('scalar source must have one channel')
        return values[...,0]
    return values


def _light_packet(binding,packed,frame,points):
    params=dict(binding.parameters);world=packed.world_matrices[binding.name][frame]
    def batched(value):return value[frame if value.shape[0]>1 else 0] if value.ndim==3 else value
    if binding.evaluator is lighting.directional:
        direction=batched(params['directions']) @ world[:3,:3].T
        data=lighting.directional(direction,batched(params['irradiance']))
        return dict(measure='delta',direction=data['direction'][None].expand(len(points),-1,-1),
            irradiance=data['irradiance'][None].expand(len(points),-1,-1),
            valid=torch.ones((len(points),len(direction)),dtype=torch.bool,device=points.device))
    if binding.evaluator is lighting.point:
        positions=batched(params['positions']) @ world[:3,:3].T+world[:3,3]
        data=lighting.point(points,positions,batched(params['intensity']))
        return {k:v for k,v in dict(data,measure='delta').items() if k!='distance_squared'}
    if not torch.equal(world,torch.eye(4,dtype=world.dtype,device=world.device)):
        raise ValueError('custom light requires explicit world-space callback; transform unsupported')
    return binding.evaluator(points,**params)


class _PrincipledRecompute(torch.autograd.Function):
    """Save only inputs; return the recomputed first VJP into their original graph."""
    @staticmethod
    def forward(ctx,keys,static,*tensors):
        ctx.keys=keys
        ctx.static=dict(static)
        ctx.save_for_backward(*tensors)
        parameters=dict(static,**dict(zip(keys,tensors[3:])))
        return materials.principled(*tensors[:3],**parameters)

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent):
        needs=ctx.needs_input_grad[2:]
        if not any(needs):return (None,None)+tuple(None for _ in needs)
        # These temporary leaves exist only to compute this block's VJP. The
        # returned gradients reconnect every original input; no cache is detached.
        temporary=[value.detach().requires_grad_(need) for value,need in zip(ctx.saved_tensors,needs)]
        with torch.enable_grad():
            parameters=dict(ctx.static,**dict(zip(ctx.keys,temporary[3:])))
            output=materials.principled(*temporary[:3],**parameters)
            partials=iter(torch.autograd.grad(output,[v for v,n in zip(temporary,needs) if n],
                                               cotangent,allow_unused=True))
        return (None,None)+tuple(next(partials) if need else None for need in needs)


def _principled_recompute(normals,view,directions,parameters):
    """Exact first-order wrapper of one admitted <=4-point public material call."""
    keys=tuple(sorted(k for k,v in parameters.items() if isinstance(v,torch.Tensor)))
    static={k:v for k,v in parameters.items() if k not in keys}
    values=(normals,view,directions,*(parameters[k] for k in keys))
    if not torch.is_grad_enabled() or not any(value.requires_grad for value in values):
        return materials.principled(normals,view,directions,**parameters)
    return _PrincipledRecompute.apply(keys,static,*values)


def _principled_blocks(normals,view,directions,parameters,*,point_block_limit=4,group_point_limit=128,_prepared=None):
    """Admitted point-major partition; known field axes, never shape guessing."""
    if type(group_point_limit)!=int or group_point_limit not in range(4,129,4):
        raise ValueError('Principled group point limit must be a multiple of four in [4,128]')
    if type(point_block_limit)!=int or point_block_limit not in (1,2,4):
        raise ValueError('Principled point block limit must be one, two or four')
    values=(normals,view,directions,*(v for v in parameters.values() if isinstance(v,torch.Tensor)))
    if point_block_limit!=4 and torch.is_grad_enabled() and any(v.requires_grad for v in values):
        raise ValueError('Principled backward partition must retain four-point limit')
    points=len(normals);samples=directions.shape[-2]
    if samples>1024:raise ValueError('Principled light sample budget exceeded')
    block=min(point_block_limit,1024//max(1,samples))
    def sliced(name,value,left,right):
        if not isinstance(value,torch.Tensor):return value
        # These are the accepted Principled field signatures, not arbitrary
        # callback tensors whose first extent might accidentally equal P.
        if name in ('albedo','f0'):
            if value.ndim==2 and value.shape[0]==points:return value[left:right]
            if value.ndim==3 and value.shape[:2]==(1,points):return value[:,left:right]
        if name=='roughness':
            if value.shape==(points,):return value[left:right]
            if value.shape==(1,points):return value[:,left:right]
        return value
    if point_block_limit==4 and samples and points>4:
        from ._principled_groups import render
        return render(normals,view,directions,parameters,block,group_point_limit=group_point_limit,_prepared=_prepared)
    pieces=[]
    for left in range(0,points,block):
        right=min(left+block,points)
        fields={k:sliced(k,v,left,right) for k,v in parameters.items()}
        pieces.append(_principled_recompute(normals[left:right],view[left:right],directions[left:right],fields))
    return torch.cat(pieces,dim=0) if pieces else normals.new_empty((0,samples,3))


def _shade(packed,camera,buffer,*,point_block_limit=4,group_point_limit=128,_shared_curves=None):
    result=buffer['world_position'].new_zeros(buffer['world_position'].shape)
    covered=buffer['_covered'];ids=buffer['material_index']
    if bool((covered & (ids<0)).any()):raise ValueError('colour requires a material on every selected face')
    for frame in range(camera.batch_size):
        for number,mat in enumerate(packed.materials):
            selected=torch.zeros_like(covered);selected[frame]=covered[frame]&(ids[frame]==number)
            if not bool(selected.any()):continue
            points=buffer['world_position'][selected];n=buffer['normal'][selected]
            view=(camera.pose[frame,:3,2].to(dtype=points.dtype).expand_as(points)
                  if camera.projection_kind=='orthographic' else
                  camera.pose[frame,:3,3].to(dtype=points.dtype)-points)
            params={k:_source(v,packed,buffer,selected) for k,v in mat.parameters.items()}
            # Shared empty reductions retain paths without overflowing finite sums.
            value=points*0
            for field in params.values():
                if isinstance(field,torch.Tensor):value=value+field.reshape(-1)[:0].sum()
            for lamp in packed.lights:
                if lamp.evaluator is lighting.spherical_harmonics:
                    if mat.evaluator is not materials.classic:raise ValueError('SH irradiance only supports explicit classic diffuse path')
                    if bool((params['specular']!=0).any()):raise ValueError('SH diffuse path requires zero classic specular')
                    coefficients=lamp.parameters['coefficients']
                    if coefficients.ndim==3:coefficients=coefficients[frame if coefficients.shape[0]>1 else 0]
                    world=packed.world_matrices[lamp.name][frame]
                    if not torch.equal(world,torch.eye(4,dtype=world.dtype,device=world.device)):
                        raise ValueError('SH light transform requires explicit coefficient rotation')
                    # SH already integrates the diffuse illumination cosine.
                    # Geometric sidedness selects the surface; a smooth shading
                    # normal must not create a second camera-visibility boundary.
                    with torch.no_grad():
                        geometric,_,_=geometry.face_geometry(packed.vertices,packed.faces,eps=0.0)
                        geometric=geometric[buffer['face_index'][selected].long()]
                        facing=sum3(geometric*view)
                        two_sided=params.get('two_sided',False)
                        active=(facing!=0) if two_sided else (facing>0)
                        sign=torch.where((facing<0)&two_sided,-torch.ones_like(facing),torch.ones_like(facing))
                    oriented=n*sign[:,None]
                    # Normal-incidence evaluation preserves all public classic
                    # field validation and zero-parameter adjoints, without the
                    # unrelated shading-normal/view hemisphere gate.
                    diffuse=materials.classic(oriented,oriented,oriented[:,None,:],**params)[:,0,:]
                    radiance=diffuse*lighting.spherical_harmonics(oriented,coefficients)
                    value=value+torch.where(active[:,None],radiance,torch.zeros_like(radiance))
                    continue
                packet=_light_packet(lamp,packed,frame,points)
                # Validate the original packet before shielding only its declared
                # invalid directions from BRDF normalization. Integration still
                # consumes the original zeros/mask; malformed packets are errors.
                direction,_,valid,_=lighting._packet(n,packet)
                safe_direction=torch.where(valid[...,None],direction,direction.new_tensor([0.,0.,1.]))
                prepared=None
                if _shared_curves is not None and isinstance(params.get('roughness'),torch.Tensor) and params['roughness'].ndim==0:
                    if torch.is_grad_enabled():raise ValueError('shared material setup requires forward-only execution')
                    prepared=_shared_curves.setdefault(number,{})
                brdf=(_principled_blocks(n,view,safe_direction,params,point_block_limit=point_block_limit,group_point_limit=group_point_limit,_prepared=prepared)
                      if mat.evaluator is materials.principled else
                      mat.evaluator(n,view,safe_direction,**params))
                oriented=n
                if params.get('two_sided',False):
                    sign=torch.where(sum3(n*view)<0,-torch.ones_like(n[:,0]),torch.ones_like(n[:,0]))
                    oriented=n*sign[:,None]
                value=value+lighting.integrate(oriented,brdf,packet)
            result[selected]=_finite(value,'shaded colour')
    return result*buffer['mask'][...,None]


def _visibility(packed,camera,buffer,epsilon):
    ndc,valid,_=buffer['_projection'];v=packed.vertices
    count=len(v)//camera.batch_size
    xy=torch.stack(((ndc[:,0]+1)*camera.image_width/2,(1-ndc[:,1])*camera.image_height/2),-1)
    x=xy[:,0].floor().long();y=xy[:,1].floor().long()
    active=valid&(x>=0)&(x<camera.image_width)&(y>=0)&(y<camera.image_height)&(ndc[:,2]>=-1)&(ndc[:,2]<=1)
    result=torch.zeros(len(v),dtype=torch.bool,device=v.device)
    if len(v):
        frame=torch.arange(camera.batch_size,device=v.device).repeat_interleave(count)
        depth=buffer['depth'][frame,y.clamp(0,camera.image_height-1),x.clamp(0,camera.image_width-1)]
        result=active & ((ndc[:,2]-depth).abs()<=epsilon)
    return result.reshape(camera.batch_size,count)


def render(scene,camera,integrator='deferred',*,soft=0.0,supersample=1,coverage=None,
           outputs=('colour','mask'),cull_mode=primitives.CULL_BACK,background=None,
           depth_epsilon=1e-4,workspace_limit_bytes=67108864,opaque_tile_pixels=None,
           cpu_workspace_allowance_bytes=67108864):
    """Render named tensors; transparent output is premultiplied linear RGB.

    Explicit background composites RGB but does not change foreground coverage.
    Whole-triangle clipping and front-fragment soft silhouettes are approximate.
    """
    if integrator!='deferred':raise ValueError('unknown integrator')
    if isinstance(outputs,str) or not isinstance(outputs,(list,tuple)):raise TypeError('outputs must be a sequence of names')
    if not outputs or any(type(k)!=str for k in outputs) or len(set(outputs))!=len(outputs) or not set(outputs)<=_OUTPUTS:
        raise ValueError('unknown, empty or duplicate outputs')
    soft=_real(soft,'soft');epsilon=_real(depth_epsilon,'depth_epsilon')
    if type(supersample)!=int:raise TypeError('supersample must be a positive integer')
    if supersample<1:raise ValueError('supersample must be positive')
    if soft:
        try:scaled=soft*supersample
        except OverflowError:raise ValueError('soft scaled radius is not representable') from None
        if soft*soft==0 or not math.isfinite(soft*soft) or not math.isfinite(scaled*scaled):
            raise ValueError('soft squared radius is not representable')
    if type(cull_mode)!=int or cull_mode not in (primitives.CULL_NONE,primitives.CULL_BACK,primitives.CULL_FRONT):
        raise ValueError('invalid cull_mode')
    if coverage is not None:
        if coverage!='opaque':raise ValueError('unknown coverage model')
        resolved,_=_resolve_camera(scene,camera)
        if resolved.projection_kind=='orthographic':
            raise NotImplementedError('orthographic opaque rendering is not qualified for this camera/material combination')
        if soft or supersample!=1:raise ValueError('opaque coverage does not combine with soft or supersampling')
        if set(outputs)-{'mask','colour'}:raise NotImplementedError('opaque geometric output composition is not qualified')
        if 'colour' in outputs:
            from ._opaque_stream import render as opaque_shade
            return opaque_shade(scene,camera,outputs=outputs,cull_mode=cull_mode,background=background,workspace_limit_bytes=workspace_limit_bytes,tile_pixels=opaque_tile_pixels)
        if opaque_tile_pixels is not None:raise ValueError('opaque_tile_pixels requires opaque colour rendering')
        if background is not None:raise ValueError('opaque mask-only output has no background colour')
        from ._opaque_scene import mask as opaque_mask
        return {'mask':opaque_mask(scene,camera,cull_mode=cull_mode,workspace_limit_bytes=workspace_limit_bytes)}
    if opaque_tile_pixels is not None:raise ValueError('opaque_tile_pixels requires opaque colour rendering')
    workload=_plan_workload(scene,camera,supersample,workspace_limit_bytes,outputs,background,soft,cpu_workspace_allowance_bytes)
    packed,camera=_prepare(scene,camera)
    _validate_bindings(packed)
    if soft:
        info=torch.finfo(packed.vertices.dtype)
        if not info.tiny*info.eps<=scaled*scaled<=info.max:
            raise ValueError('soft squared radius is not representable in scene dtype')
    if background is not None:
        if not isinstance(background,torch.Tensor) or background.dtype!=packed.vertices.dtype:raise TypeError('background must match scene dtype')
        if background.device!=packed.vertices.device or background.shape not in ((3,),(camera.batch_size,3)):raise ValueError('background shape/device mismatch')
        _finite(background,'background')
    high_camera=_camera(camera,camera.batch_size,factor=supersample)
    high=_buffers(packed,high_camera,soft*supersample,cull_mode)
    colour=(_shade(packed,high_camera,high,point_block_limit=workload['principled_point_block_limit'],group_point_limit=workload['principled_group_point_limit'])
            if 'colour' in outputs else None)
    mask=high['mask']
    if supersample>1:
        from .image import reconstruct
        if colour is not None:
            joined=reconstruct(torch.cat((colour,mask[...,None]),-1),supersample)
            colour,mask=joined[...,:3],joined[...,3]
        else:mask=reconstruct(mask[...,None],supersample)[...,0]
    geometric=set(outputs)-{'colour','mask'}
    base=_buffers(packed,camera,0.,cull_mode) if geometric and (supersample>1 or soft>0) else high
    result={k:base[k] for k in outputs if k not in ('colour','mask','visibility')}
    if 'mask' in outputs:result['mask']=mask
    if colour is not None:
        if background is not None:colour=colour+(1-mask[...,None])*background.reshape(-1,1,1,3)
        result['colour']=_finite(colour,'colour')
    if 'visibility' in outputs:result['visibility']=_visibility(packed,camera,base,epsilon)
    return {k:result[k] for k in outputs}


def mesh(vertices,faces,material,lights,camera,*,attributes=None,**options):
    if type(material)!=scene_types.Material:raise TypeError('material must be scene.Material')
    lights=tuple(lights)
    if any(type(p)!=scene_types.Light for p in lights):raise TypeError('lights must contain scene.Light')
    model=scene_types.Scene([scene_types.Mesh('__render_mesh__',vertices,faces,
        material=material.name,attributes={} if attributes is None else attributes),material,*lights])
    return render(model,camera,**options)


def visibility(scene,camera,*,depth_epsilon=1e-4,cull_mode=primitives.CULL_BACK):
    return render(scene,camera,outputs=('visibility',),depth_epsilon=depth_epsilon,cull_mode=cull_mode)['visibility']
