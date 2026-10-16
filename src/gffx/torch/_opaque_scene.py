"""Bounded nonlinear opaque composition over native visible-region quadrature."""
import math
import torch
from . import transforms
from ._opaque import opaque_coverage

class _PerspectiveWeights(torch.autograd.Function):
    """Three-term normalization with an explicit, device-identical first VJP."""
    @staticmethod
    def forward(ctx,raw,cw,active):
        u=raw/cw;den=((u[...,0]+u[...,1])+u[...,2])[...,None]
        if bool((active&(den[...,0]<=0)).any()):raise ValueError('opaque perspective denominator')
        den=torch.where(active[...,None],den,torch.ones_like(den))
        beta=torch.where(active[...,None],u/den,torch.zeros_like(u))
        ctx.save_for_backward(beta,cw,den,u,active);return beta
    @staticmethod
    def backward(ctx,g):
        if torch.is_grad_enabled():raise RuntimeError('opaque perspective supports first derivatives only')
        beta,cw,den,u,active=ctx.saved_tensors
        product=g*beta;total=((product[...,0]+product[...,1])+product[...,2])[...,None]
        gu=torch.where(active[...,None],(g-total)/den,torch.zeros_like(g))
        return gu/cw,-(gu*u)/cw,None


def _project(scene,camera,cull_mode,detach_principal=False):
    from ._deferred import _prepare
    packed,camera=_prepare(scene,camera)
    v,f=packed.vertices,packed.faces;h,w=camera.image_height,camera.image_width
    projection_camera=camera
    if detach_principal:
        from copy import copy
        projection_camera=copy(camera);projection_camera.cx=camera.cx.detach();projection_camera.cy=camera.cy.detach()
    clip=transforms.transform_points(v,projection_camera.world_to_clip().to(dtype=v.dtype).contiguous(),packed.vertex_offsets)
    ndc,valid=transforms.perspective_divide(clip)
    if len(f):
        corners=f.long();good=valid[corners] & (clip[:,3][corners]>0)
        if bool((good.any(-1) & ~good.all(-1)).any()):raise NotImplementedError('opaque mask camera-plane clipping is not qualified')
        z=ndc[:,2][corners];outside=((z < -1).all(-1)|(z > 1).all(-1)|(~good).all(-1))
        inside=good.all(-1)&(z>=-1).all(-1)&(z<=1).all(-1)
        if bool((~outside & ~inside).any()):raise NotImplementedError('opaque mask requires triangles inside the projection depth range; partial clipping is not qualified')
        keep=inside
        if cull_mode!=1:
            a=ndc[corners[:,1]]-ndc[corners[:,0]];b=ndc[corners[:,2]]-ndc[corners[:,0]]
            orientation=a[:,0]*b[:,1]-a[:,1]*b[:,0]
            keep=keep & (orientation>0 if cull_mode==2 else orientation<0)
    else:keep=torch.zeros(0,dtype=torch.bool,device=v.device)
    pixels=torch.stack(((ndc[:,0]+1)*(w/2),(1-ndc[:,1])*(h/2),ndc[:,2]),-1).contiguous()
    return packed,camera,pixels,keep,clip

def mask(scene,camera,*,cull_mode,workspace_limit_bytes):
    from ._deferred import _plan_workload
    if type(workspace_limit_bytes)!=int:raise TypeError('workspace budget must be integer bytes')
    if workspace_limit_bytes<=1048576:raise ValueError('opaque workspace budget needs polygon scratch')
    _plan_workload(scene,camera,1,workspace_limit_bytes-1048576,('mask',))
    packed,camera,pixels,keep,clip=_project(scene,camera,cull_mode)
    v,f=packed.vertices,packed.faces;h,w=camera.image_height,camera.image_width
    outputs=[]
    for b in range(camera.batch_size):
        first,last=int(packed.face_offsets[b]),int(packed.face_offsets[b+1])
        outputs.append(opaque_coverage(pixels,f[first:last][keep[first:last]].contiguous(),h,w,workspace_limit_bytes=workspace_limit_bytes))
    return torch.stack(outputs)


def _cuts(packed,clip):
    from ._deferred import AttributeSource,TextureSource
    from . import render as primitives
    f=packed.faces;v=packed.vertices;columns=[]
    iw=1/torch.where(clip[:,3]>0,clip[:,3],torch.ones_like(clip[:,3]));iw=iw[f.long()]
    def add(field,threshold,selected):
        if len(columns)==16:raise OverflowError('opaque scene material-cut capacity16 exceeded')
        columns.append(torch.where(selected[:,None],(field-threshold)*iw,torch.zeros_like(field)))
    for number,mat in enumerate(packed.materials):
        selected=packed.face_material==number
        if not bool(selected.any()):continue
        for name,source in mat.parameters.items():
            if isinstance(source,TextureSource):
                if name!='albedo':raise NotImplementedError('opaque texture sources currently qualify albedo only')
                if source.filter!=primitives.FILTER_BILINEAR or source.mip_filter!=primitives.MIP_NEAREST or source.lod is not None or source.border is not None or source.wrap_u not in (primitives.WRAP_CLAMP,primitives.WRAP_REPEAT) or source.wrap_v not in (primitives.WRAP_CLAMP,primitives.WRAP_REPEAT):
                    raise NotImplementedError('opaque texture profile: bilinear level0,clamp/repeat,explicit existing pyramid')
                uv=packed.attributes[source.uv].values
                for axis,size in enumerate((source.width,source.height)):
                    field=uv[...,axis];lo=float(field[selected].detach().amin());hi=float(field[selected].detach().amax())
                    first=math.floor(lo*size-.5)+1;last=math.ceil(hi*size-.5)-1
                    if (source.wrap_u if axis==0 else source.wrap_v)==primitives.WRAP_CLAMP:first=max(first,0);last=min(last,size-1)
                    if last-first+1>16-len(columns):raise OverflowError('opaque scene material-cut capacity16 exceeded')
                    for k in range(first,last+1):add(field,(k+.5)/size,selected)
            elif isinstance(source,AttributeSource) and name=='roughness':
                field=packed.attributes[source.name].values[...,0];lo=float(field[selected].detach().amin());hi=float(field[selected].detach().amax())
                for threshold in mat.parameters['budget']._envelope['payload']['roughness_nodes']:
                    if lo<threshold<hi:add(field,threshold,selected)
    return torch.stack(columns,-1).contiguous() if columns else v.new_empty((len(f),3,0))


def _normals(packed):
    from . import mesh as geometry
    v,f=packed.vertices,packed.faces
    if len(f):
        normal,_,_=geometry.face_geometry(v,f);normals=normal[:,None].expand(-1,3,-1)
    else:normals=v.new_zeros((0,3,3))
    if 'normals' in packed.attributes:
        a=packed.attributes['normals'];normals=torch.where(a.present[:,None,None],a.values,normals)
    return normals


def _horizon_cuts(packed,camera,clip,normals,keep,remaining=16,*,_constant=None,_activity=None):
    """Linear horizon numerators,or certified one-sign quadratic view fields.

    Attribute domain,not equal sampled values,establishes constant normals under
    differentiation. A varying normal and view make a quadratic numerator;
    Bernstein signs exclude absent zero sets; crossing quadratics remain explicit
    curved inputs, never replaced by a line.
    """
    from ._reference_channels import sum3
    v,f=packed.vertices,packed.faces;columns=[];curves=None
    if _activity is not None:_activity.update(linear=[],curved=torch.zeros_like(keep))
    if not packed.lights or not len(f):return v.new_empty((len(f),3,0)),None
    constant=_constant
    if constant is None:
        constant=torch.ones(len(f),dtype=torch.bool,device=v.device)
        a=packed.attributes.get('normals')
        if a is not None:
            for frame in range(camera.batch_size):
                for obj,domain in enumerate(a.domains):
                    if domain in ('vertex','face_varying'):
                        first,last=map(int,packed.object_face_offsets[frame,obj:obj+2]);constant[first:last]=False
    # Forward-only compaction preserves every selected face and scatters its
    # exact coefficients back to the old dense layout. The differentiated path
    # stays dense, so point/face accumulation order is unchanged.
    # A conservative magnitude guard proves unselected dot/error/interpolation
    # arithmetic finite. Extreme/nonfinite inputs use the original dense checks.
    if _constant is None and not torch.is_grad_enabled() and bool(keep.any()) and not bool(keep.all()):
        def bounded(x):return bool((torch.isfinite(x)&(x.abs()<=1e50)).all())
        safe=bounded(v) and bounded(normals) and bounded(camera.pose) and bounded(clip[:,3])
        safe=safe and bool(((clip[:,3]<=0)|(clip[:,3]>=1e-50)).all())
        if safe:
            for lamp in packed.lights:
                if not (bounded(lamp.parameters['directions']) and bounded(packed.world_matrices[lamp.name])):
                    safe=False;break
        if safe:
            from types import SimpleNamespace
            ids=keep.nonzero(as_tuple=False).flatten()
            offsets=[int(keep[:int(k)].sum()) for k in packed.face_offsets]
            small=SimpleNamespace(vertices=v,faces=f[ids],lights=packed.lights,world_matrices=packed.world_matrices,
                face_offsets=offsets,attributes={})
            activity={} if _activity is not None else None
            linear,curved=_horizon_cuts(small,camera,clip,normals[ids],torch.ones(len(ids),dtype=torch.bool,device=v.device),remaining,_constant=constant[ids],_activity=activity)
            if activity is not None:
                for mask in activity['linear']:
                    full=torch.zeros_like(keep);full.index_copy_(0,ids,mask);_activity['linear'].append(full)
                _activity['curved'].index_copy_(0,ids,activity['curved'])
            dense=v.new_zeros((len(f),3,linear.shape[-1]));dense.index_copy_(0,ids,linear)
            if curved is not None:
                full=v.new_zeros((len(f),6));full.index_copy_(0,ids,curved);curved=full
            return dense,curved
    iw=1/torch.where(clip[:,3]>0,clip[:,3],torch.ones_like(clip[:,3]));iw=iw[f.long()]
    eps=torch.finfo(v.dtype).eps
    def dot(a,b):
        products=a*b;value=sum3(products)
        error=(32*eps)*sum3(products.abs())+torch.finfo(v.dtype).tiny
        if not bool(torch.isfinite(value).all() & torch.isfinite(error).all()):raise ValueError('nonfinite horizon coefficients')
        return value,error
    def append(field,selected):
        if not bool(selected.any()):return
        if len(columns)+(curves is not None)>=remaining:raise OverflowError('opaque scene shared material/horizon-cut capacity16 exceeded')
        columns.append(torch.where(selected[:,None],field*iw,torch.zeros_like(field)))
        if _activity is not None:_activity['linear'].append(selected)
    for frame in range(camera.batch_size):
        first,last=map(int,packed.face_offsets[frame:frame+2]);selected=torch.zeros_like(keep);selected[first:last]=keep[first:last]
        if not bool(selected.any()):continue
        view=camera.pose[frame,:3,3]-v[f.long()];diag,err=dot(normals,view)
        lower=[diag[:,i]-err[:,i] for i in range(3)];upper=[diag[:,i]+err[:,i] for i in range(3)]
        for i,j in ((0,1),(0,2),(1,2)):
            a,ea=dot(normals[:,i],view[:,j]);b,eb=dot(normals[:,j],view[:,i])
            # No division needed for signs of the doubled off-diagonal coefficient.
            e=ea+eb+(32*eps)*(a.abs()+b.abs());lower.append(a+b-e);upper.append(a+b+e)
        lo=torch.stack(lower,-1).amin(-1);hi=torch.stack(upper,-1).amax(-1)
        crossing=selected & (lo<=0) & (hi>=0)
        curved=crossing & ~constant
        if bool(curved.any()):
            if _activity is not None:_activity['curved']|=curved
            if len(columns)+1>remaining:raise OverflowError('opaque scene shared material/horizon-cut capacity16 exceeded')
            values=[diag[:,i]*iw[:,i]*iw[:,i] for i in range(3)]
            for i,j in ((0,1),(0,2),(1,2)):
                a,_=dot(normals[:,i],view[:,j]);b,_=dot(normals[:,j],view[:,i]);values.append((a+b)*iw[:,i]*iw[:,j])
            row=torch.stack(values,-1)
            curves=torch.where(curved[:,None],row,torch.zeros_like(row) if curves is None else curves)
        if bool((crossing & (diag.abs().amax(-1)<=err.amax(-1))).any()):raise NotImplementedError('opaque grazing view horizon is not qualified')
        append(diag,crossing & constant)
        for lamp in packed.lights:
            direction=lamp.parameters['directions'];direction=direction[frame if direction.shape[0]>1 else 0] if direction.ndim==3 else direction
            direction=direction @ packed.world_matrices[lamp.name][frame,:3,:3].T
            for sample in direction:
                value,error=dot(normals,sample);lo=(value-error).amin(-1);hi=(value+error).amax(-1)
                crossing=selected & (lo<=0) & (hi>=0)
                if bool((crossing & (value.abs().amax(-1)<=error.amax(-1))).any()):raise NotImplementedError('opaque grazing light horizon is not qualified')
                append(value,crossing)
    return (torch.stack(columns,-1).contiguous() if columns else v.new_empty((len(f),3,0))),curves



def _select_horizons(linear,curved,activity,selected):
    # Eligibility survives even when finite coefficients round to exact zero.
    columns=[i for i,mask in enumerate(activity['linear']) if bool((mask&selected).any())]
    linear=linear[...,columns].contiguous()
    curved=curved if curved is not None and bool((activity['curved']&selected).any()) else None
    return linear,curved


def _face_bounds(pixels,faces):
    corners=pixels[faces.long(),:2]
    return (corners[...,0].amax(-1),corners[...,0].amin(-1),corners[...,1].amax(-1),corners[...,1].amin(-1))


def _select_bounds(keep,bounds,ox,oy,w,h):
    xmax,xmin,ymax,ymin=bounds
    return keep & (xmax>=ox) & (xmin<=ox+w) & (ymax>=oy) & (ymin<=oy+h)


def _window_horizons(packed,camera,pixels,clip,normals,keep,selected,bounds,remaining,shared):
    if shared is None or 'region' not in shared:
        return _horizon_cuts(packed,camera,clip,normals,selected,remaining)
    cache=shared.get('horizons')
    if cache is None:
        oy,ox,h,w=shared['region']
        whole=_select_bounds(keep,bounds,ox,oy,w,h)
        activity={}
        try:
            linear,curved=_horizon_cuts(packed,camera,clip,normals,whole,remaining,_activity=activity)
            cache=(linear,curved,activity)
        except OverflowError:
            # A union may require more columns than any individual pixel. Retain
            # the original local admission; never raise the shared cut limit.
            cache=False
        shared['horizons']=cache
    if cache is False:return _horizon_cuts(packed,camera,clip,normals,selected,remaining)
    return _select_horizons(*cache,selected)


def _extra_geometry_bytes(scene,camera,*,sh_batch_profile=16):
    """Charge opaque cuts/horizon graph beyond the shared deferred estimate.

    Per packed face: 16 three-corner cut columns, their retained operands and
    adjoints, normal/view fields, six Bernstein coefficients and error bounds,
    boolean selections and simultaneous stack/concatenation copies. At most576
    scalar slots bound this inventory,including64 slots for packed conic inputs.
    Constant-material profiles remove only impossible cut-column copies; all
    other384 slots remain reserved. Borrowed inputs are charged by the shared
    planner. No sampled ambient-free-memory budget.
    """
    from .scene import Mesh,Attribute
    batch=camera.batch_size;faces=0
    for p in scene.prims:
        transform=p.transform
        if isinstance(transform,torch.Tensor) and transform.ndim==3:batch=max(batch,transform.shape[0])
        if type(p)==Mesh:
            faces+=len(p.faces)
            if p.vertices.ndim==3:batch=max(batch,p.vertices.shape[0])
            for a in p.attributes.values():
                rank={'constant':1,'uniform':2,'vertex':2,'face_varying':3}[a.domain]
                if a.values.ndim==rank+1:batch=max(batch,a.values.shape[0])
    # Constant material bindings create no material partition columns. In that
    # profile each camera contributes at most one view column and one column per
    # directional sample. Unknown/attribute/texture profiles keep all16 columns.
    from .scene import Material,Light
    from ._deferred import AttributeSource,TextureSource
    from . import material,light
    cut_bound=0;known=True;light_samples=0
    for p in scene.prims:
        if type(p)==Material:
            if p.evaluator is not material.principled or any(isinstance(q,(AttributeSource,TextureSource)) for q in p.parameters.values()):known=False
        if type(p)==Light:
            q=p.parameters.get('directions')
            if p.evaluator is not light.directional or not isinstance(q,torch.Tensor) or q.ndim not in (2,3) or q.shape[-1]!=3:known=False
            else:light_samples+=q.shape[-2]
    cut_bound=min(16,batch*(1+light_samples)) if known else 16
    # Of the original576 slots,192 charge four three-corner cut copies for16
    # columns. Keep the other384 slots unchanged (including retained operands,
    # masks, conic inputs and their adjoints); remove only impossible columns.
    # One additional forward-only horizon coefficient set plus its exact
    # eligibility masks remains live during local selection. Charge it even on
    # paths that do not reuse it; no increase to the64MiB allowance.
    from ._opaque_sh import profile as sh_profile
    if sh_profile(scene,camera):
        if sh_batch_profile==80:
            # Full geometry/normal fields plus bounded compact construction,
            # selected cut copies/adjoints, integer masks and gather metadata.
            return 1048576+batch*faces*256*camera.fx.element_size()+512*40*30*camera.fx.element_size()+(3*1048576 if camera.fx.device.type=='cuda' else 0)
        # SH does not allocate view/light horizon coefficients, error fields or
        # conics.256 face-scalars charge geometry/normal operands and adjoints;
        # three full corner-cut copies charge9*16 slots. Local cut construction
        # is compact (at most32 faces), charged within1MiB scratch.
        return 1048576+batch*faces*(256+9*16)*camera.fx.element_size()+(3*1048576 if camera.fx.device.type=='cuda' else 0)
    material_reserve=0
    for p in scene.prims:
        if type(p)==Material and p.evaluator is material.principled:
            r=p.parameters.get('roughness');budget=p.parameters.get('budget')
            if isinstance(r,torch.Tensor) and r.ndim==0 and isinstance(budget,material.PrincipledBudget):
                material_reserve+=8*64*len(budget._data['cosine_nodes'])
    # Four exact per-face extrema are retained only within the guarded forward call.
    from ._opaque_samples import RESERVE_BYTES
    # CUDA forward may use four private polygon slabs; charge all three extra.
    lane_reserve=3*1048576 if camera.fx.device.type=='cuda' else 0
    return lane_reserve+RESERVE_BYTES+material_reserve+batch*faces*((384+12*cut_bound+3*cut_bound+6+4)*camera.fx.element_size()+cut_bound+1)


def _flux_reserve(scene,camera):
    from .scene import Mesh
    sizes=[camera.batch_size]
    for p in scene.prims:
        if isinstance(p.transform,torch.Tensor) and p.transform.ndim==3:sizes.append(p.transform.shape[0])
        if type(p)==Mesh:
            if p.vertices.ndim==3:sizes.append(p.vertices.shape[0])
            for a in p.attributes.values():
                rank={'constant':1,'uniform':2,'vertex':2,'face_varying':3}[a.domain]
                if a.values.ndim==rank+1:sizes.append(a.values.shape[0])
    return max(sizes)*camera.image_height*camera.image_width*64*camera.fx.element_size()


def _sample_integral(packed,camera,clip,normals,owners,q,workload,*,contributions=False,_shared_curves=None,_albedo_cotangent=None):
    wide=workload.get('opaque_native_window',False) and q.numel()>3200
    args=(packed,camera,clip,normals,owners,q,workload)
    options=dict(contributions=contributions,_shared_curves=_shared_curves)
    if _albedo_cotangent is not None:
        from ._opaque_samples import compose
        try:result=compose(*args,**options,_albedo_cotangent=_albedo_cotangent)
        except (RuntimeError,ValueError,TypeError) as error:
            if wide:raise
            _sample_integral_eager(*args,**options)
            raise error
        if result is not None:return result
        if wide:raise OverflowError('opaque quadrature native albedo fallback overflow; canonical pixels required')
        values=_sample_integral_eager(*args,**options)
        return (*values,None)
    if not torch.is_grad_enabled() and bool((owners>=0).any()):
        from ._opaque_samples import compose
        try:
            result=compose(*args,**options)
        except (RuntimeError,ValueError,TypeError) as error:
            if wide:raise OverflowError('opaque quadrature native fallback overflow; canonical pixels required') from error
            # Run the original checks to preserve their first diagnostic. A
            # valid reference result cannot silently hide a failed native call.
            _sample_integral_eager(*args,**options)
            raise error
        if result is not None:return result
    if wide:
        if not contributions and not bool((owners>=0).any()) and q.dtype==torch.float64 and not bool(q[...,3].view(torch.int64).any()):
            return q.new_zeros((*q.shape[:-2],3)),q.new_zeros(q.shape[:-2])
        raise OverflowError('opaque quadrature native fallback overflow; canonical pixels required')
    return _sample_integral_eager(*args,**options)


def _sample_integral_eager(packed,camera,clip,normals,owners,q,workload,*,contributions=False,_shared_curves=None):
    from . import render as primitives
    from ._deferred import _shade,_unit
    from ._curve_math import sum_last
    from ._reference_channels import multiply
    v,f=packed.vertices,packed.faces;batch=camera.batch_size
    face=owners.reshape(batch,1,-1,1).contiguous();active=face>=0;raw=q[...,:3].reshape(batch,1,-1,1,3)
    # Native quadrature has validated inputs and initialized every unused slot.
    # No selected owner means _shade would return the same positive-zero array.
    # Keep weights/reduction below unchanged, and all differentiated replay fresh.
    if not torch.is_grad_enabled() and not bool(active.any()):
        # Only the all-positive-zero bit pattern proves these reductions inert.
        # Preserve signed zero, tiny contributions, nonfinite checks and replay.
        if not contributions and q.dtype==torch.float64 and not bool(q[...,3].view(torch.int64).any()):
            return q.new_zeros((*q.shape[:-2],3)),q.new_zeros(q.shape[:-2])
        values=q.new_zeros((*q.shape[:-1],3))
    else:
        if len(f):
            cw=clip[:,3][f.long()][face.clamp_min(0).long()];cw=torch.where(active[...,None],cw,torch.ones_like(cw))
            beta=_PerspectiveWeights.apply(raw,cw,active).contiguous()
        else:beta=raw*0
        def interpolate(a):return primitives.interpolate(face,beta,a.contiguous())[...,0,:]
        world=interpolate(v[f.long()]);covered=active[...,0]
        n=interpolate(normals);unit=n.new_zeros(n.shape)
        if bool(covered.any()):unit[covered]=_unit(n[covered],'opaque interpolated normals')
        mid=torch.where(covered,packed.face_material[face[...,0].clamp_min(0).long()] if len(f) else face[...,0],-1)
        buffer=dict(face_index=face[...,0],world_position=world,normal=unit,_face=face,_weights=beta,_covered=covered,material_index=mid,mask=covered.to(v.dtype))
        values=_shade(packed,camera,buffer,point_block_limit=workload['principled_point_block_limit'],group_point_limit=workload['principled_group_point_limit'],_shared_curves=_shared_curves).reshape(batch,-1,800,3)
    weights=q[...,3];weighted=multiply(values,weights[...,None]);rgb=sum_last(weighted.transpose(-1,-2))
    alpha=sum_last(weights)
    if contributions:return rgb,alpha,torch.cat((weighted,weights[...,None]),-1)
    return rgb,alpha


class _PrincipalTransport(torch.autograd.Function):
    @staticmethod
    def forward(ctx,rgb,alpha,cx,cy,dx,dy):
        ctx.save_for_backward(dx,dy);ctx.set_materialize_grads(False)
        return rgb,alpha
    @staticmethod
    def backward(ctx,grgb,galpha):
        if torch.is_grad_enabled():raise RuntimeError('opaque camera transport supports first derivatives only')
        from ._curve_math import sum_last
        from ._reference_channels import sum3
        dx,dy=ctx.saved_tensors;result=[]
        for need,flux in zip(ctx.needs_input_grad[2:4],(dx,dy)):
            if not need:result.append(None);continue
            value=flux[...,3]*0
            if grgb is not None:value=sum3(grgb*flux[...,:3])
            if galpha is not None:value=value+galpha*flux[...,3]
            result.append(sum_last(value.reshape(len(value),-1)))
        return grgb,galpha,*result,None,None


def _principal_flux(packed,camera,pixels,keep,clip,normals,cuts,curves,h,w,ox,oy,workload,limit,batch_rule=0):
    if batch_rule in (3,4):return _principal_flux64(packed,camera,pixels,keep,clip,normals,cuts,curves,h,w,ox,oy,workload,limit,batch_rule)
    from . import render as primitives
    v,f=packed.vertices,packed.faces;batch=camera.batch_size;flux=[]
    # No retained boundary shading graph. Each edge's800-slot storage dies
    # before the next edge, and all die before ordinary volume quadrature.
    with torch.no_grad():
        for axis,needed in enumerate((camera.cx.requires_grad,camera.cy.requires_grad)):
            total=v.new_zeros((batch,h,w,4))
            if needed:
                for side in range(2):
                    owners=[];samples=[]
                    for b in range(batch):
                        first,last=int(packed.face_offsets[b]),int(packed.face_offsets[b+1]);original=torch.arange(first,last,dtype=torch.int32,device=v.device)[keep[first:last]]
                        ids,q=primitives.opaque_boundary_quadrature(pixels.detach(),f[original.long()].contiguous(),cuts[original.long()].detach().contiguous(),h,w,edge=2*axis+side,origin=(ox,oy),workspace_limit_bytes=limit,curves=None if curves is None else curves[original.long()].detach().contiguous())
                        ids=torch.where(ids>=0,original[ids.clamp_min(0).long()] if len(original) else ids.new_zeros(ids.shape),-1)
                        owners.append(ids);samples.append(q)
                    ids=torch.stack(owners);q=torch.stack(samples)
                    colour,alpha=_sample_integral(packed,camera,clip,normals,ids,q,workload)
                    edge_value=torch.cat((colour,alpha[...,None]),-1).reshape(batch,h,w,4)
                    total=total+edge_value if side==0 else total-edge_value
                    del ids,q,owners,samples,colour,alpha,edge_value
            flux.append(total)
    return flux


def _forward_selected_faces(pixels,keep,cuts,curves,selected,*,compact=False):
    """Closed AABB exclusion only, retaining the full primitive input domain."""
    from ._deferred import _finite
    # Native quadrature validates ALL projected vertices, including unreferenced
    # ones, and all kept cut/curve rows before candidate selection. Preserve
    # those obligations even when the compact call excludes off-window faces.
    # Face index validation already occurs for every face in _normals.
    _finite(pixels,'opaque quadrature coordinates')
    if bool((pixels.abs()>1048576).any()):raise ValueError('opaque quadrature coordinates must be bounded')
    _finite(cuts if compact else cuts[keep],'opaque quadrature cuts')
    if curves is not None:_finite(curves[keep],'opaque quadrature curves')
    return selected


def shade(scene,camera,*,outputs,cull_mode,background,workspace_limit_bytes,_region=None,_workload=None,_batch_cursors=None,_batch_first=True,_prepared_geometry=None,_shared_geometry=None,_empty_only=False,_albedo_cotangent=None,_batch_rule=0,_partition_pool=None,_partition_key=None):
    from . import scene as scene_types,material,light,mesh as geometry,render as primitives
    from .camera import Camera
    from ._deferred import _resolve_camera,_plan_workload,_validate_bindings,_shade,_finite,_unit,AttributeSource,TextureSource
    from ._curve_math import sum_last
    from ._reference_channels import multiply
    if _empty_only and (torch.is_grad_enabled() or _batch_cursors is not None):raise ValueError('empty preflight requires ordinary forward execution')
    from ._opaque_sh import profile as sh_profile, texture_cuts as sh_texture_cuts
    sh=sh_profile(scene,_resolve_camera(scene,camera)[0])
    # Caller-supplied shaders and unqualified illumination modes cannot silently
    # inherit this finite nonlinear Principled profile.
    for prim in scene.prims:
        if type(prim)==scene_types.Material:
            if not sh and prim.evaluator is not material.principled:raise NotImplementedError('opaque shaded scene profile requires Principled or orthographic classic SH')
            for name in ('albedo','roughness','f0'):
                value=prim.parameters.get(name)
                if isinstance(value,torch.Tensor) and value.shape!=(torch.Size([]) if name=='roughness' else torch.Size([3])):
                    raise NotImplementedError('opaque material fields use constants or explicit attribute/texture sources')
        if not sh and type(prim)==scene_types.Light and prim.evaluator is not light.directional:raise NotImplementedError('opaque shaded scene profile currently requires directional lighting')
    if sh and _empty_only:return None
    camera_base,_=_resolve_camera(scene,camera)
    if camera_base.fx.device.type not in ('cpu','cuda') or camera_base.fx.dtype!=torch.float64:raise NotImplementedError('opaque nonlinear scene profile is CPU/CUDA float64')
    if type(workspace_limit_bytes)!=int or not 1048576<workspace_limit_bytes<=67108864:raise ValueError('opaque scene workspace budget')
    # Charge every possible quadrature sample through the existing material/source/
    # lighting planner, before allocation. This charges the resident path;
    # the streaming entry supplies an already admitted window workload.
    if _workload is None:
        budget_camera=Camera.from_intrinsics(camera_base.fx,camera_base.fy,camera_base.cx,camera_base.cy,camera_base.image_height,camera_base.image_width*800,camera_base.near,camera_base.far,pose=camera_base.pose)
    #64 scalar slots/pixel conservatively cover two retained RGBA flux fields,
    #edge result/reduction copies and first-VJP products. Streaming charges this
    #against its region reserve; direct composition charges the full extent.
    transport=torch.is_grad_enabled() and (camera_base.cx.requires_grad or camera_base.cy.requires_grad)
    flux_reserve=_flux_reserve(scene,camera_base)
    workload=_workload if _workload is not None else _plan_workload(scene,budget_camera,1,workspace_limit_bytes-1048576-_extra_geometry_bytes(scene,camera_base)-flux_reserve,outputs,background)
    # Private, one-pixel forward scope only. The caller guards all input versions
    # before every batch; no prepared geometry is retained on autograd ctx.
    if _prepared_geometry is not None and (torch.is_grad_enabled() or _batch_cursors is None):
        raise ValueError('prepared opaque geometry requires a forward continuation')
    fixed_albedo=_shared_geometry is not None and 'constant_albedo' in _shared_geometry
    if _albedo_cotangent is not None and not fixed_albedo:raise ValueError('native albedo VJP requires explicit fixed replay')
    if fixed_albedo:
        from ._opaque_albedo_replay import _profile
        if _profile(scene,camera) is not _shared_geometry['constant_albedo'] or background is not None:raise ValueError('fixed albedo preparation identity differs')
    if _shared_geometry is not None and torch.is_grad_enabled() and not fixed_albedo:raise ValueError('shared opaque preparation is forward-only')
    cached=None if _prepared_geometry is None else _prepared_geometry.get('geometry')
    if cached is None:
        base=None if _shared_geometry is None else _shared_geometry.get('base')
        if base is None:
            packed,camera,pixels,keep,clip=_project(scene,camera,cull_mode,detach_principal=transport);_validate_bindings(packed)
            if packed.vertices.dtype!=torch.float64:raise NotImplementedError('opaque nonlinear scene profile is CPU/CUDA float64')
            cuts=packed.vertices.new_empty((len(packed.faces),3,0)) if sh else _cuts(packed,clip);normals=_normals(packed)
            if _shared_geometry is not None:_shared_geometry['base']=(packed,camera,pixels,keep,clip,cuts,normals)
        else:packed,camera,pixels,keep,clip,cuts,normals=base
        v,f=packed.vertices,packed.faces;h,w=camera.image_height,camera.image_width;batch=camera.batch_size
        oy,ox=0,0
        if _region is not None:oy,ox,h,w=_region
        bounds=None if _shared_geometry is None else _shared_geometry.get('face_bounds')
        if bounds is None:
            bounds=_face_bounds(pixels,f)
            if _shared_geometry is not None:_shared_geometry['face_bounds']=bounds
        selected=_select_bounds(keep,bounds,ox,oy,w,h)
        if sh:
            cuts=sh_texture_cuts(packed,pixels,selected,(oy,ox,h,w),compact=_batch_rule==4,cut_limit=40 if _batch_rule==4 else 16)
            horizon=cuts[...,:0];curves=None
        else:horizon,curves=_window_horizons(packed,camera,pixels,clip,normals,keep,selected,bounds,16-cuts.shape[-1],_shared_geometry)
        # Generated columns depend on the candidate faces in this window. A
        # neighbour's column must not change this pixel's native quadrature rule.
        # Curve-only forward windows keep the same linear-column count (zero).
        # The native rule is chosen per pixel, and foreign zero conic rows are
        # inert. Linear horizons and differentiated composition retain fallback.
        # Reuse the bounded canonical-pixel fallback; never drop a horizon or
        # change the meaning of explicitly supplied material/native cut columns.
        if not _empty_only and h*w>1 and (horizon.shape[-1] or (curves is not None and torch.is_grad_enabled() and _albedo_cotangent is None)):
            raise OverflowError('opaque quadrature window horizon layout overflow; canonical pixels required')
        cuts=torch.cat((cuts,horizon),-1).contiguous()
        # The checked partition binds the same closed candidate set in forward
        # and replay. Excluded faces are strictly off-window; retain contacts,
        # validate the entire input domain, and let indexing scatter zero VJPs.
        if not torch.is_grad_enabled() or fixed_albedo or _partition_pool is not None or _batch_rule==4:keep=_forward_selected_faces(pixels,keep,cuts,curves,selected,compact=_batch_rule==4)
        if _prepared_geometry is not None:_prepared_geometry['geometry']=(packed,camera,pixels,keep,clip,cuts,normals,curves)
    else:packed,camera,pixels,keep,clip,cuts,normals,curves=cached
    if background is not None:
        if not isinstance(background,torch.Tensor) or background.dtype!=packed.vertices.dtype:raise TypeError('background must match scene dtype')
        if background.device!=packed.vertices.device or background.shape not in ((3,),(camera.batch_size,3)):raise ValueError('background shape/device mismatch')
        _finite(background,'background')
    v,f=packed.vertices,packed.faces;h,w=camera.image_height,camera.image_width;batch=camera.batch_size
    oy,ox=0,0
    if _region is not None:oy,ox,h,w=_region
    # Closed face bounds exclude every face only after full geometry, cut and
    # binding validation. With no owners the native rule emits positive zeros.
    if (not torch.is_grad_enabled() or fixed_albedo) and _batch_cursors is None:
        if not bool(keep.any()):
            alpha=v.new_zeros((batch,h,w));rgb=v.new_zeros((batch,h,w,3))
            if background is not None:rgb=rgb+(1-alpha[...,None])*background.reshape(-1,1,1,3)
            result=dict(colour=_finite(rgb,'opaque colour'),mask=_finite(alpha,'opaque coverage'))
            return {key:result[key] for key in outputs}
        if _empty_only:return None

    owners=[];samples=[]
    if _batch_cursors is not None and (h!=1 or w!=1 or len(_batch_cursors)!=batch):raise ValueError("opaque batch scene requires one pixel and one cursor per frame")
    receipts=[]
    flux=_principal_flux(packed,camera,pixels,keep,clip,normals,cuts,curves,h,w,ox,oy,workload,workspace_limit_bytes,_batch_rule) if transport and (_batch_cursors is None or _batch_first) else None
    for b in range(batch):
        first,last=int(packed.face_offsets[b]),int(packed.face_offsets[b+1]);original=torch.arange(first,last,dtype=torch.int32,device=v.device)[keep[first:last]]
        if _batch_cursors is None:
            ids,q=primitives.opaque_material_quadrature(pixels,f[original.long()].contiguous(),_cut_rows(cuts,original,keep,_batch_rule).contiguous(),h,w,origin=(ox,oy),workspace_limit_bytes=workspace_limit_bytes,curves=None if curves is None else curves[original.long()].contiguous())
        elif _batch_cursors[b] is None:
            ids=torch.full((1,800),-1,dtype=torch.int32,device=v.device);q=v.new_zeros((1,800,4));receipts.append(None)
        elif _partition_pool is not None:
            if curves is not None:raise ValueError('partition reuse is linear-cut only')
            ids,q,receipt=_partition_pool.emit(pixels.contiguous(),f[original.long()].contiguous(),_cut_rows(cuts,original,keep,_batch_rule).contiguous(),(ox,oy),_batch_rule,_batch_cursors[b],(_partition_key,b))
            receipts.append(receipt)
        else:
            ids,q,receipt=primitives.opaque_material_quadrature_batch(pixels,f[original.long()].contiguous(),_cut_rows(cuts,original,keep,_batch_rule).contiguous(),cursor=_batch_cursors[b],rule=_batch_rule,origin=(ox,oy),workspace_limit_bytes=workspace_limit_bytes,curves=None if curves is None else curves[original.long()].contiguous())
            receipts.append(receipt)
        ids=torch.where(ids>=0,original[ids.clamp_min(0).long()] if len(original) else ids.new_zeros(ids.shape),-1)
        owners.append(ids);samples.append(q)
    owners=torch.stack(owners);q=torch.stack(samples)
    shared_curves=None if _shared_geometry is None or torch.is_grad_enabled() else _shared_geometry.setdefault('material_curves',{})
    explicit_gradient=None
    if _batch_cursors is None:
        if _albedo_cotangent is None:rgb,alpha=_sample_integral(packed,camera,clip,normals,owners,q,workload,_shared_curves=shared_curves)
        else:rgb,alpha,explicit_gradient=_sample_integral(packed,camera,clip,normals,owners,q,workload,_shared_curves=shared_curves,_albedo_cotangent=_albedo_cotangent)
    elif _albedo_cotangent is None:rgb,alpha,weighted=_sample_integral(packed,camera,clip,normals,owners,q,workload,contributions=True,_shared_curves=shared_curves)
    else:rgb,alpha,weighted,explicit_gradient=_sample_integral(packed,camera,clip,normals,owners,q,workload,contributions=True,_shared_curves=shared_curves,_albedo_cotangent=_albedo_cotangent)
    rgb=rgb.reshape(batch,h,w,3);alpha=alpha.reshape(batch,h,w)
    if flux is not None:rgb,alpha=_PrincipalTransport.apply(rgb,alpha,camera.cx,camera.cy,*flux)
    if background is not None:rgb=rgb+((1 if _batch_cursors is None or _batch_first else 0)-alpha[...,None])*background.reshape(-1,1,1,3)
    result=dict(colour=_finite(rgb,'opaque colour'),mask=_finite(alpha,'opaque coverage'))
    if _batch_cursors is not None:
        result['_contributions']=weighted.reshape(batch,800,4);result['_receipts']=tuple(receipts)
        if explicit_gradient is not None:result['_albedo_gradient']=explicit_gradient
        return result
    answer={key:result[key] for key in outputs}
    if explicit_gradient is not None:answer['_albedo_gradient']=explicit_gradient
    return answer


def _principal_flux64(packed,camera,pixels,keep,clip,normals,cuts,curves,h,w,ox,oy,workload,limit,rule=3):
    from ._opaque_quadrature import opaque_boundary_batch64,opaque_boundary_batch80
    boundary=opaque_boundary_batch80 if rule==4 else opaque_boundary_batch64
    from ._opaque_pixel import _advance
    if (h,w)!=(1,1) or curves is not None:raise ValueError('SH64 boundary stream requires one linear-cut cell')
    v,f=packed.vertices,packed.faces;batch=camera.batch_size;flux=[]
    with torch.no_grad():
        for axis,needed in enumerate((camera.cx.requires_grad,camera.cy.requires_grad)):
            total=v.new_zeros((batch,1,1,4))
            if needed:
                for side in range(2):
                    edge_value=v.new_zeros((batch,1,1,4))
                    for b in range(batch):
                        first,last=int(packed.face_offsets[b]),int(packed.face_offsets[b+1]);original=torch.arange(first,last,dtype=torch.int32,device=v.device)[keep[first:last]]
                        cursor=(0,0,0,0);carry=v.new_zeros((4,));versions=(pixels._version,f._version,cuts._version)
                        for index in range(80 if rule==4 else 64):
                            if versions!=(pixels._version,f._version,cuts._version):raise RuntimeError('SH boundary inputs changed during continuation')
                            owners,q,r=boundary(pixels.detach(),f[original.long()].contiguous(),_cut_rows(cuts,original,keep,rule).detach().contiguous(),edge=2*axis+side,cursor=cursor,origin=(ox,oy),workspace_limit_bytes=limit)
                            owners=torch.where(owners>=0,original[owners.clamp_min(0).long()] if len(original) else owners.new_zeros(owners.shape),-1)
                            # _sample_integral expects the full scene batch. Inactive frames have exact zero weights.
                            ids=torch.full((batch,1,800),-1,dtype=torch.int32,device=v.device);samples=v.new_zeros((batch,1,800,4));ids[b]=owners;samples[b]=q
                            colour,alpha,weighted=_sample_integral(packed,camera,clip,normals,ids,samples,workload,contributions=True)
                            nxt=_advance((cursor,),(r,),rule=rule)[0]
                            if index==0 and nxt is None:carry=torch.cat((colour[b].reshape(3),alpha[b].reshape(1)),-1)
                            else:carry.copy_(torch.ops.gffx.shading_ordered_rgba(carry,weighted[b].reshape(800,4)[:r[2]].contiguous()))
                            del owners,q,ids,samples,colour,alpha,weighted
                            if nxt is None:break
                            cursor=nxt
                        else:raise OverflowError('SH boundary finite continuation exhausted')
                        edge_value[b,0,0]=carry
                    total=total+edge_value if side==0 else total-edge_value
            flux.append(total)
    return flux


def _cut_rows(cuts,original,keep,rule):
    if rule!=4:return cuts[original.long()]
    ids=keep.nonzero().flatten()
    positions=torch.searchsorted(ids,original.long())
    if len(original) and (bool((positions>=len(ids)).any()) or not torch.equal(ids[positions],original.long())):raise ValueError('compact cut owner mapping changed')
    if len(cuts)!=len(ids):raise ValueError('compact cut row extent differs')
    return cuts[positions]
