"""Bounded native composition; explicit fixed-albedo replay owns its first VJP."""
import torch
RESERVE_BYTES=1048576


def window_capacity(workload,batch,requested):
    """Storage inventory: docs/OPAQUE_SAMPLE_COMPOSITION.md; scalar fallback."""
    needed=16*800*(2*4*8+3*4)+4096
    if batch!=1 or workload.get('backward',True) or workload.get('components',{}).get('buffers',0)<needed:return 0
    if workload.get('principled_point_block_limit')!=4 or workload.get('principled_group_point_limit')!=128:return 0
    return min(16,requested)


def compose(packed,camera,clip,normals,owners,q,workload,*,contributions=False,_shared_curves=None,_albedo_cotangent=None):
    from . import material,light
    from ._principled_groups import _prepare_material_curve
    from ._deferred import _light_packet
    backward=_albedo_cotangent is not None
    if (torch.is_grad_enabled() and not backward) or camera.batch_size!=1:return None
    if len(packed.materials)!=1 or len(packed.lights)!=1:return None
    mat=packed.materials[0];lamp=packed.lights[0];params=mat.parameters
    if mat.evaluator is not material.principled or lamp.evaluator is not light.directional:return None
    if set(params)-{'albedo','roughness','f0','budget','two_sided'} or params.get('two_sided',False) is not False:return None
    if set(lamp.parameters)!={'directions','irradiance'}:return None
    budget=params.get('budget');r=params.get('roughness');a=params.get('albedo');f=params.get('f0')
    if not isinstance(budget,material.PrincipledBudget) or budget._metadata['admission']!='accepted':return None
    if not all(isinstance(x,torch.Tensor) for x in (a,r,f)):return None
    if a.shape!=(3,) or f.shape!=(3,) or r.ndim!=0:return None
    count=q.numel()//3200
    if q.shape[-2:]!=(800,4) or not 1<=count<=16:return None
    if backward and contributions and count!=1:return None
    if workload.get('principled_point_block_limit')!=4 or workload.get('principled_group_point_limit')!=128:return None
    if not 1<=len(budget._data['cosine_nodes'])-1<=512:return None
    if lamp.parameters['directions'].shape not in ((1,3),(1,1,3)):return None
    values=(q,packed.vertices,clip,normals,a,r,f,*budget._data.values())
    if any(x.dtype!=torch.float64 or x.device!=q.device or not x.is_contiguous() for x in values):return None
    if any(x.dtype!=torch.int32 or x.device!=q.device or not x.is_contiguous() for x in (owners,packed.faces)):return None
    scoped=None
    if backward:
        from ._principled_groups import _active_albedo_curve,_fixed_replay_constants,_curve_storage
        scoped=_active_albedo_curve.get()
        if scoped is None or count>8 or _curve_storage(a)!=scoped['target_storage'] or budget is not scoped['budget']:return None
        if not _fixed_replay_constants(q,packed.vertices,clip,normals,r,f,camera.pose,lamp.parameters['directions'],lamp.parameters['irradiance']):return None
        g=_albedo_cotangent
        if g.numel()!=count*3 or g.dtype!=q.dtype or g.device!=q.device or g.requires_grad:raise ValueError('native albedo cotangent profile')
    if bool((packed.face_material!=0).any()):return None
    lo,hi=budget._metadata['domain']['roughness']
    if bool(((r<lo)|(r>hi)|~torch.isfinite(r)).any()):return None
    # Preparation belongs only to the guarded caller window. Every native input
    # domain is still checked on each invocation, and nothing is kept on ctx.
    state=(scoped['prepared'].setdefault('_native_samples',{}) if backward else
           ({} if _shared_curves is None else _shared_curves.setdefault('_native_samples',{})))
    inputs=(r,*budget._data.values(),packed.world_matrices[lamp.name],*lamp.parameters.values())
    signature=tuple((id(x),x._version) for x in inputs if isinstance(x,torch.Tensor))
    if state:
        if state['signature']!=signature:raise RuntimeError('opaque native preparation inputs modified')
    else:
        if backward and 'curve' in scoped['prepared']:curve=scoped['prepared']['curve']
        else:
            curve=_prepare_material_curve(budget,r.reshape(1))
            if backward:scoped['prepared']['curve']=curve
        packet=_light_packet(lamp,packed,0,packed.vertices[:1])
        if packet['direction'].shape!=(1,1,3) or not bool(packet['valid'].all()):return None
        state.update(signature=signature,curve=curve,packet=packet)
    cn,curve,partial,mean,width=state['curve'];packet=state['packet']
    args=(owners.reshape(-1,800),q.reshape(-1,800,4),packed.vertices,packed.faces,clip,normals,camera.pose[0,:3,3].contiguous(),cn[1:],budget._data['_clog'],curve,partial,mean,width,a,r.reshape(1),f,packet['direction'].reshape(3),packet['irradiance'].reshape(3))
    if backward:
        with torch.no_grad():weighted,rgba,gradient=torch.ops.gffx.shading_opaque_albedo(*args,_albedo_cotangent.contiguous().reshape(count,3))
    else:weighted,rgba=torch.ops.gffx.shading_opaque_samples(*args)
    rgb=rgba[:,:3].reshape(*q.shape[:-2],3);alpha=rgba[:,3].reshape(q.shape[:-2])
    if backward:
        gradient=gradient.reshape(3) if count==1 else gradient.reshape(*q.shape[:-2],3)
        return (rgb,alpha,weighted.reshape_as(q),gradient) if contributions else (rgb,alpha,gradient)
    return (rgb,alpha,weighted.reshape_as(q)) if contributions else (rgb,alpha)
