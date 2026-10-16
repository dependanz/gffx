"""Explicit bounded first-VJP replay, with call-local fixed geometry only."""
import torch


def _profile(scene,camera):
    from . import scene as types,material,light
    from ._opaque_stream import _snapshot
    mats=[p for p in scene.prims if type(p)==types.Material]
    lamps=[p for p in scene.prims if type(p)==types.Light]
    if len(mats)!=1 or len(lamps)!=1:raise NotImplementedError('albedo replay needs one material and one light')
    mat=mats[0];lamp=lamps[0];p=mat.parameters
    if mat.evaluator is not material.principled or lamp.evaluator is not light.directional:raise NotImplementedError('albedo replay requires Principled and directional lighting')
    if set(p)-{'albedo','roughness','f0','budget','two_sided'} or p.get('two_sided',False) is not False:raise NotImplementedError('albedo replay constant one-sided profile')
    a,r,f=(p.get(k) for k in ('albedo','roughness','f0'))
    if not all(isinstance(x,torch.Tensor) for x in (a,r,f)) or a.shape!=(3,) or r.ndim!=0 or f.shape!=(3,):raise NotImplementedError('albedo replay requires constant RGB albedo/F0 and scalar roughness')
    if a.dtype!=torch.float64 or a.device.type not in ('cpu','cuda') or camera.batch_size!=1:raise NotImplementedError('albedo replay requires batch1 CPU/CUDA float64')
    if not isinstance(p.get('budget'),material.PrincipledBudget) or p['budget']._metadata['admission']!='accepted':raise ValueError('albedo replay requires accepted resource')
    if set(lamp.parameters)!={'directions','irradiance'} or lamp.parameters['directions'].shape not in ((1,3),(1,1,3)):raise NotImplementedError('albedo replay requires one fixed directional sample')
    tensors=[];tree=_snapshot((scene,camera,None),tensors,{})
    if any(t.requires_grad and t is not a for t in tensors):raise NotImplementedError('albedo replay does not differentiate geometry or other parameters')
    index=next(i for i,t in enumerate(tensors) if t is a)
    def count(node):
        if isinstance(node,tuple) and len(node)==2 and node[0]=='tensor' and node[1]==index:return 1
        return sum(count(x) for x in node) if isinstance(node,(tuple,list)) else 0
    if count(tree)!=1:raise NotImplementedError('albedo replay target must not alias another binding')
    return a


@torch.no_grad()
def replay(scene,camera,region,colour_cotangent,mask_cotangent,*,initial_gradient=None,cull_mode=2,workspace_limit_bytes=67108864):
    from . import _opaque_stream as stream
    from ._opaque_scene import shade
    from ._deferred import _resolve_camera
    base,_=_resolve_camera(scene,camera)
    if type(region)!=tuple or len(region)!=4 or any(type(q)!=int for q in region):raise TypeError('replay region requires integer y,x,height,width')
    y,x,h,w=region
    if min(y,x)<0 or min(h,w)<1 or h*w>256 or y+h>base.image_height or x+w>base.image_width:raise ValueError('replay region outside extent or256pixel cap')
    if type(cull_mode)!=int or cull_mode not in (1,2,3):raise ValueError('invalid replay cull mode')
    if type(workspace_limit_bytes)!=int or not 1048576<workspace_limit_bytes<=67108864:raise ValueError('replay workspace budget')
    a=_profile(scene,base)
    for value,shape in ((colour_cotangent,(1,h,w,3)),(mask_cotangent,(1,h,w)),(initial_gradient,(3,))):
        if value is None:
            if shape==(3,):continue
            raise TypeError('both replay cotangents required')
        if not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=a.dtype or value.device!=a.device:raise ValueError('replay cotangent/carry shape,dtype or device')
        if value.requires_grad or not bool(torch.isfinite(value).all()):raise ValueError('replay cotangents/carry must be finite first-order constants')
    originals=[];tree=stream._snapshot((scene,camera,None),originals,{});structure=repr(tree)
    extra=[colour_cotangent,mask_cotangent]+([] if initial_gradient is None else [initial_gradient])
    signature=tuple((id(t),t._version) for t in originals+extra)
    def guard():
        current=[];current_tree=stream._snapshot((scene,camera,None),current,{})
        if repr(current_tree)!=structure or tuple((id(t),t._version) for t in current+extra)!=signature:raise RuntimeError('opaque replay inputs modified during call')
    guard();index=next(i for i,t in enumerate(originals) if t is a)
    temporary=[t.detach().requires_grad_(i==index) for i,t in enumerate(originals)]
    ss,cc,_=stream._restore(tree,temporary);target=temporary[index]
    extra_bytes=sum(t.numel()*t.element_size() for t in extra)+6*a.element_size()
    config,tensors=stream._configuration(ss,cc,outputs=('colour','mask'),cull_mode=cull_mode,background=None,workspace_limit_bytes=workspace_limit_bytes-extra_bytes,tile_pixels=1,region=region)
    # Eight native outputs/scratch fit the existing1MiB reserve; quadrature
    # arrays fit the retained one-pixel eager buffers. No wide eager graph.
    cap=min(8,w) if config['workload'].get('components',{}).get('buffers',0)>=8*800*76+4096 else 1
    if cap>1:config['workload']=dict(config['workload'],opaque_native_window=True)
    from .scene import Material
    from ._principled_groups import _fixed_albedo_curve
    params=next(p.parameters for p in ss.prims if type(p)==Material)
    # CUDA engine callbacks otherwise run outside this call's scoped context.
    # Keep this serial replay on the caller thread and restore its setting.
    with _fixed_albedo_curve(target,params['roughness'],params['budget']),torch.autograd.set_multithreading_enabled(False):
        shared={'region':region,'constant_albedo':target}
        restored_s,restored_c,_=stream._restore_forward(config,tensors,shared)
        try:
            empty=shade(restored_s,restored_c,outputs=('colour','mask'),cull_mode=cull_mode,background=None,workspace_limit_bytes=config['limit'],_region=region,_workload=config['workload'],_shared_geometry=shared,_empty_only=True)
        except OverflowError:empty=None
        guard();carry=a.new_zeros(3) if initial_gradient is None else initial_gradient.clone();zero=a.new_zeros(3)
        result={'colour':a.new_empty((1,h,w,3)),'mask':a.new_empty((1,h,w))}
        if empty is not None:
            for _ in range(h*w):carry.add_(zero)
            result.update(empty)
        else:
            for yy in range(h):
                for x0 in range(0,w,cap):
                    count=min(cap,w-x0);group=None
                    if count>1:
                        guard()
                        with torch.enable_grad():group=stream._evaluate(config,tensors,(yy,x0,1,count),shared,albedo_cotangent=colour_cotangent[:,yy:yy+1,x0:x0+count])
                        if group is not None and any(v.requires_grad for v in group.values()):raise RuntimeError('native group unexpectedly retained a graph')
                    for xx in range(x0,x0+count):
                        guard()
                        with torch.enable_grad():
                            if group is None:value=stream._evaluate(config,tensors,(yy,xx,1,1),shared,albedo_cotangent=colour_cotangent[:,yy:yy+1,xx:xx+1])
                            else:
                                j=xx-x0
                                value={k:group[k][:,:,j:j+1] for k in ('colour','mask')}
                                value['_albedo_gradient']=group['_albedo_gradient'].reshape(-1,3)[j] if '_albedo_gradient' in group else zero
                            outputs=[];cotangents=[]
                            if '_albedo_gradient' in value:
                                gradient=value.pop('_albedo_gradient')
                                if any(v.requires_grad for v in value.values()):raise RuntimeError('native replay unexpectedly retained a graph')
                            else:
                                for key,g in (('colour',colour_cotangent),('mask',mask_cotangent)):
                                    if value[key].requires_grad:outputs.append(value[key]);cotangents.append(g[:,yy:yy+1,xx:xx+1].contiguous())
                                gradient,=torch.autograd.grad(outputs,(target,),cotangents,allow_unused=True) if outputs else (None,)
                        for key in result:result[key][:,yy:yy+1,xx:xx+1].copy_(value[key])
                        carry.add_(zero if gradient is None else gradient)
                        del value,outputs,cotangents,gradient
                    del group
        guard()
        if not bool(torch.isfinite(carry).all()):raise ValueError('nonfinite replay gradient carry')
        result['albedo_gradient']=carry
        return result
