"""Bounded opaque windows; stateless forward and canonical pixel-order VJP replay."""
from collections.abc import Mapping
from dataclasses import fields,is_dataclass
from copy import copy
import torch
from .camera import Camera
from .scene import Scene
from .material import PrincipledBudget

def _snapshot(value,tensors,indices):
    if isinstance(value,torch.Tensor):
        key=id(value)
        if key not in indices:indices[key]=len(tensors);tensors.append(value)
        return ('tensor',indices[key])
    if isinstance(value,PrincipledBudget):
        for t in value._data.values():
            if t.requires_grad:raise ValueError('opaque resource tensors are immutable constants')
            _snapshot(t,tensors,indices)
        return ('constant',value)
    if isinstance(value,Camera):return ('camera',tuple(_snapshot(getattr(value,k),tensors,indices) for k in ('fx','fy','cx','cy','pose')),value.image_height,value.image_width,value.near,value.far,value.projection_kind)
    if isinstance(value,Scene):return ('scene',_snapshot(value.prims,tensors,indices))
    if is_dataclass(value):return ('record',type(value),tuple((f.name,_snapshot(getattr(value,f.name),tensors,indices)) for f in fields(value)))
    if isinstance(value,Mapping):return ('mapping',tuple((k,_snapshot(v,tensors,indices)) for k,v in value.items()))
    if isinstance(value,(tuple,list)):return ('sequence',type(value),tuple(_snapshot(v,tensors,indices) for v in value))
    return ('constant',value)

def _restore(node,tensors):
    kind=node[0]
    if kind=='tensor':return tensors[node[1]]
    if kind=='constant':return node[1]
    if kind=='camera':
        fx,fy,cx,cy,pose=(_restore(n,tensors) for n in node[1]);return (Camera.from_orthographic if node[6]=='orthographic' else Camera.from_intrinsics)(fx,fy,cx,cy,node[2],node[3],node[4],node[5],pose=pose)
    if kind=='scene':return Scene(_restore(node[1],tensors))
    if kind=='record':return node[1](**{k:_restore(v,tensors) for k,v in node[2]})
    if kind=='mapping':return {k:_restore(v,tensors) for k,v in node[1]}
    if kind=='sequence':return node[1](_restore(n,tensors) for n in node[2])
    raise AssertionError('invalid opaque snapshot')

def _restore_forward(config,tensors,shared):
    if shared is None:return _restore(config['tree'],tensors)
    if torch.is_grad_enabled() and 'constant_albedo' not in shared:raise ValueError('shared snapshot restoration requires forward-only execution')
    if 'restored' not in shared:shared['restored']=_restore(config['tree'],tensors)
    return shared['restored']


def _windows(height,width,cap):
    for y in range(height):
        for x in range(0,width,cap):yield (y,x,1,min(cap,width-x))

def _evaluate(config,tensors,region,shared=None,*,albedo_cotangent=None):
    from ._opaque_scene import shade
    s,c,b=_restore_forward(config,tensors,shared)
    oy,ox=config.get('origin',(0,0));y,x,h,w=region;region=(oy+y,ox+x,h,w)
    if albedo_cotangent is not None and (not 1<=h*w<=8 or config['shape'][0]!=1):raise ValueError('explicit native replay is at most eight pixels')
    # A capacity-refused window may contain both fitting and overflowing pixels.
    # Retry fitting pixels on the old path. Only a refused single pixel enters
    # continuation; unsupported profiles and unrelated errors still propagate.
    try:
        return shade(s,c,outputs=('colour','mask'),cull_mode=config['cull'],background=b,workspace_limit_bytes=config['limit'],_region=region,_workload=config['workload'],_shared_geometry=shared,**({} if albedo_cotangent is None else dict(_albedo_cotangent=albedo_cotangent)))
    except OverflowError as error:
        if 'quadrature' not in str(error) or 'overflow' not in str(error).lower():raise
    if h*w>1 and albedo_cotangent is not None:return None
    if h*w>1:
        rows=[]
        for yy in range(h):
            columns=[_evaluate(config,tensors,(y+yy,x+xx,1,1),shared) for xx in range(w)]
            rows.append({k:torch.cat([v[k] for v in columns],2) for k in ('colour','mask')})
        return {k:torch.cat([v[k] for v in rows],1) for k in ('colour','mask')}
    if albedo_cotangent is not None:
        from ._opaque_albedo_pixel import replay
        return replay(config,tensors,region,shared,albedo_cotangent)
    from ._opaque_pixel import pixel
    return pixel(config,tensors,region,shared)

class _Replay(torch.autograd.Function):
    @staticmethod
    def forward(ctx,config,*tensors):
        ctx.config=config;ctx.save_for_backward(*tensors);ctx.set_materialize_grads(False)
        b,h,w=config['shape'];like=tensors[config['like']]
        rgb=like.new_empty((b,h,w,3));mask=like.new_empty((b,h,w))
        oy,ox=config.get('origin',(0,0));shared={'region':(oy,ox,h,w)};versions=tuple(t._version for t in tensors)
        # Prepare the same guarded geometry once. No sample storage is allocated
        # by this probe; an occupied request continues through admitted tiles.
        from ._opaque_scene import shade
        scene,camera,background=_restore_forward(config,tensors,shared)
        try:
            empty=shade(scene,camera,outputs=('colour','mask'),cull_mode=config['cull'],background=background,workspace_limit_bytes=config['limit'],_region=(oy,ox,h,w),_workload=config['workload'],_shared_geometry=shared,_empty_only=True)
        except OverflowError:
            # A union can require more cut columns than its separate pixels.
            # Preserve local admission and its errors in the ordinary tile path.
            empty=None
        if tuple(t._version for t in tensors)!=versions:raise RuntimeError('opaque inputs modified during forward window')
        if empty is not None:
            rgb.copy_(empty['colour']);mask.copy_(empty['mask']);return rgb,mask
        for region in _windows(h,w,config['cap']):
            if tuple(t._version for t in tensors)!=versions:raise RuntimeError('opaque inputs modified during forward window')
            y,x,hh,ww=region;result=_evaluate(config,tensors,region,shared)
            rgb[:,y:y+hh,x:x+ww]=result['colour'];mask[:,y:y+hh,x:x+ww]=result['mask']
        if tuple(t._version for t in tensors)!=versions:raise RuntimeError('opaque inputs modified during forward window')
        return rgb,mask
    @staticmethod
    def backward(ctx,gcolour,gmask):
        if torch.is_grad_enabled():raise RuntimeError('opaque streaming supports first derivatives only')
        config=ctx.config;saved=ctx.saved_tensors;needs=ctx.needs_input_grad[1:];ids=[i for i,n in enumerate(needs) if n]
        accumulated=[torch.zeros_like(saved[i]) if i in ids else None for i in range(len(saved))]
        # One fixed ascending pixel order for every forward tile cap. This keeps
        # reduction partition independent and bounds each retained replay graph.
        _,h,w=config['shape']
        for region in _windows(h,w,1):
            y,x,_,_=region;temporary=[v.detach().requires_grad_(n) for v,n in zip(saved,needs)]
            with torch.enable_grad():
                result=_evaluate(config,temporary,region);outputs=[];cotangents=[]
                for name,g in (('colour',gcolour),('mask',gmask)):
                    if g is not None and result[name].requires_grad:outputs.append(result[name]);cotangents.append(g[:,y:y+1,x:x+1].contiguous())
                gradients=torch.autograd.grad(outputs,[temporary[i] for i in ids],cotangents,allow_unused=True) if outputs else [None]*len(ids)
            for i,g in zip(ids,gradients):
                if g is not None:accumulated[i].add_(g)
            del result,outputs,cotangents,gradients,temporary
        return (None,*accumulated)

def _configuration(scene,camera,*,outputs,cull_mode,background,workspace_limit_bytes,tile_pixels=None,region=None,sh_batch_profile=16):
    from ._deferred import _resolve_camera,_plan_workload
    if type(workspace_limit_bytes)!=int or not 1048576<workspace_limit_bytes<=67108864:raise ValueError('opaque stream workspace budget')
    if tile_pixels is not None and (type(tile_pixels)!=int or tile_pixels<1):raise ValueError('opaque tile pixels must be a positive integer')
    base,_=_resolve_camera(scene,camera);tensors=[];tree=_snapshot((scene,camera,background),tensors,{})
    # Outputs,cotangents,replay input adjoints and metadata all remain charged.
    # Packed batch count is returned by the planner's pixel-buffer estimate.
    sizes=[base.batch_size]
    for prim in scene.prims:
        for a in (getattr(prim,'vertices',None),prim.transform):
            if isinstance(a,torch.Tensor) and a.ndim==3:sizes.append(a.shape[0])
        for attr in getattr(prim,'attributes',{}).values():
            rank={'constant':1,'uniform':2,'vertex':2,'face_varying':3}[attr.domain]
            if attr.values.ndim==rank+1:sizes.append(attr.values.shape[0])
    batch=max(sizes);h,w=base.image_height,base.image_width;origin=(0,0)
    if region is not None:
        if type(region)!=tuple or len(region)!=4 or any(type(q)!=int for q in region):raise TypeError('opaque region must be an integer (y,x,height,width) tuple')
        y,x,hh,ww=region
        if min(y,x)<0 or min(hh,ww)<1 or y+hh>h or x+ww>w:raise ValueError('opaque region outside camera extent')
        origin=(y,x);h,w=hh,ww
    from ._opaque_scene import _extra_geometry_bytes
    from ._opaque_sh import _profile
    _profile(dict(sh_batch_profile=sh_batch_profile))
    geometry_reserve=_extra_geometry_bytes(scene,base,sh_batch_profile=sh_batch_profile)
    #32KiB/frame charges the continuation weighted RGBA buffer and receipts.
    reserve=1048576+65536+32768*batch+geometry_reserve+batch*h*w*(4*3+64)*base.fx.element_size()+sum(t.numel()*t.element_size()*2 for t in tensors if t.requires_grad)
    if sh_batch_profile==80:reserve+=h*w*(80*(4+12*batch)*4+512)+131072
    available=workspace_limit_bytes-reserve
    if available<=0:raise ValueError('opaque stream total output/input reserve exceeds budget')
    def plan(n):
        proxy=copy(base);proxy.image_height=1;proxy.image_width=n*800
        return _plan_workload(scene,proxy,1,available,('colour','mask'),background)
    workload=plan(1)
    from ._opaque_samples import window_capacity
    from ._opaque_sh import profile as sh_profile
    sh=sh_profile(scene,base)
    native_cap=0 if sh else window_capacity(workload,batch,min(w,tile_pixels or w))
    if native_cap:
        # Retain all one-pixel eager storage for fallback. Multi-pixel sample
        # composition may only use native storage (or an exact empty result).
        lo=native_cap;workload=dict(workload,opaque_native_window=True)
    else:
        lo=1;hi=1 if sh else min(w,tile_pixels or w)
        while lo<hi:
            mid=(lo+hi+1)//2
            try:candidate=plan(mid)
            except ValueError as e:
                if 'budget exceeded' not in str(e):raise
                hi=mid-1
            else:
                # Growing a geometry tile must not trade an admitted shader cohort
                # for thousands of smaller material calls. All storage is still
                # charged by the planner; a tight one-pixel fallback remains legal.
                if any(candidate[k]<workload[k] for k in ('principled_point_block_limit','principled_group_point_limit')):
                    hi=mid-1
                else:lo=mid;workload=candidate
    config=dict(tree=tree,origin=origin,shape=(batch,h,w),cap=lo,cull=cull_mode,limit=available+1048576,workload=workload,like=next(i for i,t in enumerate(tensors) if t.dtype==base.fx.dtype and t.device==base.fx.device),reserve_bytes=reserve,sh_batch_profile=sh_batch_profile,sh_storage_profile=sh_batch_profile)
    return config,tensors

def render(scene,camera,*,outputs,cull_mode,background,workspace_limit_bytes,tile_pixels=None,region=None):
    config,tensors=_configuration(scene,camera,outputs=outputs,cull_mode=cull_mode,background=background,workspace_limit_bytes=workspace_limit_bytes,tile_pixels=tile_pixels,region=region)
    from ._opaque_sh import profile as sh_profile,render as sh_render
    from ._deferred import _resolve_camera
    if sh_profile(scene,_resolve_camera(scene,camera)[0]):return sh_render(config,tensors,outputs)
    rgb,mask=_Replay.apply(config,*tensors)
    return {k:({'colour':rgb,'mask':mask}[k]) for k in outputs}


def window(scene,camera,region,*,outputs=('colour','mask'),cull_mode=2,background=None,workspace_limit_bytes=67108864,tile_pixels=None):
    """Render one explicit global-camera rectangle; no full-frame allocation or I/O.

    Output and cotangent reserves cover only this rectangle. The caller owns
    storage and global scheduling. Backward uses the same canonical pixel order.
    """
    if type(outputs) not in (tuple,list) or not outputs or any(type(k)!=str for k in outputs) or len(set(outputs))!=len(outputs) or not set(outputs)<={'colour','mask'}:raise ValueError('invalid opaque window outputs')
    if type(cull_mode)!=int or cull_mode not in (1,2,3):raise ValueError('invalid cull_mode')
    return render(scene,camera,outputs=outputs,cull_mode=cull_mode,background=background,workspace_limit_bytes=workspace_limit_bytes,tile_pixels=tile_pixels,region=region)
