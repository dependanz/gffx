"""Renderer-only ordered groups with call-local recomputation, never a public bypass."""
import torch
from ._common import materialize
from ._principled_native import _error


from contextvars import ContextVar
from contextlib import contextmanager
_active_albedo_curve=ContextVar('gffx_explicit_albedo_curve',default=None)


def _curve_storage(t):
    return (t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device))


def _curve_signature(target,roughness,budget):
    return (target._version,roughness._version,tuple((k,id(v),v._version) for k,v in sorted(budget._data.items()) if isinstance(v,torch.Tensor)))


@contextmanager
def _fixed_albedo_curve(target,roughness,budget):
    # Owned by one explicit <=256pixel call, never by a saved autograd context.
    state=dict(target=target,roughness=roughness,budget=budget,prepared={},
        target_storage=_curve_storage(target),roughness_storage=_curve_storage(roughness),
        signature=_curve_signature(target,roughness,budget))
    token=_active_albedo_curve.set(state)
    try:yield
    finally:_active_albedo_curve.reset(token)


def _call_curve_state(keys,static,values):
    state=_active_albedo_curve.get()
    if state is None or 'albedo' not in keys or 'roughness' not in keys:return None
    ai=3+keys.index('albedo');ri=3+keys.index('roughness');a=values[ai];r=values[ri]
    if any(v.requires_grad for i,v in enumerate(values) if i!=ai):return None
    if a.shape!=(3,) or r.ndim!=0 or static.get('budget') is not state['budget']:return None
    if _curve_storage(a)!=state['target_storage'] or _curve_storage(r)!=state['roughness_storage']:return None
    if _curve_signature(state['target'],state['roughness'],state['budget'])!=state['signature']:raise RuntimeError('explicit replay material inputs modified')
    return state['prepared']


def _fixed_replay_constants(*values):
    state=_active_albedo_curve.get()
    if state is None or any(v.requires_grad or v.dtype!=torch.float64 or v.device!=state['target'].device for v in values):return False
    if _curve_signature(state['target'],state['roughness'],state['budget'])!=state['signature']:raise RuntimeError('explicit replay material inputs modified')
    return True


def workspace_bound(budget,points,samples,backward,*,shared_curve=False):
    # Conservative simultaneous tensor bound, in float64 elements.64*(K+1)
    # covers selected control gathering/interpolation/stacking, input copies,
    # old+concatenated curves/partials and the largest <=32-row native scratch.
    #192 pair elements cover values, Jacobians, cotangents and both VJP results.
    # docs/GROUPED_STORAGE_BOUND.md inventories the <=48*(K+1) peak.
    # Resource-owned constants and renderer-retained inputs are charged separately.
    nodes=len(budget._data['cosine_nodes'])-1
    return 8+8*((1 if shared_curve else points)*64*(nodes+1)+points*(64+samples*(192 if backward else 64)))


def capacity(budget,block,samples,backward,limit,*,shared_curve=False):
    # Fit INSIDE the renderer's existing single-group material reserve. Grouping
    # cannot silently enlarge the whole-frame planner's memory allowance.
    reserve=3*budget.workspace(3*block if samples else 0,backward=backward)
    reserve+=8*block*samples*(192 if backward else 64)
    available=min(limit,reserve)
    groups=min(32,128//block,1024//max(1,block*samples))
    while groups>1 and workspace_bound(budget,groups*block,samples,backward,shared_curve=shared_curve)>available:groups-=1
    return groups


def _prepare_material_curve(budget,roughness):
    cn,coefficients,u,width=budget._prepare_curve(roughness,with_width=True);d=budget._data
    curves=[];partials=[];means=[]
    for left in range(0,len(roughness),32):
        curve,partial=torch.ops.gffx.shading_loss_curve(d['_clog'],coefficients[left:left+32],u[left:left+32])
        _,mean=torch.ops.gffx.shading_loss_query(cn[1:],d['_clog'],d['_weights'],curve,partial,
            roughness.new_empty((0,)),torch.empty((0,),dtype=torch.int32,device=roughness.device))
        curves.append(curve);partials.append(partial);means.append(mean)
    return cn,torch.cat(curves),torch.cat(partials),torch.cat(means),width


def _shared_curve_state(keys,values):
    if 'roughness' in keys and values[3+keys.index('roughness')].ndim==0:return {}
    return None


class _GroupedMaterial(torch.autograd.Function):
    @staticmethod
    def forward(ctx,budget,two_sided,derivatives,sizes,prepared,*fields):
        packed=[materialize(torch.cat(fields[k::6],dim=0)) for k in range(6)]
        n,v,l,a,r,f=packed;d=budget._data
        try:
            if prepared is None:
                cn,curve,partial,mean,width=_prepare_material_curve(budget,r)
            else:
                # Only original scalar bindings share. State belongs to this
                # forward OR backward invocation, never to ctx or the budget.
                if 'curve' not in prepared:prepared['curve']=_prepare_material_curve(budget,r[:1])
                cn,cv,dp,mn,wd=prepared['curve']
                curve,partial,mean,width=cv,dp,mn,wd
            operation=(torch.ops.gffx.shading_principled_shared_groups if prepared is not None else torch.ops.gffx.shading_principled_groups)
            value,jacobian=operation(cn[1:],d['_clog'],
                curve,partial,mean,width,n,v,l,a,r,f,two_sided,derivatives)
        except RuntimeError as error:raise _error(error) from error
        ctx.sizes=sizes
        if derivatives:ctx.save_for_backward(jacobian)
        return tuple(value.split(sizes,dim=0))

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,*cotangents):
        jacobian,=ctx.saved_tensors
        try:point,light=torch.ops.gffx.shading_principled_groups_backward(jacobian,materialize(torch.cat(cotangents)))
        except RuntimeError as error:raise _error(error) from error
        result=[];left=0;required=ctx.needs_input_grad[5:]
        for size in ctx.sizes:
            row=point[left:left+size];offset=len(result)
            # Keep native full-Jacobian/VJP calculation and errors unchanged.
            # Autograd discards unrequested edges; avoid allocating their copies.
            result.extend((row[:,:3].contiguous() if required[offset] else None,
                row[:,3:6].contiguous() if required[offset+1] else None,
                light[left:left+size] if required[offset+2] else None,
                row[:,9:12].contiguous() if required[offset+3] else None,
                row[:,12].contiguous() if required[offset+4] else None,
                row[:,13:16].contiguous() if required[offset+5] else None));left+=size
        return (None,None,None,None,None,*result)


def _constant_layout(n,v,l,params,sizes,backward):
    """Call-local constant fields; caller already checked every original value."""
    from .material import _principled_layout
    if not sizes or len(sizes)>32 or any(not 1<=p<=4 for p in sizes):
        raise ValueError('internal groups require nonempty four-point layouts')
    points=sum(sizes)
    if n.shape!=(points,3) or v.shape!=n.shape or l.ndim!=3 or l.shape[0]!=points or l.shape[2]!=3:
        raise ValueError('constant group geometry shape mismatch')
    q=l.shape[1]
    if points*q>1024:raise ValueError('internal group pair limit exceeded')
    # Distinct original group sizes share identical shape/budget obligations.
    # Values across ALL points have already been checked, and the native call
    # still checks full geometry/derivative arithmetic. Nothing persists.
    for size in sorted(set(sizes)):
        _,(b,p,samples),_=_principled_layout(n[:size],v[:size],l[:size],_values_checked=True,**params)
        if b!=1 or not 1<=p<=4 or not samples:raise ValueError('internal groups require nonempty four-point layouts')
        if backward:
            needed=3*params['budget'].workspace(3*p,backward=True)+8*p*samples*192
            if needed>params.get('workspace_limit_bytes',67108864):
                raise ValueError('material workspace budget exceeded before expanded allocation')
    return tuple(materialize(x) for x in (n,v,l,params['albedo'].expand(points,3),
        params['roughness'].expand(points),params['f0'].expand(points,3)))


def _material(keys,static,groups,prepared=None):
    from .material import _principled_layout
    # Only graph-free, identical constant bindings use the compact layout.
    # General fields and all eager gradient-bearing paths keep the original loop.
    if groups and not torch.is_grad_enabled():
        first=groups[0];params=dict(static,**dict(zip(keys,first[3:])))
        constants=(isinstance(params.get('albedo'),torch.Tensor) and params['albedo'].shape==(3,)
            and isinstance(params.get('roughness'),torch.Tensor) and params['roughness'].ndim==0
            and isinstance(params.get('f0'),torch.Tensor) and params['f0'].shape in ((),(3,)))
        same=all(all(a is b for a,b in zip(row[3:],first[3:])) for row in groups)
        shapes=all(row[0].ndim==2 and row[0].shape[-1]==3 and row[1].shape==row[0].shape
            and row[2].ndim==3 and row[2].shape[0]==len(row[0]) and row[2].shape[-1]==3 for row in groups)
        if constants and same and shapes:
            sizes=tuple(len(row[0]) for row in groups)
            n,v,l=(torch.cat([row[k] for row in groups]) for k in range(3))
            fields=_constant_layout(n,v,l,params,sizes,False)
            return _GroupedMaterial.apply(static['budget'],static.get('two_sided',False),False,sizes,prepared,*fields)
    fields=[];sizes=[];derivatives=False
    for tensors in groups:
        params=dict(static,**dict(zip(keys,tensors[3:])))
        layout,(batch,points,samples),needs=_principled_layout(*tensors[:3],_values_checked=True,**params)
        if batch!=1 or not 1<=points<=4 or not samples:raise ValueError('internal groups require nonempty four-point layouts')
        fields.extend(layout);sizes.append(points);derivatives|=needs
    return _GroupedMaterial.apply(static['budget'],static.get('two_sided',False),derivatives,tuple(sizes),prepared,*fields)



def _groups(keys,values,points,left,right,block):
    rows=[];selections=[]
    for start in range(left,right,block):
        stop=min(start+block,right);selected=[];slices=[]
        for index,value in enumerate(values):
            name=keys[index-3] if index>=3 else None;selection=None
            if index<3:selection=slice(start,stop)
            elif name in ('albedo','f0'):
                if value.ndim==2 and value.shape[0]==points:selection=slice(start,stop)
                elif value.ndim==3 and value.shape[:2]==(1,points):selection=(slice(None),slice(start,stop))
            elif name=='roughness':
                if value.shape==(points,):selection=slice(start,stop)
                elif value.shape==(1,points):selection=(slice(None),slice(start,stop))
            selected.append(value if selection is None else value[selection]);slices.append(selection)
        rows.append(selected);selections.append(slices)
    return rows,selections


def _validate_original(keys,static,values):
    from .material import _principled_values
    params=dict(static,**dict(zip(keys,values[3:])))
    _principled_values(*values[:3],params['albedo'],params['roughness'],params['f0'],params['budget'])


def _albedo_cohort_partials(keys,static,values,needs,left,right,block,cotangent,prepared):
    """Same full native derivatives and per-group _Expand tree, without graphs."""
    if 'albedo' not in keys:return None
    index=3+keys.index('albedo')
    if [i for i,n in enumerate(needs) if n]!=[index]:return None
    params=dict(static,**dict(zip(keys,values[3:])))
    a,r,f=(params.get(k) for k in ('albedo','roughness','f0'))
    if not (a.shape==(3,) and isinstance(r,torch.Tensor) and r.ndim==0 and isinstance(f,torch.Tensor) and f.shape in ((),(3,)) and prepared is not None and a.dtype==torch.float64 and a.device.type in ('cpu','cuda')):return None
    sizes=tuple(min(block,right-i) for i in range(left,right,block))
    n,v,l=(value[left:right] for value in values[:3])
    # Every original <=4-point shape and backward workspace obligation remains.
    nn,vv,ll,aa,rr,ff=_constant_layout(n,v,l,params,sizes,True)
    budget=static['budget'];d=budget._data
    try:
        if 'curve' not in prepared:prepared['curve']=_prepare_material_curve(budget,rr[:1])
        cn,curve,partial,mean,width=prepared['curve']
        value,jacobian=torch.ops.gffx.shading_principled_shared_groups(cn[1:],d['_clog'],curve,partial,mean,width,nn,vv,ll,aa,rr,ff,static.get('two_sided',False),True)
        point,light=torch.ops.gffx.shading_principled_groups_backward(jacobian,materialize(cotangent[left:right]))
    except RuntimeError as error:raise _error(error) from error
    from ._reference_channels import _sum_last
    result=[];offset=0;group=0
    while group<len(sizes):
        size=sizes[group];end=group+1
        while end<len(sizes) and sizes[end]==size:end+=1
        count=end-group
        rows=point[offset:offset+count*size,9:12].contiguous().reshape(count,size,3)
        result.extend(_sum_last(rows.transpose(1,2)).unbind(0))
        offset+=count*size;group=end
    return result


class _Blocks(torch.autograd.Function):
    """One explicit reverse-group accumulation, independent of engine edge order."""
    @staticmethod
    def forward(ctx,keys,static,block,group_limit,*values):
        static=dict(static);prepared=static.pop('_forward_prepared',None)
        _validate_original(keys,static,values)
        ctx.keys=keys;ctx.static=dict(static);ctx.block=block;ctx.group_limit=group_limit
        ctx.save_for_backward(*values);points=len(values[0]);parts=[]
        if prepared is None:prepared=_call_curve_state(keys,static,values)
        if prepared is None:prepared=_shared_curve_state(keys,values)
        params=dict(static,**dict(zip(keys,values[3:])))
        constants=(isinstance(params.get('albedo'),torch.Tensor) and params['albedo'].shape==(3,)
            and isinstance(params.get('roughness'),torch.Tensor) and params['roughness'].ndim==0
            and isinstance(params.get('f0'),torch.Tensor) and params['f0'].shape in ((),(3,)))
        for left in range(0,points,block*group_limit):
            right=min(points,left+block*group_limit)
            if constants:
                # Forward alone: the original values are already validated.
                # Keep each four-point layout obligation and native cohort cap;
                # avoid slicing groups only to concatenate them immediately.
                sizes=tuple(min(block,right-i) for i in range(left,right,block))
                fields=_constant_layout(*(x[left:right] for x in values[:3]),params,sizes,False)
                parts.extend(_GroupedMaterial.apply(static['budget'],static.get('two_sided',False),False,sizes,prepared,*fields))
            else:
                groups,_=_groups(keys,values,points,left,right,block)
                parts.extend(_material(keys,static,groups,prepared))
        return torch.cat(parts,dim=0)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,cotangent):
        values=ctx.saved_tensors;needs=ctx.needs_input_grad[4:];points=len(values[0]);block=ctx.block;span=block*ctx.group_limit
        gradients=[torch.zeros_like(x) if need else None for x,need in zip(values,needs)]
        _validate_original(ctx.keys,ctx.static,values)
        prepared=_call_curve_state(ctx.keys,ctx.static,values)
        if prepared is None:prepared=_shared_curve_state(ctx.keys,values)
        # Cat's prior four-point composition visits the last group first. Make
        # that declared order explicit; shared fields are returned on ONE edge.
        for left in reversed(range(0,points,span)):
            right=min(points,left+span)
            direct=_albedo_cohort_partials(ctx.keys,ctx.static,values,needs,left,right,block,cotangent,prepared)
            if direct is not None:
                target=gradients[3+ctx.keys.index('albedo')]
                for part in reversed(direct):target.add_(part)
                del direct
                continue
            groups,selections=_groups(ctx.keys,values,points,left,right,block)
            temporary=[[x.detach().requires_grad_(need) for x,need in zip(row,needs)] for row in groups]
            with torch.enable_grad():
                outputs=_material(ctx.keys,ctx.static,temporary,prepared)
                cots=tuple(cotangent[i:min(i+block,right)] for i in range(left,right,block))
                partials=iter(torch.autograd.grad(outputs,[x for row in temporary for x,need in zip(row,needs) if need],cots,allow_unused=True))
            rows=[[next(partials) if need else None for need in needs] for row in groups]
            for row,slices in zip(reversed(rows),reversed(selections)):
                for index,(gradient,selection) in enumerate(zip(row,slices)):
                    if gradient is None:continue
                    if selection is None:gradients[index].add_(gradient)
                    else:gradients[index][selection].add_(gradient)
        return (None,None,None,None,*gradients)


def _forward_curve_state(state,budget,roughness):
    # Budget is immutable by public interface; also guard private tensor mutation.
    signature=(id(budget),id(roughness),roughness._version,tuple(
        (k,id(v),v._version) for k,v in sorted(budget._data.items()) if isinstance(v,torch.Tensor)))
    if state and state['_signature']!=signature:raise RuntimeError('shared material inputs modified during forward')
    if not state:state.update(_signature=signature,_inputs=(budget,roughness))
    return state


def render(normals,view,directions,parameters,block,*,group_point_limit=128,_prepared=None):
    keys=tuple(sorted(k for k,v in parameters.items() if isinstance(v,torch.Tensor)))
    static={k:v for k,v in parameters.items() if k not in keys}
    values=(normals,view,directions,*(parameters[k] for k in keys))
    backward=torch.is_grad_enabled() and any(x.requires_grad for x in values)
    groups=capacity(static['budget'],block,directions.shape[1],backward,static.get('workspace_limit_bytes',64*1024**2),shared_curve=_shared_curve_state(keys,values) is not None)
    groups=min(groups,group_point_limit//block)
    if _prepared is not None:
        if torch.is_grad_enabled() or _shared_curve_state(keys,values) is None:
            raise ValueError('shared material curve requires scalar forward roughness')
        static['_forward_prepared']=_forward_curve_state(_prepared,static['budget'],parameters['roughness'])
    return _Blocks.apply(keys,static,block,groups,*values)
