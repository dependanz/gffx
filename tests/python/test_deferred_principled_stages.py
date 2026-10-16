"""Captured public-objective cotangents at Principled subexpression boundaries."""
import hashlib
import json
import torch
from gffx.torch import material, scene, shade
from test_deferred_cuda import fixture, cuda_required


SPECS={
    'losses':('view_cosine','light_cosine','roughness','f0','indices'),
    'ggx':('n_dot_h','roughness'),
    'visibility':('n_dot_v','n_dot_l','roughness'),
    'fresnel':('v_dot_h','f0'),
}
FUNCTIONS={'ggx':'ggx_distribution','visibility':'smith_visibility','fresnel':'fresnel_schlick'}


def raw(value):
    # contiguous() need not pack a singleton whose original stride is nonunit.
    flat=torch.empty(value.numel(),dtype=value.dtype,device='cpu')
    flat.copy_(value.detach().cpu().reshape(-1))
    return bytes(flat.view(torch.uint8).tolist())


def describe(value):
    if value is None:return None
    return dict(dtype=str(value.dtype),shape=list(value.shape),stride=list(value.stride()),
                hex=raw(value).hex(),values=value.detach().cpu().reshape(-1).tolist())


def snapshot(value,device=None):
    # Preserve the actual supplied strides as well as logical bytes. This fixed
    # one-point fixture has no multiply-addressed expanded storage at the hooks.
    if any(size>1 and stride==0 for size,stride in zip(value.shape,value.stride())):
        raise AssertionError('unexpected overlapping stage operand')
    with torch.no_grad():
        copied=torch.empty_strided(value.shape,value.stride(),dtype=value.dtype,
                                   device=value.device if device is None else device)
        copied.copy_(value.detach())
    return copied


def capture(device,monkeypatch):
    model,_,parameters=fixture(device,torch.float64,principled=True)
    mat=next(p for p in model.prims if type(p)==scene.Material)
    budget=mat.parameters['budget'];stages={}
    originals={key:getattr(shade,name) for key,name in FUNCTIONS.items()}
    loss_original=material.PrincipledBudget._material_losses

    def invoke(key,function,args):
        assert key not in stages,'fixture unexpectedly evaluates a component twice'
        output=function(*args)
        outputs=output if isinstance(output,tuple) else (output,)
        row=dict(inputs=tuple(snapshot(arg) for arg in args),
                 outputs=tuple(snapshot(item) for item in outputs),incoming=[None]*len(outputs))
        stages[key]=row
        for index,item in enumerate(outputs):
            def save(gradient,*,index=index,row=row):
                row['incoming'][index]=snapshot(gradient)
            item.register_hook(save)
        return output

    with monkeypatch.context() as patch:
        def losses(owner,*args):return invoke('losses',lambda *values:loss_original(owner,*values),args)
        patch.setattr(material.PrincipledBudget,'_material_losses',losses)
        for key,name in FUNCTIONS.items():
            def wrapped(*args,key=key):return invoke(key,originals[key],args)
            patch.setattr(shade,name,wrapped)
        normal=parameters[0].new_tensor([[0.,0.,1.]])
        view=parameters[0].new_tensor([[0.,0.,2.]])
        direction=parameters[0].new_tensor([[[0.,0.,1.]]])
        value=material.principled(normal,view,direction,**mat.parameters)
        cotangent=value.new_tensor([[[.25,.5,.75]]])
        joint=torch.autograd.grad(value,parameters,cotangent)
    assert set(stages)==set(SPECS)
    assert all(item is not None for row in stages.values() for item in row['incoming'])
    tables={name:dict(dtype=str(tensor.dtype),shape=list(tensor.shape),
                     sha256=hashlib.sha256(raw(tensor)).hexdigest())
            for name,tensor in budget._data.items() if isinstance(tensor,torch.Tensor)}
    return dict(budget=budget,stages=stages,tables=tables,value=describe(value),
        objective_cotangent=describe(cotangent),
        parameters={name:describe(tensor) for name,tensor in zip(('albedo','roughness','f0'),parameters)},
        joint={name:describe(gradient) for name,gradient in zip(('albedo','roughness','f0'),joint)})


def replay(key,captured,common_incoming):
    operands=[snapshot(value) for value in captured['stages'][key]['inputs']]
    differentiated=[];names=[]
    for name,value in zip(SPECS[key],operands):
        if value.is_floating_point():
            value.requires_grad_();differentiated.append(value);names.append(name)
    device=operands[0].device
    # These are the CPU public objective's recorded cotangents, copied verbatim
    # as explicit diagnostic inputs to both devices, never a production fallback.
    incoming=tuple(snapshot(value,device=device) for value in common_incoming)
    output=(captured['budget']._material_losses(*operands) if key=='losses' else
            getattr(shade,FUNCTIONS[key])(*operands))
    outputs=output if isinstance(output,tuple) else (output,)
    gradients=torch.autograd.grad(outputs,differentiated,incoming,allow_unused=True)
    return dict(inputs={name:describe(value) for name,value in zip(SPECS[key],operands)},
        outputs=[describe(value) for value in outputs],incoming=[describe(value) for value in incoming],
        vjps={name:describe(value) for name,value in zip(names,gradients)})


def test_deferred_principled_captured_stage_attribution(cuda_required,monkeypatch,record_property):
    captured={device:capture(device,monkeypatch) for device in ('cpu','cuda')}
    record_property('principled_public_objective',json.dumps({device:{name:value for name,value in row.items()
        if name not in ('budget','stages')} for device,row in captured.items()},sort_keys=True))
    assert captured['cpu']['tables']==captured['cuda']['tables']
    assert captured['cpu']['parameters']==captured['cuda']['parameters']
    assert captured['cpu']['objective_cotangent']==captured['cuda']['objective_cotangent']
    for key in SPECS:
        actual={device:dict(inputs={name:describe(value) for name,value in zip(SPECS[key],row['stages'][key]['inputs'])},
            outputs=[describe(value) for value in row['stages'][key]['outputs']],
            incoming=[describe(value) for value in row['stages'][key]['incoming']]) for device,row in captured.items()}
        common=captured['cpu']['stages'][key]['incoming']
        replayed={device:replay(key,row,common) for device,row in captured.items()}
        record_property('principled_stage_'+key,json.dumps(dict(actual=actual,replay=replayed,
            actual_output_equal=actual['cpu']['outputs']==actual['cuda']['outputs'],
            actual_incoming_equal=actual['cpu']['incoming']==actual['cuda']['incoming'],
            common_vjp_equal={name:replayed['cpu']['vjps'][name]==replayed['cuda']['vjps'][name]
                              for name in replayed['cpu']['vjps']}),sort_keys=True))
        assert actual['cpu']['inputs']==actual['cuda']['inputs'],'stage operand mismatch precedes its VJP'
        assert replayed['cpu']['inputs']==replayed['cuda']['inputs']
        assert replayed['cpu']['incoming']==replayed['cuda']['incoming']
        for device in ('cpu','cuda'):
            assert replayed[device]['outputs']==actual[device]['outputs'],'replay changed stage mathematics'
    # A passing diagnostic validates its capture/replay controls, not parity of
    # differing recorded components; the strict six-case gate stays separate.


def test_deferred_principled_curve_primitive_attribution(cuda_required,monkeypatch,record_property):
    """Localize matched-input curve arithmetic; diagnostic green is not shader parity."""
    from gffx.torch._loss_curve import _bezier,bounded_slopes,controls
    from gffx.torch.image import _fixed_sum
    captured={device:capture(device,monkeypatch) for device in ('cpu','cuda')}
    stages={}
    for device,row in captured.items():
        budget=row['budget'];r=snapshot(row['stages']['losses']['inputs'][2]).requires_grad_()
        d=budget._data;x=d['_rlog'];cn=d['cosine_nodes']
        i=(torch.searchsorted(d['roughness_nodes'],r.contiguous(),right=True)-1).clamp(max=len(x)-2)
        h=x[i+1]-x[i];logr=r.log();u=(logr-x[i])/h
        values={"roughness_log":logr,"roughness_u":u}
        for name,y,m,trailing in [('angular',d['_logs'],d['_slopes'],2),('horizon',d['_hlogs'],d['_hslopes'],0)]:
            width=h.reshape((-1,)+(1,)*trailing)
            control=torch.stack((y[i],y[i]+width*m[i]/3,y[i+1]-width*m[i+1]/3,y[i+1]))
            interpolated=_bezier(control,u.reshape((-1,)+(1,)*trailing));output=interpolated.exp()
            values[name+'_control']=control;values[name+'_log']=interpolated;values[name+'_exp']=output
        angular=values['angular_exp'].transpose(0,1)
        slopes=bounded_slopes(d['_clog'],angular,ordered_pairs=True)
        b=controls(d['_clog'],angular,slopes).permute(2,0,1,3)
        values['angular_slopes']=slopes;values['angular_controls']=b
        width=d['_clog'][1:]-d['_clog'][:-1]
        terms=2*cn[None,2:,None,None].square()*width[None,:,None,None]*b*d['_weights'][None,:,:,None]
        values['mean_terms']=terms;values['mean_default_sum']=terms.sum(dim=(1,2))
        values['mean_fixed_sum']=torch.stack([torch.stack([_fixed_sum(frame[...,channel]) for channel in range(2)]) for frame in terms])
        stages[device]=values
    summary={}
    for name,cpu in stages['cpu'].items():
        gpu=stages['cuda'][name].detach().cpu();host=cpu.detach();different=(host!=gpu)
        indices=different.reshape(-1).nonzero().reshape(-1).tolist()[:8]
        summary[name]=dict(shape=list(host.shape),elements=host.numel(),different_values=int(different.sum()),bytes_equal=raw(host)==raw(gpu),
            max_abs=float((host-gpu).abs().max()) if host.numel() else 0.,
            first_differences=[dict(index=j,cpu=float(host.reshape(-1)[j]),cuda=float(gpu.reshape(-1)[j])) for j in indices])
    # Feed identical mean terms to distinguish reduction order from earlier arithmetic.
    common=stages['cpu']['mean_terms'].detach()
    means={device:common.to(device).sum(dim=(1,2)) for device in ('cpu','cuda')}
    fixed={device:torch.stack([torch.stack([_fixed_sum(frame[...,channel]) for channel in range(2)]) for frame in common.to(device)]) for device in ('cpu','cuda')}
    summary['matched_mean_terms']=dict(default_cpu=describe(means['cpu']),default_cuda=describe(means['cuda']),fixed_cpu=describe(fixed['cpu']),fixed_cuda=describe(fixed['cuda']))
    assert raw(fixed['cpu'])==raw(fixed['cuda']),'fixed reduction diagnostic is inconsistent'
    record_property('principled_curve_primitives',json.dumps(summary,sort_keys=True))
