"""Actual off-axis public-objective cotangents and matched stage replays."""
import json,torch
from test_principled_cuda_qualification import cpu,transferred
from test_deferred_principled_stages import raw,snapshot


def test_offaxis_stage_common_cotangent_attribution(cpu,monkeypatch,record_property):
    from gffx.torch import material,shade
    resources={'cpu':cpu,'cuda':transferred(cpu)};lanes={}
    host=[torch.tensor(x,dtype=torch.float64) for x in ([[.1,.2,1.]],[[.2,.1,1.]],[[.3,.4,1.],[-.2,.1,1.]],[.2,.4,.6],.63,.09)]
    cot=torch.tensor([[[.17,-.29,.41],[.53,-.67,.79]]],dtype=torch.float64)
    for device in ('cpu','cuda'):
        stages={};counts={};args=[x.to(device).clone().requires_grad_() for x in host]
        def invoke(name,fn,values):
            index=counts.get(name,0);counts[name]=index+1;key=name+'/'+str(index)
            result=fn(*values);outputs=result if isinstance(result,tuple) else (result,)
            row={'inputs':tuple(snapshot(x) if isinstance(x,torch.Tensor) else x for x in values),'outputs':tuple(snapshot(x) for x in outputs),'incoming':[None]*len(outputs),'fn':fn};stages[key]=row
            for i,x in enumerate(outputs):
                if x.requires_grad:
                    def save(g,i=i,row=row):row['incoming'][i]=snapshot(g)
                    x.register_hook(save)
            return result
        def wrap(name,fn):return lambda *values:invoke(name,fn,values)
        with monkeypatch.context() as patch:
            for owner,names in ((material,('_unit','_dot')),(shade,('ggx_distribution','smith_visibility','fresnel_schlick'))):
                for name in names:patch.setattr(owner,name,wrap(name,getattr(owner,name)))
            original=material.PrincipledBudget._material_losses
            def losses(owner,*v):return invoke('losses',lambda *q:original(owner,*q),v)
            patch.setattr(material.PrincipledBudget,'_material_losses',losses)
            value=material.principled(*args[:3],albedo=args[3],roughness=args[4],f0=args[5],budget=resources[device],allow_experimental_budget=device!='cpu')
            torch.autograd.grad(value,args,cot.to(device))
        lanes[device]=stages
    rows={}
    assert lanes['cpu'].keys()==lanes['cuda'].keys()
    for name,a in lanes['cpu'].items():
        b=lanes['cuda'][name]
        if any(x is None for x in a['incoming']):continue
        def eq(x,y):return raw(x)==raw(y) if isinstance(x,torch.Tensor) else x==y
        record={'inputs_equal':all(eq(x,y) for x,y in zip(a['inputs'],b['inputs'])),'outputs_equal':all(eq(x,y) for x,y in zip(a['outputs'],b['outputs'])),'incoming_equal':all(eq(x,y) for x,y in zip(a['incoming'],b['incoming']))}
        gradients=[]
        for device in ('cpu','cuda'):
            stage=lanes[device][name];values=[snapshot(x,device=device) if isinstance(x,torch.Tensor) else x for x in a['inputs']]
            leaves=[x.requires_grad_() for x in values if isinstance(x,torch.Tensor) and x.is_floating_point()]
            value=stage['fn'](*values);outputs=value if isinstance(value,tuple) else (value,)
            gradients.append(torch.autograd.grad(outputs,leaves,tuple(snapshot(x,device=device) for x in a['incoming']),allow_unused=True))
        record['common_vjp_equal']=[(x is None and y is None) or (x is not None and y is not None and raw(x)==raw(y)) for x,y in zip(*gradients)]
        record['common_vjp_max_abs']=[None if x is None or y is None else float((x.detach().cpu()-y.detach().cpu()).abs().max()) for x,y in zip(*gradients)]
        if name.startswith('ggx_distribution'):
            record['inputs']=[x.tolist() for x in a['inputs']];record['cpu_output']=[x.tolist() for x in a['outputs']];record['cuda_output']=[x.tolist() for x in b['outputs']]
        rows[name]=record
    record_property('offaxis_stages',json.dumps(rows,sort_keys=True))
