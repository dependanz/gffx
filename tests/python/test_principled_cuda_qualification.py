"""Explicit experimental CUDA-f64 transport, never CPU-profile admission by cast."""
import pytest
import torch


@pytest.fixture(autouse=True)
def cuda_lane():
    if not torch.cuda.is_available():pytest.skip('CUDA qualification requires a real device')
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def cpu():
    from gffx.torch.material import PrincipledBudget
    return PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)


def transferred(cpu):return cpu.to(device='cuda:0',dtype=torch.float64,allow_unverified=True)


def test_cuda_transfer_is_explicit_and_cannot_inherit_cpu_admission(cpu):
    with pytest.raises(ValueError):cpu.to(device='cuda:0',dtype=torch.float64)
    gpu=transferred(cpu)
    assert gpu.metadata['admission']=='experimental'
    assert gpu.metadata['acceptance_profile_id'] is None and gpu.metadata['resource_id'] is None
    assert gpu.metadata['execution_device']=='cuda:0' and gpu.metadata['execution_dtype']=='float64'
    assert gpu.to_envelope()==cpu.to_envelope()
    assert all(value.device.type=='cuda' for value in gpu._data.values())
    back=gpu.to(device='cpu',dtype=torch.float64)
    assert back.metadata['admission']=='accepted' and back.to_envelope()==cpu.to_envelope()


@pytest.mark.parametrize('c,r,f',[(.463,.213,.47),(1e-8,.0193,.93),(.947,.827,1.)])
def test_cuda_budget_queries_and_partials_match_cpu(cpu,c,r,f):
    gpu=transferred(cpu)
    def run(resource,device):
        args=[torch.tensor(x,device=device,dtype=torch.float64,requires_grad=True) for x in (c,r,f)]
        value=resource.directional_loss(*args)+resource.mean_loss(*args[1:])
        return [value.detach().cpu(),*[g.detach().cpu() for g in torch.autograd.grad(value,args)]]
    for expected,actual in zip(run(cpu,'cpu'),run(gpu,'cuda:0')):
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)


def test_cuda_budget_rejects_wrong_query_device_and_preserves_input(cpu):
    gpu=transferred(cpu);x=torch.tensor(.5,dtype=torch.float64)
    with pytest.raises(ValueError):gpu.directional_loss(x,x,x)
    assert x.device.type=='cpu' and x.item()==.5


def test_cuda_complete_material_matches_cpu_values_and_all_partials(cpu):
    from gffx.torch.material import principled
    gpu=transferred(cpu)
    def run(resource,device,experimental):
        args=[torch.tensor(x,dtype=torch.float64,device=device,requires_grad=True) for x in
              ([[.1,.2,1.]],[[.2,.1,1.]],[[.3,.4,1.],[-.2,.1,1.]],[.2,.4,.6],.63,.09)]
        kw=dict(albedo=args[3],roughness=args[4],f0=args[5],budget=resource)
        if experimental:
            with pytest.raises(ValueError,match='experimental'):principled(*args[:3],**kw)
        value=principled(*args[:3],**kw,allow_experimental_budget=experimental)
        return [value.detach().cpu(),*[g.detach().cpu() for g in torch.autograd.grad(value.square().sum(),args)]]
    for expected,actual in zip(run(cpu,'cpu',False),run(gpu,'cuda:0',True)):
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)


def test_cuda_supervision_changes_only_host_cap_and_restores_cpu_policy(monkeypatch,tmp_path):
    from tools import cuda_budget_probe as probe,material_loss_cost as cost
    original=cost.PROFILE
    seen=[]
    def supervise(*args,**kwargs):
        seen.append(dict(cost.PROFILE))
        raise RuntimeError('simulated worker failure')
    monkeypatch.setattr(cost,'supervise',supervise)
    with pytest.raises(RuntimeError,match='simulated'):
        probe.supervise_cuda(tmp_path,tmp_path/'fresh.json',worker_args=())
    assert cost.PROFILE is original
    assert seen==[dict(original,sampled_working_set_limit_bytes=2*1024**3)]
    assert original['sampled_working_set_limit_bytes']==1024**3
    assert seen[0]['watchdog_seconds']==180


def test_cuda_budget_batched_queries_have_exact_reference_bytes(cpu,record_property):
    """Distinct query entries exercise vector arithmetic and every independent input VJP."""
    from test_deferred_principled_stages import raw
    gpu=transferred(cpu)
    host=[torch.tensor(x,dtype=torch.float64) for x in
          ([0.,1e-8,.013,.019,.219,.463,.811,1.],
           [.00179,.0193,.0371,.173,.213,.419,.613,.827],
           [0.,.04,.07,.17,.29,.47,.73,1.])]
    cot=torch.tensor([.17,-.29,.41,.53,-.67,.79,-.83,.97],dtype=torch.float64)
    def evaluate(resource,device):
        args=[x.to(device).clone().requires_grad_() for x in host]
        value=resource.directional_loss(*args)+resource.mean_loss(*args[1:])
        grads=torch.autograd.grad(value,args,grad_outputs=cot.to(device))
        return tuple(x.detach().cpu() for x in (value,*grads))
    rows=(evaluate(cpu,'cpu'),evaluate(gpu,'cuda'),evaluate(gpu,'cuda'))
    failed=[]
    for name,(a,b,c) in zip(('value','cosine_vjp','roughness_vjp','f0_vjp'),zip(*rows)):
        bad=raw(a)!=raw(b) or raw(b)!=raw(c)
        record_property(name+'_max_abs',float((a-b).abs().max()))
        if bad:failed.append(name)
    assert not failed,failed


def test_cuda_complete_material_exact_reference_bytes(cpu,record_property):
    """All independent material inputs must retain exact device gradients off axis."""
    from gffx.torch.material import principled
    from test_deferred_principled_stages import raw
    gpu=transferred(cpu)
    host=[torch.tensor(x,dtype=torch.float64) for x in
          ([[.1,.2,1.]],[[.2,.1,1.]],[[.3,.4,1.],[-.2,.1,1.]],[.2,.4,.6],.63,.09)]
    cot=torch.tensor([[[.17,-.29,.41],[.53,-.67,.79]]],dtype=torch.float64)
    def run(resource,device):
        args=[x.to(device).clone().requires_grad_() for x in host]
        out=principled(*args[:3],albedo=args[3],roughness=args[4],f0=args[5],
                       budget=resource,allow_experimental_budget=device!='cpu')
        grads=torch.autograd.grad(out,args,grad_outputs=cot.to(device))
        return tuple(x.detach().cpu() for x in (out,*grads))
    rows=(run(cpu,'cpu'),run(gpu,'cuda'),run(gpu,'cuda'))
    failures=[]
    for name,(a,b,c) in zip(('value','normal','view','light','albedo','roughness','f0'),zip(*rows)):
        bad=raw(a)!=raw(b) or raw(b)!=raw(c)
        record_property(name+'_max_abs',float((a-b).abs().max()))
        if bad:failures.append(name)
    assert not failures,failures


def test_cuda_material_vector_primitives_matched_inputs(record_property):
    from gffx.torch import material
    from test_deferred_principled_stages import raw
    host=torch.tensor([[.1,.2,1.],[.2,.1,1.],[.3,.4,1.],[-.2,.1,1.]],dtype=torch.float64)
    cot=torch.tensor([[.17,-.29,.41],[.53,-.67,.79],[.23,.37,-.61],[-.11,.31,.71]],dtype=torch.float64)
    half=material._unit(host[1:2],'v')+material._unit(host[2:],'l')
    failures=[]
    for label,source,fn,incoming in [('half_unit',half,lambda x:material._unit(x,'half'),cot[2:]),
       ('half_norm',half,lambda x:torch.linalg.vector_norm(x,dim=-1),cot[2:,0]),
       ('unit',host,lambda x:material._unit(x,'matched'),cot),
       ('norm',host,lambda x:torch.linalg.vector_norm(x,dim=-1),cot[:,0]),
       ('dot',host,lambda x:material._dot(x*.25,x*.25),cot[:,0])]:
        rows=[]
        for device in ('cpu','cuda','cuda'):
            x=source.to(device).clone().requires_grad_();y=fn(x)
            g,=torch.autograd.grad(y,x,grad_outputs=incoming.to(device))
            rows.append((y.detach().cpu(),g.detach().cpu()))
        for index,(a,b,c) in enumerate(zip(*rows)):
            bad=raw(a)!=raw(b) or raw(b)!=raw(c)
            record_property(f'{label}_{index}_max_abs',float((a-b).abs().max()))
            if bad:failures.append(f'{label}/{index}')
    assert not failures,failures


def test_cuda_offaxis_shading_primitives_matched_inputs(record_property):
    from gffx.torch import material,shade
    from test_deferred_principled_stages import raw
    n=material._unit(torch.tensor([[.1,.2,1.]],dtype=torch.float64),'n')
    v=material._unit(torch.tensor([[.2,.1,1.]],dtype=torch.float64),'v')
    l=material._unit(torch.tensor([[.3,.4,1.],[-.2,.1,1.]],dtype=torch.float64),'l')
    h=material._unit(v+l,'h')
    dot=lambda a,b:material._dot(a,b)
    r=torch.full((2,),.63,dtype=torch.float64)
    f=torch.full((2,3),.09,dtype=torch.float64)
    stages=[('ggx',shade.ggx_distribution,(dot(n,h),r)),
            ('smith',shade.smith_visibility,(dot(n,v).expand(2).clone(),dot(n,l),r)),
            ('fresnel',shade.fresnel_schlick,(dot(v,h),f))]
    failures=[]
    for label,fn,host in stages:
        count=6 if label=='fresnel' else 2
        cot=torch.linspace(-.37,.79,count,dtype=torch.float64).reshape((2,3) if count==6 else (2,))
        rows=[]
        for device in ('cpu','cuda','cuda'):
            args=[x.to(device).clone().requires_grad_() for x in host]
            y=fn(*args);g=torch.autograd.grad(y,args,grad_outputs=cot.to(device))
            rows.append(tuple(x.detach().cpu() for x in (y,*g)))
        for index,(a,b,c) in enumerate(zip(*rows)):
            bad=raw(a)!=raw(b) or raw(b)!=raw(c)
            record_property(f'{label}_{index}_max_abs',float((a-b).abs().max()))
            if bad:failures.append(f'{label}/{index}')
    assert not failures,failures


def test_cuda_offaxis_captured_component_attribution(cpu,monkeypatch,record_property):
    import json
    from gffx.torch import material,shade
    from test_deferred_principled_stages import raw
    gpu=transferred(cpu)
    host=[torch.tensor(x,dtype=torch.float64) for x in
          ([[.1,.2,1.]],[[.2,.1,1.]],[[.3,.4,1.],[-.2,.1,1.]],[.2,.4,.6],.63,.09)]
    cot=torch.tensor([[[.17,-.29,.41],[.53,-.67,.79]]],dtype=torch.float64)
    def capture(resource,device):
        rows={};counts={}
        def call(label,fn,args):
            out=fn(*args)
            outputs=out if isinstance(out,tuple) else (out,)
            if not torch.is_grad_enabled() or not all(v.requires_grad for v in outputs):return out
            n=counts.get(label,0);counts[label]=n+1;key=f'{label}{n}'
            row={'fn':fn,'args':tuple(a.detach().clone() if isinstance(a,torch.Tensor) else a for a in args),
                 'outputs':tuple(v.detach().clone() for v in outputs),'incoming':[None]*len(outputs)}
            rows[key]=row
            for i,v in enumerate(outputs):
                def save(g,i=i,row=row):row['incoming'][i]=g.detach().clone()
                v.register_hook(save)
            return out
        with monkeypatch.context() as patch:
            for scope,name in [(material,'_unit'),(material,'_dot'),(shade,'ggx_distribution'),(shade,'smith_visibility'),(shade,'fresnel_schlick')]:
                fn=getattr(scope,name)
                def wrapped(*args,fn=fn,name=name):return call(name,fn,args)
                patch.setattr(scope,name,wrapped)
            old=material.PrincipledBudget._material_losses
            def losses(owner,*args):return call('losses',lambda *xs:old(owner,*xs),args)
            patch.setattr(material.PrincipledBudget,'_material_losses',losses)
            args=[x.to(device).clone().requires_grad_() for x in host]
            # Native pair fusion deliberately has no eager component hooks.
            # Keep this historical component diagnostic on the explicit reference;
            # installed native-pair tests separately qualify its whole Jacobian.
            evaluator=getattr(material,'_principled_reference',material.principled)
            out=evaluator(*args[:3],albedo=args[3],roughness=args[4],f0=args[5],budget=resource,allow_experimental_budget=device!='cpu')
            torch.autograd.grad(out,args,grad_outputs=cot.to(device))
        assert rows and any(k.startswith('losses') for k in rows), 'component capture must not pass vacuously'
        assert all(all(g is not None for g in row['incoming']) for row in rows.values())
        return rows
    captures={'cpu':capture(cpu,'cpu'),'cuda':capture(gpu,'cuda')};assert captures['cpu'].keys()==captures['cuda'].keys()
    result={}
    for key,a in captures['cpu'].items():
        b=captures['cuda'][key]
        report={'actual_inputs_equal':[raw(x)==raw(y) if isinstance(x,torch.Tensor) else x==y for x,y in zip(a['args'],b['args'])],
                'actual_outputs_equal':[raw(x)==raw(y) for x,y in zip(a['outputs'],b['outputs'])]}
        common=[]
        for device,row in [('cpu',a),('cuda',b)]:
            args=[x.to(device).clone().requires_grad_(x.is_floating_point()) if isinstance(x,torch.Tensor) else x for x in a['args']]
            inputs=[x for x in args if isinstance(x,torch.Tensor) and x.is_floating_point()]
            out=row['fn'](*args);outputs=out if isinstance(out,tuple) else (out,)
            gradients=torch.autograd.grad(outputs,inputs,grad_outputs=[g.to(device) for g in a['incoming']],allow_unused=True)
            common.append((outputs,gradients))
        report['common_outputs_equal']=[raw(x)==raw(y) for x,y in zip(common[0][0],common[1][0])]
        report['common_vjps_equal']=[x is y if x is None or y is None else raw(x)==raw(y) for x,y in zip(common[0][1],common[1][1])]
        result[key]=report
    record_property('execution_path','explicit retained eager material reference; native pair Jacobian is qualified separately')
    record_property('offaxis_components',json.dumps(result,sort_keys=True))
