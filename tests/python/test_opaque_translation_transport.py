"""Installed full-original-material motion versus independent boundary transport."""
import os,json,math,time,hashlib
from pathlib import Path
from decimal import Decimal as D,localcontext
import torch,pytest
from tools.opaque_translation_transport import original_translation

@pytest.mark.parametrize('role',['view','normal'])
@pytest.mark.parametrize('motion',['field','geometry'])
@pytest.mark.parametrize('region',['knot','horizon','visibility'])
def test_original_material_translation_derivative(role,motion,region,record_property):
    if region=='visibility' and motion=='field':pytest.skip('visibility edge requires attached geometry motion')
    from gffx.torch.material import PrincipledBudget,principled
    from gffx.torch.render import opaque_quadrature
    from tools.material_loss_cost import check_deadline
    from tools.opaque_principled_reference import PreparedPrincipled
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu');start=time.perf_counter();t=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
    inp=Path(os.environ['GFFX_FULL_INPUT']);assert hashlib.sha256(inp.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a';data=json.loads(inp.read_bytes())['budget']
    curve=PreparedPrincipled(data,1.);k=min(range(2,len(curve.cn)-2),key=lambda i:abs(float(curve.cn[i].lo)-.8));cos=float(curve.cn[k].lo)
    z0,dz=(0.,.02) if region=='horizon' else (cos/math.sqrt(1-cos*cos),.0001)
    edge=.25 if region=='visibility' else None
    ref=original_translation(data,role=role,z0=z0,dz=dz,motion=motion,visible_edge=edge)
    budget=PrincipledBudget.bundled('ggx-smith-loss-'+device+'-f64-v1',device=device,dtype=torch.float64)
    assert {k:v.cpu().tolist() for k,v in budget._data.items()}==data
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    theta=t(0).requires_grad_();base=t([[-2,-2,.2],[6,-2,.2],[-2,6,.2]] if edge is None else [[edge,-2,.2],[6,-2,.2],[edge,6,.2]])
    vertices=base+theta*t([1,0,0]) if motion=='geometry' else base
    faces=torch.tensor([[0,1,2]],dtype=torch.int32,device=device)
    corners=z0+dz*(base[:,0]-.5)+(theta*dz if motion=='field' else theta*0)
    cuts=corners[None,:,None].contiguous() if region=='horizon' else base.new_empty((1,3,0))
    policy=os.environ.get('GFFX_TRANSLATION_POLICY','linear');assert policy in ('linear','curved','curved4','curved8','bands8','bands15')
    curves=None
    if policy.startswith('bands'):
        assert region=='horizon';count=int(policy[5:]);cuts=torch.stack([corners-dz*(j/(2*count)) for j in range(count)],-1)[None].contiguous()
    if policy.startswith('curved'):
        assert region=='horizon';curves=torch.stack((corners[0],corners[1],corners[2],corners[0]+corners[1],corners[0]+corners[2],corners[1]+corners[2]))[None];cuts=base.new_empty((1,3,0))
    owners,q=opaque_quadrature(vertices,faces,cuts,1,1,curves=curves);nodes=q[0][owners[0]>=0]
    if policy in ('curved4','curved8'):
        from tools.opaque_curved_transport_map import remap
        nodes=remap(nodes,int(policy[-1]))
    from tools.opaque_streamed_motion import evaluate
    def field(part):
        z=(part[:,:3]*corners).sum(-1);variable=torch.stack((torch.ones_like(z),torch.zeros_like(z),z),-1);fixed=t([[0,0,1]]).expand(len(z),3).contiguous();n,v=(fixed,variable) if role=='view' else (variable,fixed);light=t([[1,1,2]])
        rgb=principled(n,v,light,albedo=t([.2,.4,.6]),roughness=t(1),f0=t([.04]*3),budget=budget)[:,0,:]
        cosine=(n/n.norm(dim=-1,keepdim=True)*light/light.norm()).sum(-1).clamp_min(0)
        return rgb*cosine[:,None]*part[:,3,None]
    actual=evaluate(nodes,theta,field,checkpoint=lambda:check_deadline(start+120))
    checks=[]
    with localcontext() as ctx:
        ctx.prec=90
        for a,b in zip(actual['derivative'],ref['derivative']):
            for endpoint in (b.lo,b.hi):
                error=abs(D.from_float(a)-endpoint);limit=D('1e-10')+D('1e-9')*abs(endpoint);checks.append({'error_over_limit':float(error/limit),'passed':error<=limit})
    row=dict(policy=policy,role=role,motion=motion,region=region,reference=ref,actual=actual,checks=checks,device=device,samples=len(nodes),seconds=time.perf_counter()-start)
    if device=='cuda':torch.cuda.synchronize();row['gpu_peak_bytes']=torch.cuda.max_memory_allocated();assert row['gpu_peak_bytes']<67108864
    path=Path(os.environ['GFFX_FULL_NATIVE']);saved=json.loads(path.read_bytes()) if path.exists() else [];saved.append(row);path.write_text(json.dumps(saved,default=str,indent=2),encoding='utf8');record_property('translation_transport',json.dumps(row,default=str))
    assert all(x['passed'] for x in checks),row
