"""Branchwise installed loss tangents against independent endpoint intervals."""
import os,json,math,hashlib,time
from pathlib import Path
from decimal import Decimal as D,localcontext
import torch,pytest
@pytest.mark.parametrize('upper',[.005,.01])
def test_branchwise_loss_transport_matches_independent_endpoint(upper,record_property):
    from gffx.torch.material import PrincipledBudget
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_shading_intervals import I,Jet
    from tools.material_loss_cost import check_deadline
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu');t=lambda x:torch.tensor(x,dtype=torch.float64,device=device);start=time.perf_counter();path=Path(os.environ['GFFX_FULL_INPUT']);data=json.loads(path.read_bytes())['budget'];curve=PreparedPrincipled(data,1.)
    budget=PrincipledBudget.bundled('ggx-smith-loss-'+device+'-f64-v1',device=device,dtype=torch.float64)
    assert {k:v.cpu().tolist() for k,v in budget._data.items()}==data
    knots=data['cosine_nodes'];edges=[c for c in knots if c<upper]+[upper];nodes=[];weights=[]
    # Integral L'(c)dc: first interval linear, each later branch cubic in log(c).
    # Two-point Gauss exactly integrates its quadratic logarithmic derivative.
    for a,b in zip(edges,edges[1:]):
        if a==0:nodes.append(b/2);weights.append(b)
        else:
            lo,hi=math.log(a),math.log(b);mid=(lo+hi)/2;half=(hi-lo)/2
            for sign in (-1,1):c=math.exp(mid+sign*half/math.sqrt(3));nodes.append(c);weights.append(c*half)
    assert len(nodes)<=800
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    derivatives=[[],[]];point_values=[[],[]]
    for start_index in range(0,len(nodes),4):
        check_deadline(start+120);c=t(nodes[start_index:start_index+4]).requires_grad_();pair,mean=budget._native_pairs(t([1.]),c,torch.zeros(len(c),device=device,dtype=torch.int32));w=t(weights[start_index:start_index+4])
        for ch in range(2):
            grad,=torch.autograd.grad(pair[:,ch].sum(),c,retain_graph=ch==0);derivatives[ch].extend((grad*w).detach().cpu().tolist());point_values[ch].extend(pair[:,ch].detach().cpu().tolist())
    target=[z[0,0]-h for z,h in zip(curve.loss(Jet(upper)),curve.horizon)];actual=[math.fsum(v) for v in derivatives];checks=[]
    with localcontext() as ctx:
        ctx.prec=90
        for got,bound in zip(actual,target):
            for endpoint in (bound.lo,bound.hi):
                error=abs(D.from_float(got)-endpoint);limit=D('1e-10')+D('1e-9')*abs(endpoint);checks.append(dict(error_over_limit=float(error/limit),passed=error<=limit))
    row=dict(upper=upper,device=device,nodes=len(nodes),branches=len(edges)-1,actual=actual,reference=target,checks=checks,seconds=time.perf_counter()-start,scope='Scalar loss transport only; no production scene or forward image admission.')
    if device=='cuda':row['gpu_peak_bytes']=torch.cuda.max_memory_allocated();assert row['gpu_peak_bytes']<67108864
    p=Path(os.environ['GFFX_FULL_NATIVE']);rows=json.loads(p.read_bytes()) if p.exists() else [];rows.append(row);p.write_text(json.dumps(rows,default=str,indent=2),encoding='utf8');record_property('loss_transport',json.dumps(row,default=str));assert all(z['passed'] for z in checks),row
