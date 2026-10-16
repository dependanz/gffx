"""Installed shader checks against a saved independently certified pixel."""
import hashlib,json,math,os
from pathlib import Path
from decimal import Decimal as D,localcontext
from fractions import Fraction as Q
import torch

def test_installed_principled_weighted_integral_matches_independent_enclosures(record_property):
    from gffx.torch.material import PrincipledBudget,principled
    from tools import material_loss_cost as cost
    import time
    start=time.perf_counter()
    path=Path(os.environ['GFFX_VALIDATED_PIXEL'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['GFFX_VALIDATED_PIXEL_SHA']
    saved=json.loads(path.read_bytes())
    producer=saved['provenance']['report']
    assert producer['status']=='passed' and producer['pytest_exit_code']==0
    for name,sha in producer['source_sha256'].items():
        # The runner may acquire new modes; the two numerical producers must match.
        if name in ('tools/opaque_shading_intervals.py','tools/opaque_principled_reference.py'):
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
    assert saved['tolerance']=='1e-12' and len(saved['pieces'])==1
    part=saved['pieces'][0]
    assert part['face']==0 and part['triangle']==[['0','0'],['1','0'],['0','1']]
    assert part['gauss_points']==5 and len(part['leaves'])<=128
    b=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    a=math.sqrt(5+2*math.sqrt(10/7))/6;c=math.sqrt(5-2*math.sqrt(10/7))/6
    nodes=[.5-a,.5-c,.5,.5+c,.5+a]
    u=(322-13*math.sqrt(70))/1800;v=(322+13*math.sqrt(70))/1800
    weights=[u,v,64/225,v,u]
    samples=[]
    for cell in part['leaves']:
        x0,x1,y0,y1=map(lambda z:float(Q(z)),cell)
        for ns,ws in zip(nodes,weights):
            for nt,wt in zip(nodes,weights):
                s=x0+(x1-x0)*ns;t0=y0+(y1-y0)*nt
                samples.append([s,(1-s)*t0,(1-s)*(x1-x0)*(y1-y0)*ws*wt])
    assert len(samples)<=3200
    normals=t([[.01,.01,1.],[.05,.01,1.],[.01,.04,1.]])
    views=t([[-.07,.06,1.],[-.03,.06,1.],[-.07,.08,1.]])
    albedo=t([[.2,.4,.6],[.6,.2,.4],[.3,.7,.2]])
    direction=t([[.1,.1,1.]])
    values=[[],[],[]];grads=[[],[],[]]
    for k in range(0,len(samples),4):
        cost.check_deadline(start+120)
        chunk=t(samples[k:k+4]);x,y,w=chunk.unbind(-1)
        beta=torch.stack([1-x-y,x,y],-1)
        raw=beta*t([1.,.85,1.1]);lam=raw/raw.sum(-1,keepdim=True)
        theta=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        n=lam@(normals+theta*t([.01,0,0]));a0=lam@(albedo+theta*t([.05,0,0]));view=lam@views
        rgb=principled(n,view,direction,albedo=a0,roughness=t(.827),f0=t([.04,.05,.08]),budget=b)[:,0,:]
        cosine=(n/n.norm(dim=-1,keepdim=True)*(direction/direction.norm())).sum(-1)
        total=(rgb*(cosine*w)[:,None]).sum(0)
        for j in range(3):
            values[j].append(total[j].item())
            grads[j].append(torch.autograd.grad(total[j],theta,retain_graph=j<2)[0].item())
    got={'value':[math.fsum(q) for q in values],'derivative':[math.fsum(q) for q in grads]}
    result={'points':len(samples),'batch_size':4,'batches':math.ceil(len(samples)/4),'reference_sha256':os.environ['GFFX_VALIDATED_PIXEL_SHA'],'comparison_tolerance':{'atol':1e-10,'rtol':1e-9},'reference_radius_limit':1e-12,'channels':{}}
    for kind in got:
        rows=[]
        for value,enclosure in zip(got[kind],saved[kind][0][0]):
            lo,hi=map(D,enclosure.strip('[]').split(','));candidate=D.from_float(value)
            with localcontext() as ctx:
                ctx.prec=90;mid=(lo+hi)/2;radius=(hi-lo)/2
                distance=max(lo-candidate,candidate-hi,D(0))
                assert distance<=D('1e-10')+D('1e-9')*abs(mid)
                rows.append(dict(candidate=value,lower=str(lo),upper=str(hi),midpoint=str(mid),radius=str(radius),distance_from_interval=str(distance),midpoint_difference=str(abs(candidate-mid))))
        result['channels'][kind]=rows
    result['seconds']=time.perf_counter()-start
    record_property('native_comparison',json.dumps(result))


def test_principled_interval_tangent_agrees_with_independent_decimal_differences():
    from gffx.torch.material import PrincipledBudget
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_shading_intervals import Jet
    from tools.principled_material_oracle import material_pair
    b=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    data={k:v.tolist() for k,v in b._data.items()}
    values=[.01,.02,1.,-.07,.06,1.,.1,.1,1.,.2,.4,.6,.827,.04,.05,.08]
    tangent=[.01,0,0,0,0,0,0,0,0,.05,0,0,0,0,0,0]
    p=[Jet(x,tangent=z) for x,z in zip(values,tangent)]
    got=PreparedPrincipled(data,.827).radiance(p[:3],p[3:6],p[6:9],p[9:12],p[13:16])
    with localcontext() as ctx:
        ctx.prec=90;v=list(map(D.from_float,values));dv=list(map(D.from_float,tangent))
        def value(q):
            cosine=sum(a*b for a,b in zip(q[:3],q[6:9]))/(sum(a*a for a in q[:3])*sum(a*a for a in q[6:9])).sqrt()
            return [a*cosine for a in material_pair(data,q)]
        differences=[]
        for eps in (D('1e-12'),D('5e-13')):
            plus=value([a+eps*b for a,b in zip(v,dv)]);minus=value([a-eps*b for a,b in zip(v,dv)])
            differences.append([(a-b)/(2*eps) for a,b in zip(plus,minus)])
        for j,q in enumerate(got):
            mid=(q[0,1].lo+q[0,1].hi)/2
            assert abs(differences[0][j]-differences[1][j])<D('1e-25')
            assert abs(differences[1][j]-mid)<D('1e-25')


def test_exact_visibility_partition_keeps_hidden_and_subpixel_mass():
    from tools.opaque_principled_reference import integrate_pixel
    from tools.opaque_shading_intervals import Jet
    from tools.opaque_pixel_oracle import integrate
    vertices=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,0,1),(0,1,1),(Q(1,2),Q(1,2),-1),(Q(1,2)+Q(1,1000),Q(1,2),-1),(Q(1,2),Q(1,2)+Q(1,1000),-1)]
    faces=[(0,1,2),(3,4,5),(6,7,8)];colours=[(1,0,0),(0,1,0),(0,0,1)]
    fields=[(lambda x,y,rgb=rgb:[x.lift(c) for c in rgb]) for rgb in colours]
    row=integrate_pixel(vertices,faces,fields)
    exact=integrate(vertices,faces,colours,height=1,width=1)
    for a,b in zip(row['value'][0][0],exact['colour'][0][0]):
        with localcontext() as ctx:
            ctx.prec=90;number=D(b.numerator)/D(b.denominator)
            assert a.lo<=number<=a.hi
    assert row['value'][0][0][1].hi==0 and row['value'][0][0][2].lo>0
