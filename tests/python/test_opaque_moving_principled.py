"""Nonlinear moving-region bounds, independent boundary formulas, installed calls."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
import json,torch,pytest
from tools.opaque_moving_reference import integrate_moving_pixel,moving_perspective_field
from tools.opaque_principled_reference import PreparedPrincipled
from tools.opaque_shading_intervals import Jet
from tools.principled_material_oracle import material_pair

@pytest.fixture(scope='module')
def resource():
    from gffx.torch.material import PrincipledBudget
    b=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    data={k:v.tolist() for k,v in b._data.items()}
    return b,data,PreparedPrincipled(data,.827)

def fixture(kind):
    if kind=='silhouette':return [(Q(1,4),-1,0),(Q(1,4),3,0),(Q(17,4),-1,0)],[(0,1,2)],[(1,0,0)]*3
    xy=[(-2,-2),(6,-2),(-2,6)]
    return [(x,y,x-Q(1,4)) for x,y in xy]+[(x,y,0) for x,y in xy],[(0,1,2),(3,4,5)],[(0,0,-1)]*3+[(0,0,0)]*3

def coefficients(index):
    return ([.01,.004,.01], [.2,.05,.4,0.,.6,0.]) if index==0 else ([.02,.003,.015],[.65,-.03,.1,.01,.25,0.])

def field(curve,index):
    n,a=coefficients(index)
    def fn(x,y):
        return curve.radiance([x.lift(n[0])+x*n[1],x.lift(n[2]),x.lift(1)],[x.lift(q) for q in [-.07,.06,1.]],[x.lift(q) for q in [.1,.1,1.]],[x.lift(a[2*k])+x*a[2*k+1] for k in range(3)],[x.lift(q) for q in [.04,.05,.08]])
    return fn

def boundary(data,index):
    with localcontext() as c:
        c.prec=90;n,a=coefficients(index);n=list(map(D.from_float,n));a=list(map(D.from_float,a));t=D(1)/4
        v=[n[0]+n[1]*t,n[2],D(1)]+list(map(D.from_float,[-.07,.06,1.,.1,.1,1.]))+[a[2*k]+a[2*k+1]*t for k in range(3)]+list(map(D.from_float,[.827,.04,.05,.08]))
        cosine=sum(x*y for x,y in zip(v[:3],v[6:9]))/(sum(x*x for x in v[:3])*sum(x*x for x in v[6:9])).sqrt()
        return [x*cosine for x in material_pair(data,v)]

def native_integral(budget,kind):
    # Independent one-dimensional Gauss16, not the certified triangle partition.
    import numpy as np
    from gffx.torch.material import principled
    nodes,weights=np.polynomial.legendre.leggauss(16);nodes=(nodes+1)/2;weights=weights/2
    t=lambda q:torch.tensor(q,dtype=torch.float64)
    values=[[],[],[]];derivatives=[[],[],[]]
    segments=[(0,False)] if kind=='silhouette' else [(0,True),(1,False)]
    for index,left in segments:
        n,a=coefficients(index)
        for start in range(0,16,4):
            theta=t(0.).requires_grad_();edge=theta+.25;length=edge if left else 1-edge
            x=length*t(nodes[start:start+4].copy())+(0 if left else edge)
            normals=torch.stack([n[0]+n[1]*x,torch.full_like(x,n[2]),torch.ones_like(x)],-1)
            albedo=torch.stack([a[2*k]+a[2*k+1]*x for k in range(3)],-1)
            view=t([[-.07,.06,1.]]).expand(len(x),3).contiguous();direction=t([[.1,.1,1.]])
            rgb=principled(normals,view,direction,albedo=albedo,roughness=t(.827),f0=t([.04,.05,.08]),budget=budget)[:,0,:]
            cos=(normals/normals.norm(dim=-1,keepdim=True)*(direction/direction.norm())).sum(-1)
            total=(rgb*(cos*t(weights[start:start+4].copy())*length)[:,None]).sum(0)
            for k in range(3):
                values[k].append(total[k].item());derivatives[k].append(torch.autograd.grad(total[k],theta,retain_graph=k<2)[0].item())
    import math
    return dict(value=[math.fsum(q) for q in values],derivative=[math.fsum(q) for q in derivatives],quadrature='independent Gauss16 on x; y-independent full Principled field',samples=16*len(segments),batch_size=4)

@pytest.mark.parametrize('kind',['silhouette','intersection'])
def test_nonlinear_moving_pixel_matches_boundary_formula_and_installed_gradient(resource,kind,record_property):
    b,data,curve=resource;v,f,m=fixture(kind);fields=[field(curve,i) for i in range(len(f))]
    row=integrate_moving_pixel(v,f,m,fields)
    with localcontext() as c:
        c.prec=90;a=boundary(data,0);target=[-x for x in a] if kind=='silhouette' else [x-y for x,y in zip(a,boundary(data,1))]
        for actual,wanted in zip(row['derivative'],target):assert actual.lo<=wanted<=actual.hi
        assert all(q.width<=D('2e-12') for q in row['value']+row['derivative'])
    candidate=native_integral(b,kind);comparison={}
    for key in ('value','derivative'):
        comparison[key]=[]
        for a,b in zip(row[key],candidate[key]):
            with localcontext() as c:
                c.prec=90;v=D.from_float(b);mid=(a.lo+a.hi)/2;distance=max(a.lo-v,v-a.hi,D(0))
                assert distance<=D('1e-10')+D('1e-9')*abs(mid)
                comparison[key].append(dict(candidate=b,midpoint=str(mid),radius=str((a.hi-a.lo)/2),distance=str(distance)))
    assert any(q.hi<0 or q.lo>0 for q in row['derivative'])
    record_property('moving_'+kind,json.dumps(dict(reference=row,independent_boundary=target,installed=candidate,comparison=comparison),default=str))

@pytest.mark.parametrize('kind',['silhouette','intersection'])
def test_installed_affine_visibility_gradient_matches_new_exact_geometry(resource,kind):
    from gffx.torch.render import opaque_integrate
    v,f,m=fixture(kind);colours=[[.2,.4,.6],[.6,.1,.3]][:len(f)]
    reference=integrate_moving_pixel(v,f,m,[lambda x,y,c=c:[x.lift(z) for z in c] for c in colours])
    theta=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    vertices=torch.tensor([[float(z) for z in q] for q in v],dtype=torch.float64)+theta*torch.tensor(m,dtype=torch.float64)
    colour=torch.tensor([[c]*3 for c in colours],dtype=torch.float64)
    rgba=opaque_integrate(vertices,torch.tensor(f,dtype=torch.int32),colour,1,1)[0,0]
    for k in range(4):
        ref=reference['value'][k] if k<3 else None
        value=float((ref.lo+ref.hi)/2) if ref else float(reference['mask'])
        derivative=float((reference['derivative'][k].lo+reference['derivative'][k].hi)/2) if k<3 else float(reference['mask_derivative'])
        assert abs(rgba[k].item()-value)<=1e-12
        got=torch.autograd.grad(rgba[k],theta,retain_graph=k<3)[0].item();assert abs(got-derivative)<=1e-12


def test_moving_perspective_field_transports_attributes_with_the_triangle(resource):
    from tools.opaque_principled_reference import perspective_field
    triangle=[(.1,.1,0),(.7,.1,0),(.1,.7,0)];motion=[(1,0,0)]*3
    attributes=dict(inverse_w=[1.,.85,1.1],normals=[[.01,.01,1.],[.015,.01,1.],[.01,.015,1.]],views=[[-.07,.06,1.]]*3,albedo=[[.2,.4,.6],[.4,.3,.5],[.3,.5,.2]],f0=[.04,.05,.08],direction=[.1,.1,1.])
    curve=resource[2];moving=moving_perspective_field(curve,triangle,motion,**attributes);fixed=perspective_field(curve,triangle,**attributes)
    correct=moving(Jet(.3,tangent=1),Jet(.3));wrong=fixed(Jet(.3,tangent=1),Jet(.3))
    assert all(q[0,1].lo<=0<=q[0,1].hi for q in correct)
    assert any(not q[0,1].lo<=0<=q[0,1].hi for q in wrong)
