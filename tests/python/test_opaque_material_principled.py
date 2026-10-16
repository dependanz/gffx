"""Stored-resource roughness and piecewise material integration qualification."""
import json,time,math
from decimal import Decimal as D,localcontext
from fractions import Fraction as Q
import torch,pytest
from tools.opaque_material_reference import RoughPrincipled,partition,integrate_partition,bilinear
from tools.opaque_shading_intervals import I,Jet
from tools.principled_material_oracle import material_pair

@pytest.fixture(scope='module')
def resource():
    from gffx.torch.material import PrincipledBudget
    b=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    return b,{k:v.tolist() for k,v in b._data.items()}

def params():return [.01,.01,1.,-.07,.06,1.,.1,.1,1.,.2,.4,.6,.827,.04,.05,.08]

def field(data,r,x,albedo=None):
    p=params();curve=RoughPrincipled(data,r)
    return curve.radiance(*[[x.lift(v) for v in p[a:a+3]] for a in (0,3,6)],albedo or [x.lift(v) for v in p[9:12]],[x.lift(v) for v in p[13:]])

def test_roughness_point_and_first_derivative_match_independent_formula(resource,record_property):
    b,data=resource;started=time.monotonic();x=Jet(0);out=field(data,x.lift(.827,tangent=1),x)
    with localcontext() as c:
        c.prec=90;p=list(map(D.from_float,params()));cos=sum(a*b for a,b in zip(p[:3],p[6:9]))/(sum(a*a for a in p[:3])*sum(a*a for a in p[6:9])).sqrt()
        nominal=[v*cos for v in material_pair(data,p)]
        for a,w in zip(out,nominal):assert a[0,0].lo<=w<=a[0,0].hi
        diffs=[]
        for eps in (D('1e-15'),D('1e-18')):
            a=p.copy();z=p.copy();a[12]+=eps;z[12]-=eps
            d=[(v-w)*cos/(2*eps) for v,w in zip(material_pair(data,a),material_pair(data,z))];diffs.append(list(map(str,d)))
            for a,w in zip(out,d):assert max(a[0,1].lo-w,w-a[0,1].hi,D(0))<D('1e-22')
    record_property('roughness_point',json.dumps(dict(values=[str(q[0,0]) for q in out],derivatives=[str(q[0,1]) for q in out],finite_differences=diffs,seconds=time.monotonic()-started,n=len(data['_clog'])),default=str))


def test_spatial_roughness_has_separate_value_and_gradient_error_bounds(resource,record_property):
    from tools.opaque_material_reference import CurveEnvelope
    b,data=resource;started=time.monotonic()
    def check():
        if time.monotonic()-started>90:raise ValueError('roughness reference90s deadline')
    env=CurveEnvelope(data,Q(826,1000),Q(828,1000),checkpoint=check)
    def fn(signs):
        def sample(x,y):
            r=x.lift(Q(826,1000),tangent=1)+x*Q(2,1000)
            # The exact Duffy image lies inside the unit pixel, hence this linear
            # roughness lies in the declared convex range, even if an interval
            # rectangle loses x/(1-x) correlation. Derivatives are not clamped.
            r.c[0,0]=I(max(env.lo,r[0,0].lo),min(env.hi,r[0,0].hi))
            curve=env.at(r);p=params()
            return curve.radiance(*[[x.lift(v) for v in p[a:a+3]] for a in (0,3,6)],[x.lift(v) for v in p[9:12]],[x.lift(v) for v in p[13:]])
        return sample
    result=integrate_partition(partition([]),fn,checkpoint=check,gauss_points=3)
    assert all(q.width<=D('2e-12') for q in result['value']+result['derivative'])
    # Constant n,v,l: d/d(offset) integral L(.826+.002*x) dx
    # is exactly [L(.828)-L(.826)]/.002. No quadrature defines this target.
    with localcontext() as c:
        c.prec=90;p=list(map(D.from_float,params()));cos=sum(a*b for a,b in zip(p[:3],p[6:9]))/(sum(a*a for a in p[:3])*sum(a*a for a in p[6:9])).sqrt()
        p[12]=D('.826');a=material_pair(data,p);p[12]=D('.828');z=material_pair(data,p)
        target=[(b-a)*cos/D('.002') for a,b in zip(a,z)]
        for a,w in zip(result['derivative'],target):assert a.lo<=w<=a.hi
    from gffx.torch.material import principled
    import numpy as np
    nodes,weights=np.polynomial.legendre.leggauss(16);nodes=(nodes+1)/2;weights=weights/2
    t=lambda v:torch.tensor(v,dtype=torch.float64);p=params();values=[];grads=[]
    for start in range(0,16,4):
        offset=t(0.).requires_grad_();r=t(.826)+.002*t(nodes[start:start+4].copy())+offset
        rgb=principled(t([p[:3]]*4),t([p[3:6]]*4),t([p[6:9]]),albedo=t(p[9:12]),roughness=r,f0=t(p[13:]),budget=b)[:,0,:]
        output=(rgb*t(weights[start:start+4].copy())[:,None]*float(cos)).sum(0)
        values.append(output.detach().tolist());grads.append([torch.autograd.grad(output[k],offset,retain_graph=k<2)[0].item() for k in range(3)])
    candidate={key:[math.fsum(q[k] for q in rows) for k in range(3)] for key,rows in [('value',values),('derivative',grads)]}
    for key in candidate:
        for a,w in zip(result[key],candidate[key]):
            with localcontext() as c:
                c.prec=90;w=D.from_float(w);assert max(a.lo-w,w-a.hi,D(0))<=D('1e-10')+D('1e-9')*abs((a.lo+a.hi)/2)
    check()
    record_property('spatial_roughness',json.dumps(dict(result=result,seconds=time.monotonic()-started,boundary_derivative=target,installed=candidate),default=str))


def test_roughness_branch_diagnostic(resource,record_property):
    from tools.opaque_material_reference import RangeJet
    b,data=resource;centre=I(Q(827,1000));radius=I('-.001','.001')
    q=RangeJet(Jet.variable(I('.826','.828'),order=7),centre,radius)
    RoughPrincipled(data,q)


def texture_fixture():return [[[.2,.1,.3],[.8,.3,.2]],[[.4,.7,.5],[.6,.5,.9]]]

@pytest.mark.parametrize('mode',['material','uv','texel'])
def test_textured_material_transition_matches_exact_formula_and_installed_calls(resource,mode,record_property):
    from tools.opaque_material_reference import texture_cell
    from tools.opaque_principled_reference import PreparedPrincipled
    b,data=resource;p=params();texture=texture_fixture();starts=[];gains=[]
    for r in (.45,.827):
        curve=PreparedPrincipled(data,r);x=Jet(0);vectors=[[x.lift(v) for v in p[a:a+3]] for a in (0,3,6)];f0=[x.lift(v) for v in p[13:]]
        a=curve.radiance(*vectors,[x.lift(0)]*3,f0);z=curve.radiance(*vectors,[x.lift(1)]*3,f0)
        starts.append([q[0,0] for q in a]);gains.append([(v-w)[0,0] for v,w in zip(z,a)])
    cuts=[[(1,0),(0,0),(-v,1 if mode=='uv' else 0)] for v in (Q(1,4),Q(3,4))]+[[(0,0),(1,0),(-v,0)] for v in (Q(1,4),Q(3,4))]+[[(1,0),(0,0),(-Q(1,2),-1 if mode=='material' else 0)]]
    def field(sides):
        column=-1+sum(v>0 for v in sides[:2]);row=-1+sum(v>0 for v in sides[2:4]);mat=int(sides[4]>0)
        def f(x,y):
            u=x+x.lift(0,tangent=1 if mode=='uv' else 0);d=[[[0]*3 for _ in range(2)] for _ in range(2)];d[1][1][0]=int(mode=='texel')
            a=texture_cell(texture,u,y,column,row,tangent=d)
            return [x.lift(starts[mat][k])+a[k]*gains[mat][k] for k in range(3)]
        return f
    reference=integrate_partition(partition(cuts),field)
    # Independent ninety-digit constant-material formula and closed bilinear moments.
    with localcontext() as c:
        c.prec=90;pp=list(map(D.from_float,p));cos=sum(a*b for a,b in zip(pp[:3],pp[6:9]))/(sum(a*a for a in pp[:3])*sum(a*a for a in pp[6:9])).sqrt();ss=[];gg=[]
        for r in (.45,.827):
            pp[12]=D.from_float(r);pp[9:12]=[D(0)]*3;a=material_pair(data,pp);pp[9:12]=[D(1)]*3;z=material_pair(data,pp)
            ss.append([v*cos for v in a]);gg.append([(v-w)*cos for v,w in zip(z,a)])
        a=[(D.from_float(texture[0][0][k])+D.from_float(texture[1][0][k]))/2 for k in range(3)];z=[(D.from_float(texture[0][1][k])+D.from_float(texture[1][1][k]))/2 for k in range(3)]
        expected=[(ss[0][k]+ss[1][k])/2+gg[0][k]*(7*a[k]+z[k])/16+gg[1][k]*(a[k]+7*z[k])/16 for k in range(3)]
        gradient=[ss[0][k]-ss[1][k]+(gg[0][k]-gg[1][k])*(a[k]+z[k])/2 if mode=='material' else (gg[0][k]+gg[1][k])*(z[k]-a[k])/2 if mode=='uv' else (gg[0][k]+7*gg[1][k])/32 if k==0 else D(0) for k in range(3)]
        for key,targets in [('value',expected),('derivative',gradient)]:
            for a,w in zip(reference[key],targets):assert a.lo<=w<=a.hi
    installed=native_texture_integral(b,mode,float(cos))
    for key in installed:
        for a,w in zip(reference[key],installed[key]):
            with localcontext() as c:
                c.prec=90;w=D.from_float(w);assert max(a.lo-w,w-a.hi,D(0))<=D('1e-10')+D('1e-9')*abs((a.lo+a.hi)/2)
    record_property('texture_'+mode,json.dumps(dict(reference=reference,formula=dict(value=expected,derivative=gradient),installed=installed),default=str))

def native_texture_integral(budget,mode,cos):
    # A separate rectangular Gauss2 rule, installed texture and installed Principled.
    from gffx.torch.render import texture_pyramid,texture,WRAP_CLAMP
    from gffx.torch.material import principled
    t=lambda a:torch.tensor(a,dtype=torch.float64);p=params();nodes=[.5-.5/math.sqrt(3),.5+.5/math.sqrt(3)];values=[];grads=[]
    for ix in range(4):
        for iy in range(3):
            theta=t(0.).requires_grad_();uv_shift=theta if mode=='uv' else theta*0;edge=theta+.5 if mode=='material' else theta*0+.5
            xs=[theta*0,.25-uv_shift,edge,.75-uv_shift,theta*0+1];ys=[0,.25,.75,1]
            xx=[xs[ix]+(xs[ix+1]-xs[ix])*n for n in nodes for _ in nodes];yy=[ys[iy]+(ys[iy+1]-ys[iy])*n for _ in nodes for n in nodes]
            coordinates=torch.stack([torch.stack(xx)+uv_shift,t(yy)],-1)
            tex=t(texture_fixture());d=torch.zeros_like(tex);d[1,1,0]=1;tex=tex+(theta*d if mode=='texel' else theta*0)
            pyramid,offsets=texture_pyramid(tex,levels=1);albedo=texture(pyramid,offsets,2,2,coordinates.contiguous(),wrap_u=WRAP_CLAMP,wrap_v=WRAP_CLAMP)
            rgb=principled(t([p[:3]]*4),t([p[3:6]]*4),t([p[6:9]]),albedo=albedo,roughness=t(.45 if ix<2 else .827),f0=t(p[13:]),budget=budget)[:,0,:]
            output=rgb.sum(0)*((xs[ix+1]-xs[ix])*(ys[iy+1]-ys[iy])*.25*cos)
            values.append(output.detach().tolist());grads.append([torch.autograd.grad(output[k],theta,retain_graph=k<2)[0].item() for k in range(3)])
    return {key:[math.fsum(q[k] for q in rows) for k in range(3)] for key,rows in [('value',values),('derivative',grads)]}
