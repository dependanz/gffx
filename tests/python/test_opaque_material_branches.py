"""Finite piecewise-constant material branches with independent pixel/flux bounds."""
import json,math,torch,pytest
from decimal import Decimal as D,localcontext
from fractions import Fraction as Q
from tools.opaque_shading_intervals import Jet,I
from tools.opaque_principled_reference import PreparedPrincipled
from tools.opaque_material_reference import partition,integrate_partition
from tools.principled_material_oracle import material_pair

@pytest.fixture(scope='module')
def resource():
    from gffx.torch.material import PrincipledBudget
    b=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    return b,{k:v.tolist() for k,v in b._data.items()}

@pytest.mark.parametrize('roughness,fresnel,two_sided,back',[(.15,0.,False,False),(.45,.04,False,False),(.7,1.,False,False),(.95,.08,False,False),(.45,.04,True,True),(.45,.04,False,True)])
def test_opaque_material_branch_pixel_and_boundary_flux(resource,roughness,fresnel,two_sided,back,record_property):
    from tools.opaque_material_branches import radiance
    from gffx.torch.material import principled
    from gffx.torch.render import opaque_quadrature
    b,data=resource;x=Jet(0);params=[.01,.01,-1. if back else 1.,-.07,.06,1.,.1,.1,1.,.2,.4,.6,roughness,fresnel,fresnel,fresnel]
    curve=PreparedPrincipled(data,roughness)
    q=radiance(curve,*[[x.lift(v) for v in params[a:a+3]] for a in (0,3,6,9,13)],two_sided=two_sided)
    radii=[a[0,0] for a in q];background=[I('.03'),I('.07'),I('.11')]
    regions=partition([[(1,0),(0,0),(-Q(37,100),-1)]])
    ref=integrate_partition(regions,lambda sides:lambda xx,yy:[xx.lift(a) for a in (radii if sides[0]<0 else background)])
    with localcontext() as ctx:
        ctx.prec=90;p=list(map(D.from_float,params));normal=p[:3]
        if two_sided and sum(a*b for a,b in zip(normal,p[3:6]))<0:normal=[-v for v in normal]
        cos=max(D(0),sum(a*b for a,b in zip(normal,p[6:9]))/(sum(a*a for a in normal)*sum(a*a for a in p[6:9])).sqrt())
        expected=[v*cos for v in material_pair(data,p,two_sided=two_sided)]
        target=[D('.37')*a+D('.63')*z.lo for a,z in zip(expected,background)];flux=[a-z.lo for a,z in zip(expected,background)]
        for row,targets in ((ref['value'],target),(ref['derivative'],flux)):
            for a,w in zip(row,targets):assert a.lo<=w<=a.hi and a.width<=D('2e-12')
    t=lambda a:torch.tensor(a,dtype=torch.float64);theta=t(0).requires_grad_();v=t([[0,0,.2],[1,0,.2],[1,1,.2],[0,1,.2]]);f=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32)
    cuts=(v[f.long(),0]-.37-theta)[...,None].contiguous();owners,samples=opaque_quadrature(v,f,cuts,1,1)
    xx=(samples[0,:,:3]*v[f[owners[0].clamp_min(0).long()].long(),0]).sum(-1)
    colour=principled(t([params[:3]]),t([params[3:6]]),t([params[6:9]]),albedo=t(params[9:12]),roughness=t(roughness),f0=t(params[13:]),budget=b,two_sided=two_sided)[0,0]*float(cos)
    values=torch.where((xx<.37+theta)[:,None],colour,t([.03,.07,.11]))
    image=(values*samples[0,:,3,None]).sum(0);gradient=torch.stack([torch.autograd.grad(image[k],theta,retain_graph=k<2)[0] for k in range(3)])
    for row,actual in ((ref['value'],image),(ref['derivative'],gradient)):
        for a,w in zip(row,actual.tolist()):
            with localcontext() as ctx:
                ctx.prec=90;w=D.from_float(w);assert max(a.lo-w,w-a.hi,D(0))<=D('1e-10')+D('1e-9')*abs((a.lo+a.hi)/2)
    record_property('branch_pixel',json.dumps(dict(roughness=roughness,f0=fresnel,two_sided=two_sided,back=back,reference=ref,installed=image.tolist(),gradient=gradient.tolist()),default=str))


def test_opaque_material_unresolved_horizon_and_angular_crossings_refuse(resource):
    from tools.opaque_material_branches import radiance
    b,data=resource;curve=PreparedPrincipled(data,.45);x=Jet(0)
    with pytest.raises(ValueError,match='horizon|branch'):
        radiance(curve,[x.lift(0),x.lift(0),x.lift(1)],[x.lift(1),x.lift(0),x.lift(I('-.01','.01'))],[x.lift(0),x.lift(0),x.lift(1)],[x.lift(.3)]*3,[x.lift(.04)]*3)
    node=data['cosine_nodes'][300]
    with pytest.raises(ValueError,match='knot'):curve.loss(x.lift(I(node-1e-8,node+1e-8)))
