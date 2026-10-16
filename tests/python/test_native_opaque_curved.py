"""Installed shared curved quadrature: analytic area, flux, and all input VJPs."""
import os,torch,pytest
from decimal import Decimal,localcontext

def device():return os.environ.get('GFFX_CURVED_DEVICE','cpu')
def fixture():
    v=torch.tensor([[.1,.1,0.],[.9,.1,0.],[.1,.9,0.]],dtype=torch.float64,device=device(),requires_grad=True)
    f=torch.tensor([[0,1,2]],dtype=torch.int32,device=device())
    # q=b1+b2-a-b*b2^2, homogeneous in barycentric coordinates.
    a=torch.tensor(.25,dtype=v.dtype,device=v.device,requires_grad=True)
    b=torch.tensor(.1,dtype=v.dtype,device=v.device,requires_grad=True)
    q=torch.stack([-a,1-a,1-a-b,1-2*a,1-2*a,2-2*a])[None]
    return v,f,a,b,q

def evaluate(v,f,c):
    from gffx.torch.render import opaque_quadrature
    return opaque_quadrature(v,f,v.new_empty((len(f),3,0)),1,1,curves=c)

def test_curved_area_boundary_flux_and_complement(record_property):
    v,f,a,b,c=fixture();ids,q=evaluate(v,f,c);beta=q[...,:3]
    front=beta[...,1]+beta[...,2]-a-b*beta[...,2]**2>0
    area=(q[...,3]*front).sum();ga,gb,gv=torch.autograd.grad(area,(a,b,v),retain_graph=True)
    with localcontext() as ctx:
        ctx.prec=80;aa=Decimal('.25');bb=Decimal('.1');rr=2*aa/(1+(1-4*aa*bb).sqrt())
        expected=Decimal('.64')*(Decimal('.5')-aa*rr+rr*rr/2-bb*rr**3/3)
    assert abs(area.item()-float(expected))<1e-11
    assert abs(ga.item()+.64*float(rr))<1e-11
    assert abs(gb.item()+.64*float(rr**3)/3)<1e-11
    total=q[...,3].sum();assert abs(total.item()-.32)<1e-13
    ca,cb=torch.autograd.grad(total,(a,b));assert abs(ca.item())<1e-13 and abs(cb.item())<1e-13
    assert int((ids>=0).sum())==288
    record_property('area',area.item());record_property('boundary_derivative',ga.item())

def test_curved_all_geometry_and_coefficient_gradients():
    v,f,a,b,c=fixture();c=c.detach().requires_grad_()
    def fn(v,c):
        ids,q=evaluate(v,f,c);beta=q[...,:3]
        sign=(beta[...,0]**2*c[0,0]+beta[...,1]**2*c[0,1]+beta[...,2]**2*c[0,2]+beta[...,0]*beta[...,1]*c[0,3]+beta[...,0]*beta[...,2]*c[0,4]+beta[...,1]*beta[...,2]*c[0,5])>0
        return ((beta[...,0]**2+beta[...,1]**3+2*sign)*q[...,3]).sum()
    assert torch.autograd.gradcheck(fn,(v,c),eps=1e-6,atol=1e-6,rtol=1e-4)

def test_curved_refuses_tangent_uncertified_and_combined_capacity():
    from gffx.torch.render import opaque_quadrature
    v,f,a,b,c=fixture()
    # Exact dyadic screen edges keep the intended vertex event exact after
    # clipping; the decimal fixture's rounded clip vertex lies slightly inside.
    v=v.detach().new_tensor([[.25,.25,0],[.75,.25,0],[.25,.75,0]])
    for curve in ([0,1,1,1,1,2],[1,1,1,-4,-4,-4]):
        with pytest.raises(OverflowError):evaluate(v,f,v.new_tensor([curve]))
    with pytest.raises(ValueError,match='capacity'):
        opaque_quadrature(v,f,v.new_ones((1,3,16)),1,1,curves=c)
    with pytest.raises(ValueError,match='finite'):evaluate(v,f,c*float('nan'))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_curved_shared_kernel_bytes_and_vjps(dtype,record_property):
    from gffx.torch.render import opaque_quadrature
    v,f,a,b,c=fixture();saved=[]
    for dev in ('cpu','cuda'):
        vv=v.detach().to(device=dev,dtype=dtype).requires_grad_();cc=c.detach().to(device=dev,dtype=dtype).requires_grad_();ff=f.to(dev)
        ids,q=evaluate(vv,ff,cc)
        g=torch.arange(q.numel(),device=dev,dtype=q.dtype).reshape(q.shape)/q.new_tensor(4096)
        gv,gc=torch.autograd.grad(q,(vv,cc),g)
        saved.append([x.detach().cpu().numpy().tobytes() for x in (ids,q,gv,gc)])
    assert saved[0]==saved[1]
    record_property('exact_curved','owners,samples,all vertex and six coefficient VJPs byte-identical')


@pytest.mark.parametrize('count',[1,2,3,6,7])
def test_curved_radial_refinement_keeps_shared_capacity(count):
    from gffx.torch.render import opaque_quadrature
    vertices=[]
    for i in range(count):
        x=(i%3)*.25+.0625;y=(i//3)*.25+.0625
        vertices.extend([[x,y,0],[x+.125,y,0],[x,y+.125,0]])
    v=torch.tensor(vertices,dtype=torch.float64,device=device());f=torch.arange(count*3,dtype=torch.int32,device=device()).reshape(count,3)
    # A known conic on disjoint dyadic triangles; exact whole-area1/128 each.
    c=v.new_tensor([[-.25,.75,.65,.5,.5,1.5]]).repeat(count,1).requires_grad_()
    if count==7:
        with pytest.raises(OverflowError):opaque_quadrature(v,f,v.new_empty((count,3,0)),1,1,curves=c)
        return
    ids,q=opaque_quadrature(v,f,v.new_empty((count,3,0)),1,1,curves=c)
    assert q.shape==(1,800,4) and ids.shape==(1,800)
    expected=count*(288 if count<=2 else 120)
    assert int((ids>=0).sum())==expected and expected<=800
    area=q[...,3].sum();assert abs(area.item()-count/128)<1e-13
    g,=torch.autograd.grad(area,c);assert g.abs().max()<1e-13
