"""Installed depth-ordered pixel integrals and complete first reverse derivatives."""
from fractions import Fraction as Q
import pytest,torch
from test_native_opaque_coverage import fixture
from tools.opaque_pixel_oracle import integrate

def api():
    from gffx.torch.render import opaque_integrate
    return opaque_integrate

def tensor(rows,dtype=torch.float64,grad=False):
    return torch.tensor(rows,dtype=dtype,requires_grad=grad)

def expected(v,f,c,h,w,affine=False):
    from tools.opaque_pixel_oracle import integrate_affine
    row=(integrate_affine if affine else integrate)(v,f,c,height=h,width=w)
    return [[[*row['colour'][y][x],row['mask'][y][x]] for x in range(w)] for y in range(h)]

@pytest.mark.parametrize('name',['diagonal_a','diagonal_b','subdivision','silhouette','overlap','intersection','subpixel'])
@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
def test_native_opaque_colour_matches_depth_ordered_reference(name,dtype):
    _,v,f,c,m,h,w=fixture(name);v0=[[float(q) for q in p] for p in v]
    vt=tensor(v0,dtype,True);ft=torch.tensor(f,dtype=torch.int32);ct=tensor([[list(map(float,q))]*3 for q in c],dtype,True)
    actual=api()(vt,ft,ct,h,w)
    ref=tensor([[[float(z) for z in p] for p in row] for row in expected(v,f,c,h,w)],dtype)
    atol=1e-12 if dtype==torch.float64 else 1e-6
    torch.testing.assert_close(actual,ref,rtol=0,atol=atol)
    cot=torch.arange(1,h*w*4+1,dtype=dtype).reshape(h,w,4)/32
    gv,gc=torch.autograd.grad((actual*cot).sum(),(vt,ct));motion=tensor(m,dtype)
    derivatives=[]
    for eps in (Q(1,1024),Q(1,4096)):
        samples=[]
        for sign in (-1,1):
            moved=[tuple(Q(q)+sign*eps*Q(s) for q,s in zip(p,d)) for p,d in zip(v,m)]
            rgba=expected(moved,f,c,h,w)
            samples.append(sum(Q(4*(y*w+x)+k+1,32)*rgba[y][x][k] for y in range(h) for x in range(w) for k in range(4)))
        derivatives.append((samples[1]-samples[0])/(2*eps))
    assert derivatives[0]==derivatives[1]
    assert float((gv*motion).sum())==pytest.approx(float(derivatives[0]),abs=atol,rel=0)
    assert torch.isfinite(gc).all()

@pytest.mark.parametrize('faces',[[(0,1,2),(0,2,3)],[(0,1,3),(1,2,3)],[(0,1,4),(1,2,4),(2,3,4),(3,0,4)]])
def test_native_opaque_affine_colour_has_no_tessellation_seam(faces):
    v=[[-1,-1,0],[3,-1,0],[3,3,0],[-1,3,0],[.3,.7,0]]
    def radiance(x,y):return [(x+1)/4,(y+1)/4,(x+y+2)/8]
    c=[[radiance(*v[i][:2]) for i in f] for f in faces]
    out=api()(tensor(v),torch.tensor(faces,dtype=torch.int32),tensor(c),2,2)
    ref=expected(v,faces,c,2,2,True)
    torch.testing.assert_close(out,tensor([[[float(z) for z in p] for p in row] for row in ref]),rtol=0,atol=1e-12)
    for y in range(2):
        for x in range(2):torch.testing.assert_close(out[y,x],tensor([*radiance(x+.5,y+.5),1]),rtol=0,atol=1e-12)

def test_native_opaque_colour_all_geometry_depth_and_radiance_partials():
    v=tensor([[-1.2,-1.1,-1.4],[3.2,-.9,2.1],[-1.1,3.4,-1.2],[-1.3,-1.3,.1],[3.4,-1.2,.1],[-1.2,3.5,.1]],grad=True)
    f=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32)
    c=tensor([[[.3,.1,.7],[.9,.4,.2],[.1,.8,.5]],[[.7,.3,.2],[.2,.9,.1],[.4,.1,.8]]],grad=True)
    assert torch.autograd.gradcheck(lambda vv,cc:api()(vv,f,cc,1,1),(v,c),eps=1e-6,atol=1e-6,rtol=1e-4)

def test_native_opaque_hidden_colour_and_depth_tie_have_explicit_ownership():
    v=tensor([[-1,-1,0],[3,-1,0],[-1,3,0],[-1,-1,1],[3,-1,1],[-1,3,1]],grad=True)
    f=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32);c=tensor([[[1,0,0]]*3,[[0,0,1]]*3],grad=True)
    out=api()(v,f,c,1,1);assert torch.equal(out,tensor([[[1,0,0,1]]]))
    gv,gc=torch.autograd.grad(out.sum(),(v,c));assert torch.equal(gc[1],torch.zeros_like(gc[1])) and torch.equal(gv[3:],torch.zeros_like(gv[3:]))
    v=v.detach();v[3:,2]=0
    assert torch.equal(api()(v,f,c.detach(),1,1),out.detach())
    assert torch.equal(api()(v,f.flip(0).contiguous(),c.detach().flip(0).contiguous(),1,1),tensor([[[0,0,1,1]]]))

def test_native_opaque_colour_windows_keep_ordered_reverse_contributions():
    _,v,f,c,_,h,w=fixture('intersection');v=tensor([[float(q) for q in p] for p in v],grad=True);f=torch.tensor(f,dtype=torch.int32);c=tensor([[q]*3 for q in c],grad=True)
    cot=torch.arange(1,h*w*4+1,dtype=v.dtype).reshape(h,w,4)/16
    full=api()(v,f,c,h,w);gv,gc=torch.autograd.grad((full*cot).sum(),(v,c));gotv=torch.zeros_like(v);gotc=torch.zeros_like(c);rows=[]
    for y in range(h):
        row=[]
        for x in range(w):
            value=api()(v,f,c,1,1,origin=(x,y));row.append(value)
            a,b=torch.autograd.grad((value[0,0]*cot[y,x]).sum(),(v,c));gotv+=a;gotc+=b
        rows.append(torch.cat(row,1))
    assert torch.equal(full,torch.cat(rows,0)) and torch.equal(gv,gotv) and torch.equal(gc,gotc)

def test_native_opaque_colour_refusals_and_empty_outputs():
    v=tensor([[.1,.1,0],[.8,.1,0],[.1,.8,0]],grad=True);f=torch.tensor([[0,1,2]],dtype=torch.int32);c=torch.ones((1,3,3),dtype=v.dtype,requires_grad=True)
    with pytest.raises((ValueError,RuntimeError)):api()(v,f,c[:,:2],1,1)
    with pytest.raises((ValueError,RuntimeError)):api()(v,f,c,1,1,workspace_limit_bytes=1)
    with pytest.raises(OverflowError):api()(v,f.expand(33,3).contiguous(),c.expand(33,3,3).contiguous(),1,1)
    assert torch.equal(api()(v,f[:0],c[:0],2,2),torch.zeros((2,2,4),dtype=v.dtype))
    with pytest.raises(RuntimeError,match='first derivatives'):
        torch.autograd.grad(api()(v,f,c,1,1).sum(),v,create_graph=True)

@pytest.mark.parametrize('origin',[1024,65536,1048575])
def test_native_opaque_colour_small_positive_integral_survives_translation(origin):
    local=tensor([[.25,.25,0],[.25+1/128,.25,0],[.25,.25+1/128,0]],grad=True)
    shifted=local.detach().clone();shifted[:,:2]+=origin;shifted.requires_grad_()
    f=torch.tensor([[0,1,2]],dtype=torch.int32);c=tensor([[[1,0,0],[0,1,0],[0,0,1]]],grad=True)
    a=api()(local,f,c,1,1);b=api()(shifted,f,c,1,1,origin=(origin,origin))
    assert a[0,0,3]==1/32768 and (a[0,0,:3]>0).all()
    assert torch.equal(a,b)
    ga=torch.autograd.grad(a.sum(),(local,c),retain_graph=True);gb=torch.autograd.grad(b.sum(),(shifted,c))
    assert all(torch.equal(x,y) for x,y in zip(ga,gb))

def test_native_opaque_colour_reversed_winding_preserves_corner_field():
    v=tensor([[.125,.25,0],[1.5,.125,0],[.25,1.5,0]],grad=True);f=torch.tensor([[0,1,2]],dtype=torch.int32)
    c=tensor([[[.2,.3,.9],[.8,.1,.4],[.6,.9,.2]]],grad=True)
    a=api()(v,f,c,2,2);b=api()(v,f.flip(1).contiguous(),c.flip(1).contiguous(),2,2)
    torch.testing.assert_close(a,b,rtol=0,atol=1e-12)
    cot=torch.arange(16,dtype=v.dtype).reshape(2,2,4)/16
    ga=torch.autograd.grad((a*cot).sum(),(v,c));gb=torch.autograd.grad((b*cot).sum(),(v,c))
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=0,atol=1e-12)
