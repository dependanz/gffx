"""Installed native opaque coverage; independent rational images and derivatives."""
from fractions import Fraction as Q
import importlib.util
from pathlib import Path
import pytest,torch
from tools.opaque_pixel_oracle import integrate

def fixture(name):
    p=Path(__file__).parents[1]/'packaging/test_opaque_boundary_models.py'
    spec=importlib.util.spec_from_file_location('opaque_fixtures',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return next(c for c in m.CASES if c[0]==name)

@pytest.mark.parametrize('name',['diagonal_a','diagonal_b','subdivision','silhouette','overlap','intersection','subpixel'])
@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
def test_native_opaque_coverage_matches_independent_pixel_integrals(name,dtype):
    from gffx.torch.render import opaque_coverage
    _,v,f,rgb,d,h,w=fixture(name)
    expected=integrate(v,f,rgb,height=h,width=w)['mask']
    t=torch.tensor([[float(q) for q in p] for p in v],dtype=dtype,requires_grad=True);faces=torch.tensor(f,dtype=torch.int32)
    got=opaque_coverage(t,faces,h,w)
    assert got.shape==(h,w)
    assert torch.allclose(got,torch.tensor([[float(q) for q in row] for row in expected],dtype=dtype),rtol=0,atol=1e-12 if dtype==torch.float64 else 1e-6)
    # Nonuniform cotangent avoids accepting merely correct total-area gradients.
    cot=torch.arange(1,h*w+1,dtype=dtype).reshape(h,w)/16
    grad=torch.autograd.grad((got*cot).sum(),t)[0];motion=torch.tensor(d,dtype=dtype)
    exact=[]
    for e in (Q(1,1024),Q(1,4096)):
        a=integrate([tuple(Q(q)-e*Q(s) for q,s in zip(p,m)) for p,m in zip(v,d)],f,rgb,height=h,width=w)['mask']
        b=integrate([tuple(Q(q)+e*Q(s) for q,s in zip(p,m)) for p,m in zip(v,d)],f,rgb,height=h,width=w)['mask']
        exact.append(sum((b[y][x]-a[y][x])*Q(y*w+x+1,16)/(2*e) for y in range(h) for x in range(w)))
    assert exact[0]==exact[1]
    assert float((grad*motion).sum())==pytest.approx(float(exact[0]),rel=0,abs=1e-12 if dtype==torch.float64 else 1e-6)

def test_native_opaque_gradient_matches_all_coordinates():
    from gffx.torch.render import opaque_coverage
    v=torch.tensor([[.13,.21,0.],[1.71,.27,.1],[.38,1.64,.2]],dtype=torch.float64,requires_grad=True)
    f=torch.tensor([[0,1,2]],dtype=torch.int32)
    assert torch.autograd.gradcheck(lambda x:opaque_coverage(x,f,2,2),(v,),eps=1e-6,atol=1e-6,rtol=1e-4)

def test_native_opaque_refuses_capacity_instead_of_dropping_geometry():
    from gffx.torch.render import opaque_coverage
    v=torch.tensor([[.1,.1,0.],[.8,.1,0.],[.1,.8,0.]],dtype=torch.float64)
    f=torch.tensor([[0,1,2]]*33,dtype=torch.int32)
    with pytest.raises(OverflowError,match='candidate|capacity|overflow'):opaque_coverage(v,f,1,1)
    with pytest.raises((ValueError,RuntimeError),match='workspace|budget|limit'):opaque_coverage(v,f[:1],1,1,workspace_limit_bytes=1)

def test_native_opaque_invalid_and_empty_inputs_are_explicit():
    from gffx.torch.render import opaque_coverage
    v=torch.tensor([[.1,.1,0.],[.8,.1,0.],[.1,.8,0.]],dtype=torch.float64)
    f=torch.tensor([[0,1,2]],dtype=torch.int32)
    assert torch.equal(opaque_coverage(v,f[:0],2,2),torch.zeros((2,2),dtype=v.dtype))
    for bad in (0,-1,True,1.5):
        with pytest.raises((ValueError,TypeError)):opaque_coverage(v,f,bad,2)
    with pytest.raises((ValueError,RuntimeError)):opaque_coverage(v,torch.tensor([[0,1,9]],dtype=torch.int32),2,2)
    bad=v.clone();bad[0,0]=float('nan')
    with pytest.raises((ValueError,RuntimeError)):opaque_coverage(bad,f,2,2)

def test_native_opaque_global_pixel_windows_preserve_reverse_accumulation():
    from gffx.torch.render import opaque_coverage
    _,v,f,_,_,h,w=fixture('silhouette');t=torch.tensor([[float(q) for q in row] for row in v],dtype=torch.float64,requires_grad=True);f=torch.tensor(f,dtype=torch.int32)
    whole=opaque_coverage(t,f,h,w);cot=torch.arange(1,h*w+1,dtype=t.dtype).reshape(h,w)/16
    expected=torch.autograd.grad((whole*cot).sum(),t)[0];pieces=[];actual=torch.zeros_like(t)
    for y in range(h):
        row=[]
        for x in range(w):
            q=opaque_coverage(t,f,1,1,origin=(x,y));row.append(q)
            actual+=torch.autograd.grad(q[0,0]*cot[y,x],t)[0]
        pieces.append(torch.cat(row,1))
    assert torch.equal(whole,torch.cat(pieces,0))
    assert torch.equal(expected,actual)


def test_native_opaque_small_area_survives_large_pixel_origin():
    from gffx.torch.render import opaque_coverage
    # All coordinates and the known area are exactly representable in float64.
    # Global shoelace subtraction can nevertheless lose this small positive area.
    v=torch.tensor([[.125,.125,0.],[.1328125,.125,0.],[.125,.1328125,0.]],dtype=torch.float64,requires_grad=True)
    f=torch.tensor([[0,1,2]],dtype=torch.int32)
    reference=opaque_coverage(v,f,1,1);gradient=torch.autograd.grad(reference.sum(),v)[0]
    assert reference.item()==1/32768
    for origin in (1024.,65536.,1048575.):
        shifted=(v.detach()+v.new_tensor([origin,origin,0.])).requires_grad_()
        actual=opaque_coverage(shifted,f,1,1,origin=(origin,origin))
        assert torch.equal(actual,reference)
        assert torch.equal(torch.autograd.grad(actual.sum(),shifted)[0],gradient)
