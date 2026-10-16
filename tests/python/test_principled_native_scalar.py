"""Standalone arithmetic, independent oracle and exact installed-reference bridge."""
import ctypes
from decimal import Decimal,localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import pytest


def bits(x):
    return struct.pack('<d',x)


@pytest.fixture
def probe(record_property):
    path=os.environ.get('GFFX_PRINCIPLED_SCALAR_PROBE')
    if not path:pytest.skip('explicit standalone native library required')
    path=Path(path);lib=ctypes.CDLL(str(path));ptr=ctypes.POINTER(ctypes.c_double)
    lib.principled_test_log.argtypes=[ctypes.c_double,ptr]
    lib.principled_test_log.restype=ctypes.c_int
    lib.principled_test_curve.argtypes=[ptr,ctypes.c_double,ctypes.c_double,ptr,ptr]
    lib.principled_test_curve.restype=ctypes.c_int
    record_property('principled_scalar_library',json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    return lib


def cases():
    controls=[(0.,0.,0.,0.),(-1.,-1.,-1.,-1.),(-745.,)*4,
              (-.01,-.3,-1.,-2.),(-2.,-1.,-.3,-.01),
              (-745.,-744.,-740.,-730.),(-700.,-500.,-100.,0.),
              (-.0001,-.00011,-.00009,0.)]
    coordinates=[0.,math.nextafter(0.,1.),2**-20,.01,.125,.3,.5,.73,.99,math.nextafter(1.,0.),1.]
    for b in controls:
        for u in coordinates:
            for g in (-.75,0.,.125,1.):yield b,u,g


def evaluate(probe,b,u,g):
    value,gradient=ctypes.c_double(91.),ctypes.c_double(92.)
    assert probe.principled_test_curve((ctypes.c_double*4)(*b),u,g,ctypes.byref(value),ctypes.byref(gradient))==0
    return value.value,gradient.value


def test_native_curve_independent_decimal(probe,record_property):
    count=0
    with localcontext() as ctx:
        ctx.prec=180
        for b,u,g in cases():
            c=[Decimal.from_float(x) for x in b];t=Decimal.from_float(u);s=1-t
            # Bernstein polynomial and its analytic derivative, not de Casteljau.
            p=s**3*c[0]+3*s*s*t*c[1]+3*s*t*t*c[2]+t**3*c[3]
            dp=3*(s*s*(c[1]-c[0])+2*s*t*(c[2]-c[1])+t*t*(c[3]-c[2]))
            exact=p.exp();vjp=Decimal.from_float(g)*exact*dp
            value,gradient=evaluate(probe,b,u,g)
            # Existing bounded exp error allowance plus cubic rounding and
            # final subnormal rounding. This scalar screen does not relax the
            # independent physical material criteria or the strict byte bridge.
            eps=Decimal(2)**-53;tiny=Decimal(2)**-1074
            assert abs(Decimal.from_float(value)-exact)<=4096*eps*exact+tiny
            assert abs(Decimal.from_float(gradient)-vjp)<=8192*eps*abs(vjp)+2048*tiny
            if b==(0.,)*4:assert value==1. and gradient==0.
            count+=1
    record_property('independent_scalar_cases',str(count))


def test_native_log_independent_and_installed_bytes(probe,record_property):
    import torch
    from gffx.torch._curve_math import log_unit
    base=[math.ldexp(1.,k) for k in range(-1074,1)]+[i/1024 for i in range(1,1025)]
    inputs=sorted({x for v in base for x in (v,math.nextafter(v,0.),math.nextafter(v,1.)) if 0<x<=1})
    expected=log_unit(torch.tensor(inputs,dtype=torch.float64)).tolist()
    with localcontext() as ctx:
        ctx.prec=90
        for x,reference in zip(inputs,expected):
            out=ctypes.c_double(91.)
            assert probe.principled_test_log(x,ctypes.byref(out))==0
            assert bits(out.value)==bits(reference),(x,out.value,reference)
            exact=Decimal.from_float(x).ln()
            assert abs(Decimal.from_float(out.value)-exact)<=128*Decimal(2)**-53*(1+abs(exact))
    record_property('log_inputs',str(len(inputs)))


def test_native_curve_installed_value_and_vjp_bytes(probe,record_property):
    import torch
    from gffx.torch._curve_math import roughness_interpolate
    rows=list(cases());b=torch.tensor([v[0] for v in rows],dtype=torch.float64).T.contiguous()
    u=torch.tensor([v[1] for v in rows],dtype=torch.float64,requires_grad=True)
    g=torch.tensor([v[2] for v in rows],dtype=torch.float64)
    y=roughness_interpolate(b,u);du,=torch.autograd.grad(y,u,g)
    for row,reference,grad in zip(rows,y.tolist(),du.tolist()):
        value,gradient=evaluate(probe,*row)
        assert bits(value)==bits(reference),(row,value,reference)
        assert bits(gradient)==bits(grad),(row,gradient,grad)
    record_property('installed_reference_cases',str(len(rows)))


def test_native_curve_errors_preserve_outputs(probe):
    for b,u,g in [((float('nan'),0.,0.,0.),.5,1.),((-746.,)*4,.5,1.),((.01,)*4,.5,1.),((-1.,)*4,-.01,1.),((-1.,)*4,1.01,1.),((-1.,)*4,.5,float('inf'))]:
        out,grad=ctypes.c_double(91.),ctypes.c_double(92.)
        assert probe.principled_test_curve((ctypes.c_double*4)(*b),u,g,ctypes.byref(out),ctypes.byref(grad))!=0
        assert out.value==91. and grad.value==92.
    for value in (0.,-1.,1.01,float('inf'),float('nan')):
        out=ctypes.c_double(91.)
        assert probe.principled_test_log(value,ctypes.byref(out))!=0 and out.value==91.


def test_native_curve_cuda_exact_shared_header(probe,record_property):
    import torch
    from gffx.torch._curve_math import log_unit,roughness_interpolate
    path=os.environ.get('GFFX_PRINCIPLED_CUDA_PROBE')
    if not path:pytest.skip('explicit standalone CUDA library required')
    assert torch.cuda.is_available()
    library=ctypes.CDLL(path)
    library.principled_test_cuda.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*6
    library.principled_test_cuda.restype=ctypes.c_int
    rows=list(cases());n=len(rows)
    b=torch.tensor([r[0] for r in rows],dtype=torch.float64,device='cuda')
    u=torch.tensor([r[1] for r in rows],dtype=torch.float64,device='cuda',requires_grad=True)
    g=torch.tensor([r[2] for r in rows],dtype=torch.float64,device='cuda')
    xs=[math.ldexp(1.,-1074+(i%1075)) for i in range(n)]
    x=torch.tensor(xs,dtype=torch.float64,device='cuda')
    out=torch.full((n,3),91.,dtype=torch.float64,device='cuda')
    stream=torch.cuda.current_stream()
    assert library.principled_test_cuda(n,b.data_ptr(),u.data_ptr(),g.data_ptr(),x.data_ptr(),out.data_ptr(),stream.cuda_stream)==0
    stream.synchronize()
    value=roughness_interpolate(b.T,u);gradient,=torch.autograd.grad(value,u,g)
    reference=torch.stack((value,gradient,log_unit(x)),dim=1)
    assert torch.equal(out.view(torch.uint8),reference.contiguous().view(torch.uint8))
    for row,xv,result in zip(rows,xs,out.cpu().tolist()):
        y,dy=evaluate(probe,*row);log=ctypes.c_double()
        assert probe.principled_test_log(xv,ctypes.byref(log))==0
        assert [bits(v) for v in result]==[bits(v) for v in (y,dy,log.value)]
    record_property('cuda_scalar_cases',str(n))
    record_property('cuda_library',json.dumps(dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())))
