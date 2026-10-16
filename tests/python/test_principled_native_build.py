"""Complete call-local curve construction: standalone native, not material acceptance."""
import ctypes,hashlib,json,os
from pathlib import Path
import pytest,torch


@pytest.fixture
def probe(record_property):
    path=os.environ.get('GFFX_PRINCIPLED_SCALAR_PROBE')
    if not path:pytest.skip('explicit native library required')
    lib=ctypes.CDLL(path);ptr=ctypes.POINTER(ctypes.c_double)
    lib.principled_test_build.argtypes=[ctypes.c_int,ptr,ptr,ctypes.c_double,ptr,ptr,ptr]
    lib.principled_test_build.restype=ctypes.c_int
    record_property('native_library',json.dumps(dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())))
    return lib


def inputs(budget,r):
    from gffx.torch._curve_math import log_unit
    d=budget._data;rn=d['roughness_nodes'];i=int((torch.searchsorted(rn,r,right=True)-1).clamp(max=len(rn)-2));width=d['_rlog'][i+1]-d['_rlog'][i];u=(log_unit(r)-d['_rlog'][i])/width
    pieces=[]
    for y,m in ((d['_logs'],d['_slopes']),(d['_hlogs'],d['_hslopes'])):
        pieces.append(torch.stack((y[i],y[i]+width*m[i]/width.new_tensor(3.),y[i+1]-width*m[i+1]/width.new_tensor(3.),y[i+1]),dim=-1).reshape(-1,4))
    return d['_clog'],torch.cat(pieces),u


def reference(axis,b,u):
    from gffx.torch._curve_math import roughness_interpolate
    from gffx.torch._loss_curve import bounded_slopes,controls
    n=len(axis);vals=roughness_interpolate(b.T,u.expand(len(b)))
    angular=vals[:2*n].reshape(n,2);horizon=torch.stack((vals[-1],vals[-1]*0))
    m=bounded_slopes(axis,angular,ordered_pairs=True);c=controls(axis,angular,m)
    return torch.cat((angular.flatten(),horizon,c.flatten()))


def native(lib,axis,b,u):
    n=len(axis);count=10*n-6;ptr=ctypes.POINTER(ctypes.c_double)
    x=(ctypes.c_double*n)(*axis.tolist());c=(ctypes.c_double*b.numel())(*b.flatten().tolist());out=(ctypes.c_double*count)();dot=(ctypes.c_double*count)();scratch=(ctypes.c_double*(4*n))()
    assert lib.principled_test_build(n,x,c,float(u),out,dot,scratch)==0
    return torch.tensor(list(out),dtype=torch.float64),torch.tensor(list(dot),dtype=torch.float64)


def test_native_full_curve_bundled_values_and_direction(probe,record_property):
    from gffx.torch.material import PrincipledBudget
    budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    rn=budget._data['roughness_nodes'];rs=[float(rn[0]),.213,.427,.827,1.]
    for r in rs:
        rv=torch.tensor(r,dtype=torch.float64);axis,b,u=inputs(budget,rv);out,dot=native(probe,axis,b,u)
        uq=u.detach().clone().requires_grad_();expected=reference(axis,b,uq)
        # Only discarded nodal slopes have tensor-vector zero-sign differences;
        # all actual curve outputs and their dimensions are exact.
        assert torch.equal(out.view(torch.uint8),expected.contiguous().view(torch.uint8))
        for mode in range(3):
            ids=torch.arange(len(out))
            weights=torch.ones_like(out) if mode==0 else (torch.where(ids.remainder(2)==0,.125,-.25).to(out) if mode==1 else (ids.remainder(7)-3).to(out)/8)
            du,=torch.autograd.grad(expected,uq,weights,retain_graph=True)
            torch.testing.assert_close((dot*weights).sum(),du,atol=1e-11,rtol=1e-9)
        if 1e-4<float(u)<1-1e-4:
            for h in (1e-5,5e-6):
                fd=(reference(axis,b,u+h)-reference(axis,b,u-h))/(2*h)
                torch.testing.assert_close(dot,fd,atol=1e-8,rtol=1e-5)
        _,values,horizon,controls=budget._curve(rv.reshape(1))
        legacy=torch.cat((values.flatten(),horizon.flatten(),controls.flatten()))
        assert torch.equal(out.view(torch.uint8),legacy.contiguous().view(torch.uint8))
    record_property('full_curve_rows',str(len(rs)));record_property('angular_nodes',str(len(axis)));record_property('payload_sha256',budget.to_envelope()['payload_sha256'])


def test_native_two_node_curve_closed_form(probe):
    # One positive angular node and a horizon: exp(linear interpolation in log
    # loss). Constant log controls make every u derivative exactly zero.
    axis=torch.tensor([0.],dtype=torch.float64);b=torch.tensor([[-.25]*4,[-.75]*4,[-.5]*4],dtype=torch.float64)
    out,dot=native(probe,axis,b,torch.tensor(.37,dtype=torch.float64))
    from decimal import Decimal,localcontext
    with localcontext() as c:
        c.prec=90
        for actual,x in zip(out.tolist(),[-.25,-.75,-.5,None]):
            if x is None:assert actual==0.
            else:assert abs(Decimal.from_float(actual)-Decimal.from_float(x).exp())<Decimal('1e-15')
    assert torch.equal(dot,torch.zeros_like(dot))


def test_native_full_curve_cuda_value_and_direction_bytes(probe,record_property):
    from gffx.torch.material import PrincipledBudget
    path=os.environ.get('GFFX_PRINCIPLED_CUDA_PROBE')
    if not path:pytest.skip('explicit standalone CUDA probe required')
    lib=ctypes.CDLL(path);lib.principled_test_build_cuda.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_double]+[ctypes.c_void_p]*4
    lib.principled_test_build_cuda.restype=ctypes.c_int
    budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    for r in (float(budget._data['roughness_nodes'][0]),.213,.427,.827,1.):
        axis,b,u=inputs(budget,torch.tensor(r,dtype=torch.float64));cpu,dcpu=native(probe,axis,b,u);n=len(axis)
        a=axis.cuda();c=b.cuda();out=torch.empty_like(cpu,device='cuda');dot=torch.empty_like(dcpu,device='cuda');scratch=torch.empty(4*n,dtype=torch.float64,device='cuda');stream=torch.cuda.current_stream()
        assert lib.principled_test_build_cuda(n,a.data_ptr(),c.data_ptr(),float(u),out.data_ptr(),dot.data_ptr(),scratch.data_ptr(),stream.cuda_stream)==0
        stream.synchronize()
        assert torch.equal(out.cpu().view(torch.uint8),cpu.view(torch.uint8))
        assert torch.equal(dot.cpu().view(torch.uint8),dcpu.view(torch.uint8))
        uq=u.cuda().requires_grad_();expected=reference(a,c,uq)
        assert torch.equal(out.view(torch.uint8),expected.contiguous().view(torch.uint8))
        ids=torch.arange(len(out),device='cuda');weights=(ids.remainder(7)-3).to(out)/8
        du,=torch.autograd.grad(expected,uq,weights)
        torch.testing.assert_close((dot*weights).sum(),du,atol=1e-11,rtol=1e-9)
    record_property('cuda_full_curve_rows','5');record_property('cuda_library',json.dumps(dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())))
