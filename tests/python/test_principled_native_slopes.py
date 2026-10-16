"""Standalone angular limiter controls, not a material derivative acceptance."""
import ctypes
from fractions import Fraction as F
import hashlib,json,os
from pathlib import Path
import pytest
import torch


@pytest.fixture
def probe(record_property):
    path=os.environ.get('GFFX_PRINCIPLED_SCALAR_PROBE')
    if not path:pytest.skip('standalone probe required')
    lib=ctypes.CDLL(path);ptr=ctypes.POINTER(ctypes.c_double)
    lib.principled_test_slopes.argtypes=[ctypes.c_int]+[ptr]*5
    lib.principled_test_slopes.restype=ctypes.c_int
    record_property('native_library',json.dumps(dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())))
    return lib


def rational_slopes(axis,values,*,with_branches=False):
    x=list(map(F,axis));y=[list(map(F,v)) for v in values];n=len(x)
    if n==1:return ([[F(0),F(0)]],[("singleton",)]) if with_branches else [[F(0),F(0)]]
    h=[x[i+1]-x[i] for i in range(n-1)]
    sec=[[(y[i+1][c]-y[i][c])/h[i] for c in range(2)] for i in range(n-1)]
    out=[];branches=[]
    for i in range(n):
        m=[];signature=[]
        for c in range(2):
            before=sec[max(i-1,0)][c];after=sec[min(i,n-2)][c]
            if n==2:raw=before
            elif i==0:raw=((2*h[0]+h[1])*before-h[0]*sec[1][c])/(h[0]+h[1])
            elif i==n-1:raw=((2*h[-1]+h[-2])*after-h[-1]*sec[-2][c])/(h[-1]+h[-2])
            else:raw=(h[i]*before+h[i-1]*after)/(h[i-1]+h[i])
            if n==2:m.append(raw);signature.append("two_node");continue
            sign=1 if before>0 and after>0 else -1 if before<0 and after<0 else 0
            bound=F(3,2)*min(abs(before),abs(after))
            signature.append((sign,abs(before)<abs(after),abs(before)==abs(after),sign*raw>0,sign*raw==0, max(sign*raw,0)<bound,max(sign*raw,0)==bound))
            m.append(sign*min(max(sign*raw,0),bound))
        gap=y[i][0]-y[i][1];delta=m[0]-m[1]
        risk=delta*(h[i-1] if i else 0) if delta>0 else -delta*(h[i] if i<n-1 else 0)
        scale=min(F(1),3*gap/risk) if risk>0 else F(1)
        signature.append((delta>0,risk>0,scale<1,scale==1))
        if scale<1:scale*=1-8*F(2)**-52
        out.append([v*scale for v in m]);branches.append(tuple(signature))
    return (out,branches) if with_branches else out


def fixtures():
    yield [0.],[[.5,.25]],[[.1,-.03]]
    yield [-2.,0.],[[.8,.3],[.4,.1]],[[.04,.01],[-.02,.03]]
    yield [-3.,-2.,-1.,0.],[[.8,.4],[.65,.3],[.5,.2],[.35,.1]],[[.04,.03],[.03,.02],[.02,.01],[.01,0.]]
    yield [-3.,-2.,-1.,0.],[[.8,.4]]*4,[[.1,.05]]*4
    yield [-4.,-2.5,-1.,-.2,0.],[[.9,.4],[.8,.39],[.5,.35],[.48,.34],[.45,.33]],[[.03,-.01],[.02,.01],[.05,.02],[-.01,.03],[.02,-.01]]
    yield [-4.,-2.,-1.,0.],[[.8,.7],[.5,.499],[.7,.4],[.2,.199]],[[.01,.005],[-.01,.004],[.008,-.009],[.007,.001]]


def native(lib,x,y,dy):
    n=len(x);a=(ctypes.c_double*n)(*x);v=(ctypes.c_double*(2*n))(*[z for row in y for z in row]);d=(ctypes.c_double*(2*n))(*[z for row in dy for z in row]);out=(ctypes.c_double*(2*n))();dot=(ctypes.c_double*(2*n))()
    assert lib.principled_test_slopes(n,a,v,d,out,dot)==0
    return torch.tensor(list(out),dtype=torch.float64).reshape(n,2),torch.tensor(list(dot),dtype=torch.float64).reshape(n,2)


def test_native_slopes_rational_and_nonzero_value_bytes(probe,record_property):
    from gffx.torch._loss_curve import bounded_slopes
    count=0
    for x,y,dy in fixtures():
        out,dot=native(probe,x,y,dy)
        expected=bounded_slopes(torch.tensor(x,dtype=torch.float64),torch.tensor(y,dtype=torch.float64),ordered_pairs=True)
        # Slope-zero signs are not stable across the tensor reference's scalar
        # and vector min/max paths. Preserve numerical zeros here; the separate
        # control-output test still requires complete byte identity.
        nonzero=(out!=0)|(expected!=0)
        assert torch.equal(out[nonzero].contiguous().view(torch.uint8),expected[nonzero].contiguous().view(torch.uint8)),(x,out,expected)
        assert torch.equal(out==0,expected==0)
        exact=rational_slopes(x,y)
        for a,b in zip(out.flatten().tolist(),[v for row in exact for v in row]):assert abs(F(a)-b)<=F(1,10**13)*(1+abs(b))
        assert torch.isfinite(dot).all();count+=1
    record_property('angular_value_fixtures',str(count))


def test_native_slopes_directional_derivative(probe,record_property):
    from gffx.torch._loss_curve import bounded_slopes
    count=0;checks=[]
    for x,y,dy in fixtures():
        axis=torch.tensor(x,dtype=torch.float64);values=torch.tensor(y,dtype=torch.float64);direction=torch.tensor(dy,dtype=torch.float64)
        _,actual=native(probe,x,y,dy)
        f=lambda value:bounded_slopes(axis,value,ordered_pairs=True)
        _,expected=torch.autograd.functional.jvp(f,values,direction)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)
        # Certify matching branch selections at both perturbations. The retained
        # five-node fixture crosses the raw-slope zero limiter at its first node;
        # a two-sided difference there is not the derivative of either branch.
        for h in (F(1,100000),F(1,200000)):
            plus,bp=rational_slopes(x,[[F(v)+h*F(d) for v,d in zip(row,dr)] for row,dr in zip(y,dy)],with_branches=True)
            minus,bm=rational_slopes(x,[[F(v)-h*F(d) for v,d in zip(row,dr)] for row,dr in zip(y,dy)],with_branches=True)
            fd=torch.tensor([[float((p-m)/(2*h)) for p,m in zip(pr,mr)] for pr,mr in zip(plus,minus)],dtype=torch.float64)
            stable=torch.tensor([a==b for a,b in zip(bp,bm)])
            assert bool(stable.any())
            torch.testing.assert_close(actual[stable],fd[stable],atol=1e-8,rtol=1e-6)
            checks.append(dict(nodes=len(x),step=str(h),stable_nodes=int(stable.sum()),switch_nodes=[i for i,v in enumerate(stable.tolist()) if not v]))
        count+=1
    record_property('angular_derivative_fixtures',str(count));record_property('finite_difference_branch_certificate',json.dumps(checks))


def test_native_slopes_zero_sign_diagnostic(probe,record_property):
    from gffx.torch._loss_curve import bounded_slopes
    import struct
    x,y,dy=list(fixtures())[-1];actual,_=native(probe,x,y,dy)
    expected=bounded_slopes(torch.tensor(x,dtype=torch.float64),torch.tensor(y,dtype=torch.float64),ordered_pairs=True)
    mismatches=[dict(index=i,native=a.hex(),reference=b.hex(),native_bits=struct.pack('<d',a).hex(),reference_bits=struct.pack('<d',b).hex()) for i,(a,b) in enumerate(zip(actual.flatten().tolist(),expected.flatten().tolist())) if struct.pack('<d',a)!=struct.pack('<d',b)]
    probes=[]
    for op in (torch.minimum,torch.maximum):
        probes.append(dict(operation=op.__name__,negative_first=float(op(torch.tensor(-0.,dtype=torch.float64),torch.tensor(0.,dtype=torch.float64))).hex(),positive_first=float(op(torch.tensor(0.,dtype=torch.float64),torch.tensor(-0.,dtype=torch.float64))).hex()))
    record_property('exact_mismatches',json.dumps(mismatches));record_property('torch_zero_ties',json.dumps(probes))
    assert mismatches
    assert all(float.fromhex(row['native'])==float.fromhex(row['reference'])==0 for row in mismatches)


def test_native_slopes_zero_reaches_no_control_change(probe,record_property):
    from gffx.torch._loss_curve import bounded_slopes,controls
    all_rows=[]
    for x,y,dy in fixtures():
        axis=torch.tensor(x,dtype=torch.float64);values=torch.tensor(y,dtype=torch.float64)
        actual,_=native(probe,x,y,dy);expected=bounded_slopes(axis,values,ordered_pairs=True)
        ac=controls(axis,values,actual);ex=controls(axis,values,expected)
        assert torch.equal(ac.contiguous().view(torch.uint8),ex.contiguous().view(torch.uint8))
        all_rows.append(dict(nodes=len(x),controls=int(ac.numel())))
    record_property('exact_control_rows',json.dumps(all_rows))


def test_native_slopes_cuda_shared_value_and_directional_bytes(probe,record_property):
    from gffx.torch._loss_curve import bounded_slopes,controls
    path=os.environ.get('GFFX_PRINCIPLED_CUDA_PROBE')
    if not path:pytest.skip('explicit CUDA standalone library required')
    lib=ctypes.CDLL(path);lib.principled_test_slopes_cuda.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*6
    lib.principled_test_slopes_cuda.restype=ctypes.c_int
    count=0
    for x,y,dy in fixtures():
        axis=torch.tensor(x,dtype=torch.float64,device='cuda');values=torch.tensor(y,dtype=torch.float64,device='cuda');direction=torch.tensor(dy,dtype=torch.float64,device='cuda')
        out=torch.full_like(values,91.);dot=torch.full_like(values,92.);stream=torch.cuda.current_stream()
        assert lib.principled_test_slopes_cuda(len(x),axis.data_ptr(),values.data_ptr(),direction.data_ptr(),out.data_ptr(),dot.data_ptr(),stream.cuda_stream)==0
        stream.synchronize();cpu,dcpu=native(probe,x,y,dy)
        assert torch.equal(out.cpu().contiguous().view(torch.uint8),cpu.contiguous().view(torch.uint8))
        assert torch.equal(dot.cpu().contiguous().view(torch.uint8),dcpu.contiguous().view(torch.uint8))
        expected=bounded_slopes(axis,values,ordered_pairs=True)
        ac=controls(axis,values,out);ex=controls(axis,values,expected)
        assert torch.equal(ac.contiguous().view(torch.uint8),ex.contiguous().view(torch.uint8))
        _,reference=torch.autograd.functional.jvp(lambda v:bounded_slopes(axis,v,ordered_pairs=True),values,direction)
        torch.testing.assert_close(dot,reference,atol=1e-12,rtol=1e-10)
        count+=1
    record_property('angular_cuda_fixtures',str(count));record_property('cuda_library',json.dumps(dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())))
