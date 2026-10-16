"""Standalone native directional/mean endpoint pairs and first partials."""
import ctypes,os
import pytest,torch
from test_principled_native_build import probe,inputs,native,reference


def query(lib,axis,cn,weights,curve,dcurve,c):
    ptr=ctypes.POINTER(ctypes.c_double)
    lib.principled_test_query.argtypes=[ctypes.c_int]+[ptr]*5+[ctypes.c_double,ptr,ptr]
    lib.principled_test_query.restype=ctypes.c_int
    arrays=[(ctypes.c_double*v.numel())(*v.flatten().tolist()) for v in (cn,axis,weights,curve,dcurve)]
    out=(ctypes.c_double*10)();scratch=(ctypes.c_double*(8*len(axis)))()
    assert lib.principled_test_query(len(axis),*arrays,float(c),out,scratch)==0
    return torch.tensor(list(out),dtype=torch.float64)


def reference_query(budget,axis,b,u,c):
    n=len(axis);data=reference(axis,b,u)
    curve=(budget._data['cosine_nodes'],data[:2*n].reshape(1,n,2),data[2*n:2*n+2].reshape(1,2),data[2*n+2:].reshape(1,n-1,4,2))
    p,a=budget._curve_pairs(curve,c.reshape(1),direction=True,mean=True)
    return torch.cat((p.flatten(),a.flatten()))


def test_native_loss_query_bundled_values_and_first_partials(probe,record_property):
    from gffx.torch.material import PrincipledBudget
    budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64);count=0
    for roughness in (.213,.427,.827,1.):
        axis,b,u=inputs(budget,torch.tensor(roughness,dtype=torch.float64));curve,dcurve=native(probe,axis,b,u);cn=budget._data['cosine_nodes'][1:];weights=budget._data['_weights']
        for cosine in (0.,float(cn[0])/2,float(cn[0]),.013,.3,.99,1.):
            out=query(probe,axis,cn,weights,curve,dcurve,cosine)
            uq=u.detach().clone().requires_grad_();cq=torch.tensor(cosine,dtype=torch.float64,requires_grad=True);expected=reference_query(budget,axis,b,uq,cq)
            assert torch.equal(out[:4].view(torch.uint8),expected.detach().contiguous().view(torch.uint8))
            cot=torch.tensor([.125,-.75,.3,-.5],dtype=torch.float64)
            gu,gc=torch.autograd.grad(expected,(uq,cq),cot)
            torch.testing.assert_close((out[4:8]*cot).sum(),gu,atol=1e-11,rtol=1e-9)
            torch.testing.assert_close((out[8:]*cot[:2]).sum(),gc,atol=1e-10,rtol=1e-9)
            count+=1
    record_property('native_query_rows',str(count))


def test_native_loss_query_two_node_analytic(probe):
    from decimal import Decimal,localcontext
    axis=torch.tensor([0.],dtype=torch.float64);cn=torch.tensor([1.],dtype=torch.float64);w=torch.empty((0,4),dtype=torch.float64)
    b=torch.tensor([[-.25]*4,[-.75]*4,[-.5]*4],dtype=torch.float64);curve,dcurve=native(probe,axis,b,torch.tensor(.37,dtype=torch.float64))
    for c in (0.,.17,.8,1.):
        out=query(probe,axis,cn,w,curve,dcurve,c)
        with localcontext() as ctx:
            ctx.prec=90;v=[Decimal('-.25').exp(),Decimal('-.75').exp()];h=[Decimal('-.5').exp(),Decimal(0)];t=Decimal.from_float(c)
            expected=[(1-t)*h[i]+t*v[i] for i in range(2)]+[(h[i]+2*v[i])/3 for i in range(2)]
            for a,e in zip(out[:4].tolist(),expected):assert abs(Decimal.from_float(a)-e)<Decimal('1e-15')
            for a,e in zip(out[8:].tolist(),[v[i]-h[i] for i in range(2)]):assert abs(Decimal.from_float(a)-e)<Decimal('1e-15')
        assert torch.equal(out[4:8],torch.zeros(4,dtype=torch.float64))


def test_native_query_cosine_independent_diagnostic(probe,record_property):
    from decimal import Decimal,localcontext
    import json
    from gffx.torch.material import PrincipledBudget
    from gffx.torch._curve_math import log_unit
    budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    rows=[]
    for roughness in (.213,.427,.827,1.):
        axis,b,u=inputs(budget,torch.tensor(roughness,dtype=torch.float64));curve,dcurve=native(probe,axis,b,u);cn=budget._data['cosine_nodes'][1:];weights=budget._data['_weights'];n=len(axis)
        for cosine in (0.,float(cn[0])/2,float(cn[0]),.013,.3,.99,1.):
            out=query(probe,axis,cn,weights,curve,dcurve,cosine)
            cq=torch.tensor(cosine,dtype=torch.float64,requires_grad=True);expected=reference_query(budget,axis,b,u,cq);cot=torch.tensor([.125,-.75,.3,-.5],dtype=torch.float64);gc,=torch.autograd.grad(expected,cq,cot)
            with localcontext() as ctx:
                ctx.prec=150;D=Decimal.from_float
                if cosine<float(cn[0]):partials=[(D(float(curve[j]))-D(float(curve[2*n+j])))/D(float(cn[0])) for j in range(2)]
                else:
                    index=min(int(torch.searchsorted(cn,cq.detach(),right=True))-1,n-2);width=float(axis[index+1]-axis[index]);t=float((log_unit(cq.detach())-axis[index])/width);v=D(t);q=1-v
                    controls=curve[2*n+2:].reshape(n-1,4,2)[index]
                    partials=[]
                    for channel in range(2):
                        z=[D(float(x)) for x in controls[:,channel]]
                        polynomial=3*(q*q*(z[1]-z[0])+2*q*v*(z[2]-z[1])+v*v*(z[3]-z[2]))
                        partials.append(polynomial/D(width)/D(cosine))
                exact=sum(D(float(cot[j]))*partials[j] for j in range(2));actual=float((out[8:]*cot[:2]).sum());err=abs(D(actual)-exact);old=abs(D(float(gc))-exact)
                assert err<=Decimal('1e-10')+Decimal('1e-9')*abs(exact)
                rows.append(dict(roughness=roughness,cosine=cosine,native=actual,installed=float(gc),independent=str(exact),native_absolute_error=str(err),installed_absolute_error=str(old)))
    mismatches=[r for r in rows if abs(r['native']-r['installed'])>1e-10+1e-9*abs(r['installed'])]
    assert len(rows)==28
    # Reference mismatches are diagnostic data; angular20 repairs the captured
    # vram19 cancellation. The independent bound above remains mandatory.
    record_property('independent_query_cosine_rows',json.dumps(rows));record_property('reference_bridge_mismatches',json.dumps(mismatches))


def test_native_loss_query_cuda_exact_values_and_partials(probe,record_property):
    import hashlib,json
    from pathlib import Path
    from gffx.torch.material import PrincipledBudget
    path=os.environ.get('GFFX_PRINCIPLED_CUDA_PROBE')
    if not path:pytest.skip('explicit standalone CUDA library required')
    lib=ctypes.CDLL(path);lib.principled_test_query_cuda.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*5+[ctypes.c_double]+[ctypes.c_void_p]*3
    lib.principled_test_query_cuda.restype=ctypes.c_int
    budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64);count=0
    for r in (.213,.427,.827,1.):
        axis,b,u=inputs(budget,torch.tensor(r,dtype=torch.float64));curve,dcurve=native(probe,axis,b,u);cn=budget._data['cosine_nodes'][1:];weights=budget._data['_weights'];n=len(axis)
        args=[t.cuda() for t in (cn,axis,weights,curve,dcurve)];out=torch.empty(10,dtype=torch.float64,device='cuda');scratch=torch.empty(8*n,dtype=torch.float64,device='cuda');stream=torch.cuda.current_stream()
        for c in (0.,float(cn[0])/2,float(cn[0]),.013,.3,.99,1.):
            assert lib.principled_test_query_cuda(n,*[t.data_ptr() for t in args],c,out.data_ptr(),scratch.data_ptr(),stream.cuda_stream)==0
            stream.synchronize();expected=query(probe,axis,cn,weights,curve,dcurve,c)
            assert torch.equal(out.cpu().view(torch.uint8),expected.view(torch.uint8))
            count+=1
    record_property('cuda_endpoint_query_rows',str(count));record_property('cuda_library',json.dumps(dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())))
