"""Exact rational and fixed-order controls for camera inverse first VJP."""
import hashlib,json,os
from pathlib import Path
from fractions import Fraction as Q
import pytest
import torch

SOURCE=Path("D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque128-20260924/camera-cpu3-values.json")
SHA="646aaf5137e4fe7cf0d45bf41e1d77032a68f92e23f0279222d915acdc775c74"

def ordered(a,b):
    terms=[a[..., :,k,None]*b[...,None,k,:] for k in range(4)]
    return (terms[0]+terms[1])+(terms[2]+terms[3])

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('strided',[False,True])
def test_saved_inverse_vjp_has_fixed_order_and_exact_rational_bound(dtype,strided,record_property):
    from gffx.torch.camera import _Backend
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    row=json.loads(SOURCE.read_bytes());device=os.environ.get('GFFX_KNOT_DEVICE','cpu')
    pose=torch.tensor(row['inputs']['pose'],dtype=dtype,device=device).requires_grad_()
    saved=torch.tensor(row['values']['inverse'],dtype=dtype,device=device)
    errors=[]
    for channel,g in enumerate(saved):
        if strided:
            packed=torch.zeros((*g.shape[:-1],8),dtype=dtype,device=device);packed[...,::2]=g;g=packed[...,::2];assert not g.is_contiguous()
        inverse=_Backend.inverse(pose);actual=torch.autograd.grad(inverse,pose,g)[0]
        tr=inverse.detach().transpose(-1,-2);expected=-ordered(ordered(tr,g),tr)
        # Independent exact rational triple product, with the *binary* inputs.
        a=tr.cpu().tolist()[0];b=g.cpu().tolist()[0];got=actual.cpu().tolist()[0]
        for i in range(4):
            for j in range(4):
                exact=-sum((Q(a[i][k])*Q(b[k][l])*Q(a[l][j]) for k in range(4) for l in range(4)),Q(0))
                error=abs(Q(got[i][j])-exact);limit=Q(1,10**10)+Q(1,10**9)*abs(exact) if dtype==torch.float64 else Q(8)*Q(float(torch.finfo(dtype).eps))*max(Q(1),abs(exact))
                assert error<=limit;errors.append(float(error))
        assert actual.cpu().contiguous().numpy().tobytes()==expected.cpu().contiguous().numpy().tobytes(),(device,dtype,channel,'inverse backward accumulation differs')
    record_property('inverse_order',json.dumps(dict(device=device,dtype=str(dtype),strided=strided,max_exact_rational_error=max(errors),source_sha256=SHA)))

def test_inverse_double_backward_preserves_existing_eager_support():
    from gffx.torch.camera import _Backend
    device=os.environ.get('GFFX_KNOT_DEVICE','cpu')
    x=(torch.eye(4,dtype=torch.float64,device=device)*2+torch.arange(16,dtype=torch.float64,device=device).reshape(4,4)/100).unsqueeze(0).requires_grad_()
    assert torch.autograd.gradcheck(_Backend.inverse,(x,))
    assert torch.autograd.gradgradcheck(_Backend.inverse,(x,))
