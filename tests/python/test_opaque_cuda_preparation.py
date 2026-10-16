"""Complete CUDA domain validation and exact output initialization."""
import pytest,torch
from gffx.torch import render


def fixture():
    v=torch.full((5003,3),10.,device='cuda',dtype=torch.float64)
    v[:3]=v.new_tensor([[10.,10.,0.],[11.,10.,0.],[10.,11.,0.]])
    f=torch.tensor([[0,1,2]],device='cuda',dtype=torch.int32).repeat(1401,1)
    c=v.new_zeros((len(f),3,1));return v,f,c


def call(v,f,c,batch,g=None):
    if batch:
        op=torch.ops.gffx.opaque_material_batch if g is None else torch.ops.gffx.opaque_material_batch_backward
        return op(v,f,c,*(() if g is None else (g,)),0.,0.,0,0,0,0,67108864)
    op=torch.ops.gffx.opaque_material_quadrature if g is None else torch.ops.gffx.opaque_material_quadrature_backward
    return op(v,f,c,*(() if g is None else (g,)),1,1,0.,0.,67108864)


@pytest.mark.parametrize('batch',[False,True])
@pytest.mark.parametrize('bad',['vertex','range','face','cut','gradient'])
def test_native_domain_checks_complete_input_tails(batch,bad):
    v,f,c=fixture();g=None
    if bad=='vertex':v[-1,-1]=float('nan')
    elif bad=='range':v[-1,-1]=1048577.
    elif bad=='face':f[-1,-1]=len(v)
    elif bad=='cut':c[-1,-1,-1]=float('inf')
    else:g=v.new_zeros((1,800,4));g[-1,-1,-1]=float('nan')
    with pytest.raises(RuntimeError):call(v,f,c,batch,g)


@pytest.mark.parametrize('batch',[False,True])
def test_native_empty_outputs_and_large_adjoint_padding_are_exact(batch):
    v,f,c=fixture();out=call(v,f,c,batch)
    assert torch.equal(out[0],torch.full_like(out[0],-1))
    assert torch.equal(out[1].view(torch.uint8),torch.zeros_like(out[1]).view(torch.uint8))
    g=v.new_ones((1,800,4));grads=call(v,f,c,batch,g)
    for a in grads[:2]:assert torch.equal(a.view(torch.uint8),torch.zeros_like(a).view(torch.uint8))
