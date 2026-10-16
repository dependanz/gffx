"""Composite-rule capacity and polynomial moments, independent exact integrals."""
import os,torch,pytest
@pytest.mark.parametrize('count',[1,2,3,8,16,32,33])
def test_composite_plain_capacity_preserves_area_and_linear_moment(count):
    from gffx.torch.render import opaque_quadrature
    dev=os.environ.get('GFFX_CURVED_DEVICE','cpu');vertices=[]
    for i in range(count):
        x=(i%8)/8+1/64;y=(i//8)/8+1/64
        vertices.extend([[x,y,0],[x+1/16,y,0],[x,y+1/16,0]])
    v=torch.tensor(vertices,dtype=torch.float64,device=dev,requires_grad=True);f=torch.arange(3*count,dtype=torch.int32,device=dev).reshape(count,3)
    if count==33:
        with pytest.raises(OverflowError):opaque_quadrature(v,f,v.new_empty((count,3,0)),1,1)
        return
    ids,q=opaque_quadrature(v,f,v.new_empty((count,3,0)),1,1)
    panels=max(p for p in (1,2,3,4) if 25*count*p*p<=800)
    assert int((ids>=0).sum())==25*count*panels*panels
    assert q.shape==(1,800,4)
    total=q[...,3].sum();assert abs(total.item()-count/512)<1e-13
    # Every barycentric coordinate integrates to exactly area/3.
    for j in range(3):assert abs((q[...,j]*q[...,3]).sum().item()-count/1536)<1e-13
    grad,=torch.autograd.grad(total,v)
    expected=v.new_tensor([[-1/32,-1/32,0],[1/32,0,0],[0,1/32,0]]).repeat(count,1)
    torch.testing.assert_close(grad,expected,atol=1e-13,rtol=0)

# Retired candidate diagnostic, pinned so default qualification cannot use it.
import os
pytestmark=pytest.mark.skipif(os.environ.get("GFFX_OPAQUE_WHEEL_SHA")!="61b05ea6562fe6a2f625f266f47a8a4c4610611c24b88faf5ed74660ad674313",reason="requires retired opaque124 candidate1 installed wheel")
