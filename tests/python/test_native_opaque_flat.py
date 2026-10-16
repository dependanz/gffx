"""Installed capacity-counted flat quadrature, closed moments and shared first VJPs."""
import os,torch,pytest

def fixture(count,curved=0,dev=None):
    dev=dev or os.environ.get('GFFX_CURVED_DEVICE','cpu');vertices=[]
    for i in range(count):
        x=(i%8)/8+1/64;y=(i//8)/8+1/64
        vertices.extend([[x,y,0],[x+1/32,y,0],[x,y+1/32,0]])
    v=torch.tensor(vertices,dtype=torch.float64,device=dev,requires_grad=True);f=torch.arange(count*3,dtype=torch.int32,device=dev).reshape(count,3)
    c=v.new_zeros((count,6));c[:curved]=v.new_tensor([-.25,.75,.65,.5,.5,1.5]);return v,f,c.requires_grad_()

def evaluate(v,f,c):
    from gffx.torch.render import opaque_quadrature
    return opaque_quadrature(v,f,v.new_empty((len(f),3,0)),1,1,curves=c)

@pytest.mark.parametrize('count,samples',[(1,40),(2,80),(8,320),(16,640),(20,800),(21,735),(22,770),(23,690),(26,780),(27,675),(32,800)])
def test_flat_capacity_keeps_closed_moments_and_area_motion(count,samples):
    v,f,c=fixture(count);ids,q=evaluate(v,f,c)
    assert int((ids>=0).sum())==samples and q.shape==(1,800,4)
    area=q[...,3].sum();assert abs(area.item()-count/2048)<1e-13
    # Exact triangle barycentric fourth moment: area/15.
    for k in range(3):assert abs((q[...,k]**4*q[...,3]).sum().item()-count/(2048*15))<1e-13
    gv,gc=torch.autograd.grad(area,(v,c));expected=v.new_tensor([[-1/64,-1/64,0],[1/64,0,0],[0,1/64,0]]).repeat(count,1)
    torch.testing.assert_close(gv,expected,atol=1e-13,rtol=0);assert torch.count_nonzero(gc)==0

@pytest.mark.parametrize('curved,flat,samples',[(1,12,768),(1,13,743),(2,4,736),(2,5,776),(3,7,640),(4,7,760),(5,7,775),(6,2,800)])
def test_flat_refinement_preserves_curved_priority_and_combined_capacity(curved,flat,samples):
    v,f,c=fixture(curved+flat,curved);ids,q=evaluate(v,f,c);assert int((ids>=0).sum())==samples<=800
    area=q[...,3].sum();assert abs(area.item()-(curved+flat)/2048)<1e-13
    g,=torch.autograd.grad(area,c);assert g.abs().max()<1e-13

@pytest.mark.parametrize('curved,flat',[(6,3),(7,0),(0,33)])
def test_flat_refinement_retains_independent_face_and_logical_block_refusals(curved,flat):
    v,f,c=fixture(curved+flat,curved)
    with pytest.raises(OverflowError):evaluate(v,f,c)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('count',[1,21,26,32])
def test_flat_shared_cpu_cuda_sample_and_first_vjp_bytes(count,dtype,record_property):
    rows=[]
    for dev in ('cpu','cuda'):
        if dev=='cuda':torch.cuda.reset_peak_memory_stats()
        v,f,c=fixture(count,dev=dev);v=v.detach().to(dtype).requires_grad_();c=c.detach().to(dtype).requires_grad_();ids,q=evaluate(v,f,c);g=torch.arange(q.numel(),device=dev,dtype=q.dtype).reshape(q.shape)/4096
        gv,gc=torch.autograd.grad(q,(v,c),g);rows.append([x.detach().cpu().numpy().tobytes() for x in (ids,q,gv,gc)])
        if dev=='cuda':assert torch.cuda.max_memory_allocated()<=67108864
    assert rows[0]==rows[1];record_property('exact_flat','owners,samples,vertex/cut first VJPs byte-identical')
