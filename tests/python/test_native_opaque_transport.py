"""Installed material-integral transport; raw node differentiation stays separate."""
import torch,pytest,os,math


@pytest.mark.parametrize('reverse',[False,True])
def test_material_transport_fixed_pixel_has_closed_polynomial_shape_derivatives(record_property,reverse):
    from gffx.torch.render import opaque_material_quadrature
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');t=torch.tensor(0.,dtype=torch.float64,device=dev,requires_grad=True)
    v=torch.tensor([[-2,-2,0],[6,-2,0],[-2,6,0]],dtype=torch.float64,device=dev)+t*torch.tensor([1.,0.,0.],dtype=torch.float64,device=dev)
    if reverse:v=v.flip(0)
    f=torch.tensor([[0,1,2]],dtype=torch.int32,device=dev);ids,q=opaque_material_quadrature(v,f,v.new_empty((1,3,0)),1,1)
    assert q.shape==(1,800,4) and int((ids>=0).sum())==88
    for degree in range(5):
        value=(q[...,1]**degree*q[...,3]).sum();g=torch.autograd.grad(value,t,retain_graph=True)[0]
        exact=(3**(degree+1)-2**(degree+1))/((degree+1)*8**degree);slope=-(3**degree-2**degree)/8**degree if degree else 0
        assert abs(value.item()-exact)<3e-13 and abs(g.item()-slope)<3e-13
    record_property('transport','fixed pixel with translated affine material; exact polynomial boundary flux')


def test_material_transport_visible_silhouette_has_closed_area_derivative():
    from gffx.torch.render import opaque_material_quadrature
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');t=torch.tensor(0.,dtype=torch.float64,device=dev,requires_grad=True)
    v=torch.tensor([[-.25,-.5,0],[1.75,-.5,0],[-.25,1.5,0]],dtype=torch.float64,device=dev)+t*torch.tensor([1.,0.,0.],dtype=torch.float64,device=dev)
    f=torch.tensor([[0,1,2]],dtype=torch.int32,device=dev);ids,q=opaque_material_quadrature(v,f,v.new_empty((1,3,0)),1,1);area=q[...,3].sum();g=torch.autograd.grad(area,t)[0]
    assert abs(area.item()-.71875)<3e-13 and abs(g.item()-.75)<3e-13


def test_material_transport_preserves_first_only_and_capacity_refusal():
    from gffx.torch.render import opaque_material_quadrature
    from tests.python.test_native_opaque_flat import fixture
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');v,f,c=fixture(1,dev=dev);ids,q=opaque_material_quadrature(v,f,v.new_empty((1,3,0)),1,1)
    with pytest.raises(RuntimeError,match='first'):torch.autograd.grad(q[...,3].sum(),v,create_graph=True)
    v,f,c=fixture(33,dev=dev)
    with pytest.raises(OverflowError):opaque_material_quadrature(v,f,v.new_empty((33,3,0)),1,1)


@pytest.mark.parametrize('count,samples',[(1,44),(8,352),(18,792),(19,760),(20,800),(21,735),(32,800)])
def test_material_transport_capacity_retains_area_and_fallback(count,samples):
    from gffx.torch.render import opaque_material_quadrature
    from tests.python.test_native_opaque_flat import fixture
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');v,f,c=fixture(count,dev=dev)
    ids,q=opaque_material_quadrature(v,f,v.new_empty((count,3,0)),1,1)
    assert int((ids>=0).sum())==samples
    area=q[...,3].sum();g,=torch.autograd.grad(area,v)
    expected=v.new_tensor([[-1/64,-1/64,0],[1/64,0,0],[0,1/64,0]]).repeat(count,1)
    assert abs(area.item()-count/2048)<1e-13
    torch.testing.assert_close(g,expected,atol=1e-13,rtol=0)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_material_transport_cpu_cuda_sample_and_first_vjp_bytes(dtype):
    if os.environ.get('GFFX_KNOT_DEVICE','cpu')!='cuda':pytest.skip('paired device test charged to CUDA worker')
    from gffx.torch.render import opaque_material_quadrature
    rows=[]
    for dev in ('cpu','cuda'):
        if dev=='cuda':torch.cuda.reset_peak_memory_stats()
        v=torch.tensor([[-.25,-.5,0],[1.75,-.5,0],[-.25,1.5,0]],dtype=dtype,device=dev,requires_grad=True)
        f=torch.tensor([[0,1,2]],dtype=torch.int32,device=dev);cuts=v.new_empty((1,3,0)).requires_grad_()
        ids,q=opaque_material_quadrature(v,f,cuts,1,1)
        upstream=(torch.arange(q.numel(),dtype=dtype).reshape(q.shape)/4096).to(dev)
        gv,gc=torch.autograd.grad(q,(v,cuts),upstream)
        rows.append([x.detach().cpu().contiguous().numpy().tobytes() for x in (ids,q,gv,gc)])
        if dev=='cuda':assert torch.cuda.max_memory_allocated()<=67108864
    assert rows[0]==rows[1]



def test_material_transport_linear_cut_retains_one_sided_interior_flux():
    from gffx.torch.render import opaque_material_quadrature,opaque_quadrature
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');theta=torch.tensor(0.,dtype=torch.float64,device=dev,requires_grad=True)
    v=theta.new_tensor([[-2,-2,0],[6,-2,0],[-2,6,0]]);f=torch.tensor([[0,1,2]],dtype=torch.int32,device=dev)
    cuts=(v[:,0]-.5-theta).reshape(1,3,1)
    ids,q=opaque_material_quadrature(v,f,cuts,1,1)
    # The value at the jump is zero. Boundary samples cannot supply its two traces.
    x=(q[...,:3]*v[:,0]).sum(-1);value=(torch.where(x>.5+theta,torch.ones_like(x),torch.zeros_like(x))*q[...,3]).sum()
    g,=torch.autograd.grad(value,theta,retain_graph=True)
    assert abs(value.item()-.5)<3e-13 and abs(g.item()+1)<3e-13
    raw_ids,raw=opaque_quadrature(v,f,cuts,1,1)
    assert torch.equal(ids,raw_ids) and torch.equal(q.contiguous().view(torch.uint8),raw.contiguous().view(torch.uint8))
