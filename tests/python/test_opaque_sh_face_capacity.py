"""Analytical SH-only candidate capacity, preserved legacy admission and replay."""
import torch,pytest,os

def fixture(n):
    # Dyadic, strictly separated triangles: independent exact area=2**-13 each.
    v=[]
    for i in range(n):
        x=(i%32)/32+1/256;y=(i//32)/32+1/256
        v.extend([[x,y,0],[x+1/64,y,0],[x,y+1/64,0]])
    v=torch.tensor(v,dtype=torch.float64,device=os.environ.get("GFFX_CAPACITY_DEVICE","cpu"))
    return v,torch.arange(3*n,dtype=torch.int32,device=v.device).reshape(n,3),v.new_empty((n,3,0))

def batches(v,f,c):
    from gffx.torch._opaque_quadrature import opaque_material_quadrature_batch as run
    cursor=(0,0,0,0)
    for i in range(16):
        ids,q,r=run(v,f,c,rule=1,cursor=cursor)
        assert r[2]<=800 and r[3]<=32 and r[4]<=12800 and r[5]<=512
        yield ids,q,r
        if r[1]:return
        cursor=(r[0],i+1,r[4],r[5])
    pytest.fail('pixel did not finish')

@pytest.mark.parametrize('n',[33,65,512])
def test_all_separated_owners_and_exact_area(n):
    v,f,c=fixture(n);owners=[];area=0.;last=None
    for ids,q,r in batches(v,f,c):
        owners.extend(ids[0,:r[2]].tolist());area+=q[0,:r[2],3].sum().item();last=r
    assert set(owners)==set(range(n)) and len(owners)==n*25
    assert abs(area-n/8192)<1e-14 and last[5]==n

def test_33_owner_geometry_replay():
    v,f,c=fixture(33);v.requires_grad_();area=v.new_zeros(())
    for ids,q,r in batches(v,f,c):area=area+q[0,:r[2],3].sum()
    g,=torch.autograd.grad(area,v)
    expected=v.new_tensor([[-1/128,-1/128,0],[1/128,0,0],[0,1/128,0]]).repeat(33,1)
    torch.testing.assert_close(g,expected,atol=1e-13,rtol=0)

@pytest.mark.parametrize('n,rule',[(33,0),(513,1)])
def test_legacy_and_new_capacity_refuse(n,rule):
    from gffx.torch._opaque_quadrature import opaque_material_quadrature_batch as run
    v,f,c=fixture(n)
    with pytest.raises(OverflowError):run(v,f,c,rule=rule)


def test_reversed_shared_edge_is_one_geometric_boundary():
    xy=[[0.14669857241022655,0.5074879747687575],[0.1824733988898437,0.664910051990514],[0.07692107003536819,0.5149961959496032],[0.08123546760221956,0.6759134593123548]]
    dev=os.environ.get('GFFX_CAPACITY_DEVICE','cpu')
    v=torch.tensor([[x,y,0.] for x,y in xy],dtype=torch.float64,device=dev,requires_grad=True)
    f=torch.tensor([[0,3,1],[0,2,3]]+[[0,0,0]]*31,dtype=torch.int32,device=dev);c=v.new_empty((33,3,0))
    area=v.new_zeros(())
    for ids,q,r in batches(v,f,c):area=area+q[0,:r[2],3].sum()
    triangles=v[f[:2].long()];ab=triangles[:,1]-triangles[:,0];ac=triangles[:,2]-triangles[:,0]
    expected=((ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]).abs()/2).sum()
    g,=torch.autograd.grad(area,v);ge,=torch.autograd.grad(expected,v)
    torch.testing.assert_close(area,expected,atol=1e-14,rtol=0)
    torch.testing.assert_close(g,ge,atol=1e-12,rtol=0)


def test_saved_crowded_shared_edge_continuation_does_not_invent_parallel_hit():
    import numpy as np
    from gffx.torch._opaque_quadrature import opaque_material_quadrature_batch as run
    a=np.load('D:/devspace/gffit/qualification-20261003/boundary-sh6/crowded512-diagnostic1-native01.npz')
    dev=os.environ.get('GFFX_CAPACITY_DEVICE','cpu')
    v=torch.tensor(a['vertices'],device=dev);faces=torch.tensor(a['faces'],device=dev);c=torch.tensor(a['cuts'],device=dev)
    ids,q,r=run(v,faces,c,rule=1,origin=tuple(a['origin'].tolist()),cursor=tuple(a['cursor'].tolist()))
    assert r[2]>0 and torch.isfinite(q).all() and q[0,:r[2],3].min()>0


@pytest.mark.parametrize('power',[40,48])
def test_distinct_parallel_cuts_keep_positive_strip_and_shape_gradient(power):
    dev=os.environ.get('GFFX_CAPACITY_DEVICE','cpu');eps=2.**-power
    v=torch.tensor([[.125,.125,0.],[.625,.125,0.],[.125,.625,0.]],dtype=torch.float64,device=dev,requires_grad=True)
    f=torch.tensor([[0,1,2]]+[[0,0,0]]*32,dtype=torch.int32,device=dev)
    c=v.new_zeros((33,3,2));c[0]=v.new_tensor([[-eps,-2*eps],[1-eps,1-2*eps],[-eps,-2*eps]])
    total=v.new_zeros(())
    for ids,q,r in batches(v,f,c):
        n=r[2];blocks=q[0,:n].reshape(-1,25,4);s=blocks[:,:,1].mean(1)
        # Classify whole integration regions; rounded endpoint-near nodes can
        # equal a knot even though their region has strictly positive width.
        total=total+blocks[(s>eps)&(s<2*eps),:,3].sum()
    expected=.25*(eps-1.5*eps*eps);g,=torch.autograd.grad(total,v)
    assert total>0
    torch.testing.assert_close(total,total.new_tensor(expected),atol=0,rtol=1e-12)
    torch.testing.assert_close(g,g.new_tensor([[-2*expected,-2*expected,0],[2*expected,0,0],[0,2*expected,0]]),atol=1e-28,rtol=1e-12)


def test_expanded_cpu_cuda_bytes_and_first_vjp():
    if os.environ.get('GFFX_CAPACITY_DEVICE')!='cuda':pytest.skip('device comparison belongs to CUDA worker')
    saved=[]
    try:
        for dev in ('cpu','cuda'):
            os.environ['GFFX_CAPACITY_DEVICE']=dev;v,f,c=fixture(33);v.requires_grad_();rows=[]
            for ids,q,r in batches(v,f,c):
                upstream=torch.arange(q.numel(),dtype=q.dtype,device=dev).reshape(q.shape)/4096
                g,=torch.autograd.grad(q,v,upstream)
                rows.append((ids.cpu().numpy().tobytes(),q.detach().cpu().numpy().tobytes(),g.cpu().numpy().tobytes(),tuple(r)))
            saved.append(rows)
    finally:os.environ['GFFX_CAPACITY_DEVICE']='cuda'
    assert saved[0]==saved[1]
