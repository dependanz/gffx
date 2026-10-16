"""Prospective native endpoint queries: ownership, physical values and first VJP."""
import pytest,torch
from test_principled_native_operation import packed,raw

def fixture():
    from gffx.torch.material import PrincipledBudget
    b=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    x,controls,u=packed();v,d=torch.ops.gffx.shading_loss_curve(x,controls,u)
    c=torch.tensor([0.,float(b._data['cosine_nodes'][1])/2,.013,.3,1.,.99,.15],dtype=torch.float64)
    owners=torch.tensor([0,0,1,2,4,2,3],dtype=torch.int32)
    return b,x,controls,u,v,d,c,owners

def test_native_query_values_vjp_and_duplicate_owners():
    budget,x,controls,u,v,d,c,owners=fixture();n=len(x)
    q,m=torch.ops.gffx.shading_loss_query(budget._data['cosine_nodes'][1:],x,budget._data['_weights'],v,d,c,owners)
    r=u.clone().requires_grad_();cos=c.clone().requires_grad_()
    from test_principled_native_build import reference
    curves=torch.stack([reference(x,controls[i],r[i]) for i in range(len(r))])
    curve=(budget._data['cosine_nodes'],curves[:,:2*n].reshape(-1,n,2),curves[:,2*n:2*n+2],curves[:,2*n+2:].reshape(-1,n-1,4,2))
    expected,mean=budget._curve_pairs(curve,cos,direction=True,mean=True,indices=owners)
    assert torch.equal(raw(q[:,:2]),raw(expected));assert torch.equal(raw(m[:,:2]),raw(mean))
    g=(torch.arange(len(c)*2).reshape(-1,2).remainder(7)-3).to(c)/8
    h=(torch.arange(len(u)*2).reshape(-1,2).remainder(5)-2).to(c)/8
    gu,gc=torch.ops.gffx.shading_loss_query_backward(q,m,owners,g,h)
    er,ec=torch.autograd.grad((expected*g).sum()+(mean*h).sum(),(r,cos))
    torch.testing.assert_close(gu,er,atol=1e-10,rtol=1e-9);torch.testing.assert_close(gc,ec,atol=1e-10,rtol=1e-9)

def test_native_query_cuda_exact_and_caller_stream():
    budget,x,controls,u,v,d,c,owners=fixture()
    inputs=(budget._data['cosine_nodes'][1:],x,budget._data['_weights'],v,d,c,owners)
    q,m=torch.ops.gffx.shading_loss_query(*inputs)
    g=torch.arange(len(c)*2,dtype=torch.float64).reshape(-1,2)/16;h=torch.ones(len(u),2,dtype=torch.float64)/8
    gu,gc=torch.ops.gffx.shading_loss_query_backward(q,m,owners,g,h)
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        cq,cm=torch.ops.gffx.shading_loss_query(*(t.cuda() for t in inputs))
        cu,cc=torch.ops.gffx.shading_loss_query_backward(cq,cm,owners.cuda(),g.cuda(),h.cuda())
    stream.synchronize()
    for a,b in ((q,cq),(m,cm),(gu,cu),(gc,cc)):assert torch.equal(raw(a),raw(b))

def test_native_query_invalid_domains_and_empty():
    budget,x,controls,u,v,d,c,owners=fixture()
    inputs=[budget._data['cosine_nodes'][1:],x,budget._data['_weights'],v,d,c,owners]
    for slot,bad in ((0,inputs[0].flip(0)),(5,c+2),(6,owners+99),(6,owners.long()),(3,v.float())):
        candidate=inputs.copy();candidate[slot]=bad
        with pytest.raises(RuntimeError):torch.ops.gffx.shading_loss_query(*candidate)
    q,m=torch.ops.gffx.shading_loss_query(*inputs[:3],v[:0],d[:0],c[:0],owners[:0])
    assert q.shape==(0,6) and m.shape==(0,4)
    gu,gc=torch.ops.gffx.shading_loss_query_backward(q,m,owners[:0],torch.empty(0,2,dtype=torch.float64),torch.empty(0,2,dtype=torch.float64))
    assert gu.numel()==gc.numel()==0

@pytest.mark.parametrize('nodes',[1,2,512])
def test_native_query_full_cap_fixed_owner_reduction(nodes):
 import gffx.torch.image
 from gffx.torch._curve_math import log_unit
 cn=torch.linspace(.01,1.,nodes,dtype=torch.float64) if nodes>1 else torch.ones(1,dtype=torch.float64)
 logs=log_unit(cn);b=32;q=1056
 controls=torch.full((b,2*nodes+1,4),-.5,dtype=torch.float64);u=torch.linspace(0.,1.,b,dtype=torch.float64)
 v,d=torch.ops.gffx.shading_loss_curve(logs,controls,u)
 from gffx.torch._loss_curve import bernstein_weights
 w=bernstein_weights(logs[1:]-logs[:-1]);c=torch.linspace(0.,1.,q,dtype=torch.float64);owners=torch.arange(q,dtype=torch.int32).remainder(b)
 inputs=(cn,logs,w,v,d,c,owners);y,m=torch.ops.gffx.shading_loss_query(*inputs)
 cy,cm=torch.ops.gffx.shading_loss_query(*(t.cuda() for t in inputs))
 g=torch.linspace(-1.,1.,q*2,dtype=torch.float64).reshape(q,2);h=torch.ones(b,2,dtype=torch.float64)
 a,z=torch.ops.gffx.shading_loss_query_backward(y,m,owners,g,h)
 ca,cz=torch.ops.gffx.shading_loss_query_backward(cy,cm,owners.cuda(),g.cuda(),h.cuda())
 for x,t in ((y,cy),(m,cm),(a,ca),(z,cz)):assert torch.equal(raw(x),raw(t))
 assert torch.equal(a,torch.zeros_like(a))

def test_public_queries_and_material_use_one_native_query(monkeypatch):
 from gffx.torch.material import PrincipledBudget
 budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
 original=torch.ops.gffx.shading_loss_query;calls=[]
 def observed(*args):calls.append((args[3].shape[0],len(args[5])));return original(*args)
 monkeypatch.setattr(torch.ops.gffx,'shading_loss_query',observed)
 r=torch.tensor([.213,.827],dtype=torch.float64,requires_grad=True);f=torch.tensor([[.1,.4,.8]]*2,dtype=torch.float64,requires_grad=True)
 c=torch.tensor([0.,.77],dtype=torch.float64,requires_grad=True);l=torch.tensor([.01,.9,.3],dtype=torch.float64,requires_grad=True);ids=torch.tensor([0,1,0])
 values=budget._material_losses(c,l,r,f,ids)
 assert calls==[(2,5)]
 gradients=torch.autograd.grad(sum(x.sum() for x in values),(r,f,c,l))
 assert all(torch.isfinite(x).all() for x in gradients)
 calls.clear();budget.directional_loss(c,r,f[:,0]);assert calls==[(2,2)]
