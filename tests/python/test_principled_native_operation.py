"""Installed curve C ABI/provider adapter: exact values and bounded first VJP."""
import pytest
import torch
from test_principled_native_build import inputs,reference


def packed():
    from gffx.torch.material import PrincipledBudget
    budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    rows=[inputs(budget,torch.tensor(r,dtype=torch.float64)) for r in (float(budget._data['roughness_nodes'][0]),.213,.427,.827,1.)]
    return rows[0][0],torch.stack([r[1] for r in rows]),torch.stack([r[2] for r in rows])


def raw(x):
    return x.detach().cpu().contiguous().flatten().view(torch.uint8)


def test_installed_native_curve_values_and_vjp(record_property):
    import gffx.torch.image
    axis,b,u=packed()
    value,partial=torch.ops.gffx.shading_loss_curve(axis,b,u)
    for row in range(len(u)):
        q=u[row].clone().requires_grad_();expected=reference(axis,b[row],q)
        assert torch.equal(raw(value[row]),raw(expected))
        for mode in range(3):
            ids=torch.arange(len(expected))
            cot=torch.ones_like(value) if mode==0 else (torch.arange(value.numel()).reshape(value.shape).remainder(7)-3).to(value)/8 if mode==1 else -torch.ones_like(value)*.125
            actual=torch.ops.gffx.shading_loss_curve_backward(partial,cot)
            want,=torch.autograd.grad(expected,q,cot[row],retain_graph=True)
            torch.testing.assert_close(actual[row],want,atol=1e-11,rtol=1e-9)
    record_property('native_curve_rows',str(len(u)))


def test_installed_native_curve_empty_and_domains():
    import gffx.torch.image
    axis,b,u=packed()
    y,dy=torch.ops.gffx.shading_loss_curve(axis,b[:0],u[:0])
    assert y.shape==dy.shape==(0,10*len(axis)-6)
    assert torch.ops.gffx.shading_loss_curve_backward(dy,y).shape==(0,)
    for x,c,q in ((axis.flip(0),b,u),(axis,b,u+2),(axis,b-800,u),(axis.float(),b,u),(axis,b.repeat(7,1,1),u.repeat(7))):
        with pytest.raises(RuntimeError):torch.ops.gffx.shading_loss_curve(x,c,q)
    with pytest.raises(RuntimeError):torch.ops.gffx.shading_loss_curve_backward(torch.ones(2,4),torch.ones(2,4))


def test_installed_native_curve_cuda_stream_exact():
    import gffx.torch.image
    assert torch.cuda.is_available()
    axis,b,u=packed();cpu,dcpu=torch.ops.gffx.shading_loss_curve(axis,b,u)
    cot=(torch.arange(cpu.numel()).reshape(cpu.shape).remainder(11)-5).to(cpu)/8
    gcpu=torch.ops.gffx.shading_loss_curve_backward(dcpu,cot)
    stream=torch.cuda.Stream()
    for repeat in range(2):
        with torch.cuda.stream(stream):
            out,dout=torch.ops.gffx.shading_loss_curve(axis.cuda(),b.cuda(),u.cuda())
            grad=torch.ops.gffx.shading_loss_curve_backward(dout,cot.cuda())
        stream.synchronize()
        assert torch.equal(raw(out),raw(cpu))
        assert torch.equal(raw(dout),raw(dcpu))
        assert torch.equal(raw(grad),raw(gcpu))


def test_material_curve_dispatches_native_and_retains_reference(monkeypatch):
    from gffx.torch.material import PrincipledBudget
    budget=PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)
    original=torch.ops.gffx.shading_loss_curve;calls=[]
    def observed(*args):
        calls.append(tuple(args[-1].shape));return original(*args)
    monkeypatch.setattr(torch.ops.gffx,'shading_loss_curve',observed)
    r=torch.tensor([.213,.827],dtype=torch.float64,requires_grad=True)
    actual=budget._curve(r)
    assert calls==[(2,)]
    expected=budget._curve_reference(r)
    for x,y in zip(actual,expected):assert torch.equal(raw(x),raw(y))
    cot=[(torch.arange(x.numel()).reshape(x.shape).remainder(7)-3).to(x)/8 for x in actual[1:]]
    ga=torch.autograd.grad(sum((x*w).sum() for x,w in zip(actual[1:],cot)),r)[0]
    ge=torch.autograd.grad(sum((x*w).sum() for x,w in zip(expected[1:],cot)),r)[0]
    torch.testing.assert_close(ga,ge,atol=1e-10,rtol=1e-9)


@pytest.mark.parametrize('nodes',[1,2,512])
def test_native_curve_full_cap_node_parallel_bytes(nodes):
    import gffx.torch.image
    axis=torch.linspace(-20.,0.,nodes,dtype=torch.float64) if nodes>1 else torch.zeros(1,dtype=torch.float64)
    b=torch.linspace(-3.,-.01,32*(2*nodes+1)*4,dtype=torch.float64).reshape(32,2*nodes+1,4)
    u=torch.linspace(0.,1.,32,dtype=torch.float64)
    y,dy=torch.ops.gffx.shading_loss_curve(axis,b,u)
    cy,cd=torch.ops.gffx.shading_loss_curve(axis.cuda(),b.cuda(),u.cuda())
    assert torch.equal(raw(y),raw(cy));assert torch.equal(raw(dy),raw(cd))
    cot=(torch.arange(y.numel()).reshape(y.shape).remainder(7)-3).to(y)/8
    g=torch.ops.gffx.shading_loss_curve_backward(dy,cot)
    cg=torch.ops.gffx.shading_loss_curve_backward(cd,cot.cuda())
    assert torch.equal(raw(g),raw(cg))
