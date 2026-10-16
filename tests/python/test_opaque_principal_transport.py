import os
from dataclasses import replace
import pytest,torch


def fixture(batch=1,named=False,principal=True):
    from tests.python.test_opaque_scene_horizons import fixture as original
    from gffx.torch import scene
    s,c,n=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));mesh=s.prims[0]
    v=mesh.vertices.detach()*mesh.vertices.new_tensor([2,2,1])
    if batch>1:v=v[None].repeat(batch,1,1)
    s=scene.Scene([replace(mesh,vertices=v),*s.prims[1:]])
    c.cx=(c.cx.detach()+.03).requires_grad_(principal);c.cy=(c.cy.detach()+.01).requires_grad_(principal)
    if named:s=scene.Scene([*s.prims,scene.Camera('camera',c)]);return s,'camera',c,n
    return s,c,c,n


@pytest.mark.parametrize('named',[False,True])
def test_principal_transport_preserves_other_parameter_gradients(named):
    from gffx.torch import render
    answers=[]
    for enabled in (False,True):
        s,cam,c,n=fixture(named=named,principal=enabled)
        result=render.render(s,cam,coverage='opaque',cull_mode=render.CULL_NONE)
        g=torch.autograd.grad(result['colour'].sum(),n)[0]
        answers.append((result,g))
    for k in answers[0][0]:assert torch.equal(answers[0][0][k].detach().contiguous().view(torch.uint8),answers[1][0][k].detach().contiguous().view(torch.uint8))
    assert torch.equal(answers[0][1].reshape(-1).view(torch.uint8),answers[1][1].reshape(-1).view(torch.uint8))


def test_principal_transport_batch_loss_and_alpha_have_explicit_scaling():
    from gffx.torch import render
    grads=[]
    for batch in (1,2):
        s,cam,c,n=fixture(batch=batch)
        result=render.render(s,cam,coverage='opaque',cull_mode=render.CULL_NONE,background=c.fx.new_tensor([.2,.3,.4]))
        alpha=torch.autograd.grad(result['mask'].sum(),(c.cx,c.cy),retain_graph=True)
        assert all(g.abs().max()<2e-14 for g in alpha)
        grads.append(torch.cat(torch.autograd.grad(result['colour'].sum()/batch,(c.cx,c.cy))))
    assert torch.equal(grads[0].contiguous().view(torch.uint8),grads[1].contiguous().view(torch.uint8))


def test_principal_transport_second_derivatives_refuse():
    from gffx.torch import render
    s,cam,c,n=fixture();result=render.render(s,cam,coverage='opaque',cull_mode=render.CULL_NONE)
    with pytest.raises(RuntimeError,match='first derivatives'):torch.autograd.grad(result['colour'].sum(),c.cx,create_graph=True)
