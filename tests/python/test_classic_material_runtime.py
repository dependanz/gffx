"""Resource-independent classic composition and tagged lighting acceptance."""
import importlib
import math

import pytest
import torch


def t(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try: yield
    finally: torch.set_num_threads(old)


@pytest.mark.parametrize('shape', [(), (3,), (3, 1), (1, 3, 1), (2, 3, 1), (2, 3, 3)])
def test_classic_grey_rgb_sharing_keeps_independent_scene_gradients(shape):
    m = importlib.import_module('gffx.torch.material')
    n = t([0., 0., 1.]).expand(2, 3, 3)
    v, l, a = n[0], t([[.3, .2, 1.], [.1, -.3, 1.]]), t([.2, .4, .6], True)
    s = torch.full(shape, .2, dtype=torch.float64, requires_grad=True)
    e = t([2., 3., 4.], True)
    batched = m.classic(n, v, l, albedo=a, specular=s, exponent=e)
    separate = torch.stack([m.classic(n[i], v, l, albedo=a,
        specular=s[i] if len(shape)==3 and shape[0]==2 else s, exponent=e) for i in range(2)])
    torch.testing.assert_close(batched, separate, atol=1e-12, rtol=1e-12)
    for x, y in zip(torch.autograd.grad(batched.sum(), (a,s,e)), torch.autograd.grad(separate.sum(), (a,s,e))):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-12)


def test_inactive_classic_extreme_finite_vectors_have_zero_gradients():
    m = importlib.import_module('gffx.torch.material')
    n, v, l = t([[0.,0.,1e-310]],True), t([[0.,0.,1e-310]],True), t([[0.,0.,-1e-310]],True)
    a,s,e = t([.2,.4,.6],True),t(.2,True),t(1e308,True)
    out=m.classic(n,v,l,albedo=a,specular=s,exponent=e)
    assert torch.equal(out,torch.zeros_like(out))
    assert all(torch.equal(g,torch.zeros_like(g)) for g in torch.autograd.grad(out.sum(),(n,v,l,a,s,e)))


@pytest.mark.parametrize('measure',['delta','solid_angle'])
def test_tagged_integration_all_input_gradcheck_and_horizon(measure):
    light=importlib.import_module('gffx.torch.light')
    n,d,brdf,weight,omega=t([[.2,.1,1.]],True),t([[[.3,.2,1.]]],True),t([[[.2,.4,.6]]],True),t([[[.4,.6,.8]]],True),t([[.7]],True)
    def evaluate(n,d,brdf,weight,omega):
        packet=dict(measure=measure,direction=d,valid=torch.ones((1,1),dtype=torch.bool))
        packet['irradiance' if measure=='delta' else 'radiance']=weight
        if measure=='solid_angle': packet['solid_angle']=omega
        return light.integrate(n,brdf,packet)
    assert torch.autograd.gradcheck(evaluate,(n,d,brdf,weight,omega),eps=1e-6,atol=1e-5,rtol=1e-3)
    n,d=t([[0.,0.,1.]],True),t([[[1.,0.,0.]]],True)
    out=evaluate(n,d,brdf,weight,omega)
    args=(n,d,brdf,weight,omega) if measure=='solid_angle' else (n,d,brdf,weight)
    assert torch.equal(out,torch.zeros_like(out))
    assert all(torch.equal(g,torch.zeros_like(g)) for g in torch.autograd.grad(out.sum(),args))


def test_classic_demo_compares_public_images_without_regenerating_reference():
    # Small exact normal-incidence reference: test the producer independent of
    # persistent corpus availability. Actual pinned head evidence is a separate run.
    demo=importlib.import_module('tools.testing_dashboard.classic_material_demo')
    buffers=dict(covered=torch.ones(128,128,dtype=torch.bool),
        normal=t([0.,0.,1.]).expand(128,128,3),view=t([0.,0.,1.]).expand(128,128,3))
    c=2/math.sqrt(6); h=math.sqrt((1+c)/2)
    images={}
    for source in ('constant','vertex','gradient','checker'):
        colour=t([.2,.4,.6]).expand(128,128,3)
        buffers['albedo/'+source]=colour
        for model,spec in (('diffuse',0.),('classic',.1)):
            images[model+'/'+source]=(colour/math.pi+spec*10/(8*math.pi)*h*h)*c
    result=demo.evaluate(dict(buffers=buffers,images=images))
    assert len(result)==8 and all(item['metrics']['passed'] for item in result.values())
    assert all(item['actual'].shape==(128,128,3) for item in result.values())
