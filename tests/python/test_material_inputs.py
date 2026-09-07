"""Material-field and environment quadrature contracts, before public APIs."""
import importlib
import math
import sys
from pathlib import Path

import pytest
torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def recipe():
    return importlib.import_module('tools.material_contract_reference')


def tensor(value, dtype=torch.float64, grad=False):
    return torch.tensor(value, dtype=dtype, requires_grad=grad)


def sample_material_maps(render, texture, uv):
    # Five channels here are only test storage: native samplers read 3/1/1 maps.
    samples=[]
    for start,end in ((0,3),(3,4),(4,5)):
        pyramid,offsets=render.texture_pyramid(texture[...,start:end],levels=1)
        samples.append(render.texture(pyramid,offsets,2,2,uv,channel_count=end-start))
    return torch.cat(samples,-1)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_material_fields_expand_only_declared_axes(dtype):
    mod = recipe()
    n = tensor([[[0., 0., 1.]]*3]*2, dtype)
    result = mod.parameters(n, model='principled', albedo=tensor([.2,.4,.6], dtype),
                            roughness=tensor(.7, dtype), f0=tensor([[.1],[.2],[.3]], dtype))
    assert result['albedo'].shape == result['f0'].shape == (2,3,3)
    assert result['roughness'].shape == (2,3)
    assert torch.equal(result['f0'][0,:,0], tensor([.1,.2,.3], dtype))
    assert torch.equal(result['f0'][0,:,0], result['f0'][0,:,2])
    classic = mod.parameters(n[0], model='classic', albedo=tensor([.2,.4,.6], dtype),
                             specular=tensor(.1, dtype), exponent=tensor([4.,8.,16.], dtype))
    assert classic['specular'].shape == (3,3)
    assert classic['exponent'].shape == (3,)
    assert torch.equal(classic['specular'], torch.full((3,3), .1, dtype=dtype))


def test_material_fields_preserve_shared_gradients_and_noncontiguous_inputs():
    n = tensor([[[0.,0.,1.]]*4]*2)[:,::2]
    albedo = tensor([.2,.4,.6], grad=True)
    f0 = tensor([[.04],[.08]], grad=True)
    r = tensor(.7, grad=True)
    values = recipe().parameters(n, model='principled', albedo=albedo, f0=f0, roughness=r)
    sum(value.sum() for value in values.values()).backward()
    assert torch.equal(albedo.grad, tensor([4.,4.,4.]))
    assert torch.equal(f0.grad, tensor([[6.],[6.]]))
    assert r.grad == 4
    assert torch.equal(albedo, tensor([.2,.4,.6]))


@pytest.mark.parametrize('bad', ['model','extra','missing','albedo','roughness','batch','ambiguous','empty'])
def test_material_fields_reject_invalid_or_ambiguous_parameters(bad):
    n = tensor([[[0.,0.,1.]]*2]*2)
    kwargs = dict(model='principled', albedo=tensor([.2,.4,.6]), roughness=tensor(.7), f0=tensor(.04))
    if bad == 'model': kwargs['model'] = 'unknown'
    if bad == 'extra': kwargs['exponent'] = tensor(8.)
    if bad == 'missing': del kwargs['roughness']
    if bad == 'albedo': kwargs['albedo'] = tensor(.2)
    if bad == 'roughness': kwargs['roughness'] = tensor(0.)
    if bad == 'batch': kwargs['f0'] = torch.ones(3,2,3,dtype=torch.float64)
    if bad == 'ambiguous': kwargs['f0'] = tensor([.1,.2])  # use [P,1], not [P]
    if bad == 'empty': n=n[:,:0]; kwargs['roughness']=tensor(float('nan'))
    with pytest.raises(ValueError): recipe().parameters(n, **kwargs)


def test_environment_radiance_is_not_cosine_convolved_irradiance():
    directions = tensor([[0.,0.,1.],[0.,0.,-2.],[1.,0.,0.]])
    coefficients = torch.zeros(9,3,dtype=torch.float64)
    coefficients[0] = math.sqrt(4*math.pi)
    coefficients[2] = tensor([.5,2.,-.5])/math.sqrt(3/(4*math.pi))
    radiance = recipe().environment_radiance(directions, coefficients)
    assert torch.allclose(radiance, tensor([[1.5,3.,.5],[.5,-1.,1.5],[1.,1.,1.]]), atol=1e-12)
    assert radiance[1,1] == pytest.approx(-1.)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_environment_integral_uses_solid_angle_weights_and_cosine_once(dtype):
    n=tensor([[0.,0.,2.]],dtype)
    directions=tensor([[0.,0.,3.],[1.,0.,1.],[0.,0.,-1.]],dtype)
    radiance=tensor([[2.,3.,4.]]*3,dtype)
    brdf=torch.ones(1,3,3,dtype=dtype)/math.pi
    weights=tensor([.2,.3,.5],dtype)
    result=recipe().integrate_environment(n,directions,radiance,brdf,weights)
    expected=tensor([[2.,3.,4.]],dtype)*(.2+.3/math.sqrt(2))/math.pi
    assert torch.allclose(result,expected,atol=1e-6,rtol=1e-6)
    assert torch.allclose(recipe().integrate_environment(n,directions,radiance,brdf,2*weights),2*result)


def test_environment_interior_gradients_and_shared_batches():
    mod=recipe()
    args=(tensor([[.2,.1,1.]],grad=True),tensor([[.1,.2,1.],[.3,-.2,1.]],grad=True),
          tensor([[1.,2.,3.],[2.,1.,.5]],grad=True),tensor([[[.2,.3,.4],[.1,.2,.3]]],grad=True),
          tensor([.4,.6],grad=True))
    assert torch.autograd.gradcheck(mod.integrate_environment,args)
    coefficients=torch.zeros(9,3,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(mod.environment_radiance,(args[1],coefficients))
    n=args[0].detach().expand(2,1,3)
    result=mod.integrate_environment(n,args[1],args[2],args[3],args[4])
    assert torch.equal(result[0],result[1])
    grad=torch.autograd.grad(result.sum(),args[2])[0]
    single=torch.autograd.grad(mod.integrate_environment(*args).sum(),args[2])[0]
    assert torch.allclose(grad,2*single)


def test_environment_horizon_masks_have_zero_gradients():
    args=(tensor([[0.,0.,1.]],grad=True),tensor([[1.,0.,0.],[0.,0.,-1.]],grad=True),
          tensor([[1.,1.,1.]]*2,grad=True),tensor([[[1.,1.,1.]]*2],grad=True),tensor([1.,1.],grad=True))
    value=recipe().integrate_environment(*args)
    assert torch.equal(value,torch.zeros_like(value))
    assert all(torch.equal(g,torch.zeros_like(g)) for g in torch.autograd.grad(value.sum(),args))


def test_environment_empty_repeat_and_validation_contracts():
    mod=recipe(); n=tensor([[0.,0.,1.]])
    d=tensor([[0.,0.,1.]]); l=tensor([[1.,1.,1.]]); b=tensor([[[.2,.2,.2]]]); w=tensor([1.])
    assert torch.equal(mod.integrate_environment(n,d,l,b,w),mod.integrate_environment(n,d,l,b,w))
    assert torch.equal(mod.integrate_environment(n,d[:0],l[:0],b[:,:0],w[:0]),torch.zeros(1,3,dtype=n.dtype))
    assert mod.integrate_environment(n[:0],d,l,b[:0],w).shape == (0,3)
    for args in ((n,d,l,b,-w),(n,d,l,-b,w),(n*0,d,l,b,w),(n,d*0,l,b,w),
                 (n,d,l,b,w.expand(2)),(n,d,l,b.expand(2,1,3),w),
                 (n,d,l*float('inf'),b,w)):
        with pytest.raises(ValueError): mod.integrate_environment(*args)
    with pytest.raises(TypeError): mod.integrate_environment(n.float(),d,l,b,w)
    with pytest.raises(ValueError): mod.integrate_environment(n.to('meta'),d,l,b,w)
    with pytest.raises(ValueError,match='represent'):
        mod.integrate_environment(n,d,l*1e308,b*1e308,w)


def test_full_coupled_environment_matches_analytic_nonconstant_light(request):
    from tools.shading_energy_probe import _gauss_unit
    mu,weights=_gauss_unit(64)
    directions=torch.stack((torch.sqrt(1-mu.square()),torch.zeros_like(mu),mu),-1)
    c=.4; k=.5; ev=1-c*math.log((1+c)/c); mean=4*(1-math.log(2))/3
    el=1-mu*torch.log((1+mu)/mu)
    albedo=tensor([0.,.4,1.])
    spec=1/(2*math.pi*(c+mu))
    brdf=spec[:,None]+albedo*(1-ev)*(1-el[:,None])/(math.pi*(1-mean))
    coefficients=torch.zeros(9,3,dtype=torch.float64); coefficients[0]=math.sqrt(4*math.pi)
    coefficients[2]=k/math.sqrt(3/(4*math.pi))
    mod=recipe(); incident=mod.environment_radiance(directions,coefficients)
    value=mod.integrate_environment(tensor([[0.,0.,1.]]),directions,incident,brdf[None],2*math.pi*weights)[0]
    i2=2*math.log(2)/3-1/6
    expected=ev+k*(.5-c+c*c*math.log((1+c)/c))+albedo*(1-ev)*(1+k*(5/24)/i2)
    assert torch.allclose(value,expected,atol=1e-9,rtol=1e-9)
    wrong=(ev+albedo*(1-ev))*(1+2*k/3)
    assert (value-wrong).abs().max() > .001
    white=mod.integrate_environment(tensor([[0.,0.,1.]]),directions,torch.ones_like(incident),brdf[None],2*math.pi*weights)[0]
    assert torch.allclose(white,ev+albedo*(1-ev),atol=1e-9,rtol=1e-9)
    plugin=request.config.pluginmanager.get_plugin('tools.testing_dashboard.pytest_plugin')
    if plugin: plugin.Evidence(request.node.nodeid).record('full-environment-versus-lambert-shortcut',dict(
        integrated=value.tolist(),analytic=expected.tolist(),incorrect_lambert_shortcut=wrong.tolist(),
        scope='roughness-one unit-Fresnel exact-budget control; not general public material acceptance'))


def test_native_vertex_and_constant_texture_material_fields_agree():
    import gffx
    render=gffx.torch.render
    values=tensor([.2,.4,.6,.7,.04])
    attributes=values.expand(1,3,5).clone().requires_grad_()
    face_index=torch.zeros((1,1,2,1),dtype=torch.int32)
    bary=tensor([[[[[.2,.3,.5]],[[.4,.2,.4]]]]],grad=True)
    per_vertex=render.interpolate(face_index,bary,attributes).reshape(2,5)
    texture=values.expand(2,2,5).clone().requires_grad_()
    uv=tensor([[.35,.45],[.45,.55]],grad=True)
    sampled=sample_material_maps(render,texture,uv)
    n=tensor([[0.,0.,1.]]*2)
    def prepare(x):
        return recipe().parameters(n,model='principled',albedo=x[:,:3],roughness=x[:,3],f0=x[:,4:])
    a,b=prepare(per_vertex),prepare(sampled)
    assert all(torch.allclose(a[key],b[key],atol=1e-14,rtol=1e-14) for key in a)
    sum(value.sum() for value in b.values()).backward()
    assert texture.grad is not None and torch.isfinite(texture.grad).all()
    assert torch.allclose(uv.grad,torch.zeros_like(uv),atol=1e-14)
    sum(value.sum() for value in a.values()).backward()
    assert torch.isfinite(attributes.grad).all() and torch.isfinite(bary.grad).all()


def test_native_texture_to_material_field_gradients():
    import gffx
    render=gffx.torch.render
    texture=tensor([[[.2,.3,.4,.3,.05],[.3,.4,.5,.5,.08]],
                    [[.4,.5,.6,.6,.10],[.5,.6,.7,.8,.12]]],grad=True)
    uv=tensor([[.35,.45]],grad=True)
    def evaluate(tex,coords):
        sample=sample_material_maps(render,tex,coords)
        fields=recipe().parameters(tensor([[0.,0.,1.]]),model='principled',albedo=sample[:,:3],
                                   roughness=sample[:,3],f0=sample[:,4:])
        return torch.cat((fields['albedo'].flatten(),fields['roughness'],fields['f0'].flatten()))
    assert torch.autograd.gradcheck(evaluate,(texture,uv),eps=1e-6,atol=1e-5,rtol=1e-3)
    duv=torch.autograd.grad(evaluate(texture,uv).sum(),uv)[0]
    assert bool((duv.abs()>1e-5).all())


def test_material_environment_demo_and_lesson_are_finite_and_explicit():
    demo=importlib.import_module('tools.testing_dashboard.material_environment_demo')
    data=demo.examples()
    assert len(data['cosines']) == 33
    assert set(data['curves']) == {'full_black','full_grey','full_white','shortcut_white'}
    assert all(len(values)==33 and all(math.isfinite(x) for x in values) for values in data['curves'].values())
    assert max(abs(a-b) for a,b in zip(data['curves']['full_white'],data['curves']['shortcut_white'])) > .001
    assert 'roughness one' in data['scope']
    from tools.testing_dashboard.build_lessons import material_section
    lesson=material_section()
    assert 'solid angle' in lesson and 'level 0' in lesson and 'not a PDF' in lesson
