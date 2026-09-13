"""First-order recomputation controls against the admitted point material."""
import torch
from gffx.torch import material,scene,light
from gffx.torch.camera import Camera
from test_deferred import api


def resource():
    return material.PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)


def fields():
    def t(value):return torch.tensor(value,dtype=torch.float64,requires_grad=True)
    return (t([[.1,.2,1.],[-.1,.15,1.]]),t([[.2,.1,1.],[.1,-.2,1.]]),
            t([[[.2,.3,1.]],[[.15,.2,1.]]]),t([[.2,.3,.4],[.3,.4,.5]]),t(.613),t(.071))


def test_deferred_principled_six_pixels_fit_without_retaining_all_point_graphs():
    d=api();dtype=torch.float64
    def t(value):return torch.tensor(value,dtype=dtype)
    roughness=t(.613).requires_grad_()
    mat=scene.Material('p',material.principled,dict(albedo=t([.2,.3,.4]),roughness=roughness,f0=t(.071),budget=resource()))
    key=scene.Light('key',light.directional,dict(directions=t([[.2,.3,1.]]),irradiance=t([[1.,1.,1.]])))
    v=t([[-2.,-2.,-2.],[2.,-2.,-2.],[2.,2.,-2.],[-2.,2.,-2.]])
    f=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32)
    model=scene.Scene([scene.Mesh('q',v,f,material='p'),mat,key])
    camera=Camera.from_intrinsics(t(2.),t(2.),t(1.5),t(1.),2,3,.1,10.)
    result=d.render(model,camera)['colour']
    gradient,=torch.autograd.grad(result.sum(),roughness)
    assert result.shape==(1,2,3,3) and torch.isfinite(result).all()
    assert torch.isfinite(gradient) and gradient!=0


def test_deferred_recompute_matches_unchunked_values_and_every_fitted_gradient():
    d=api();budget=resource();arguments=fields()
    n,v,l,a,r,f=arguments
    parameters=dict(albedo=a,roughness=r,f0=f,budget=budget)
    direct=material.principled(n,v,l,**parameters)
    actual=d._principled_recompute(n,v,l,parameters)
    assert torch.equal(direct,actual)
    weights=torch.arange(1,direct.numel()+1,dtype=n.dtype).reshape_as(direct)/10
    expected=torch.autograd.grad((direct*weights).sum(),arguments)
    observed=torch.autograd.grad((actual*weights).sum(),arguments)
    for old,new in zip(expected,observed):
        assert torch.equal(old,new)
        assert torch.isfinite(new).all() and bool((new!=0).any())


def test_deferred_recompute_saves_inputs_not_material_intermediate_graph():
    d=api();n,v,l,a,r,f=fields();budget=resource();saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor:(saved.append(tensor),tensor)[1],lambda tensor:tensor):
        output=d._principled_recompute(n,v,l,dict(albedo=a,roughness=r,f0=f,budget=budget))
    expected={x.untyped_storage().data_ptr() for x in (n,v,l,a,r,f)}
    assert saved and {x.untyped_storage().data_ptr() for x in saved}==expected
    assert sum(x.numel()*x.element_size() for x in saved)==sum(x.numel()*x.element_size() for x in (n,v,l,a,r,f))
    assert torch.isfinite(torch.autograd.grad(output.sum(),r)[0])


def test_deferred_recompute_roughness_derivative_matches_two_finite_differences():
    d=api();n,v,l,a,r,f=fields();budget=resource()
    def value(rr):return d._principled_recompute(n,v,l,dict(albedo=a,roughness=rr,f0=f,budget=budget)).sum()
    derivative,=torch.autograd.grad(value(r),r)
    with torch.no_grad():
        for step in (1e-5,5e-6):
            difference=(value(r+step)-value(r-step))/(2*step)
            torch.testing.assert_close(derivative,difference,atol=1e-7,rtol=1e-4)
