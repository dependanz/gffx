"""Bounded cubic loss interpolation: shape, analytic integrals and derivatives."""
import math
import pytest
torch=pytest.importorskip('torch')


def t(value,grad=False):
    return torch.tensor(value,dtype=torch.float64,requires_grad=grad)


@pytest.fixture(autouse=True)
def serial():
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(old)


def test_cubic_slopes_reproduce_quadratics_without_overshoot():
    from tools.material_loss_smooth import bounded_slopes, controls
    x=t([0.,.3,.7,1.])
    y=.2+.1*x+.01*x.square()
    m=bounded_slopes(x,y)
    assert torch.allclose(m,.1+.02*x,atol=1e-14,rtol=1e-14)
    b=controls(x,y,m)
    assert bool((b[:,1:]>=b[:,:-1]).all())
    assert bool((b>=y[:-1,None]).all() and (b<=y[1:,None]).all())


def test_ordered_cubic_controls_handle_plateaus_turns_and_touching_fields():
    from tools.material_loss_smooth import bounded_slopes, controls
    x=t([0.,.1,.4,.8,1.])
    y=t([[.2,.1],[.4,.4],[.4,.2],[.9,.8],[.5,.5]],True)
    m=bounded_slopes(x,y,ordered_pairs=True)
    b=controls(x,y,m)
    assert bool((b[...,0]>=b[...,1]).all())
    lo=torch.minimum(y[:-1],y[1:])[:,None,:]
    hi=torch.maximum(y[:-1],y[1:])[:,None,:]
    assert bool((b>=lo-1e-15).all() and (b<=hi+1e-15).all())
    assert torch.isfinite(torch.autograd.grad(b.sum(),y)[0]).all()


@pytest.mark.parametrize('width',[1e-9,.03,1.,4.,10.,100.])
def test_bernstein_integrals_match_independent_quadrature(width):
    from tools.material_loss_smooth import bernstein_weights
    from tools.shading_energy_probe import _gauss_unit
    nodes,w=_gauss_unit(256)
    powers=torch.stack(((1-nodes)**3,3*nodes*(1-nodes)**2,3*nodes**2*(1-nodes),nodes**3),dim=-1)
    numeric=(torch.exp(-2*width*(1-nodes))[:,None]*powers*w[:,None]).sum(0)
    observed=bernstein_weights(t([width]))[0]
    assert bool((observed>0).all())
    assert torch.allclose(observed,numeric,atol=1e-13,rtol=1e-11)
    assert abs(observed.sum().item()-(-math.expm1(-2*width)/(2*width)))<1e-13


def specimen():
    from tools.material_loss_smooth import SmoothLossResource
    rn=t([.01,.02,.04,.08]);cn=t([0.,.1,.2,.5,1.])
    rows=[]
    for r in rn:
        values=[torch.stack((.4*r**.2,r*0))]
        for c in cn[1:]:
            q=c.log()
            values.append(torch.stack(((.5+.01*q+.001*q*q)*r**.2,(.1+.02*q+.002*q*q)*r**.2)))
        rows.append(torch.stack(values))
    return SmoothLossResource(rn,cn,torch.stack(rows))


def test_smooth_resource_exact_field_and_query_gradients_without_quadrature(monkeypatch):
    import tools.material_loss_smooth as module
    resource=specimen()
    r,c,f=t(.031,True),t(.37,True),t(.4,True)
    actual=resource.evaluate(c,r,f)
    q=c.log()
    expected=r**.2*((1-f)*(.5+.01*q+.001*q*q)+f*(.1+.02*q+.002*q*q))
    assert torch.allclose(actual['loss'],expected,atol=1e-13,rtol=1e-13)
    assert torch.autograd.gradcheck(lambda c,r,f:resource.evaluate(c,r,f)['loss'],(c,r,f),atol=1e-8)
    numeric=resource.integrate_for_test(r,f,order=32)
    assert torch.allclose(actual['mean_loss'],numeric,atol=1e-11,rtol=1e-11)
    a=torch.autograd.grad(actual['mean_loss'],(r,f));b=torch.autograd.grad(numeric,(r,f))
    assert all(torch.allclose(x,y,atol=1e-11,rtol=1e-11) for x,y in zip(a,b))
    monkeypatch.setattr(module,'_gauss_unit',lambda *_:(_ for _ in ()).throw(AssertionError('query quadrature')))
    assert torch.isfinite(resource.evaluate(c,r,f)['mean_loss'])


def test_smooth_resource_positive_knot_slopes_are_continuous_and_tiny_loss_survives():
    from tools.material_loss_smooth import SmoothLossResource
    resource=specimen()
    for value in (.2,.5):
        slopes=[]
        for side in (-1,1):
            c=t(value+side*1e-8,True)
            slopes.append(torch.autograd.grad(resource.evaluate(c,t(.031),t(.4))['loss'],c)[0])
        assert torch.allclose(slopes[0],slopes[1],atol=1e-8,rtol=1e-6)
    tiny=SmoothLossResource(t(resource.roughness_nodes),t(resource.cosine_nodes),t(resource.losses)*1e-25)
    value=tiny.evaluate(t(.37),t(.031),t(1.))
    assert value['loss']>0 and value['mean_loss']>0 and value['energy']==1
    with pytest.raises(ValueError):resource.evaluate(t(.3),t(.009),t(.5))


def test_smooth_resource_minimum_shape_endpoints_and_storage_are_valid():
    from dataclasses import FrozenInstanceError
    from tools.material_loss_smooth import SmoothLossResource
    r=t([.01,.02]);c=t([0.,1.]);v=t([[[.8,0.],[.7,.3]]]*2)
    resource=SmoothLossResource(r,c,v)
    for view in (0.,.25,1.):
        actual=resource.evaluate(t(view),t(.015),t(.4))
        expected=.6*((1-view)*.8+view*.7)+.4*view*.3
        assert abs(actual['loss'].item()-expected)<1e-14
        assert abs(actual['mean_loss'].item()-(.6*(.8+2*.7)/3+.4*2*.3/3))<1e-14
    r.fill_(1);c.fill_(1);v.fill_(1)
    assert resource.roughness_nodes==(.01,.02)
    with pytest.raises(FrozenInstanceError):resource._logs=()


def test_smooth_resource_constant_unit_budget_has_exact_normalization():
    from tools.material_loss_smooth import SmoothLossResource
    r=t([.01,.02,.04,.08]);c=t([0.,.00001,.03,.17,.71,1.])
    v=torch.stack((torch.ones_like(c),.8*c.square()),dim=-1).repeat(4,1,1)
    resource=SmoothLossResource(r,c,v)
    for view in (0.,.00001,.19,1.):
        actual=resource.evaluate(t(view),t(.031),t(0.))
        assert actual['loss']==1
        assert abs(actual['mean_loss'].item()-1)<1e-14
