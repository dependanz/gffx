"""Direct-loss reference repair; old unresolved quadratures remain controls."""
import importlib
import math
import sys
from pathlib import Path

import pytest
torch=pytest.importorskip("torch")
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))
from tools.testing_dashboard.pytest_plugin import Evidence


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    try: yield
    finally: torch.set_num_threads(old)


def t(x,grad=False): return torch.tensor(x,dtype=torch.float64,requires_grad=grad)
def ref(): return importlib.import_module("tools.material_loss_reference")


@pytest.mark.parametrize("roughness",[.001,.01,.2,1.])
def test_rejected_probability_has_exact_normal_view(roughness):
    value=ref().rejected_probability(t(1.),t(roughness),resolution=64)
    a2=roughness**4
    assert abs(value.item()-a2/(1+a2)) <= 1e-12*a2/(1+a2)


@pytest.mark.parametrize("cosine",[.13,.47,.86])
def test_loss_has_exact_rough_limit_and_derivatives(cosine):
    c=t(cosine,True)
    result=ref().directional(c,t(1.),t(1.),resolution=128)
    expected=cosine*math.log((1+cosine)/cosine)
    assert abs(result.item()-expected)<1e-9
    expected_grad=math.log((1+cosine)/cosine)-1/(1+cosine)
    assert abs(torch.autograd.grad(result,c)[0].item()-expected_grad)<1e-8


@pytest.mark.parametrize("roughness",[.001,.003,.01])
@pytest.mark.parametrize("f0",[1-1e-8,1.])
def test_loss_mean_resolves_both_coordinates(roughness,f0,request):
    args=t(roughness,True),t(f0,True)
    rows=[]
    for method in ("sinh","log"):
        values=[];gradients=[]
        for n,v in ((64,32),(128,64)):
            result=ref().average(*args,resolution=n,view_resolution=v,method=method)
            gradients.append([x.item() for x in torch.autograd.grad(result,args)])
            values.append(result.item())
        rows.append(dict(method=method,values=values,gradients=gradients,
                         relative_change=abs(values[0]-values[1])/abs(values[1])))
    Evidence(request.node.nodeid).record("direct-loss",dict(roughness=roughness,f0=f0,rows=rows))
    for row in rows:
        assert 0<row["values"][1]<1,row
        assert row["relative_change"]<=.001,row
        assert all(abs(a-b)<=.001+.002*abs(b) for a,b in zip(*row["gradients"])),row
    assert abs(rows[0]["values"][1]-rows[1]["values"][1])<=.001*rows[0]["values"][1]


def test_loss_mean_exact_control_and_explicit_limits():
    mod=ref()
    value=mod.average(t(1.),t(1.),resolution=64,view_resolution=32)
    assert abs(value.item()-(1-4*(1-math.log(2))/3))<2e-9
    for kwargs in (dict(method="automatic"),dict(resolution=256,view_resolution=64)):
        with pytest.raises(ValueError):mod.average(t(.01),t(1.),**kwargs)


def test_loss_keeps_parameter_gradients():
    args=t(.003,True),t(.29,True)
    assert torch.autograd.gradcheck(lambda *x:ref().average(*x,resolution=32,view_resolution=16),
                                   args,eps=1e-6,atol=1e-6,rtol=1e-4)


@pytest.mark.parametrize('roughness',[.001,.01,.3,1.])
@pytest.mark.parametrize('cosine',[.03,.43,.97])
def test_rejected_tail_matches_direct_density_integral(roughness,cosine):
    from tools.shading_energy_probe import _gauss_unit
    # Separate direct integration of Heitz eq.3 over the rejected inverse-slope
    # interval, not evaluation of the production tail antiderivative.
    x,w=_gauss_unit(128)
    phi=torch.cat((x,x+1))*(math.pi/2)
    pw=torch.cat((w,w))/2
    a=roughness**2;c=cosine
    b=math.sqrt(1-c*c)*phi.cos()
    # The quadratic root for inverse slope avoids sharing the production tmax.
    upper=a*(torch.sqrt(b*b+c*c)-b)/c
    lower=torch.where(b<0,-a*b/c,0.)
    s=lower+(upper-lower)*x[:,None]
    density=4*(c*s+a*b)/((c+math.sqrt(a*a+(1-a*a)*c*c))*(1+s*s)**2)
    direct=(density*(upper-lower)*w[:,None]*pw).sum()
    actual=ref().rejected_probability(t(c),t(roughness),resolution=128)
    assert torch.allclose(actual,direct,atol=1e-15,rtol=1e-9)


@pytest.mark.parametrize('roughness,cosine',
    [(r,c) for r in (.001,.01,.199,1.) for c in (.025,.43,.97)]+
    [(.001,.000001),(.01,.000001)])
def test_loss_complements_existing_energy_and_partial_gradients(roughness,cosine):
    from tools import material_budget_sharp as sharp
    for f0 in (0.,.04,1.):
        args=tuple(t(x,True) for x in (cosine,roughness,f0))
        loss=ref().directional(*args,resolution=256)
        energy=sharp.directional(*args,resolution=512)
        gl=torch.autograd.grad(loss,args);ge=torch.autograd.grad(energy,args)
        assert 0<=loss.item()<=1.001
        assert abs(loss.item()+energy.item()-1)<=5e-5
        assert all(abs(a.item()+b.item())<=.001+.002*abs(b.item()) for a,b in zip(gl,ge))


@pytest.mark.parametrize('roughness',[.199,1.])
def test_extreme_grazing_loss_refines_without_using_unresolved_old_gradient(roughness,request):
    rows=[]
    for f0 in (0.,.04,1.):
        args=tuple(t(x,True) for x in (1e-6,roughness,f0))
        results=[]
        for method,n in (('sinh',256),('sinh',512),('log',512)):
            value=ref().directional(*args,resolution=n,method=method)
            results.append(dict(value=value.item(),gradients=[x.item() for x in torch.autograd.grad(value,args)]))
        rows.append(dict(f0=f0,results=results))
    Evidence(request.node.nodeid).record('extreme-grazing-loss',dict(roughness=roughness,rows=rows,
        scope='Cross-coordinate refinement, not independent physical models. Old directional gradients are unresolved here.'))
    for row in rows:
        for result in row['results'][:-1]:
            other=row['results'][-1]
            assert abs(result['value']-other['value'])<5e-5
            assert all(abs(a-b)<=.001+.002*abs(b) for a,b in zip(result['gradients'],other['gradients']))


def test_old_extreme_grazing_gradient_fails_exact_rough_control():
    from tools import material_budget_sharp as sharp
    c=t(1e-6,True)
    old=sharp.directional(c,t(1.),t(1.),resolution=512)
    old_gradient=torch.autograd.grad(old,c)[0].item()
    exact=math.log(1000001)-1/1.000001
    assert abs(old_gradient+exact)>.001+.002*exact
    new=ref().directional(c,t(1.),t(1.),resolution=512)
    assert abs(new.item()-1e-6*math.log(1000001))<5e-5
    assert abs(torch.autograd.grad(new,c)[0].item()-exact)<.001+.002*exact


def test_old_visible_normal_sampler_misses_known_small_tail(request):
    from tools.shading_energy_probe import _gauss_unit
    a2=.001**4
    x,w=_gauss_unit(128)
    u=x*(2-x);weights=w*2*(1-x)
    rejected=(weights*(u>1/(1+a2))).sum().item()
    exact=a2/(1+a2)
    Evidence(request.node.nodeid).record('missed-tail',dict(sampled=rejected,exact=exact,
        scope='Normal-view rejected probability alone; confirms a missed rare event, not all errors of the old oracle.'))
    assert rejected==0 and exact>0
    assert abs(ref().rejected_probability(t(1.),t(.001)).item()-exact)<exact*1e-12


def test_loss_preserves_legacy_mean_failure_as_negative_control():
    from tools import material_budget_sharp as sharp
    a,b=[1-sharp.average(t(.003),t(1.),resolution=n,view_resolution=v).item()
         for n,v in ((64,32),(128,64))]
    assert abs(a-b)/abs(b)>.001


@pytest.mark.parametrize('roughness',[.001,.003,.01])
def test_tiny_loss_gradient_matches_relative_finite_difference(roughness,request):
    r=t(roughness,True);f=t(1.,True)
    result=ref().average(r,f)
    dr,df=torch.autograd.grad(result,(r,f))
    estimates=[]
    for relative_step in (.001,.0005):
        h=roughness*relative_step
        estimates.append((ref().average(t(roughness+h),t(1.))-
                          ref().average(t(roughness-h),t(1.))).item()/(2*h))
    # Affine Fresnel dependence provides an exact slope for this quadrature.
    affine=(ref().average(t(roughness),t(1.))-
            ref().average(t(roughness),t(0.))).item()
    Evidence(request.node.nodeid).record('tiny-loss-gradients',dict(roughness=roughness,
        roughness_gradient=dr.item(),finite_difference=estimates,fresnel_gradient=df.item(),affine=affine))
    assert all(abs(g-dr.item())<=.001*abs(dr.item()) for g in estimates)
    assert abs(df.item()-affine)<1e-10


def test_loss_rejects_invalid_inputs_without_changing_domain():
    mod=ref()
    for point in ((0.,.01,1.),(1.,.01,1.),(.4,0.,1.),(.4,1e-100,1.),(.4,.01,1.1)):
        with pytest.raises(ValueError):mod.directional(*(t(x) for x in point))
    with pytest.raises(TypeError):mod.average(torch.tensor(.01),t(1.))
    with pytest.raises(ValueError):mod.directional(t(.43),t(.01),t(1.),resolution=True)


def test_residual_study_retains_legacy_and_adds_direct_loss(monkeypatch):
    from tools import material_residual_probe as probe
    monkeypatch.setattr(probe,'residual',lambda *args,**kwargs:t(1e-7))
    monkeypatch.setattr(probe.loss,'average',lambda *args,**kwargs:t(2e-7))
    data=probe.study()
    assert len(data['rows'])==6
    for row in data['rows']:
        assert set(row['methods'])=={'legacy','direct','transformed','independent','loss-sinh','loss-log'}
        assert row['loss_coordinate_disagreement']==0
    assert 'not an independent physical model' in data['scope']
