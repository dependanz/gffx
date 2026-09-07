"""Sharp-material reference acceptance; no public fast-path admission."""
import importlib
import math
import sys
from pathlib import Path

import pytest

torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.append(str(ROOT))
from tools import material_budget_reference as direct
from tools import shading_energy_probe as independent


@pytest.fixture(autouse=True)
def bounded_threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(old)


def recipe():return importlib.import_module('tools.material_budget_sharp')
def t(x,grad=False):return torch.tensor(x,dtype=torch.float64,requires_grad=grad)


def evidence(request,name,value):
    plugin=request.config.pluginmanager.get_plugin('tools.testing_dashboard.pytest_plugin')
    if plugin:plugin.Evidence(request.node.nodeid).record(name,value)


@pytest.mark.parametrize('cosine',[.13,.47,.86])
def test_sharp_reference_has_exact_rough_limit(cosine):
    c=t(cosine,True)
    value=recipe().directional(c,t(1.),t(1.),resolution=256)
    assert abs(value.item()-(1-cosine*math.log((1+cosine)/cosine))) < 1e-9
    expected=-math.log((1+cosine)/cosine)+1/(1+cosine)
    assert abs(torch.autograd.grad(value,c)[0].item()-expected)<1e-8


@pytest.mark.parametrize('roughness',[.01,.05,.1,.199])
@pytest.mark.parametrize('cosine',[.025,.11,.43,.79,.97])
def test_sharp_reference_resolves_values_and_gradients(cosine,roughness,request):
    rows=[]
    for f0 in (0.,.04,1.):
        args=tuple(t(x,True) for x in (cosine,roughness,f0))
        a=recipe().directional(*args,resolution=256)
        b=recipe().directional(*args,resolution=512)
        ga=torch.autograd.grad(a,args);gb=torch.autograd.grad(b,args)
        va=independent.directional_albedo(cosine,roughness,f0=f0,resolution=256,radial_rule='tail-dense')
        vb=independent.directional_albedo(cosine,roughness,f0=f0,resolution=512,radial_rule='tail-dense')
        rows.append(dict(cosine=cosine,roughness=roughness,f0=f0,value=b.item(),
            value_change=abs(a.item()-b.item()),gradients=[x.item() for x in gb],
            gradient_change=[abs(x.item()-y.item()) for x,y in zip(ga,gb)],
            independent_value=vb,independent_change=abs(va-vb)))
    evidence(request,'sharp-refinement',dict(rows=rows,scope='Finite scalar reference screening, not every positive roughness or runtime/table admission.'))
    for row in rows:
        assert row['value_change']<=5e-5,row
        assert all(e<=.001+.002*abs(g) for e,g in zip(row['gradient_change'],row['gradients'])),row
        assert row['independent_change']<=5e-5,row
        assert abs(row['value']-row['independent_value'])<=5e-5,row
        assert 0<=row['value']<=1.001,row


@pytest.mark.parametrize('point',[(.11,.213,.04),(.43,.613,.29)])
def test_sharp_reference_agrees_with_direct_derivatives(point):
    args=tuple(t(x,True) for x in point)
    a=recipe().directional(*args,resolution=512)
    b=direct.directional(*args,resolution=512)
    assert abs(a.item()-b.item())<5e-5
    ga=torch.autograd.grad(a,args);gb=torch.autograd.grad(b,args)
    assert all(abs(x.item()-y.item())<=.001+.002*abs(y.item()) for x,y in zip(ga,gb))


def test_sharp_reference_keeps_moving_boundary_gradients():
    args=tuple(t(x,True) for x in (.043,.073,.29))
    assert torch.autograd.gradcheck(lambda *xs:recipe().directional(*xs,resolution=64),args,eps=1e-6,atol=1e-6,rtol=1e-4)
    a=recipe().directional(*args,resolution=64)
    b=recipe().directional(*args,resolution=64)
    assert torch.equal(a,b)
    z=recipe().directional(args[0],args[1],t(0.),resolution=64)
    o=recipe().directional(args[0],args[1],t(1.),resolution=64)
    assert torch.allclose(a,z+args[2]*(o-z),atol=1e-13,rtol=0)


def test_sharp_reference_approaches_mirror_without_roughness_clamp():
    c,r,f=t(.43,True),t(.001,True),t(.04,True)
    value=recipe().directional(c,r,f,resolution=256)
    expected=f+(1-f)*(1-c).pow(5)
    assert abs(value.item()-expected.item())<1e-5
    assert torch.isfinite(torch.autograd.grad(value,(c,r,f))[1])


def test_sharp_reference_rejects_invalid_and_unrepresentable_inputs():
    mod=recipe()
    for resolution in (True,0,15,513):
        with pytest.raises(ValueError):mod.directional(t(.43),t(.1),t(.04),resolution=resolution)
    for point in ((0.,.1,.04),(1.,.1,.04),(.43,0.,.04),(.43,1.1,.04),(.43,1e-100,.04),(.43,.1,-.1),(.43,float('nan'),.04)):
        with pytest.raises(ValueError):mod.directional(*(t(x) for x in point))
    with pytest.raises(TypeError):mod.directional(torch.tensor(.43),t(.1),t(.04))
    with pytest.raises(ValueError):mod.directional(t([.43]),t(.1),t(.04))
    with pytest.raises(ValueError,match='representable'):
        mod.directional(t(1e-310),t(.1),t(.04))


def test_sharp_independent_rule_retains_legacy_nonconvergence():
    for c in (.43,.79,.97):
        old=[independent.directional_albedo(c,.199,f0=1.,resolution=n) for n in (256,512)]
        new=[independent.directional_albedo(c,.199,f0=1.,resolution=n,radial_rule='tail-dense') for n in (256,512)]
        assert abs(old[1]-old[0])>5e-5
        assert abs(new[1]-new[0])<=5e-5
    with pytest.raises(ValueError,match='radial_rule'):
        independent.directional_albedo(.43,.1,radial_rule='automatic')
    for rule in ('uniform','tail-dense'):
        assert abs(independent.directional_albedo(0.,1.,f0=1.,resolution=128,radial_rule=rule)-1)<1e-12


def test_sharp_average_has_exact_control_and_work_cap():
    result=recipe().average(t(1.),t(1.),resolution=64,view_resolution=32)
    assert abs(result.item()-4*(1-math.log(2))/3)<2e-9
    for n,v in ((256,64),(128,65),(True,16)):
        with pytest.raises(ValueError):recipe().average(t(.1),t(.04),resolution=n,view_resolution=v)


@pytest.mark.parametrize('roughness',[.01,.05,.1,.199])
def test_sharp_average_refines_values_and_gradients(roughness,request):
    r,f=t(roughness,True),t(.04,True)
    a=recipe().average(r,f,resolution=64,view_resolution=32)
    b=recipe().average(r,f,resolution=128,view_resolution=64)
    ga=torch.autograd.grad(a,(r,f));gb=torch.autograd.grad(b,(r,f))
    row=dict(roughness=roughness,value=b.item(),value_change=abs(a.item()-b.item()),
        gradients=[g.item() for g in gb],gradient_change=[abs(x.item()-y.item()) for x,y in zip(ga,gb)])
    evidence(request,'sharp-average',row)
    assert row['value_change']<=5e-5,row
    assert all(e<=.001+.002*abs(g) for e,g in zip(row['gradient_change'],row['gradients'])),row


@pytest.mark.parametrize('roughness',[.01,.05,.1,.199])
def test_sharp_coupling_keeps_independent_furnace_limits(roughness,request):
    from tools.shading_energy_probe import _gauss_unit
    rows=[]
    with torch.no_grad():
        nodes,weights=_gauss_unit(64)
        for f0 in (0.,.04,1.):
            average=recipe().average(t(roughness),t(f0),resolution=128,view_resolution=32)
            incoming=torch.stack([recipe().directional(c,t(roughness),t(f0),resolution=128) for c in nodes])
            residual_integral=2*((1-incoming)*nodes*weights).sum()
            assert 0<=average<1
            for c in (.025,.43,.97):
                budget=recipe().directional(t(c),t(roughness),t(f0),resolution=256)
                specular=independent.directional_albedo(c,roughness,f0=f0,resolution=512,radial_rule='tail-dense')
                for albedo in (0.,.4,1.):
                    energy=specular+albedo*(1-budget)*residual_integral/(1-average)
                    rows.append(dict(cosine=c,roughness=roughness,f0=f0,albedo=albedo,
                        energy=energy.item(),average=average.item()))
    evidence(request,'sharp-furnace',dict(rows=rows,scope='Independent VNDF specular plus separately integrated coupled residual, finite 108-case sharp study; not runtime acceptance.'))
    for row in rows:
        assert 0<=row['energy']<=1.001,row
        if row['albedo']==1.:assert abs(row['energy']-1)<.001,row


def test_sharp_demo_exposes_surfaces_and_reference_limits():
    demo=importlib.import_module('tools.testing_dashboard.sharp_material_demo')
    data=demo.examples()
    assert data['image'].shape==(360,1280,3) and data['image'].dtype==torch.uint8
    assert data['roughnesses']==[.01,.05,.1,.199]
    assert len(data['rows'])==16
    assert all(0<=row['energy']<=1.001 for row in data['rows'])
    assert 'not furnace acceptance' in data['scope']
    assert '512' in data['scope']
    for column in range(4):
        tile=data['image'][:,column*320:(column+1)*320]
        assert len(torch.unique(tile.reshape(-1,3),dim=0))>100


def test_sharp_publisher_obeys_real_dashboard_protocol(tmp_path,monkeypatch):
    from tools.testing_dashboard.server import TestingServer
    demo=importlib.import_module('tools.testing_dashboard.sharp_material_demo')
    monkeypatch.setattr(demo,'examples',lambda:dict(image=torch.zeros((2,2,3),dtype=torch.uint8),
        roughnesses=[.01],rows=[dict(roughness=.01,cosine=.43,energy=.1,direct_grid=.01,roughness_gradient=.2)],scope='test fixture'))
    with TestingServer(tmp_path,port=0,allow_loopback_for_tests=True) as server:
        demo.publish('http://'+server.address,'sharp-publisher-test')
        page=server.index.page('shading/sharp-demo')
        assert page['evidence']['state']=='finished' and page['evidence']['exit_code']==0
        assert len(page['artifacts'])==6
