"""Full-angle experimental loss resource: exact mean first, physical accuracy separate."""
import math
import json
import hashlib
import os
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from tools.testing_dashboard.pytest_plugin import Evidence


def t(x,grad=False):return torch.tensor(x,dtype=torch.float64,requires_grad=grad)


SCREENS = {
    'development': ((.00163,.0137,.173,.733),(.025,.43,.97),(0.,.04,1.),.01),
    'fresh': ((.00231,.0371,.287,.891),(.007,.219,.811,.999),(.17,.73,1.),.003),
    'holdout2': ((.00473,.0813,.419,.963),(.003,.337,.907,.995),(.29,.61,1.),.0015),
}


@pytest.fixture(autouse=True)
def serial():
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(old)


@pytest.mark.parametrize('roughness',[.001,.01,.199,1.])
def test_endpoint_loss_limits_refinement_and_independent_values(roughness,request):
    from tools.material_loss_endpoints import endpoints
    from tools import material_loss_reference as ref, shading_energy_probe as independent
    rows=[]
    for cosine in (0.,1.):
        a=endpoints(cosine,t(roughness),resolution=128)
        b=endpoints(cosine,t(roughness),resolution=256)
        for f in (0.,.04,1.):
            loss=(1-f)*b[0]+f*b[1]
            close=ref.directional(t(1-1e-8 if cosine else 1e-6*roughness**2),t(roughness),t(f),resolution=512,method='log')
            oracle=independent.directional_albedo(cosine,roughness,f0=f,resolution=512,radial_rule='tail-dense')
            rows.append(dict(cosine=cosine,roughness=roughness,f0=f,loss=loss.item(),refinement=abs(((1-f)*(a[0]-b[0])+f*(a[1]-b[1])).item()),
                             continuity=abs((loss-close).item()),independent_error=abs(1-loss.item()-oracle)))
    Evidence(request.node.nodeid).record('endpoint-limits',dict(rows=rows))
    assert all(0<=r['loss']<=1 and r['refinement']<=5e-5 and r['continuity']<=5e-5 and r['independent_error']<=5e-5 for r in rows),rows


def test_endpoint_exact_controls_and_roughness_gradients():
    from tools.material_loss_endpoints import endpoints
    assert endpoints(0.,t(1.))[1]==0
    assert abs(endpoints(1.,t(1.))[1].item()-math.log(2))<1e-12
    for cosine in (0.,1.):
        r=t(.173,True)
        assert torch.autograd.gradcheck(lambda r:endpoints(cosine,r), (r,),eps=1e-6,atol=1e-7,rtol=1e-5)


def test_paired_loss_geometry_agrees_with_scalar_reference():
    from tools.material_loss_endpoints import endpoints
    from tools import material_loss_reference as ref
    for r,c in ((.001,.43),(.0137,.0001),(.173,.43),(.733,.97)):
        pair=endpoints(c,t(r),resolution=256)
        for f in (0.,.04,1.):
            expected=ref.directional(t(c),t(r),t(f),resolution=256)
            assert torch.allclose((1-f)*pair[0]+f*pair[1],expected,atol=5e-9,rtol=1e-7)


def test_full_angle_resource_exact_mean_and_gradients():
    from tools.material_loss_resource import LossResource
    rn=t([.01,.02]);cn=t([0,.1,.3,1.])
    values=torch.stack([torch.stack([torch.stack((.2+r*c*c,.1*r*c*c)) for c in cn]) for r in rn])
    resource=LossResource(rn,cn,values)
    r,f=t(.013,True),t(.4,True)
    result=resource.evaluate(t(.23,True),r,f)
    # Independent closed form: L(c)=.1*r*c^2 except the first linear segment.
    known=.1*r*(.5+.1**4/6)
    assert torch.allclose(resource.evaluate(t(.23),r,t(1.))['mean_loss'],known,atol=1e-12,rtol=1e-12)
    measured=resource.integrate_for_test(r,f,order=32)
    assert torch.allclose(result['mean_loss'],measured,atol=1e-11,rtol=1e-11)
    a=torch.autograd.grad(result['mean_loss'],(r,f));b=torch.autograd.grad(measured,(r,f))
    assert all(torch.allclose(x,y,atol=1e-11,rtol=1e-11) for x,y in zip(a,b))
    assert torch.autograd.gradcheck(lambda r,f:resource.evaluate(t(.23),r,f)['mean_loss'],(r,f),atol=1e-8)
    values.fill_(1);rn.fill_(1);cn.fill_(1)
    assert torch.equal(resource.evaluate(t(.23),r,f)['mean_loss'],result['mean_loss'])


def test_resource_removable_singularity_tiny_loss_and_knot_sides():
    from tools.material_loss_resource import LossResource
    rn=t([.01,.02]);cn=t([0.,.1,.5,1.])
    # c^-2 segments make the integrated exponent exactly zero.
    values=t([[[.9,0.],[.1,.01],[.004,.0004],[.001,.0001]]]*2)
    resource=LossResource(rn,cn,values)
    r,f=t(.013,True),t(.5,True)
    assert torch.allclose(resource.evaluate(t(.3),r,f)['mean_loss'],resource.integrate_for_test(r,f,order=64),atol=1e-12)
    for c,sign in ((0.,1),(.1,1),(.5,1),(1.,-1)):
        x=t(c,True);v=resource.evaluate(x,r,f)['loss'];g=torch.autograd.grad(v,x)[0]
        h=1e-7
        fd=(-3*v.detach()+4*resource.evaluate(t(c+sign*h),r,f)['loss']-resource.evaluate(t(c+sign*2*h),r,f)['loss'])/(2*sign*h)
        assert torch.allclose(g,fd,atol=1e-8,rtol=1e-6)
    tiny=values*1e-25;small=LossResource(t([.01,.02]),t([0.,.1,.5,1.]),tiny)
    assert small.evaluate(t(.3),r,t(1.))['loss']>0
    assert small.evaluate(t(.3),r,t(1.))['energy']==1


def test_resource_domain_and_generation_work_guards():
    from tools.material_loss_resource import LossResource, generate
    for kwargs in (dict(roughness_nodes=130),dict(cosine_nodes=514),dict(resolution=1024),dict(roughness_nodes=129,cosine_nodes=513,resolution=512)):
        with pytest.raises(ValueError):generate(**kwargs)
    good=t([[[.8,0.],[.7,.3]],[[.8,0.],[.7,.3]]])
    for bad in (good.float(),good*float('nan'),good*2,good[:1]):
        with pytest.raises(ValueError):LossResource(t([.01,.02]),t([0.,1.]),bad)
    r=LossResource(t([.01,.02]),t([0.,.1,1.]),torch.cat((good[:,:1],good[:,1:],good[:,1:]),dim=1))
    with pytest.raises(ValueError):r.evaluate(t(.3),t(.009),t(.5))


@pytest.fixture(scope='module')
def resources(tmp_path_factory):
    from tools.material_loss_resource import generate, LossResource
    from tools.material_loss_smooth import SmoothLossResource
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        fixture_dir=os.environ.get('GFFX_LOSS_RESOURCE_FIXTURE_DIR')
        if fixture_dir:
            raw=[]
            # Conditional-allocation v1, regenerated after the first fresh guard failure.
            # Legacy fixed-allocation files must not silently satisfy this fixture.
            for name,expected in (('coarse','404ef671cca98bb95c636e7167cc8fb796fbd220f43f89b69fc5489382cf8628'),
                                  ('fine','c85381a9e78538d2367a13134c895dbc0c7c4180516e504ae8a5b0b4cfce9046')):
                blob=(Path(fixture_dir)/(name+'.json')).read_bytes()
                assert hashlib.sha256(blob).hexdigest()==expected,'Frozen generation fixture changed'
                value=json.loads(blob)
                raw.append(LossResource(t(value['roughness_nodes']),t(value['cosine_nodes']),t(value['losses'])))
            coarse,fine=raw
        else:
            coarse=generate(roughness_nodes=97,cosine_nodes=385,resolution=192,azimuth_resolution=32,grazing_azimuth=True)
            fine=generate(roughness_nodes=129,cosine_nodes=513,resolution=280,azimuth_resolution=32,grazing_azimuth=True)
        coarse,fine=map(SmoothLossResource.from_resource,(coarse,fine))
    finally:torch.set_num_threads(old)
    root=tmp_path_factory.mktemp('loss-resource')
    for name,resource in (('coarse',coarse),('fine',fine)):
        (root/(name+'.json')).write_text(json.dumps(dict(
            roughness_nodes=resource.roughness_nodes,cosine_nodes=resource.cosine_nodes,
            losses=resource.losses)),encoding='utf-8')
    return coarse,fine,root


def save_rows(request,root,name,rows):
    root.mkdir(parents=True,exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
    Evidence(request.node.nodeid).record(name,dict(rows=rows))


def check_rows(request,rows,checks):
    failures=[]
    for name,predicate in checks:
        try:
            with Evidence(request.node.nodeid).check(name):
                assert all(predicate(row) for row in rows),rows
        except AssertionError:
            failures.append(name)
    assert not failures,failures


@pytest.mark.parametrize('screen',SCREENS)
def test_full_resource_held_out_directional_accuracy_and_gradients(resources,request,screen):
    from tools.material_loss_reference import directional
    coarse,fine,root=resources
    rows=[]
    roughs, fixed_views, fresnels, coupled = SCREENS[screen]
    for rough in roughs:
        views=(*fixed_views,coupled*rough**2)
        for view in views:
            for f in fresnels:
                c,r,f0=t(view,True),t(rough,True),t(f,True)
                observed=fine.evaluate(c,r,f0)['loss']
                prior=coarse.evaluate(c,r,f0)['loss']
                reference=directional(c,r,f0,resolution=512,method='log')
                og=torch.autograd.grad(observed,(c,r,f0))
                rg=torch.autograd.grad(reference,(c,r,f0))
                rows.append(dict(roughness=rough,cosine=view,f0=f,loss=observed.item(),
                    reference=reference.item(),absolute_error=abs((observed-reference).item()),
                    relative_error=abs((observed/reference-1).item()),
                    refinement=abs(((observed-prior)/observed).item()),
                    gradient_ratios=[abs((a-b).item())/(.01+.02*abs(b.item())) for a,b in zip(og,rg)]))
    save_rows(request,root if screen=='development' else root/screen,'full-angle-holdouts',rows)
    check_rows(request,rows,[
        ('Directional physical accuracy',lambda x:x['absolute_error']<=5e-4 and x['relative_error']<=.001),
        ('Directional gradient accuracy',lambda x:max(x['gradient_ratios'])<=1),
        ('Directional grid refinement',lambda x:x['refinement']<=.001)])


@pytest.mark.parametrize('screen',SCREENS)
def test_full_resource_mean_matches_reference_and_its_gradients(resources,request,screen):
    from tools.material_mean_validation import average_value_and_gradients
    _,fine,root=resources
    rows=[]
    roughs, _, fresnels, _ = SCREENS[screen]
    for rough in roughs:
        for f in fresnels:
            r,f0=t(rough,True),t(f,True)
            observed=fine.evaluate(t(.43),r,f0)['mean_loss']
            numerical=fine.integrate_for_test(r,f0,order=32)
            og=torch.autograd.grad(observed,(r,f0))
            ng=torch.autograd.grad(numerical,(r,f0))
            reference,rg=average_value_and_gradients(r,f0,resolution=128,view_resolution=64)
            rows.append(dict(roughness=rough,f0=f,mean=observed.item(),reference=reference.item(),
                relative_error=abs((observed/reference-1).item()),
                numerical_error=abs((observed-numerical).item()),
                numerical_gradient_errors=[abs((a-b).item()) for a,b in zip(og,ng)],
                gradient_ratios=[abs((a-b).item())/(.01+.02*abs(b.item())) for a,b in zip(og,rg)]))
    save_rows(request,root if screen=='development' else root/screen,'full-angle-mean',rows)
    check_rows(request,rows,[
        ('Mean physical accuracy',lambda x:x['relative_error']<=.001),
        ('Mean gradient accuracy',lambda x:max(x['gradient_ratios'])<=1),
        ('Exact curve mean consistency',lambda x:x['numerical_error']<=1e-11 and max(x['numerical_gradient_errors'])<=1e-11)])


@pytest.mark.parametrize('screen',SCREENS)
def test_full_resource_independently_integrated_furnace(resources,request,screen):
    from tools.shading_energy_probe import directional_albedo
    coarse,fine,root=resources
    rows=[]
    roughs, views, fresnels, _ = SCREENS[screen]
    for rough in roughs:
        for view in views:
            for f in fresnels:
                specular=directional_albedo(view,rough,f0=f,resolution=512,radial_rule='tail-dense')
                values=[]
                for resource in (coarse,fine):
                    value=resource.evaluate(t(view),t(rough),t(f))
                    integrated=resource.integrate_for_test(t(rough),t(f),order=16)
                    values.append((value['loss']*integrated/value['mean_loss']).item())
                for albedo in (0.,.4,1.):
                    energy=specular+albedo*values[1]
                    rows.append(dict(roughness=rough,cosine=view,f0=f,albedo=albedo,energy=energy,
                        white_error=abs(energy-1) if albedo==1 else 0,
                        refinement=albedo*abs(values[1]-values[0])))
    save_rows(request,root if screen=='development' else root/screen,'full-angle-furnace',rows)
    check_rows(request,rows,[
        ('Furnace energy and white accuracy',lambda x:0<=x['energy']<=1.001 and x['white_error']<.001),
        ('Furnace grid refinement',lambda x:x['refinement']<.001)])


@pytest.mark.parametrize('screen',SCREENS)
def test_dense_generation_quadrature_is_resolved(request,tmp_path,screen):
    from tools.material_loss_resource import generation_pair
    rows=[]
    roughs, views, _, coupled = SCREENS[screen]
    for r in roughs:
        for c in (0.,1.,*views,coupled*r*r):
            a=generation_pair(c,t(r),resolution=280,azimuth_resolution=32)
            b=generation_pair(c,t(r),resolution=512,azimuth_resolution=64)
            for f in (0,1):
                error=abs((a[f]-b[f]).item())
                rows.append(dict(roughness=r,cosine=c,f0=f,loss=b[f].item(),error=error,
                                 limit=1e-12 if b[f]==0 else .0001*b[f].item()))
    Evidence(request.node.nodeid).record('generation-quadrature',dict(rows=rows))
    (tmp_path/'generation-quadrature.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    assert all(x['error']<=x['limit'] for x in rows),rows


def test_original_power_resource_retains_directional_failure(resources):
    from tools.material_loss_resource import LossResource
    from tools.material_loss_reference import directional
    _,fine,_=resources
    old=LossResource(t(fine.roughness_nodes),t(fine.cosine_nodes),t(fine.losses))
    for rough,view in ((.173,.025),(.733,.97)):
        reference=directional(t(view),t(rough),t(1.),resolution=512,method='log')
        assert abs((old.evaluate(t(view),t(rough),t(1.))['loss']/reference-1).item())>.001


def test_resource_demo_publishes_saved_failures_without_promoting_them(tmp_path):
    from tools.testing_dashboard.loss_resource_demo import publish
    from tools.testing_dashboard.server import TestingServer
    row=dict(roughness=.173,cosine=.025,f0=1.,loss=.1,reference=.101,absolute_error=.001,
             relative_error=.01,refinement=.02,gradient_ratios=[0.,2.,0.])
    for name,data in (('full-angle-holdouts',[row]),('full-angle-mean',[]),('full-angle-furnace',[])):
        (tmp_path/(name+'.json')).write_text(json.dumps(data),encoding='utf-8')
    with TestingServer(tmp_path/'server',port=0,allow_loopback_for_tests=True) as server:
        publish('http://'+server.address,'resource-fixture',tmp_path,label='synthetic failed screen')
        page=server.index.page('shading/loss-resource-demo')
        assert page['evidence']['state']=='finished' and len(page['artifacts'])>=5
        assert 'not acceptance' in server.state.get('loss-resource/read-me',run='resource-fixture')['value']
        assert server.state.get('loss-resource/map',run='resource-fixture')['data'].shape==(64,64,3)
        assert '0.173' in server.state.get('loss-resource/read-me',run='resource-fixture')['value']
