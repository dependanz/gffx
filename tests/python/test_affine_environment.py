"""World-space coloured lighting: bounded independent reference before images."""
import copy
import importlib
import math
import os
from pathlib import Path
import time

import pytest
import torch


def tool():
    return importlib.import_module('tools.affine_environment')


@pytest.fixture(scope='module')
def resources():
    from gffx.torch.material import PrincipledBudget
    from tools.principled_direct_images import PreparedCurve, RESOURCE, scalar
    budget=PrincipledBudget.bundled(RESOURCE,device='cpu',dtype=torch.float64)
    return budget,PreparedCurve(scalar.LossCurve(budget.to_envelope()['payload']),roughness=1.)


@pytest.mark.parametrize('split',[.25,.5])
def test_full_sphere_quadrature_preserves_hemisphere_moments(split):
    q,w=tool().quadrature(16,split)
    assert q.shape==(1024,3) and w.shape==(1024,)
    assert bool((w>0).all()) and bool((q[:,2]>0).all())
    torch.testing.assert_close(q.norm(dim=1),torch.ones(1024,dtype=torch.float64),atol=1e-14,rtol=0)
    assert w.sum().item()==pytest.approx(2*math.pi,abs=1e-13)
    torch.testing.assert_close((q*w[:,None]).sum(0),torch.tensor([0.,0.,math.pi],dtype=torch.float64),atol=1e-12,rtol=0)
    torch.testing.assert_close((q.square()*w[:,None]).sum(0),torch.full((3,),2*math.pi/3,dtype=torch.float64),atol=1e-12,rtol=0)


def test_affine_radiance_is_positive_and_matches_independent_sh_ordering():
    from tools.shading_environment_reference import sh_radiance
    q,_=tool().quadrature(8,.5)
    actual=tool().field(q)
    expected=torch.tensor([sh_radiance(d,tool().coefficients()) for d in q.tolist()],dtype=torch.float64)
    torch.testing.assert_close(actual,expected,atol=1e-14,rtol=0)
    assert bool((actual>0).all()) and not torch.equal(actual[:,0],actual[:,1])


@pytest.mark.parametrize('bands',[[[1.,0.,0.]]*3,[[math.nan,0.,0.]]*3,[[.1,.2]],[[10**500,0.,0.]]*3])
def test_affine_field_rejects_unqualified_coefficients(bands):
    with pytest.raises(ValueError):tool().field(torch.tensor([[0.,0.,1.]],dtype=torch.float64),bands)


@pytest.mark.parametrize('normal,view', [([0.,0.,1.],[0.,0.,1.]),([1.,2.,3.],[-.5,1.,2.]),([0.,1.,0.],[1.,.02,0.])])
def test_view_frame_is_orthonormal_and_retains_world_orientation(normal,view):
    c,basis=tool().frame(normal,view)
    torch.testing.assert_close(basis@basis.T,torch.eye(3,dtype=torch.float64),atol=1e-13,rtol=0)
    assert torch.linalg.det(basis).item()==pytest.approx(1.,abs=1e-13)
    v=torch.tensor(view,dtype=torch.float64);v=v/v.norm()
    torch.testing.assert_close(v@basis.T,torch.tensor([math.sqrt(max(0.,1-c*c)),0.,c],dtype=torch.float64),atol=2e-13,rtol=0)


@pytest.mark.parametrize('normal,view',[([0,0,0],[0,0,1]),([0,0,1],[1,0,0]),([0,0,1],[0,0,-1]),([math.inf,0,1],[0,0,1])])
def test_reference_rejects_invalid_or_backfacing_geometry(normal,view):
    with pytest.raises(ValueError):tool().frame(normal,view)


def test_affine_lambert_integral_matches_public_sh_convolution():
    from gffx.torch import light
    n=torch.tensor([[.2,-.4,1.]],dtype=torch.float64);n=n/n.norm(dim=1,keepdim=True)
    _,basis=tool().frame(n[0].tolist(),n[0].tolist())
    q,w=tool().quadrature(16,.25);world=q@basis
    value=(tool().field(world)*(q[:,2]*w)[:,None]/math.pi).sum(0)
    expected=1+2/3*(torch.tensor(tool().BANDS,dtype=torch.float64)@n[0])
    torch.testing.assert_close(value,expected,atol=1e-12,rtol=0)
    convolved=light.spherical_harmonics(n,torch.tensor(tool().coefficients(),dtype=torch.float64))[0]/math.pi
    torch.testing.assert_close(value,convolved,atol=1e-12,rtol=0)


def test_weighted_loss_moments_match_known_polynomial_and_keep_history():
    class Curve:
        c=[0.,.1,1.]
        def loss(self,mu,r,f):return .2+.3*mu
        def mean(self,r,f):return .4
    row=tool().moments(Curve(),deadline=time.perf_counter()+20)
    assert row['values']==pytest.approx([.4,2*.2/3+.3/2],abs=1e-13)
    assert row['history'][-1]['streak']==2 and len(row['history'])==3


def test_weighted_loss_moments_reject_disagreeing_mean_and_exhausted_work():
    from tools.shading_environment_reference import ReferenceConvergenceError
    class Curve:
        c=[0.,.1,1.]
        def loss(self,mu,r,f):return .5
        def mean(self,r,f):return .1
    with pytest.raises(ValueError,match='mean'):tool().moments(Curve(),deadline=time.perf_counter()+20)
    with pytest.raises(ReferenceConvergenceError,match='work_limit'):
        tool().moments(Curve(),deadline=time.perf_counter()+20,max_evaluations=1)


def test_affine_reference_keeps_work_and_time_guards():
    from tools.shading_environment_reference import ReferenceConvergenceError
    with pytest.raises(ReferenceConvergenceError) as error:
        tool().reference([0.,0.,1.],[.9,0.,.1],model='classic',curve=None,deadline=time.perf_counter()+30,max_evaluations=512)
    assert error.value.report['evaluations']==512 and len(error.value.report['history'])==1
    with pytest.raises(TimeoutError):
        tool().reference([0.,0.,1.],[0.,0.,1.],model='classic',curve=None,deadline=time.perf_counter()-1)


@pytest.mark.parametrize('model',['classic','principled'])
def test_affine_reference_matches_independent_scalar_at_normal_view(model,resources):
    from tools.shading_environment_reference import environment_reference
    from tools.principled_direct_images import scalar
    _,curve=resources
    def brdf(n,v,l):
        if model=='classic':return scalar.classic(n,v,l,albedo=[.2,.4,.6],specular=.1,exponent=2.)
        return scalar.principled(n,v,l,albedo=[.2,.4,.6],roughness=1.,f0=.04,budget=curve)
    # A polynomial synthetic residual is an independent integration control, not resource admission.
    if model=='principled':
        class Polynomial:
            r=[.001,1.];c=[0.,.1,1.]
            def loss(self,mu,r,f):return .4+.2*mu
            def mean(self,r,f):return .4+2*.2/3
        curve=Polynomial()
    n=[0.,0.,1.]
    other=environment_reference(n,n,tool().coefficients(),brdf=brdf)
    ref=tool().reference(n,n,model=model,curve=curve,deadline=time.perf_counter()+30)
    assert ref['radiance']==pytest.approx(other['radiance'],abs=1e-10,rel=1e-9)


@pytest.mark.parametrize('model',['classic','principled'])
def test_affine_public_complete_material_matches_reference_at_grazing_view(model,resources):
    budget,curve=resources;n=[.2,-.3,1.];v=[1.,0.,-.18]
    ref=tool().reference(n,v,model=model,curve=curve,deadline=time.perf_counter()+30)
    value=tool().candidate(n,v,model=model,budget=budget,ref=ref,deadline=time.perf_counter()+30)
    assert len(value['estimates'])==2 and value['directions']<=131072
    assert all(e<=t for error in value['errors'] for e,t in zip(error,value['limits']))
    wrong=copy.deepcopy(ref);wrong['radiance']=[10.,10.,10.]
    with pytest.raises(ValueError):tool().validate_reference(wrong)


def test_affine_reference_rotates_lighting_with_geometry_and_detects_fixed_world_light(resources):
    _,curve=resources;n=[.2,-.3,1.];v=[.6,.1,1.]
    r=torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]],dtype=torch.float64)
    nr=(torch.tensor(n,dtype=torch.float64)@r).tolist();vr=(torch.tensor(v,dtype=torch.float64)@r).tolist()
    b=(torch.tensor(tool().BANDS,dtype=torch.float64)@r).tolist()
    a=tool().reference(n,v,model='principled',curve=curve,deadline=time.perf_counter()+30)
    rotated=tool().reference(nr,vr,model='principled',curve=curve,bands=b,deadline=time.perf_counter()+30)
    fixed=tool().reference(nr,vr,model='principled',curve=curve,deadline=time.perf_counter()+30)
    assert a['radiance']==pytest.approx(rotated['radiance'],abs=1e-11)
    assert a['radiance']!=pytest.approx(fixed['radiance'],abs=1e-4)


def test_affine_cohort_refuses_existing_output_and_reserves_watchdog(tmp_path):
    with pytest.raises(FileExistsError):tool().run(tmp_path,tmp_path,tmp_path/'absent.whl')
    with pytest.raises(TimeoutError):tool().launch_guard(time.perf_counter()-421)


def test_affine_replay_refuses_changed_report_before_loading(tmp_path):
    path=tmp_path/'point.json';path.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='identity'):tool().read_point(path,'0'*64)


@pytest.fixture(scope='module')
def checked_reference(resources):
    return tool().reference([0.,0.,1.],[.6,0.,.8],model='principled',curve=resources[1],deadline=time.perf_counter()+30)


@pytest.mark.parametrize('mutation',['mean_nan','count','moment_shape','moment_value','angular_count','angular_shape'])
def test_affine_reference_replay_rejects_tampered_histories(checked_reference,mutation):
    ref=copy.deepcopy(checked_reference)
    if mutation=='mean_nan':ref['moments']['adaptive_mean']=math.nan
    elif mutation=='count':ref['moments']['history'][0]['evaluations']=1
    elif mutation=='moment_shape':ref['moments']['history'][0]['values'].pop()
    elif mutation=='moment_value':ref['moments']['values'][0]+=.1
    elif mutation=='angular_count':ref['history'][0]['evaluations']+=1
    else:ref['history'][0]['estimates'][0].pop()
    with pytest.raises(ValueError):tool().validate_reference(ref)


def test_world_lighting_cannot_use_doubled_half_sphere_shortcut():
    from tools.environment_full_images import reference_brdf
    q,w=tool().quadrature(16,.5);half=len(q)//2
    terms=reference_brdf(.4,q,model='classic')*tool().field(q)*(q[:,2]*w)[:,None]
    assert (2*terms[:half].sum(0)-terms.sum(0)).abs().max().item()>.01


def test_affine_cohort_replay_rejects_changed_summary(tmp_path):
    (tmp_path/'cohort.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='identity'):tool().replay(tmp_path,'0'*64,tmp_path)


def test_saved_affine_cohort_replays_all_predeclared_points():
    root=os.environ.get('GFFX_AFFINE_REPORTS');digest=os.environ.get('GFFX_AFFINE_SHA256')
    corpus=os.environ.get('GFFX_FULL_ENV_CORPUS')
    if not all((root,digest,corpus)):pytest.skip('explicit measured affine cohort required')
    result=tool().replay(Path(root),digest,Path(corpus))
    assert len(result['rows'])==16
    assert {r['model'] for r in result['rows']}=={'classic','principled'}
    assert all(r['reference']['history'][-1]['streak']==2 for r in result['rows'])


def test_affine_demo_map_is_finite_and_has_opposite_direction_colour_balance():
    demo=importlib.import_module('tools.testing_dashboard.affine_environment_demo')
    linear=demo.environment_map()
    assert linear.shape==(192,384,3) and linear.dtype==torch.float64
    assert bool(torch.isfinite(linear).all()) and bool((linear>0).all())
    opposite=torch.roll(linear.flip(0),192,dims=1)
    torch.testing.assert_close(linear+opposite,torch.full_like(linear,2.),atol=1e-14,rtol=0)
