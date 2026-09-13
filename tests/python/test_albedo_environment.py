"""Nonconstant material-colour preflight, never full-image acceptance by points."""
import copy
import importlib
import os
from pathlib import Path
import time
import pytest
import torch


def tool():return importlib.import_module('tools.albedo_environment')


@pytest.fixture(autouse=True)
def serial():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope='module')
def budget():
    from gffx.torch.material import PrincipledBudget
    return PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)


def test_albedo_preflight_schedule_and_work_caps_are_fixed():
    assert tool().schedule()==[(m,e) for m in ('classic','principled') for e in ('uniform','affine')]
    assert tool().PROFILE['comparisons']==96
    assert tool().PROFILE['maximum_samples']==29343744
    assert tool().PROFILE['albedos']==['vertex','gradient','checker']
    with pytest.raises(TimeoutError):tool().launch_guard(time.perf_counter()-421)


@pytest.mark.parametrize('value',[[0.,1.,.5],[-.1,0.,0.],[1.1,0.,0.],[float('nan'),0.,0.],[True,0.,0.],[.2,.3]])
def test_albedo_validation_is_strict(value):
    if value==[0.,1.,.5]:assert tool().colour(value)==(0.,1.,.5)
    else:
        with pytest.raises(ValueError):tool().colour(value)


@pytest.mark.parametrize('model',['classic','principled'])
def test_generalized_reference_preserves_the_qualified_constant_colour(model,budget):
    from tools import affine_tiles as previous
    from tools.principled_direct_images import PreparedCurve,scalar
    curve=PreparedCurve(scalar.LossCurve(budget.to_envelope()['payload']),roughness=1.)
    deadline=time.perf_counter()+30;n=[.2,-.1,1.];v=[.1,.3,1.]
    old=previous.Prepared(model,curve,deadline=deadline).point(n,v,deadline=deadline)
    new=tool().Prepared(model,curve,'affine',deadline=deadline).point(n,v,[.2,.4,.6],deadline=deadline)
    tool().validate_reference(new)
    torch.testing.assert_close(torch.tensor(new['radiance'],dtype=torch.float64),torch.tensor(old['radiance'],dtype=torch.float64),atol=1e-12,rtol=1e-12)


def test_uniform_classic_albedo_has_independent_closed_form():
    p=tool().Prepared('classic',None,'uniform',deadline=time.perf_counter()+10)
    for a in ([0.,0.,0.],[1.,.2,.6]):
        row=p.point([0.,0.,1.],[0.,0.,1.],a,deadline=time.perf_counter()+10)
        assert row['radiance']==pytest.approx([x+.1*25/24 for x in a],abs=1e-11,rel=1e-9)


def test_albedo_reference_honours_work_and_deadline_limits():
    p=tool().Prepared('classic',None,'uniform',deadline=time.perf_counter()+10)
    with pytest.raises(tool().old.scalar.ReferenceConvergenceError):
        p.point([0.,0.,1.],[0.,0.,1.],[.3,.4,.5],deadline=time.perf_counter()+10,max_evaluations=512)
    with pytest.raises(TimeoutError):
        p.point([0.,0.,1.],[0.,0.,1.],[.3,.4,.5],deadline=time.perf_counter()-1)


def test_reference_and_candidate_refuse_changed_albedo_identity(budget):
    p=tool().Prepared('classic',None,'uniform',deadline=time.perf_counter()+10)
    n=[0.,0.,1.];a=[.1,.7,.3]
    ref=p.point(n,n,a,deadline=time.perf_counter()+10)
    with pytest.raises(ValueError,match='identity'):
        tool().candidate(n,n,[.4,.4,.4],model='classic',environment='uniform',budget=budget,ref=ref,deadline=time.perf_counter()+10)
    bad=copy.deepcopy(ref);bad['history'][-1]['streak']=1
    with pytest.raises(ValueError):tool().validate_reference(bad)


def test_albedo_cohort_refuses_overwrite(tmp_path):
    with pytest.raises(FileExistsError):tool().run(tmp_path,tmp_path,tmp_path/'wheel')


@pytest.mark.parametrize('model,environment',[(m,e) for m in ('classic','principled') for e in ('uniform','affine')])
def test_public_nonconstant_colour_matches_reference_and_detects_corruption(model,environment,budget):
    from tools.principled_direct_images import PreparedCurve,scalar
    curve=PreparedCurve(scalar.LossCurve(budget.to_envelope()['payload']),roughness=1.)
    deadline=time.perf_counter()+30;n=[0.,0.,1.];v=[.8,0.,.6];a=[.8,.1,.35]
    ref=tool().Prepared(model,curve,environment,deadline=deadline).point(n,v,a,deadline=deadline)
    actual=tool().candidate(n,v,a,model=model,environment=environment,budget=budget,ref=ref,deadline=deadline)
    row=dict(reference=ref,actual=actual,point=dict(normal=n,view=v),albedo=a)
    tool().validate_row(row,model,environment)
    for key,value in [('albedo',[.3,.2,.1]),('environment','bogus')]:
        bad=copy.deepcopy(row);bad['reference'][key]=value
        with pytest.raises(ValueError):tool().validate_row(bad,model,environment)
    for key,value in [('directions',0),('calls',0),('errors',[[0.,0.,0.]]),('estimates',[[float('nan')]*3]*2)]:
        bad=copy.deepcopy(row);bad['actual'][key]=value
        with pytest.raises(ValueError):tool().validate_row(bad,model,environment)


def test_saved_albedo_input_selection_is_exact_and_preserves_fields():
    corpus=os.environ.get('GFFX_FULL_ENV_CORPUS')
    if not corpus:pytest.skip('explicit immutable corpus required')
    fixed=tool().old.fixed
    buffers=fixed.load_bundle(Path(corpus),path=fixed.MANIFEST,sha256=fixed.MANIFEST_HASH)['buffers']
    rows=tool().inputs(buffers)
    assert len(rows)==24
    for offset,name in enumerate(('vertex','gradient','checker')):
        group=rows[offset*8:(offset+1)*8]
        assert len({r['point']['flat_index'] for r in group})==8
        for row in group:
            assert row['albedo_name']==name
            assert row['albedo']==buffers['albedo/'+name].reshape(-1,3)[row['point']['flat_index']].tolist()


def test_saved_albedo_preflight_replays_exact_fields_and_all_comparisons():
    root=os.environ.get('GFFX_ALBEDO_ENV');digest=os.environ.get('GFFX_ALBEDO_ENV_SHA256')
    corpus=os.environ.get('GFFX_FULL_ENV_CORPUS')
    if not all((root,digest,corpus)):pytest.skip('explicit completed preflight required')
    result=tool().replay(Path(root),digest,Path(corpus))
    assert len(result['workers'])==4
    assert sum(len(w['rows']) for w in result['workers'])==96


def test_albedo_swatches_preserve_input_reference_and_public_order():
    from tools.testing_dashboard.albedo_environment_demo import swatches
    from tools.testing_dashboard.image_reference_demo import display_rgb
    row=dict(albedo=[.8,.1,.2],reference=dict(radiance=[.3,.4,.5]),actual=dict(estimates=[[.6,.7,.8]]*2))
    image=swatches(dict(rows=[copy.deepcopy(row) for _ in range(24)]))
    assert image.shape==(216,384,3) and image.dtype==torch.uint8
    expected=display_rgb(torch.tensor([[.8,.1,.2],[.3,.4,.5],[.6,.7,.8]],dtype=torch.float64)/2)
    for group in range(3):
        for part in range(3):
            assert torch.equal(image[group*72+part*24,0],expected[part])
    with pytest.raises(ValueError):swatches(dict(rows=[]))


def test_saved_albedo_replay_rejects_changed_field_order_and_worker_evidence(monkeypatch):
    import hashlib,json
    root=os.environ.get('GFFX_ALBEDO_ENV');digest=os.environ.get('GFFX_ALBEDO_ENV_SHA256');corpus=os.environ.get('GFFX_FULL_ENV_CORPUS')
    if not all((root,digest,corpus)):pytest.skip('explicit completed preflight required')
    root=Path(root);original=tool().read_worker
    for mutation in ('order','colour','field'):
        def changed(path,sha):
            row=copy.deepcopy(original(path,sha))
            if mutation=='order':row['rows'][0],row['rows'][1]=row['rows'][1],row['rows'][0]
            elif mutation=='colour':row['rows'][0]['albedo']=[0.,0.,0.]
            else:row['rows'][0]['albedo_name']='constant'
            return row
        with monkeypatch.context() as m:
            m.setattr(tool(),'read_worker',changed)
            with pytest.raises(ValueError,match='identity'):tool().replay(root,digest,Path(corpus))
    saved=json.loads((root/'classic-uniform.json').read_text())
    for key,value in [('failure','injected failure'),('sampled_peak_working_set_bytes',1024**3+1),('monitored_pid',-1)]:
        bad=copy.deepcopy(saved);bad['supervisor'][key]=value;blob=json.dumps(bad).encode()
        with monkeypatch.context() as m:
            m.setattr(tool(),'_read',lambda *_:blob)
            with pytest.raises(ValueError,match='unqualified'):original(root/'classic-uniform.json',hashlib.sha256(blob).hexdigest())
