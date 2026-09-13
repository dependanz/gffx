"""Prepared reference reuse and bounded partial-image evidence."""
import copy
import importlib
import os
from pathlib import Path
import time

import pytest
import torch


def tool():return importlib.import_module('tools.affine_tiles')


@pytest.fixture(scope='module')
def curve():
    from gffx.torch.material import PrincipledBudget
    from tools.principled_direct_images import RESOURCE, PreparedCurve, scalar
    budget=PrincipledBudget.bundled(RESOURCE,device='cpu',dtype=torch.float64)
    return PreparedCurve(scalar.LossCurve(budget.to_envelope()['payload']),roughness=1.)


def test_tile_schedule_is_fixed_partial_and_includes_short_final_tile():
    assert tool().schedule()==[(m,i) for m in ('classic','principled') for i in (0,26,51)]
    assert sum(tool().tile_range(i)[1]-tool().tile_range(i)[0] for _,i in tool().schedule())==528
    assert tool().PROFILE['maximum_samples']==161488896


@pytest.mark.parametrize('model',['classic','principled'])
def test_prepared_reference_preserves_fresh_point_history(model,curve):
    from tools.affine_environment import reference
    prepared=tool().Prepared(model,curve,deadline=time.perf_counter()+30)
    n,v=[.2,-.3,1.],[1.,0.,-.18]
    expected=reference(n,v,model=model,curve=curve,deadline=time.perf_counter()+30)
    actual=prepared.point(n,v,deadline=time.perf_counter()+30)
    assert actual==expected


def test_prepared_setup_runs_once_and_does_not_share_returned_history(monkeypatch,curve):
    from tools import affine_environment as old
    original=old.moments;calls=[]
    def observed(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(old,'moments',observed)
    prepared=tool().Prepared('principled',curve,deadline=time.perf_counter()+30)
    first=prepared.point([0,0,1],[.6,0,.8],deadline=time.perf_counter()+30)
    expected=copy.deepcopy(first);first['moments']['values'][0]=999
    assert prepared.point([0,0,1],[.6,0,.8],deadline=time.perf_counter()+30)==expected
    assert len(calls)==1


def test_prepared_reference_owns_copy_of_caller_curve(curve):
    owned=copy.deepcopy(curve);prepared=tool().Prepared('principled',owned,deadline=time.perf_counter()+30)
    expected=prepared.point([0,0,1],[.6,0,.8],deadline=time.perf_counter()+30)
    owned.means=(0.,0.);owned.rows[0][0]=999
    assert prepared.point([0,0,1],[.6,0,.8],deadline=time.perf_counter()+30)==expected


def test_prepared_reference_still_checks_each_pixel_and_deadline(curve):
    prepared=tool().Prepared('principled',curve,deadline=time.perf_counter()+30)
    with pytest.raises(ValueError):prepared.point([0,0,1],[0,0,-1],deadline=time.perf_counter()+30)
    with pytest.raises(TimeoutError):prepared.point([0,0,1],[0,0,1],deadline=time.perf_counter()-1)
    a=prepared.point([0,0,1],[.6,0,.8],deadline=time.perf_counter()+30)
    b=prepared.point([0,0,1],[1.,0,.02],deadline=time.perf_counter()+30)
    assert a['radiance']!=b['radiance']


def test_prepared_reference_matches_all_saved_point_histories(curve):
    from tools.affine_environment import replay
    root=os.environ.get('GFFX_AFFINE_REPORTS');digest=os.environ.get('GFFX_AFFINE_SHA256');corpus=os.environ.get('GFFX_FULL_ENV_CORPUS')
    if not all((root,digest,corpus)):pytest.skip('explicit frozen affine preflight required')
    rows=replay(Path(root),digest,Path(corpus))['rows']
    setups={m:tool().Prepared(m,curve,deadline=time.perf_counter()+30) for m in ('classic','principled')}
    for row in rows:
        p=row['point']
        assert setups[row['model']].point(p['normal'],p['view'],deadline=time.perf_counter()+30)==row['reference']


def test_tile_replay_rejects_wrong_identity_before_parsing(tmp_path):
    path=tmp_path/'tile.json';path.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='identity'):tool().read_tile(path,'0'*64,{})


def test_tile_run_refuses_overwrite_and_reserves_watchdog(tmp_path):
    with pytest.raises(FileExistsError):tool().run(tmp_path,tmp_path,tmp_path/'wheel.whl')
    with pytest.raises(TimeoutError):tool().launch_guard(time.perf_counter()-721)


def test_partial_tile_cohort_cannot_claim_full_images(tmp_path):
    (tmp_path/'cohort.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='identity'):tool().replay(tmp_path,'0'*64,tmp_path)


def test_saved_tile_qualification_replays_all_selected_pixels():
    root=os.environ.get('GFFX_AFFINE_TILES');digest=os.environ.get('GFFX_AFFINE_TILES_SHA256');corpus=os.environ.get('GFFX_FULL_ENV_CORPUS')
    if not all((root,digest,corpus)):pytest.skip('explicit measured tile qualification required')
    result=tool().replay(Path(root),digest,Path(corpus))
    assert len(result['tiles'])==6 and sum(len(row['rows']) for row in result['tiles'])==528
    assert result['record']['status']==tool().COMPLETE
    assert 'images' not in result['record']


@pytest.mark.parametrize('mutation',['index','normal','estimates','count','errors','nan','moment_count'])
def test_pixel_replay_rejects_corrupted_ownership_and_numerics(curve,mutation):
    old=tool().old;n=[0.,0.,1.];v=[.6,0.,.8]
    ref=tool().Prepared('principled',curve,deadline=time.perf_counter()+30).point(n,v,deadline=time.perf_counter()+30)
    from gffx.torch.material import PrincipledBudget
    budget=PrincipledBudget.bundled(tool().RESOURCE,device='cpu',dtype=torch.float64)
    actual=old.candidate(n,v,model='principled',budget=budget,ref=ref,deadline=time.perf_counter()+30)
    pixel=dict(flat_index=17,reference=ref,actual=actual)
    tool().validate_pixel(pixel,17,n,v,'principled')
    if mutation=='index':pixel['flat_index']=18
    elif mutation=='normal':ref['normal']=[0.,1.,0.]
    elif mutation=='estimates':actual['estimates'].pop()
    elif mutation=='count':actual['directions']+=1
    elif mutation=='errors':actual['errors'][0][0]=999.
    elif mutation=='nan':actual['estimates'][0][0]=float('nan')
    else:ref['moments']['evaluations']+=1
    with pytest.raises(ValueError):tool().validate_pixel(pixel,17,n,v,'principled')


def test_tile_demo_marks_uncomputed_pixels_and_rejects_duplicate_ownership():
    demo=importlib.import_module('tools.testing_dashboard.affine_tiles_demo')
    rows=[dict(rows=[dict(flat_index=1),dict(flat_index=5)])]
    pixels=demo.coverage_map(rows)
    assert pixels.shape==(128,128,3) and pixels.dtype==torch.uint8
    assert pixels.reshape(-1,3)[1].tolist()==[40,170,100]
    assert pixels.reshape(-1,3)[2].tolist()==[96,96,96]
    assert int((pixels[:,:,1]==170).sum())==2
    with pytest.raises(ValueError):demo.coverage_map(rows+rows)
