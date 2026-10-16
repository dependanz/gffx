"""Exact-integral and hostile-evidence controls for the separate offline campaign."""
from copy import deepcopy
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I

def fixture():
    from tools import opaque_extended_campaign as m
    from tools.opaque_reference_profile import EXTENDED
    from tests.packaging.test_opaque_reference_campaign import fixture as original,canonical
    data,old,rows=original();s=m.make_spec(canonical(data),old['sources'])
    for i,r in enumerate(rows):
        r['result'].update(protocol='original-piece-extended-v1',spec_sha256=m.digest(s),reference_profile=EXTENDED.record())
        r['result_sha256']=m.digest(r['result']);w=r['worker'];w.update(mode='extended-campaign-piece',reference_profile=EXTENDED.record(),worker_pid=100+i,profile=dict(threads=1,host_bytes=1073741824,cooperative_seconds=930,watchdog_seconds=960))
        w['campaign'].update(spec_sha256=m.digest(s),result_sha256=r['result_sha256']);w['supervisor']['monitored_pid']=100+i;r['worker_sha256']=m.digest(w)
    return data,s,rows

def rehash(r):
    from tools.opaque_extended_campaign import digest
    r['result_sha256']=digest(r['result']);r['worker']['campaign']['result_sha256']=r['result_sha256'];r['worker_sha256']=digest(r['worker'])

def test_extended_campaign_conserves_exact_rgb_and_albedo_with_same_tolerance():
    from tools import opaque_extended_campaign as m
    data,s,rows=fixture();r=m.compose(s,rows);area=sum((Q(i+1,26) for i in range(13)),Q(0))
    for (lo,hi),a in zip(r['value'],data['material']['albedo']):assert Q(lo)<=area*(2+3*Q(a))<=Q(hi) and Q(hi)-Q(lo)<=Q(2,10**12)
    assert Q(r['albedo_derivative'][0])<=3*area<=Q(r['albedo_derivative'][1])
    assert s['limits']==dict(jobs=13,cells_per_job=4096,depth_per_job=14,reference_seconds_per_job=900,worker_seconds_per_job=930,watchdog_seconds=960,host_bytes=1073741824,threads=1,cells=53248,reference_seconds=11700,worker_seconds=12090)
    assert r['profile']=='original13-offline-extended-v1'

@pytest.mark.parametrize('field,value',[('cells',4097),('cells',True),('cells',-1),('max_depth',15),('reference_seconds',900.01),('reference_seconds',-1),('elapsed_seconds',930.01),('sampled_peak_working_set_bytes',1073741825),('exit_code',1),('failure','deadline'),('status','stopped'),('monitored_pid',999)])
def test_extended_campaign_refuses_failed_or_overbudget_receipts(field,value):
    from tools import opaque_extended_campaign as m
    _,s,rows=fixture();r=rows[0];target=r['result'] if field in ('cells','max_depth','reference_seconds') else r['worker'] if field=='status' else r['worker']['supervisor'];target[field]=value;rehash(r)
    with pytest.raises(ValueError):m.compose(s,rows)

@pytest.mark.parametrize('kind',['missing','duplicate','order','input','table','geometry','source','tolerance','wide','result_hash','worker_hash','old_profile','worker_profile','result_profile','boolean_limit'])
def test_extended_campaign_refuses_mismatch_and_counterfeit_evidence(kind):
    from tools import opaque_extended_campaign as m
    _,s,rows=fixture();r=rows[0]
    if kind=='missing':rows.pop()
    elif kind=='duplicate':rows[1]=deepcopy(rows[0])
    elif kind=='order':rows[0],rows[1]=rows[1],rows[0]
    elif kind in ('input','table','geometry'):s[kind+'_sha256']='c'*64
    elif kind=='source':r['result']['sources']={}
    elif kind=='tolerance':r['result']['tolerance']='1e-6'
    elif kind=='wide':r['result']['basis'][0]=['0','1']
    elif kind=='result_hash':r['result_sha256']='c'*64
    elif kind=='worker_hash':r['worker_sha256']='c'*64
    elif kind=='old_profile':s['profile']='original13-offline-v1'
    elif kind=='worker_profile':r['worker']['profile']['cooperative_seconds']=120
    elif kind=='result_profile':r['result']['reference_profile']['cells']=256
    elif kind=='boolean_limit':s['limits']['threads']=True
    if kind not in ('result_hash','worker_hash'):rehash(r)
    with pytest.raises(ValueError):m.compose(s,rows)

def test_extended_prefix_reserves_next_whole_job_and_refuses_partial_admission():
    from tools import opaque_extended_campaign as m
    _,s,rows=fixture();usage=m.check_prefix(s,rows[:12]);left=m.remaining_allowance(s,usage)
    assert left['jobs']==1 and left['cells']==53236 and left['worker_seconds']==12078
    with pytest.raises(ValueError):m.compose(s,rows[:12])
    with pytest.raises(ValueError):m.remaining_allowance(s,dict(jobs=12,cells=53248,reference_seconds=0,worker_seconds=0))

def test_standard_campaign_contract_remains_unchanged():
    from tools import opaque_reference_campaign as old
    from tests.packaging.test_opaque_reference_campaign import fixture as original
    _,s,r=original();assert old.compose(s,r)['profile']=='original13-offline-v1'
    assert old.LIMITS['cells_per_job']==256 and old.LIMITS['reference_seconds_per_job']==90


@pytest.mark.parametrize('index',[0,8,11,12])
def test_extended_piece_worker_routes_only_ninth_to_late_proof(monkeypatch,tmp_path,index):
    import os,json,hashlib
    from pathlib import Path
    from contextlib import nullcontext
    from types import SimpleNamespace
    from tools import opaque_extended_campaign as m
    from tools import opaque_full_material_reference as ref
    from tools.opaque_reference_profile import EXTENDED
    from tests.packaging import test_opaque_extended_piece as worker
    raw=Path(os.environ['GFFX_FULL_INPUT']).read_bytes();sources={'tools/opaque_extended_campaign.py':hashlib.sha256(Path('tools/opaque_extended_campaign.py').read_bytes()).hexdigest()};spec=m.make_spec(raw,sources);sp=tmp_path/'spec.json';h=m.write_new(sp,spec);dest=tmp_path/'mock-routing-result.json'
    monkeypatch.setenv('GFFX_CAMPAIGN_SPEC',str(sp));monkeypatch.setenv('GFFX_CAMPAIGN_SPEC_SHA',h);monkeypatch.setenv('GFFX_CAMPAIGN_PIECE',str(index));monkeypatch.setenv('GFFX_CAMPAIGN_RESULT',str(dest))
    seen={}
    def square(field,**kw):seen['square']=kw
    def full(data,**kw):
        seen['full']=kw;ref.integrate_square(None,tolerance=spec['pieces'][index]['tolerance'],correction_integrator=SimpleNamespace())
        return dict(pieces=[dict(piece=index,max_depth=0)],basis=[I(0),I(0)],cells=1,seconds=0,cpu_seconds=0)
    monkeypatch.setattr(ref,'integrate_square',square);monkeypatch.setattr(ref,'full_pixel',full);monkeypatch.setattr(worker,'decimal_convolution',nullcontext)
    worker.test_extended_original_piece_independent_certificate(monkeypatch,lambda *a:None)
    assert seen['full']==dict(selected_piece=index,reference_profile=EXTENDED.name,material_only_inputs=index in (8,11,12))
    if index==8:
        from tools.opaque_late_boundary import LateBoundary
        assert seen['square']['seed_shape']==(32,4) and isinstance(seen['square']['correction_integrator'],LateBoundary)
    elif index in (11,12):
        from tools.opaque_direction_precheck import DirectionBoundary
        from tools.opaque_tail_boundary import TailBoundary
        proof=seen['square']['correction_integrator']
        assert seen['square']['seed_shape']==(64,8)
        assert type(proof) is (DirectionBoundary if index==11 else TailBoundary)
        assert proof.directions==(0,)
    else:assert 'seed_shape' not in seen['square']
    row=json.loads(dest.read_bytes());assert row['reference_profile']==EXTENDED.record() and row['piece']==index
    # Mock zeros establish routing only. No extended-piece supervisor receipt
    # exists, so this temporary result is not numerical or campaign evidence.

@pytest.mark.parametrize('mode',['offline-preflight','offline-late-preflight','extended-campaign-piece'])
def test_extended_worker_modes_require_matching_profile(mode):
    from tools.opaque_reference_profile import worker_profile,EXTENDED
    assert worker_profile(mode,EXTENDED.name)==EXTENDED
    with pytest.raises(ValueError):worker_profile(mode,'standard')
