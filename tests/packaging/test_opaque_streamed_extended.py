"""Controls for explicitly larger offline limits without looser numerical gates."""
import json,copy
import pytest
from tools.opaque_shading_intervals import I
NEW='streamed22-offline-extended-v1'

def test_extended_streamed_profile_requires_explicit_mode_and_keeps_old_limits():
    from tools.opaque_reference_profile import profile,worker_profile
    p=profile(NEW)
    assert (p.cells,p.depth,p.reference_seconds,p.worker_seconds,p.watchdog_seconds,p.host_bytes,p.threads)==(4096,14,900,930,960,1073741824,1)
    assert worker_profile('streamed-extended-piece',NEW)==p
    for mode,name in [('streamed-piece',NEW),('cpu',NEW),('streamed-extended-piece','standard'),('streamed-extended-piece','streamed22-offline-v1')]:
        with pytest.raises(ValueError):worker_profile(mode,name)
    assert profile('standard').reference_seconds==90
    assert profile('streamed22-offline-v1').cells==256


def test_extended_streamed_quadrature_keeps_accuracy_and_finite_work():
    from fractions import Fraction as Q
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    r=integrate_square(lambda x,y:[(x+x.lift(1,tangent=1))*y**3],reference_profile=NEW,max_cells=4096,max_depth=14)
    for key,want in [('value',Q(3,8)),('derivative',Q(1,4))]:
        value=r[key][0];assert Q(value.lo)<=want<=Q(value.hi) and value.width<=I('2e-12').lo
    for kw in [dict(max_cells=4097),dict(max_depth=15),dict(tolerance='1e-5')]:
        with pytest.raises(ValueError):integrate_square(lambda x,y:[x],reference_profile=NEW,**kw)
    def bad(x,y):raise ValueError('unresolved')
    with pytest.raises(Unresolved) as e:integrate_square(bad,reference_profile=NEW,max_cells=4,max_depth=14)
    assert e.value.details['cells']==4


def test_extended_supervisor_restores_standard_limits_after_worker(tmp_path,monkeypatch):
    from tools import opaque_streamed_reference_campaign as m
    from tools import material_loss_cost as cost
    before=dict(cost.PROFILE);seen=[]
    def launch(*a,**kw):
        seen.append(dict(cost.PROFILE));raise RuntimeError('owned worker failed')
    with pytest.raises(RuntimeError):m.supervised_launch(tmp_path,tmp_path/'out',reference_profile=NEW,piece=1,launch=launch)
    assert seen[0]['cooperative_seconds']==930 and seen[0]['watchdog_seconds']==960
    assert seen[0]['sampled_working_set_limit_bytes']==1073741824 and cost.PROFILE==before
    with pytest.raises(RuntimeError):m.supervised_launch(tmp_path,tmp_path/'out',reference_profile=NEW,piece=22,launch=launch)
    assert seen[1]['cooperative_seconds']==120 and seen[1]['watchdog_seconds']==180 and cost.PROFILE==before


def test_extended_campaign_runs_failed_piece_first_then_fresh_remaining22(tmp_path,monkeypatch):
    from tools import opaque_streamed_reference_campaign as m
    monkeypatch.setattr(m,'load_manifest',lambda *a:({'profile':NEW},{}))
    monkeypatch.setattr(m,'certificate_header',lambda *a,**kw:{})
    monkeypatch.setattr(m,'validate_certificate',lambda *a:None)
    seen=[]
    def launch(fixtures,out,**kw):
        index=int(kw['worker_args'][-1]);seen.append(index)
        out.write_text(json.dumps(dict(status='passed',accuracy_admitted=index==22)));return 0
    out=tmp_path/'campaign';assert m.campaign(tmp_path/'manifest','a'*64,out,launch=launch)==0
    assert seen==[1,0,*range(2,23)] and len(set(seen))==23
    assert json.loads((out/'campaign.json').read_text())['completed_pieces']==22


def test_extended_campaign_first_failure_launches_nothing_else(tmp_path,monkeypatch):
    from tools import opaque_streamed_reference_campaign as m
    monkeypatch.setattr(m,'load_manifest',lambda *a:({'profile':NEW},{}))
    seen=[]
    def launch(fixtures,out,**kw):
        seen.append(int(kw['worker_args'][-1]));out.write_text(json.dumps(dict(status='stopped')));return 1
    out=tmp_path/'campaign';assert m.campaign(tmp_path/'manifest','a'*64,out,launch=launch)==1
    assert seen==[1] and not (out/'aggregate.json').exists()

def test_extended_certificate_uses_larger_work_limits_but_same_error_share():
    import os
    from pathlib import Path
    from tools.opaque_streamed_reference_contract import certificate_header,validate_certificate
    data=json.loads(Path(os.environ['GFFX_STREAMED_INPUT']).read_bytes());token='a'*64
    old=certificate_header(data,1,token);new=certificate_header(data,1,token,reference_profile=NEW)
    assert old['allocated_basis_radius']==new['allocated_basis_radius'] and old['allocation_sha256']==new['allocation_sha256']
    row=dict(new,status='passed',basis=[['0','0']]*2,cells=4096,reference_seconds=899,seconds=929,supervisor=dict(exit_code=0,failure=None,elapsed_seconds=959,sampled_peak_working_set_bytes=1073741824))
    validate_certificate(row,new)
    with pytest.raises(ValueError):validate_certificate(row,old)
    for key,value in [('cells',4097),('reference_seconds',901),('seconds',931)]:
        bad=copy.deepcopy(row);bad[key]=value
        with pytest.raises(ValueError):validate_certificate(bad,new)
    for key,value in [('elapsed_seconds',961),('sampled_peak_working_set_bytes',1073741825)]:
        bad=copy.deepcopy(row);bad['supervisor'][key]=value
        with pytest.raises(ValueError):validate_certificate(bad,new)
    bad=copy.deepcopy(row);bad['basis'][0]=['0','1e-12']
    with pytest.raises(ValueError):validate_certificate(bad,new)


def test_extended_full_piece_binding_preserves_allocation_and_charges_deadline(monkeypatch):
    import os
    from pathlib import Path
    from tools import opaque_full_material_reference as m
    from tools.opaque_streamed_reference_contract import allocation
    from tools.opaque_shading_intervals import Unresolved
    data=json.loads(Path(os.environ['GFFX_STREAMED_INPUT']).read_bytes());seen=[]
    def square(field,**kw):seen.append(kw);return dict(value=[I(0),I(0)],cells=1)
    monkeypatch.setattr(m,'integrate_square',square)
    result=m.full_pixel(data,reference_profile=NEW,streamed_piece=1,aggregate_residual=True,smooth_partition=(4,4),stable_view_base=True,tight_cosine_slopes=True)
    assert seen[0]['max_cells']==4096 and seen[0]['max_depth']==14
    assert seen[0]['tolerance']==str(allocation(data)[1])==result['allocated_basis_radius']
    now=[0.];monkeypatch.setattr(m.time,'monotonic',lambda:now[0])
    def overdue(*a,**kw):now[0]=901;return dict(value=[I(0),I(0)],cells=1)
    monkeypatch.setattr(m,'integrate_square',overdue)
    with pytest.raises(Unresolved,match='deadline'):m.full_pixel(data,reference_profile=NEW,streamed_piece=1)
