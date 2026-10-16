"""Two-hour offline profile: explicit work/time extension, unchanged accuracy and storage."""
import json,os,copy
from pathlib import Path
import pytest
from tools.opaque_shading_intervals import I,Unresolved
HOUR='streamed22-offline-two-hour-v1'

def test_two_hour_profile_changes_only_time_and_refuses_other_modes():
    from tools.opaque_reference_profile import profile,worker_profile
    old=profile('streamed22-offline-hour-v1');new=profile(HOUR)
    a=old.record();b=new.record()
    for k in ('name','cells','reference_seconds','worker_seconds','watchdog_seconds'):a.pop(k);b.pop(k)
    assert a==b and (new.reference_seconds,new.worker_seconds,new.watchdog_seconds)==(7200,7230,7260)
    assert worker_profile('streamed-two-hour-piece',HOUR)==new
    for mode,name in [('cpu',HOUR),('streamed-extended-piece',HOUR),('streamed-piece',HOUR),('streamed-two-hour-piece','streamed22-offline-extended-v1')]:
        with pytest.raises(ValueError):worker_profile(mode,name)


def test_two_hour_certificate_keeps_error_allocation_and_rejects_excess():
    from tools.opaque_streamed_reference_contract import certificate_header,validate_certificate
    data=json.loads(Path(os.environ['GFFX_STREAMED_INPUT']).read_bytes());token='a'*64
    old=certificate_header(data,1,token,reference_profile='streamed22-offline-extended-v1');header=certificate_header(data,1,token,reference_profile=HOUR)
    for key in ('input_sha256','piece_sha256','allocation_sha256','allocated_basis_radius'):assert old[key]==header[key]
    row=dict(header,status='passed',basis=[['0','0']]*2,cells=8192,reference_seconds=7199,seconds=7229,supervisor=dict(exit_code=0,failure=None,elapsed_seconds=7259,sampled_peak_working_set_bytes=1073741824))
    validate_certificate(row,header)
    for key,value in [('cells',8193),('reference_seconds',7201),('seconds',7231),('basis',[['0','1e-12']]*2)]:
        bad=copy.deepcopy(row);bad[key]=value
        with pytest.raises(ValueError):validate_certificate(bad,header)
    for key,value in [('elapsed_seconds',7261),('sampled_peak_working_set_bytes',1073741825)]:
        bad=copy.deepcopy(row);bad['supervisor'][key]=value
        with pytest.raises(ValueError):validate_certificate(bad,header)
    with pytest.raises(ValueError):validate_certificate(row,old)


def test_two_hour_full_piece_and_quadrature_enforce_time_work_and_accuracy(monkeypatch):
    from tools import opaque_full_material_reference as m
    from tools.opaque_square_reference import integrate_square
    from fractions import Fraction as Q
    r=integrate_square(lambda x,y:[x*y],reference_profile=HOUR,max_cells=8192,max_depth=14)
    assert Q(r['value'][0].lo)<=Q(1,4)<=Q(r['value'][0].hi) and r['value'][0].width<=I('2e-12').lo
    for kw in [dict(max_cells=8193),dict(max_depth=15),dict(tolerance='1e-5')]:
        with pytest.raises(ValueError):integrate_square(lambda x,y:[x],reference_profile=HOUR,**kw)
    data=json.loads(Path(os.environ['GFFX_STREAMED_INPUT']).read_bytes());now=[0.];seen=[]
    monkeypatch.setattr(m.time,'monotonic',lambda:now[0])
    def overdue(field,**kw):seen.append(kw);now[0]=7201;return dict(value=[I(0),I(0)],cells=1)
    monkeypatch.setattr(m,'integrate_square',overdue)
    with pytest.raises(Unresolved,match='deadline'):m.full_pixel(data,reference_profile=HOUR,streamed_piece=1)
    assert seen[0]['max_cells']==8192 and seen[0]['max_depth']==14


def test_two_hour_controller_keeps_first_failure_order_and_restores_worker_limits(tmp_path,monkeypatch):
    from tools import opaque_streamed_reference_campaign as m
    from tools import material_loss_cost as cost
    assert m.execution_order(HOUR)==[1,0,*range(2,22)]
    saved=dict(cost.PROFILE);seen=[]
    def launch(*a,**kw):seen.append(dict(cost.PROFILE));raise RuntimeError('stopped worker')
    for index in (1,22):
        with pytest.raises(RuntimeError):m.supervised_launch(tmp_path,tmp_path/'out',reference_profile=HOUR,piece=index,launch=launch)
        assert cost.PROFILE==saved
    assert (seen[0]['cooperative_seconds'],seen[0]['watchdog_seconds'])==(7230,7260)
    assert (seen[1]['cooperative_seconds'],seen[1]['watchdog_seconds'])==(120,180)
    assert all(r['sampled_working_set_limit_bytes']==1073741824 for r in seen)


def test_two_hour_campaign_fresh_serial_identity_and_stop(tmp_path,monkeypatch):
    from tools import opaque_streamed_reference_campaign as m
    manifest=dict(profile=HOUR,checker='horizon-correction-v1')
    assert m.checker_options(manifest)==dict(centered_correction=True,directional_correction=True,horizon_correction=True)
    expected=[3,2,0,1,*range(4,22),22]
    assert m.campaign_order(manifest)==expected[:-1]
    monkeypatch.setattr(m,'certificate_header',lambda *a,**kw:{})
    monkeypatch.setattr(m,'validate_certificate',lambda *a:None)
    checks=[];calls=[]
    monkeypatch.setattr(m,'load_manifest',lambda *a:checks.append(a) or (manifest,{}))
    def launch(fixtures,out,**kw):
        index=int(kw['worker_args'][-1]);calls.append(index)
        out.write_text(json.dumps(dict(status='passed',accuracy_admitted=index==22)))
        return 0
    out=tmp_path/'complete'
    assert m.campaign(tmp_path/'manifest','a'*64,out,launch=launch)==0
    assert calls==expected and len(checks)==47
    with pytest.raises(FileExistsError):m.campaign(tmp_path/'manifest','a'*64,out,launch=launch)
    calls.clear()
    def fail(fixtures,out,**kw):
        calls.append(int(kw['worker_args'][-1]));out.write_text(json.dumps(dict(status='stopped')));return 1
    assert m.campaign(tmp_path/'manifest','a'*64,tmp_path/'fail',launch=fail)==1
    assert calls==[3] and not (tmp_path/'fail/aggregate.json').exists()
    calls.clear();checks.clear()
    def changed(*a):
        checks.append(a)
        if len(checks)==4:raise ValueError('changed candidate')
        return manifest,{}
    monkeypatch.setattr(m,'load_manifest',changed)
    assert m.campaign(tmp_path/'manifest','a'*64,tmp_path/'changed',launch=launch)==1
    assert calls==[3]


def test_two_hour_combined_allocation_requires_all_fresh_pieces():
    from tools.opaque_streamed_reference_contract import certificate_headers,aggregate
    data=json.loads(Path(os.environ['GFFX_STREAMED_INPUT']).read_bytes());token='a'*64
    headers=certificate_headers(data,token,reference_profile=HOUR)
    rows=[dict(h,status='passed',basis=[['0','0']]*2,cells=8192,reference_seconds=7200,seconds=7230,supervisor=dict(exit_code=0,failure=None,elapsed_seconds=7260,sampled_peak_working_set_bytes=1073741824)) for h in headers]
    result=aggregate(data,rows,token,reference_profile=HOUR)
    assert result['cells']==180224 and result['reference_seconds']==158400
    for bad in (rows[:-1],list(reversed(rows)),[rows[0]]*22):
        with pytest.raises(ValueError):aggregate(data,bad,token,reference_profile=HOUR)
    with pytest.raises(ValueError):aggregate(data,rows,'b'*64,reference_profile=HOUR)
