"""One-hour offline time extension: unchanged work, storage and numerical gates."""
import json,os,copy
from pathlib import Path
import pytest
from tools.opaque_shading_intervals import I,Unresolved
HOUR='streamed22-offline-hour-v1'

def test_hour_profile_changes_only_time_and_refuses_other_modes():
    from tools.opaque_reference_profile import profile,worker_profile
    old=profile('streamed22-offline-extended-v1');new=profile(HOUR)
    a=old.record();b=new.record()
    for k in ('name','reference_seconds','worker_seconds','watchdog_seconds'):a.pop(k);b.pop(k)
    assert a==b and (new.reference_seconds,new.worker_seconds,new.watchdog_seconds)==(3600,3630,3660)
    assert worker_profile('streamed-hour-piece',HOUR)==new
    for mode,name in [('cpu',HOUR),('streamed-extended-piece',HOUR),('streamed-piece',HOUR),('streamed-hour-piece','streamed22-offline-extended-v1')]:
        with pytest.raises(ValueError):worker_profile(mode,name)


def test_hour_certificate_keeps_error_allocation_and_rejects_excess():
    from tools.opaque_streamed_reference_contract import certificate_header,validate_certificate
    data=json.loads(Path(os.environ['GFFX_STREAMED_INPUT']).read_bytes());token='a'*64
    old=certificate_header(data,1,token,reference_profile='streamed22-offline-extended-v1');header=certificate_header(data,1,token,reference_profile=HOUR)
    for key in ('input_sha256','piece_sha256','allocation_sha256','allocated_basis_radius'):assert old[key]==header[key]
    row=dict(header,status='passed',basis=[['0','0']]*2,cells=4096,reference_seconds=3599,seconds=3629,supervisor=dict(exit_code=0,failure=None,elapsed_seconds=3659,sampled_peak_working_set_bytes=1073741824))
    validate_certificate(row,header)
    for key,value in [('cells',4097),('reference_seconds',3601),('seconds',3631),('basis',[['0','1e-12']]*2)]:
        bad=copy.deepcopy(row);bad[key]=value
        with pytest.raises(ValueError):validate_certificate(bad,header)
    for key,value in [('elapsed_seconds',3661),('sampled_peak_working_set_bytes',1073741825)]:
        bad=copy.deepcopy(row);bad['supervisor'][key]=value
        with pytest.raises(ValueError):validate_certificate(bad,header)
    with pytest.raises(ValueError):validate_certificate(row,old)


def test_hour_full_piece_and_quadrature_enforce_time_work_and_accuracy(monkeypatch):
    from tools import opaque_full_material_reference as m
    from tools.opaque_square_reference import integrate_square
    from fractions import Fraction as Q
    r=integrate_square(lambda x,y:[x*y],reference_profile=HOUR,max_cells=4096,max_depth=14)
    assert Q(r['value'][0].lo)<=Q(1,4)<=Q(r['value'][0].hi) and r['value'][0].width<=I('2e-12').lo
    for kw in [dict(max_cells=4097),dict(max_depth=15),dict(tolerance='1e-5')]:
        with pytest.raises(ValueError):integrate_square(lambda x,y:[x],reference_profile=HOUR,**kw)
    data=json.loads(Path(os.environ['GFFX_STREAMED_INPUT']).read_bytes());now=[0.];seen=[]
    monkeypatch.setattr(m.time,'monotonic',lambda:now[0])
    def overdue(field,**kw):seen.append(kw);now[0]=3601;return dict(value=[I(0),I(0)],cells=1)
    monkeypatch.setattr(m,'integrate_square',overdue)
    with pytest.raises(Unresolved,match='deadline'):m.full_pixel(data,reference_profile=HOUR,streamed_piece=1)
    assert seen[0]['max_cells']==4096 and seen[0]['max_depth']==14


def test_hour_controller_keeps_first_failure_order_and_restores_worker_limits(tmp_path,monkeypatch):
    from tools import opaque_streamed_reference_campaign as m
    from tools import material_loss_cost as cost
    assert m.execution_order(HOUR)==[1,0,*range(2,22)]
    saved=dict(cost.PROFILE);seen=[]
    def launch(*a,**kw):seen.append(dict(cost.PROFILE));raise RuntimeError('stopped worker')
    for index in (1,22):
        with pytest.raises(RuntimeError):m.supervised_launch(tmp_path,tmp_path/'out',reference_profile=HOUR,piece=index,launch=launch)
        assert cost.PROFILE==saved
    assert (seen[0]['cooperative_seconds'],seen[0]['watchdog_seconds'])==(3630,3660)
    assert (seen[1]['cooperative_seconds'],seen[1]['watchdog_seconds'])==(120,180)
    assert all(r['sampled_working_set_limit_bytes']==1073741824 for r in seen)
