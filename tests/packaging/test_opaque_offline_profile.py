"""Independent controls for explicit offline-only budgets and retained defaults."""
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Unresolved

OFFLINE='original13-offline-extended-v1'

def test_profile_is_explicit_immutable_and_exact():
    from tools.opaque_reference_profile import profile
    old=profile();new=profile(OFFLINE)
    assert (old.cells,old.depth,old.reference_seconds,old.worker_seconds,old.watchdog_seconds)==(256,10,90,120,180)
    assert (new.cells,new.depth,new.reference_seconds,new.worker_seconds,new.watchdog_seconds)==(4096,14,900,930,960)
    assert old.host_bytes==new.host_bytes==1073741824 and old.threads==new.threads==1
    with pytest.raises(AttributeError):new.cells=9999
    assert new.record()['name']==OFFLINE

@pytest.mark.parametrize('name',[None,True,{},'unknown','original13-offline-extended-v2'])
def test_unknown_profile_fails_closed(name):
    from tools.opaque_reference_profile import profile
    with pytest.raises(ValueError):profile(name)

@pytest.mark.parametrize('kw',[dict(max_cells=4097),dict(max_depth=15),dict(max_cells=True),dict(max_depth=True),dict(max_cells=0)])
def test_larger_profile_still_refuses_growth_before_field(kw):
    from tools.opaque_square_reference import integrate_square
    def bad(*a):pytest.fail('field called before invalid cap refusal')
    with pytest.raises(ValueError):integrate_square(bad,reference_profile=OFFLINE,**kw)

def test_larger_seed_cover_is_exact_and_charged():
    from tools.opaque_square_reference import integrate_square
    r=integrate_square(lambda x,y:[x*y],reference_profile=OFFLINE,max_cells=512,max_depth=14,seed_shape=(32,16))
    assert r['cells']==r['accepted']==512 and r['max_depth']==9
    assert Q(r['value'][0].lo)<=Q(1,4)<=Q(r['value'][0].hi)
    assert r['reference_profile']['name']==OFFLINE
    assert r['error_accounting']['leaves']==512 and len(r['error_accounting']['largest_leaves'])==5

def test_extended_depth_charges_failed_domains_and_stops():
    from tools.opaque_square_reference import integrate_square
    seen=[]
    def bad(x,y):seen.append(1);raise ValueError('deliberately unresolved domain')
    with pytest.raises(Unresolved,match='depth budget') as e:integrate_square(bad,reference_profile=OFFLINE,max_cells=4096,max_depth=14)
    assert e.value.details['cells']==15 and len(seen)==15

def test_extended_cell_cap_charges_failed_domains_and_stops():
    from tools.opaque_square_reference import integrate_square
    def bad(x,y):raise ValueError('deliberately unresolved domain')
    with pytest.raises(Unresolved,match='cell budget') as e:integrate_square(bad,reference_profile=OFFLINE,max_cells=4,max_depth=14)
    assert e.value.details['cells']==4

@pytest.mark.parametrize('name,limit',[('standard',90),(OFFLINE,900)])
def test_deadline_checks_after_last_field_not_just_between_cells(monkeypatch,name,limit):
    from tools import opaque_square_reference as m
    now=[0];monkeypatch.setattr(m.time,'monotonic',lambda:now[0])
    def field(x,y):now[0]=limit+1;return [x.lift(0)]
    with pytest.raises(Unresolved,match='local deadline'):m.integrate_square(field,reference_profile=name)

@pytest.mark.parametrize('name',['standard',OFFLINE])
def test_profile_does_not_relax_closed_integral_tangent_or_accuracy(name):
    from tools.opaque_square_reference import integrate_square
    r=integrate_square(lambda x,y:[(x+x.lift(1,tangent=1))*x*y**3],reference_profile=name)
    for key,want in [('value',Q(5,24)),('derivative',Q(1,8))]:
        z=r[key][0];assert Q(z.lo)<=want<=Q(z.hi) and z.width<=I('2e-12').lo
    with pytest.raises(ValueError):integrate_square(lambda x,y:[x],reference_profile=name,tolerance='1e-5')

def test_default_caps_and_environment_do_not_select_extended(monkeypatch):
    from tools.opaque_square_reference import integrate_square
    monkeypatch.setenv('GFFX_REFERENCE_PROFILE',OFFLINE)
    with pytest.raises(ValueError):integrate_square(lambda x,y:[x],max_cells=257)
    assert integrate_square(lambda x,y:[x])['reference_profile']['name']=='standard'

@pytest.mark.parametrize('mode',['native','opaque-cuda','campaign-piece','teaching'])
def test_worker_profile_cannot_raise_rendering_or_other_test_limits(mode):
    from tools.opaque_reference_profile import worker_profile
    with pytest.raises(ValueError):worker_profile(mode,OFFLINE)

@pytest.mark.parametrize('mode',['offline-preflight','offline-late-preflight'])
def test_worker_limits_bind_the_single_offline_mode(mode):
    from tools.opaque_reference_profile import worker_profile
    p=worker_profile(mode,OFFLINE)
    assert p.worker_seconds==930 and p.watchdog_seconds==960 and p.host_bytes==1073741824
    with pytest.raises(ValueError):worker_profile(mode,'standard')

@pytest.mark.parametrize('failure',['watchdog','memory','identity','exit'])
def test_supervisor_refuses_resource_and_identity_failures(monkeypatch,tmp_path,failure):
    from tools import material_loss_cost as m
    from tools.opaque_reference_profile import profile
    import json
    p=profile(OFFLINE);monkeypatch.setattr(m,'PROFILE',dict(cooperative_seconds=p.worker_seconds,watchdog_seconds=p.watchdog_seconds,sampled_working_set_limit_bytes=p.host_bytes))
    clock=[0];monkeypatch.setattr(m.time,'perf_counter',lambda:clock[0]);monkeypatch.setattr(m.time,'sleep',lambda _:None)
    output=tmp_path/'report.json';killed=[];closed=[]
    class Child:
        pid=123
        returncode=None
        def poll(self):return self.returncode
        def kill(self):killed.append(True);self.returncode=-1
        def wait(self):return self.returncode
    child=Child()
    def spawn(*a,**k):
        clock[0]=0
        if failure in ('identity','exit'):
            child.returncode=0 if failure=='identity' else 1
            output.write_text(json.dumps(dict(status='passed',worker_pid=999 if failure=='identity' else 123)))
        return child
    monkeypatch.setattr(m.subprocess,'Popen',spawn)
    def read():
        clock[0]=p.watchdog_seconds+1
        return p.host_bytes+1 if failure=='memory' else 100
    monkeypatch.setattr(m,'working_set_reader',lambda pid:(read,lambda:closed.append(True)))
    assert m.supervise(tmp_path,output)==1
    report=json.loads(output.read_text());assert report['status']=='stopped' and closed
    if failure in ('watchdog','memory'):assert killed
    expected={'watchdog':'deadline','memory':'working-set','identity':'identity'}
    if failure!='exit':assert expected[failure] in report['supervisor']['failure']

def test_worker_routes_caps_and_profile_to_supervisor(monkeypatch,tmp_path):
    from tools import opaque_coverage_probe as p
    from tools import material_loss_cost as cost
    from tools.opaque_reference_profile import profile
    import sys
    seen={};monkeypatch.setattr(cost,'PROFILE',dict(cost.PROFILE))
    def run(*a,**kw):seen.update(kw);seen['limits']=dict(cost.PROFILE);return 0
    monkeypatch.setattr(cost,'supervise',run)
    monkeypatch.setattr(sys,'argv',['probe','--fixtures',str(tmp_path),'--output',str(tmp_path/'r.json'),'--mode','offline-preflight','--reference-profile',OFFLINE])
    assert p.main()==0 and seen['limits']['watchdog_seconds']==960 and seen['limits']['cooperative_seconds']==930
    assert seen['worker_args']==('--mode','offline-preflight','--reference-profile',OFFLINE)

def test_full_piece_profile_binding_preserves_original_share(monkeypatch):
    import json,os
    from pathlib import Path
    from tools import opaque_full_material_reference as m
    from tools.opaque_reference_profile import profile
    data=json.loads(Path(os.environ['GFFX_FULL_INPUT']).read_bytes());seen={}
    class Captured(Exception):pass
    def capture(field,**kw):seen.update(kw);raise Captured
    monkeypatch.setattr(m,'integrate_square',capture)
    with pytest.raises(Captured):m.full_pixel(data,selected_piece=8,reference_profile=OFFLINE)
    assert seen['max_cells']==4096 and seen['max_depth']==14 and seen['reference_profile']==OFFLINE
    assert seen['tolerance']==str(m.original_piece_tolerances(data['selected']['visible_parts'])[8])

def test_standard_worker_restores_limits_after_offline_selection(monkeypatch,tmp_path):
    from tools import opaque_coverage_probe as p
    from tools import material_loss_cost as cost
    import sys
    seen=[];monkeypatch.setattr(cost,'PROFILE',dict(cost.PROFILE))
    def run(*a,**kw):seen.append(dict(cost.PROFILE));return 0
    monkeypatch.setattr(cost,'supervise',run)
    for mode,name in [('offline-preflight',OFFLINE),('offline-profile-controls','standard')]:
        monkeypatch.setattr(sys,'argv',['probe','--fixtures',str(tmp_path),'--output',str(tmp_path/'r.json'),'--mode',mode,'--reference-profile',name]);assert p.main()==0
    assert seen[0]['watchdog_seconds']==960
    assert seen[1]['watchdog_seconds']==180 and seen[1]['cooperative_seconds']==120

def test_outer_checkpoint_retains_charged_work_and_complete_cover():
    from tools.opaque_square_reference import integrate_square
    calls=[0]
    def check():
        calls[0]+=1
        if calls[0]==60:raise Unresolved('injected between-cells stop')
    class Correction:
        def __call__(self,box):return [I('-1e-8','1e-8')]
    with pytest.raises(Unresolved) as exc:integrate_square(lambda x,y:[x.lift(0)],checkpoint=check,correction_integrator=Correction())
    assert exc.value.details['cells']>=1


def test_progress_reports_only_complete_cover_and_cannot_change_acceptance():
    from tools.opaque_square_reference import integrate_square
    calls=[]
    def progress(row):
        calls.append(dict(row));row['widths']=[['0','0']];row['cells']=999
    r=integrate_square(lambda x,y:[x*y],seed_shape=(2,2),progress=progress)
    assert len(calls)==1 and calls[0]['cells']==4 and calls[0]['leaves']==4
    assert r['cells']==4 and Q(r['value'][0].lo)<=Q(1,4)<=Q(r['value'][0].hi)
