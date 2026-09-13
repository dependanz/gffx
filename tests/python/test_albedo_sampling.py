"""Public multi-point lighting sampler correctness, limits and accounting."""
import copy
import importlib
import os
from pathlib import Path
import time
import pytest
import torch

def tool():return importlib.import_module('tools.albedo_sampling')

@pytest.fixture(autouse=True)
def serial():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)

@pytest.fixture
def saved():
    from tools import albedo_environment as base
    from gffx.torch.material import PrincipledBudget
    budget=PrincipledBudget.bundled(base.old.RESOURCE,device='cpu',dtype=torch.float64)
    curve=base.old.PreparedCurve(base.old.LossCurve(budget.to_envelope()['payload']),roughness=1.)
    def make(model,env,size):
        deadline=time.perf_counter()+30;prepared=base.Prepared(model,curve,env,deadline=deadline);rows=[]
        for i in range(size):
            n=[.03*i,0.,1.];v=[.2,.05*i,1.];a=[.2+.02*i,.4,.6]
            ref=prepared.point(n,v,a,deadline=deadline)
            actual=base.candidate(n,v,a,model=model,environment=env,budget=budget,ref=ref,deadline=deadline)
            rows.append(dict(point=dict(flat_index=i,normal=n,view=v),albedo=a,reference=ref,actual=actual))
        return rows,budget
    return make

@pytest.mark.parametrize('model,env',[('classic','uniform'),('classic','affine'),('principled','uniform'),('principled','affine')])
def test_batched_public_sampling_matches_independent_and_saved_outputs(saved,model,env):
    rows,budget=saved(model,env,3)
    result=tool().sample(rows,model=model,environment=env,budget=budget,deadline=time.perf_counter()+30)
    tool().validate(result,rows)
    assert result['directions']==sum(r['actual']['directions'] for r in rows)
    assert result['max_pairs']<=1024 and result['max_points']<=8

@pytest.mark.parametrize('model',['classic','principled'])
def test_sampling_preserves_partial_groups_and_rejects_bad_input_before_calls(saved,monkeypatch,model):
    rows,budget=saved(model,'affine',1);rows=[copy.deepcopy(rows[0]) for _ in range(9)]
    from gffx.torch import material
    original=getattr(material,model);calls=[]
    def observed(n,v,d,**kwargs):
        calls.append((n.shape[0],d.shape[1]));assert n.shape[0]<=8 and n.shape[0]*d.shape[1]<=1024
        return original(n,v,d,**kwargs)
    monkeypatch.setattr(material,model,observed)
    result=tool().sample(rows,model=model,environment='affine',budget=budget,deadline=time.perf_counter()+30)
    tool().validate(result,rows)
    assert {p for p,q in calls}=={8,1} and result['calls']==len(calls)
    count=len(calls);bad=copy.deepcopy(rows);bad[-1]['albedo']=[-1.,0.,0.]
    with pytest.raises(ValueError):tool().sample(bad,model=model,environment='affine',budget=budget,deadline=time.perf_counter()+30)
    assert len(calls)==count
    with pytest.raises(TimeoutError):tool().sample(rows,model=model,environment='affine',budget=budget,deadline=time.perf_counter()-1)

def test_sampling_groups_keep_quadrature_orders_separate():
    rows=[dict(reference=dict(history=[dict(n_mu=n)])) for n in (64,32,64,128,32)]
    result=tool().groups(rows,8)
    assert [(g['order'],g['indices']) for g in result]==[(32,[1,4]),(64,[0,2]),(128,[3])]
    for width in (0,9,True,1.5):
        with pytest.raises(ValueError):tool().groups(rows,width)

def test_sampling_validation_rejects_forged_work_and_pixels(saved):
    rows,budget=saved('classic','uniform',2)
    result=tool().sample(rows,model='classic',environment='uniform',budget=budget,deadline=time.perf_counter()+30)
    for key,value in [('calls',0),('directions',0),('max_pairs',1025)]:
        bad=copy.deepcopy(result);bad[key]=value
        with pytest.raises(ValueError):tool().validate(bad,rows)
    bad=copy.deepcopy(result);bad['estimates'][0][0][0]=float('nan')
    with pytest.raises(ValueError):tool().validate(bad,rows)

def test_sampling_screen_schedule_and_history_guard(tmp_path):
    assert len(tool().PARENT_HASHES)==6
    assert tool().PROFILE['pixels']==72 and tool().PROFILE['maximum_directions']==75497472
    with pytest.raises(FileExistsError):tool().run(tmp_path,tmp_path,tmp_path,tmp_path/'wheel')

def test_saved_sampling_screen_replays_all_profiles():
    root=os.environ.get('GFFX_ALBEDO_SAMPLING');sha=os.environ.get('GFFX_ALBEDO_SAMPLING_SHA256')
    if not root or not sha:pytest.skip('explicit completed sampling screen required')
    result=tool().replay(Path(root),sha,Path(os.environ['GFFX_ALBEDO_TILES']),Path(os.environ['GFFX_FULL_ENV_CORPUS']))
    assert len(result['workers'])==4

def test_saved_sampling_replay_rejects_changed_parent_timing_and_supervision(monkeypatch):
    import hashlib,json
    root=os.environ.get('GFFX_ALBEDO_SAMPLING')
    if not root:pytest.skip('explicit completed sampling screen required')
    path=Path(root)/'classic-uniform.json';row=json.loads(path.read_text())
    expected=tool().load_rows(Path(os.environ['GFFX_ALBEDO_TILES']),Path(os.environ['GFFX_FULL_ENV_CORPUS']),'classic','uniform')
    for mutation in ('parent','rows','order','time','decision','memory','failure'):
        bad=copy.deepcopy(row)
        if mutation=='parent':bad['parent_hashes'][0]='0'*64
        elif mutation=='rows':bad['rows'].reverse()
        elif mutation=='order':bad['passes'].reverse()
        elif mutation=='time':bad['passes'][1]['duration_seconds']=-1.
        elif mutation=='decision':bad['prefer_batched']=not bad['prefer_batched']
        elif mutation=='memory':bad['supervisor']['sampled_peak_working_set_bytes']=1024**3+1
        else:bad['supervisor']['failure']='injected'
        blob=json.dumps(bad).encode()
        with monkeypatch.context() as m:
            m.setattr(tool(),'_read',lambda *_:blob)
            with pytest.raises(ValueError):tool().read_worker(path,hashlib.sha256(blob).hexdigest(),expected)
