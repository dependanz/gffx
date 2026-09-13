"""Full-image colour fitting must reach source texels, not just rendered pixels."""
import importlib
import math
import time
import pytest
import torch

def ref(): return importlib.import_module('tools.image_loss_reference')
def tool(): return importlib.import_module('tools.image_loss_gradients')

@pytest.fixture(autouse=True)
def serial():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)

def test_bilinear_stencil_repeat_half_texel_and_partition():
    assert ref().stencil(2,2,[.25,.25])==[(0,0,1.),(0,1,0.),(1,0,0.),(1,1,0.)]
    assert ref().stencil(2,2,[1.25,-.75])==ref().stencil(2,2,[.25,.25])
    assert sum(w for _,_,w in ref().stencil(2,2,[0.,0.]))==1.
    for uv in ([float('nan'),0.],[0.],['0',0.]):
        with pytest.raises((ValueError,TypeError)):ref().stencil(2,2,uv)

def test_reference_source_adjoint_and_covered_denominator_known_answer():
    texture=[[[.2,.4,.6],[.3,.5,.7]],[[.4,.6,.8],[.5,.7,.9]]]
    weights=[ref().stencil(2,2,[.25,.25])]
    result=ref().evaluate(texture,weights,[[0.]*3],[[1.]*3],[[0.]*3])
    assert result['loss']==pytest.approx((.2**2+.4**2+.6**2)/3)
    assert result['gradient'][0][0]==pytest.approx([2*.2/3,2*.4/3,2*.6/3])
    assert result['gradient'][1][1]==[0.]*3
    with pytest.raises(ValueError):ref().evaluate(texture,[],[],[],[])

@pytest.mark.parametrize('direction',range(3))
def test_reference_adjoint_matches_both_finite_difference_scales(direction):
    texture=tool().texture().tolist();weights=[ref().stencil(32,32,u) for u in ([.2,.3],[1.,0.],[.6,.9])]
    b=[[.1]*3]*3;s=[[.3,.5,.7]]*3;target=[[.2,.2,.2]]*3
    result=ref().evaluate(texture,weights,b,s,target)
    rows=ref().finite_differences(texture,weights,b,s,target,result['gradient'])
    assert rows[direction]['passed'] and len(rows[direction]['estimates'])==2

@pytest.mark.parametrize('model',['classic','principled'])
def test_streamed_texture_loss_matches_reference_and_second_step(model):
    n=torch.tensor([[.1,.1,1.]]*5,dtype=torch.float64);v=n.clone()
    uv=torch.tensor([[.1,.2],[.3,.4],[.5,.6],[.7,.8],[1.,0.]],dtype=torch.float64)
    result=tool().evaluate(n,v,uv,model,deadline=time.perf_counter()+60)
    assert result['passed'] and result['pixels']==5 and result['calls']==4
    assert result['passes'][1]['loss']<result['passes'][0]['loss']
    assert result['max_points']==4
    assert all(row['passed'] for row in result['finite_differences'])

@pytest.mark.parametrize('failure',['empty','nan','size','domain','model','deadline'])
def test_invalid_fixture_fails_before_public_shader(monkeypatch,failure):
    from gffx.torch import material
    n=torch.tensor([[0.,0.,1.]],dtype=torch.float64);uv=torch.tensor([[.2,.3]],dtype=torch.float64)
    v=n.clone();model='classic';deadline=time.perf_counter()+30
    if failure=='empty':n=n[:0];v=v[:0];uv=uv[:0]
    elif failure=='nan':uv[0,0]=float('nan')
    elif failure=='size':uv=uv.expand(2,2)
    elif failure=='domain':n.zero_()
    elif failure=='model':model='unknown'
    else:deadline=time.perf_counter()-1
    def forbidden(*a,**kw):raise AssertionError('public shader called before validation')
    monkeypatch.setattr(material,'classic',forbidden)
    with pytest.raises((ValueError,TimeoutError)):tool().evaluate(n,v,uv,model,deadline=deadline)

def test_exact_schedule_and_create_only_execution(tmp_path):
    assert tool().schedule()==[('classic',64),('principled',64),('classic',6536),('principled',6536)]
    with pytest.raises(FileExistsError):tool().run(tmp_path,tmp_path,tmp_path)

def test_failed_preflight_never_launches_full_image(monkeypatch,tmp_path):
    calls=[]
    def stopped(root,path,**kwargs):
        calls.append(kwargs['worker_args'])
        tool().write_report(path,dict(status='stopped'))
        return 1
    monkeypatch.setattr(tool(),'supervise',stopped)
    with pytest.raises(RuntimeError,match='no automatic retry'):
        tool().run(tmp_path,tmp_path/'new',tmp_path/'fake.whl')
    assert len(calls)==1 and calls[0][-1]=='64'
    import json
    saved=json.loads((tmp_path/'new/cohort.json').read_text())
    assert saved['status']=='stopped' and len(saved['workers'])==1

@pytest.mark.parametrize('field',['values','gradient','loss','untouched'])
def test_pass_validation_rejects_wrong_measurements(field):
    tex=tool().texture().tolist();weights=[ref().stencil(32,32,[.5/32,.5/32])]
    b=[[0.]*3];s=[[1.]*3];target=[[0.]*3]
    r=ref().evaluate(tex,weights,b,s,target)
    if field=='values':r['values'][0][0]+=.01
    elif field=='gradient':r['gradient'][0][0][0]+=.01
    elif field=='loss':r['loss']+=.01
    else:r['gradient'][31][31][0]=1e-15
    with pytest.raises(ValueError):
        tool().inspect_pass(tex,weights,b,s,target,r['values'],r['gradient'],r['loss'])

def test_saved_image_loss_evidence_rejects_corruption(monkeypatch):
    import os,json,hashlib
    from pathlib import Path
    root=os.environ.get('GFFX_IMAGE_LOSS_ROOT')
    if not root:pytest.skip('explicit saved image-loss cohort required')
    sha=os.environ['GFFX_IMAGE_LOSS_SHA256'];path=Path(root)/'cohort.json'
    result=tool().replay(Path(root),sha)
    assert result['status']==tool().PASSED and len(result['workers'])==4
    raw=path.read_bytes();bad=json.loads(raw);bad['workers'][0]['sha256']='0'*64
    blob=json.dumps(bad).encode()
    with pytest.raises(ValueError):tool().replay(Path(root),hashlib.sha256(blob).hexdigest(),blob=blob)
    # Even a newly hashed wrapper must not make forged measurements valid.
    import copy
    worker_path=Path(root)/'worker-0.json'
    original_worker=json.loads(worker_path.read_text())
    original_read=Path.read_text;original_digest=tool().digest
    for mutation in ('memory','pid','mask','calls','texture','gradient','rgb','loss','fd','budget','wheel'):
        forged=copy.deepcopy(original_worker)
        if mutation=='memory':forged['supervisor']['sampled_peak_working_set_bytes']=1024**3+1
        elif mutation=='pid':forged['worker_pid']+=1
        elif mutation=='mask':forged['indices'][0]+=1
        elif mutation=='calls':forged['result']['calls']+=1
        elif mutation=='texture':forged['result']['passes'][0]['texture'][0][0][0]+=.01
        elif mutation=='gradient':forged['result']['passes'][0]['gradient'][0][0][0]+=.01
        elif mutation=='rgb':forged['result']['passes'][0]['values'][0][0]+=.01
        elif mutation=='loss':forged['result']['passes'][0]['loss']+=.01
        elif mutation=='fd':forged['result']['finite_differences'][0]['analytic']+=.01
        elif mutation=='budget':forged['result']['budget']['admission']='accepted by guess'
        else:forged['producer']['wheel']['sha256']='0'*64
        encoded=json.dumps(forged);forged_sha=hashlib.sha256(encoded.encode()).hexdigest()
        wrapper=json.loads(raw);wrapper['workers'][0]['sha256']=forged_sha
        blob=json.dumps(wrapper).encode()
        with monkeypatch.context() as patch:
            patch.setattr(Path,'read_text',lambda path,*a,**kw:encoded if path==worker_path else original_read(path,*a,**kw))
            patch.setattr(tool(),'digest',lambda path:forged_sha if path==worker_path else original_digest(path))
            with pytest.raises(ValueError):tool().replay(Path(root),hashlib.sha256(blob).hexdigest(),blob=blob)
