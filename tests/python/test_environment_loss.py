"""Environment objectives need the complete radiance before their cotangent."""
import importlib
import time
import pytest
import torch


def tool(): return importlib.import_module('tools.environment_loss')


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_angular_loss_cotangent_is_not_a_sum_of_chunk_losses():
    x=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    pieces=[x,3*x];target=torch.tensor(1.,dtype=torch.float64)
    total=sum(pieces)
    correct,=torch.autograd.grad((total-target).square()/3,x)
    streamed=tool().streamed_scalar_example(2.,1.,[1.,3.])
    assert streamed==pytest.approx(float(correct))
    wrong=sum(2*(a*2-1)*a/3 for a in (1.,3.))
    assert streamed!=pytest.approx(wrong)


@pytest.mark.parametrize('environment',['uniform','affine'])
def test_environment_colour_slope_matches_classic_closed_form(environment):
    from tools import albedo_environment as base
    n=[0.,0.,1.];v=[0.,0.,1.]
    slope=tool().slope(n,v,'classic',environment,None)
    assert slope==pytest.approx([1+2*b[2]/3 for b in base.bands(environment)])


@pytest.mark.parametrize('model',['classic','principled'])
def test_streamed_native_texture_gradient_matches_small_complete_graph(model):
    from gffx.torch import material
    tex=tool().direct.texture()
    n=torch.tensor([[0.,0.,1.]],dtype=torch.float64);v=n.clone()
    uv=torch.tensor([[.23,.34]],dtype=torch.float64)
    target=torch.tensor([[.15,.2,.1]],dtype=torch.float64)
    budget=material.PrincipledBudget.bundled(tool().direct.RESOURCE,device='cpu',dtype=torch.float64) if model=='principled' else None
    args=dict(model=model,environment='affine',budget=budget,orders=[8],split=.5,deadline=time.perf_counter()+30,chunk_size=64)
    actual=tool().angular_pass(tex,n,v,uv,target,**args)
    leaf=tex.clone().requires_grad_(True)
    from tools import affine_environment as old
    q,w=old.quadrature(8,.5)
    rgb=tool().contribution(leaf,n,v,uv,q,w,model,'affine',budget)
    expected,=torch.autograd.grad((rgb-target).square().sum()/3,leaf)
    assert torch.allclose(actual['gradient'],expected,atol=1e-13,rtol=1e-11)
    assert torch.allclose(actual['values'],rgb.detach(),atol=1e-14,rtol=1e-12)
    assert actual['calls']==8 and actual['directions']==512


@pytest.mark.parametrize('bad',['unknown','count','deadline','order'])
def test_environment_preflight_rejects_invalid_work_before_shader(monkeypatch,bad):
    n=torch.tensor([[0.,0.,1.]],dtype=torch.float64);uv=torch.tensor([[.2,.3]],dtype=torch.float64)
    kwargs=dict(model='classic',environment='uniform',budget=None,orders=[8],split=.5,deadline=time.perf_counter()+10)
    if bad=='unknown':kwargs['environment']='guess'
    elif bad=='count':kwargs['orders']=[8,8]
    elif bad=='deadline':kwargs['deadline']=time.perf_counter()-1
    else:kwargs['orders']=[256]
    def forbidden(*a,**k):raise AssertionError('shader reached')
    monkeypatch.setattr(tool(),'contribution',forbidden)
    with pytest.raises((ValueError,TimeoutError)):
        tool().angular_pass(tool().direct.texture(),n,n,uv,torch.zeros_like(n),**kwargs)


def test_environment_schedule_and_existing_root_fail_closed(tmp_path):
    assert tool().schedule()==[('classic','uniform'),('classic','affine'),('principled','uniform'),('principled','affine')]
    with pytest.raises(FileExistsError):tool().run(tmp_path,tmp_path,tmp_path)


def test_environment_failed_preflight_preserves_stop_and_prevents_next_profile(monkeypatch,tmp_path):
    calls=[]
    def stop(root,path,**kwargs):
        calls.append(path);tool().write_report(path,dict(status='stopped'));return 1
    monkeypatch.setattr(tool(),'supervise',stop)
    with pytest.raises(RuntimeError,match='no automatic retry'):tool().run(tmp_path,tmp_path/'new',tmp_path/'fake.whl')
    assert len(calls)==1
    import json
    result=json.loads((tmp_path/'new/cohort.json').read_text())
    assert result['status']=='stopped' and len(result['workers'])==1


def test_saved_environment_loss_requires_explicit_verified_evidence(monkeypatch):
    import os
    from pathlib import Path
    root=os.environ.get('GFFX_ENV_LOSS_ROOT')
    if not root:pytest.skip('explicit environment adjoint cohort required')
    result=tool().replay(Path(root),os.environ['GFFX_ENV_LOSS_SHA256'])
    assert result['status']==tool().PASSED and len(result['workers'])==4
    with pytest.raises(ValueError):tool().replay(Path(root),'0'*64)
    import json,copy,hashlib
    path=Path(root)/'worker-0.json';saved=json.loads(path.read_text())
    original_read=Path.read_text;original_digest=tool().direct.digest
    for mutation in ('pixel','memory','pid','calls','target','gradient','loss','offset','step','fd'):
        worker=copy.deepcopy(saved)
        if mutation=='pixel':worker['indices'][0]+=1
        elif mutation=='memory':worker['supervisor']['sampled_peak_working_set_bytes']=1024**3+1
        elif mutation=='pid':worker['worker_pid']+=1
        elif mutation=='calls':worker['result']['calls']+=1
        elif mutation=='target':worker['result']['inputs']['target'][0][0]+=.1
        elif mutation=='gradient':worker['result']['passes'][0]['offsets'][0]['gradient'][0][0][0]+=.1
        elif mutation=='loss':worker['result']['passes'][0]['offsets'][0]['loss']+=.1
        elif mutation=='offset':worker['result']['passes'][0]['offsets'][0]['split']=.125
        elif mutation=='step':worker['result']['passes'][1]['texture'][0][0][0]+=.01
        else:worker['result']['finite_differences'][0]['analytic']+=.1
        text=json.dumps(worker);sha=hashlib.sha256(text.encode()).hexdigest()
        wrapper=copy.deepcopy(result);wrapper['workers'][0]['sha256']=sha
        blob=json.dumps(wrapper).encode()
        with monkeypatch.context() as p:
            p.setattr(Path,'read_text',lambda current,*a,**kw:text if current==path else original_read(current,*a,**kw))
            p.setattr(tool().direct,'digest',lambda current:sha if current==path else original_digest(current))
            with pytest.raises(ValueError):tool().replay(Path(root),hashlib.sha256(blob).hexdigest(),blob=blob)
