"""Keep the complete lighting Jacobian before applying an image-loss cotangent."""
import importlib
import time
import pytest
import torch


def tool():return importlib.import_module('tools.environment_image_adjoint')


@pytest.fixture(autouse=True)
def serial():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_angular_jacobian_keeps_cross_channel_derivatives():
    a=torch.tensor([[.2,.3,.4]],dtype=torch.float64)
    matrices=[a.new_tensor([[1.,2.,0.],[0.,1.,3.],[4.,0.,1.]]),a.new_tensor([[.5,0.,1.],[2.,.5,0.],[0.,3.,.5]])]
    total,jac=tool().sum_jacobians(a,[lambda x,m=m:x@m.T for m in matrices],deadline=time.perf_counter()+10)
    torch.testing.assert_close(total,a@sum(matrices).T)
    torch.testing.assert_close(jac,sum(matrices))
    target=torch.ones_like(total)
    actual=(2*(total-target)/3)@jac
    leaf=a.clone().requires_grad_();value=sum(leaf@m.T for m in matrices)
    expected,=torch.autograd.grad((value-target).square().mean(),leaf)
    torch.testing.assert_close(actual,expected)


@pytest.mark.parametrize('bad',['empty','deadline','dtype','shape'])
def test_angular_jacobian_rejects_invalid_work_before_callback(bad):
    a=torch.tensor([[.2,.3,.4]],dtype=torch.float64)
    def forbidden(x):raise AssertionError('callback reached')
    chunks=[forbidden];deadline=time.perf_counter()+10
    if bad=='empty':chunks=[]
    elif bad=='deadline':deadline=0
    elif bad=='dtype':a=a.float()
    else:a=a.flatten()
    with pytest.raises((ValueError,TimeoutError)):
        tool().sum_jacobians(a,chunks,deadline=deadline)


@pytest.mark.parametrize('model',['classic','principled'])
def test_environment_image_adjoint_matches_two_pass_texture_gradient(model):
    from tools import environment_loss as old
    from gffx.torch.material import PrincipledBudget
    budget=PrincipledBudget.bundled(old.direct.RESOURCE,device='cpu',dtype=torch.float64)
    n=torch.tensor([[.12,.18,1.]],dtype=torch.float64);v=torch.tensor([[.21,.09,1.]],dtype=torch.float64)
    uv=torch.tensor([[.23,.34]],dtype=torch.float64);target=torch.tensor([[.15,.2,.1]],dtype=torch.float64)
    kw=dict(model=model,environment='affine',budget=budget,orders=[8],split=.5,deadline=time.perf_counter()+30,chunk_size=64)
    expected=old.angular_pass(old.direct.texture(),n,v,uv,target,**kw)
    actual=tool().angular_pass(old.direct.texture(),n,v,uv,target,**kw)
    torch.testing.assert_close(actual['values'],expected['values'],atol=1e-14,rtol=1e-12)
    torch.testing.assert_close(actual['gradient'],expected['gradient'],atol=1e-13,rtol=1e-11)
    assert actual['calls']==4 and actual['directions']==256
    assert actual['colour_backwards']==12 and actual['texture_backwards']==1


def test_environment_image_adjoint_rejects_work_before_material(monkeypatch):
    module=tool()
    def forbidden(*args,**kwargs):raise AssertionError('public material reached')
    monkeypatch.setattr(module,'contribution',forbidden)
    n=torch.tensor([[0.,0.,1.]],dtype=torch.float64);uv=torch.tensor([[.2,.3]],dtype=torch.float64)
    with pytest.raises(ValueError):
        module.angular_pass(module.old.direct.texture(),n,n,uv,n,model='classic',environment='uniform',
                            budget=None,orders=[256],split=.5,deadline=time.perf_counter()+10)
