"""Installed public shading chains with the accepted budget and native mip maps."""
import math
import pytest
import torch


@pytest.fixture(autouse=True)
def serial():
    prior=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


@pytest.fixture(scope='module')
def budget():
    from gffx.torch.material import PrincipledBudget
    return PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64)


def fields():
    def t(x):return torch.tensor(x,dtype=torch.float64,requires_grad=True)
    return (t([[[.2,.3,.4],[.3,.4,.5]],[[.4,.5,.6],[.5,.6,.7]]]),
        t([[.35,.45]]),t([[.12,.18,1.]]),t([[.21,.09,1.]]),
        t([[.3,.4,1.],[-.2,.1,1.]]),t(.63),t(.09),t([[.7,.8,.9],[.4,.5,.6]]))


def evaluate(values,model,budget):
    from gffx.torch import material,render,light
    tex,uv,n,v,d,r,f,incoming=values
    pyramid,offsets=render.texture_pyramid(tex.contiguous(),levels=2)
    colour=render.texture(pyramid,offsets,2,2,uv.contiguous(),channel_count=3,
        lod=uv.new_tensor([.35]),filter=render.FILTER_BILINEAR,mip_filter=render.MIP_LINEAR)
    if model=='principled':
        brdf=material.principled(n,v,d,albedo=colour,roughness=r,f0=f,budget=budget,
            allow_experimental_budget=budget.metadata['admission']=='experimental')
    else:
        brdf=material.classic(n,v,d,albedo=colour,specular=f,exponent=r*4)
    packet=dict(measure='solid_angle',direction=d[None],radiance=incoming[None],
        valid=torch.ones((1,2),dtype=torch.bool,device=n.device),solid_angle=n.new_tensor([[.6,.4]]))
    return light.integrate(n,brdf,packet)


@pytest.mark.parametrize('model',['classic','principled'])
def test_complete_shading_mip_chain_all_input_gradcheck(model,budget):
    args=fields()
    assert torch.autograd.gradcheck(lambda *xs:evaluate(xs,model,budget),args,eps=1e-6,atol=1e-5,rtol=1e-3)
    grads=torch.autograd.grad(evaluate(args,model,budget).sum(),args)
    assert all(bool(torch.isfinite(g).all()) and bool((g!=0).any()) for g in grads)


@pytest.mark.parametrize('model',['classic','principled'])
def test_two_shading_steps_rebuild_mip_graph_and_reduce_loss(model,budget):
    args=fields();before=evaluate(args,model,budget).square().sum()
    gradient,=torch.autograd.grad(before,args[0])
    with torch.no_grad():args[0].sub_(.01*gradient)
    after=evaluate(args,model,budget).square().sum()
    again,=torch.autograd.grad(after,args[0])
    assert after<before and bool(torch.isfinite(again).all())


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_classic_and_sh_cuda_values_and_gradients_match_cpu(dtype):
    if not torch.cuda.is_available():pytest.skip('explicit CUDA lane required')
    from gffx.torch import light,material
    def run(device):
        def t(x):return torch.tensor(x,device=device,dtype=dtype,requires_grad=True)
        n=t([[.2,.1,1.],[-.1,.2,1.]]);v=t([[0.,.1,1.],[.1,0.,1.]])
        d=t([[.4,.3,1.]]);a=t([.2,.4,.6]);s=t(.1);e=t(2.)
        coefficients=torch.arange(27,device=device,dtype=dtype).reshape(9,3)/100
        coefficients[0]=math.sqrt(4*math.pi);coefficients.requires_grad_()
        rgb=material.classic(n,v,d,albedo=a,specular=s,exponent=e)
        sh=light.spherical_harmonics(n,coefficients)
        loss=rgb.square().sum()+sh.square().sum()
        return [rgb.detach().cpu(),sh.detach().cpu(),*[g.detach().cpu() for g in torch.autograd.grad(loss,(n,v,d,a,s,e,coefficients))]]
    cpu,gpu=run('cpu'),run('cuda')
    for expected,actual in zip(cpu,gpu):
        torch.testing.assert_close(actual,expected,atol=1e-5 if dtype==torch.float32 else 1e-12,
                                   rtol=1e-5 if dtype==torch.float32 else 1e-12)


@pytest.mark.parametrize('model',['classic','principled'])
def test_cuda_mip_texture_shading_chain_matches_cpu_all_partials(model,budget):
    if not torch.cuda.is_available():pytest.skip('explicit CUDA lane required')
    gpu_budget=budget.to(device='cuda:0',dtype=torch.float64,allow_unverified=True)
    cpu=fields();gpu=tuple(x.detach().to('cuda:0').requires_grad_() for x in cpu)
    expected=evaluate(cpu,model,budget);actual=evaluate(gpu,model,gpu_budget)
    cpu_grads=torch.autograd.grad(expected.square().sum(),cpu)
    gpu_grads=torch.autograd.grad(actual.square().sum(),gpu)
    for a,b in zip((expected,*cpu_grads),(actual,*gpu_grads)):
        torch.testing.assert_close(b.cpu(),a,atol=1e-12,rtol=1e-10)
