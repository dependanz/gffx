"""Strict off-axis light integration and independent first-derivative controls."""
import pytest,torch

@pytest.mark.parametrize('samples',[1,3,7])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_light_integrate_nondyadic_values_and_all_first_vjps(dtype,samples):
    from gffx.torch import light
    p=33
    n=torch.arange(1,p*3+1,dtype=dtype).reshape(p,3)*.017+torch.tensor([.013,.127,.031],dtype=dtype)
    d=torch.arange(1,p*samples*3+1,dtype=dtype).reshape(p,samples,3)*.003+torch.tensor([.217,.131,.533],dtype=dtype)
    b=torch.arange(p*samples*3,dtype=dtype).reshape(p,samples,3).remainder(11)*.017+.08
    incoming=b*.3+.4;cot=torch.arange(p*3,dtype=dtype).reshape(p,3).remainder(7)*.125-.375
    lanes=[]
    for device in ('cpu','cuda'):
        leaves=[z.to(device).detach().requires_grad_() for z in (n,d,b,incoming)]
        nn,dd,bb,ii=leaves
        value=light.integrate(nn,bb,{'measure':'delta','direction':dd,'irradiance':ii,'valid':torch.ones((p,samples),dtype=torch.bool,device=device)})
        grad=torch.autograd.grad(value,leaves,cot.to(device));lanes.append([z.detach().cpu().contiguous() for z in (value,*grad)])
    for cpu,gpu in zip(*lanes):assert torch.equal(cpu.reshape(-1).view(torch.uint8),gpu.reshape(-1).view(torch.uint8))
    # Independent normalized-dot value and analytic VJP, without runtime helpers.
    ln=n.square().sum(-1,keepdim=True).sqrt();ld=d.square().sum(-1,keepdim=True).sqrt();nh=n/ln;dh=d/ld
    cosine=(nh[:,None]*dh).sum(-1);weight=(cot[:,None]*b*incoming).sum(-1)
    gn=(weight[...,None]*(dh-cosine[...,None]*nh[:,None])/ln[:,None]).sum(1)
    gd=weight[...,None]*(nh[:,None]-cosine[...,None]*dh)/ld
    expected=[(b*incoming*cosine[...,None]).sum(1),gn,gd,cot[:,None]*incoming*cosine[...,None],cot[:,None]*b*cosine[...,None]]
    for actual,reference in zip(lanes[0],expected):torch.testing.assert_close(actual,reference,atol=2e-6 if dtype==torch.float32 else 1e-10,rtol=1e-5 if dtype==torch.float32 else 1e-9)
