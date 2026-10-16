"""Installed public native continuation: exact area, motion and device ownership."""
import os,math,torch,pytest

def fixture(device=None):
    device=device or os.environ.get('GFFX_BATCH_DEVICE','cpu');n=32
    v=[[.5,.5,0.]]+[[.5+2*math.cos(2*math.pi*i/n),.5+2*math.sin(2*math.pi*i/n),0.] for i in range(n)]
    f=[[0,i+1,(i+1)%n+1] for i in range(n)]
    return torch.tensor(v,dtype=torch.float64,device=device),torch.tensor(f,dtype=torch.int32,device=device)

def function():
    from gffx.torch.render import opaque_material_quadrature_batch
    return opaque_material_quadrature_batch

def batches(v,f,c=None,curves=None):
    cursor=(0,0,0,0)
    if c is None:c=v.new_empty((len(f),3,0))
    for i in range(16):
        ids,q,r=function()(v,f,c,cursor=cursor,curves=curves)
        assert ids.device==v.device and q.device==v.device
        assert len(r)==8 and 0<=r[2]<=800 and 0<=r[3]<=32 and r[4]<=12800 and r[5]<=512
        yield ids,q,r
        if r[1]:return
        cursor=(r[0],i+1,r[4],r[5])
    pytest.fail('incomplete pixel')

@pytest.mark.parametrize('scale',[1.,1e-20,-1e-20])
def test_public_batch_area_and_material_translation(scale):
    v,f=fixture();colour=v[:,0].clone();shift=v.new_tensor(0.,requires_grad=True);moving=v+torch.stack((shift,shift*0,shift*0))
    value=v.new_zeros(());area=v.new_zeros(());count=0
    for ids,q,r in batches(moving,f):
        n=r[2];col=colour[f[ids[0,:n].long()].long()];sample=(q[0,:n,:3]*col).sum(-1)
        value=value+(sample*q[0,:n,3]*scale).sum();area=area+q[0,:n,3].sum();count+=1
    derivative,=torch.autograd.grad(value,shift)
    assert count>1 and abs(area.item()-1)<1e-12 and abs(value.item()/scale-.5)<1e-12 and abs(derivative.item()/scale+1)<2e-12

def test_public_batch_prefix_budget_and_first_only_guards():
    v,f=fixture();v.requires_grad_();c=v.new_empty((len(f),3,0));fn=function();ids,q,r=fn(v,f,c)
    with pytest.raises((OverflowError,ValueError)):fn(v,f,c,cursor=(r[0],1,r[4]-1,r[5]))
    with pytest.raises(ValueError):fn(v,f,c,workspace_limit_bytes=1)
    with pytest.raises(ValueError):fn(v,f,c,cursor=(0,16,0,0))
    with pytest.raises(RuntimeError,match='first'):torch.autograd.grad(q.sum(),v,create_graph=True)

def test_public_batch_input_mutation_is_detected_before_backward():
    v,f=fixture();v.requires_grad_();_,q,_=function()(v,f,v.new_empty((len(f),3,0)))
    with torch.no_grad():v[0,0]+=.001
    with pytest.raises(RuntimeError,match='modified'):q.sum().backward()

def test_public_batch_cpu_cuda_bytes_and_first_derivatives():
    saved=[]
    for dev in ('cpu','cuda'):
        v,f=fixture(dev);v.requires_grad_();parts=[]
        for ids,q,r in batches(v,f):
            upstream=torch.arange(q.numel(),dtype=q.dtype,device=q.device).reshape(q.shape)/4096
            g,=torch.autograd.grad(q,v,upstream)
            parts.append((ids.detach().cpu().numpy().tobytes(),q.detach().cpu().numpy().tobytes(),g.detach().cpu().numpy().tobytes(),r))
        saved.append(parts)
    assert saved[0]==saved[1]


def test_public_batch_curved_area_and_motion_have_closed_reference(record_property):
    from decimal import Decimal,localcontext
    import json
    dev=os.environ.get('GFFX_BATCH_DEVICE','cpu');vertices=[]
    for i in range(7):
        x=(i%3)*.25+.0625;y=(i//3)*.25+.0625;vertices.extend([[x,y,0],[x+.125,y,0],[x,y+.125,0]])
    v=torch.tensor(vertices,dtype=torch.float64,device=dev);f=torch.arange(21,dtype=torch.int32,device=dev).reshape(7,3)
    curves=v.new_tensor([-.25,.75,.65,.5,.5,1.5]).expand(7,6).clone().requires_grad_()
    area=v.new_zeros(());full=v.new_zeros(());count=0
    for ids,q,r in batches(v,f,curves=curves):
        n=r[2];beta=q[0,:n,:3];side=(beta[:,1]+beta[:,2]-.25-.1*beta[:,2]**2)>0
        area=area+(q[0,:n,3]*side).sum();full=full+q[0,:n,3].sum();count+=1
    g,=torch.autograd.grad(area,curves);ga=(g*v.new_tensor([-1,-1,-1,-2,-2,-2])).sum();gb=-g[:,2].sum()
    with localcontext() as context:
        context.prec=80;a=Decimal('.25');b=Decimal('.1');root=2*a/(1+(1-4*a*b).sqrt());det=Decimal(7)/64
        expected=float(det*(Decimal('.5')-a*root+root**2/2-b*root**3/3));ea=float(-det*root);eb=float(-det*root**3/3)
    assert count==4 and abs(full.item()-7/128)<1e-13
    assert abs(area.item()-expected)<1e-12 and abs(ga.item()-ea)<1e-12 and abs(gb.item()-eb)<1e-12
    record_property('curved_batch',json.dumps(dict(device=dev,area=area.item(),full=full.item(),da=ga.item(),db=gb.item(),batches=count)))


@pytest.mark.parametrize('raw_rule',[False,True])
@pytest.mark.parametrize('field,expected',[('constant',(0.,0.)),('x',(-1.,0.)),('y',(0.,-1.))])
def test_public_batch_translation_preserves_closed_area_and_both_moments(field,expected,raw_rule):
    v,f=fixture();v.requires_grad_();colour=torch.ones_like(v[:,0]) if field=='constant' else v[:,0 if field=='x' else 1].detach().clone();value=v.new_zeros(())
    for ids,q,r in batches(v,f,v.new_zeros((len(f),3,1)) if raw_rule else None):
        sample=(q[0,:,:3]*colour[f[ids[0].clamp_min(0).long()].long()]).sum(-1);value=value+(sample*q[0,:,3]).sum()
    g,=torch.autograd.grad(value,v)
    torch.testing.assert_close(g[:,:2].sum(0),v.new_tensor(expected),atol=2e-12,rtol=0)
