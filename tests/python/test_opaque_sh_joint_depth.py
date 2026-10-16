"""Joint depth/edge certificates remove only provably empty occlusion."""
import os,pytest,torch
from test_opaque_sh_face_capacity import batches

def fixture(include=True,cross=.25,texture=False):
 dev=os.environ.get('GFFX_CAPACITY_DEVICE','cpu')
 v=torch.tensor([[-2.,-2.,0.],[6.,-2.,0.],[-2.,6.,0.],[.5,-2.,.5-cross],[4.,-2.,4.-cross],[.5,4.,.5-cross],[.125,.125,0.]],dtype=torch.float64,device=dev,requires_grad=True)
 f=torch.tensor([[0,1,2],[3,4,5] if include else [6,6,6]]+[[6,6,6]]*31,dtype=torch.int32,device=dev)
 c=v.new_zeros((33,3,int(texture)))
 if texture:
  c[0,:,0]=v.detach()[f[0].long(),1]-.375
  c.requires_grad_()
 xy=v[f.long(),:2];lo=xy.amin(1);hi=xy.amax(1)
 assert int(((hi[:,0]>=0)&(lo[:,0]<=1)&(hi[:,1]>=0)&(lo[:,1]<=1)).sum())==33
 return v,f,c

@pytest.mark.parametrize('texture',[False,True])
def test_joint_depth_nonocclusion_preserves_polygon_and_texture_regions(texture):
 counts=[]
 for include in (False,True):
  v,f,c=fixture(include,texture=texture);value=v.new_zeros(());lower=v.new_zeros(());nslots=0
  for ids,q,r in batches(v,f,c):
   n=r[2];mask=ids[0,:n]==0;nslots+=int(mask.sum());value=value+q[0,:n,3][mask].sum()
   if texture:
    blocks=q[0,:n].reshape(-1,25,4);owner=ids[0,:n].reshape(-1,25)[:,0]
    y=(blocks[:,:,:3]*v[f[0].long(),1]).sum(-1)
    below=(y.mean(1)<.375)&(owner==0)
    # Every selected region stays on its own side of the required texture knot.
    assert torch.all(y[below]<=.375+1e-14)
    assert torch.all(y[(~below)&(owner==0)]>=.375-1e-14)
    lower=lower+blocks[below,:,3].sum()
  g,=torch.autograd.grad(value,v,retain_graph=texture)
  torch.testing.assert_close(value,value.new_tensor(1.),atol=1e-13,rtol=0)
  torch.testing.assert_close(g[3:6],torch.zeros_like(g[3:6]),atol=1e-12,rtol=0)
  if texture:
   torch.testing.assert_close(lower,lower.new_tensor(.375),atol=1e-13,rtol=0)
   dc,=torch.autograd.grad(lower,c)
   torch.testing.assert_close(dc[0,:,0].sum(),lower.new_tensor(-1.),atol=1e-12,rtol=0)
  counts.append(nslots)
 assert counts[0]==counts[1],counts

@pytest.mark.parametrize('cross',[.25,.5-2.**-40,.5+2.**-30,.75])
def test_joint_depth_guard_keeps_real_occlusion_and_gradient(cross):
 v,f,c=fixture(True,cross);value=v.new_zeros(())
 for ids,q,r in batches(v,f,c):
  n=r[2];value=value+q[0,:n,3][ids[0,:n]==0].sum()
 g,=torch.autograd.grad(value,v)
 expected=1.-max(0.,cross-.5)
 torch.testing.assert_close(value,value.new_tensor(expected),atol=1e-13,rtol=0)
 torch.testing.assert_close(g[3:6,2].sum(),value.new_tensor(float(cross>.5)),atol=1e-12,rtol=0)
 assert torch.isfinite(g).all()

@pytest.mark.parametrize('texture',[False,True])
def test_joint_depth_cpu_cuda_bytes_and_first_vjp(texture):
 if os.environ.get('GFFX_CAPACITY_DEVICE')!='cuda':pytest.skip('CUDA worker only')
 saved=[]
 try:
  for dev in ('cpu','cuda'):
   os.environ['GFFX_CAPACITY_DEVICE']=dev;v,f,c=fixture(True,texture=texture);rows=[]
   for ids,q,r in batches(v,f,c):
    upstream=torch.arange(q.numel(),dtype=q.dtype,device=dev).reshape(q.shape)/4096
    gradients=torch.autograd.grad(q,(v,c) if texture else (v,),upstream)
    rows.append((ids.cpu().numpy().tobytes(),q.detach().cpu().numpy().tobytes(),tuple(g.cpu().numpy().tobytes() for g in gradients),tuple(r)))
   saved.append(rows)
 finally:os.environ['GFFX_CAPACITY_DEVICE']='cuda'
 assert saved[0]==saved[1]
