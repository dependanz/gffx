"""Unnecessary subtraction must not fragment a provably separated piece."""
import torch,os,pytest
from test_opaque_sh_face_capacity import batches

def fixture(include=True):
 dev=os.environ.get('GFFX_CAPACITY_DEVICE','cpu')
 rows=[[-2.,-2.,0.],[6.,-2.,0.],[-2.,6.,0.],[.25,-2.,-1.],[4.,-2.,-1.],[.25,4.,-1.],[.5,.5,-.5],[.9,.1,-.5],[.9,.9,-.5],[.125,.125,0.]]
 v=torch.tensor(rows,dtype=torch.float64,device=dev,requires_grad=True)
 fs=[[0,1,2],[3,4,5],[6,7,8] if include else [9,9,9]]+[[9,9,9]]*30
 f=torch.tensor(fs,dtype=torch.int32,device=dev)
 xy=v[f.long(),:2];lo=xy.amin(1);hi=xy.amax(1)
 assert int(((hi[:,0]>=0)&(lo[:,0]<=1)&(hi[:,1]>=0)&(lo[:,1]<=1)).sum())==33
 return v,f,v.new_empty((33,3,0))

def test_separated_occluder_does_not_fragment_surviving_polygon():
 rows=[]
 for include in (False,True):
  v,f,c=fixture(include);parts=[];rgb=v.new_zeros(())
  for ids,q,r in batches(v,f,c):
   n=r[2];parts.append(int((ids[0,:n]==0).sum()))
   # The front occluder starts at x=.25. Integrate the background area.
   rgb=rgb+q[0,:n,3][ids[0,:n]==0].sum()
  grad,=torch.autograd.grad(rgb,v)
  torch.testing.assert_close(rgb,rgb.new_tensor(.25),atol=1e-13,rtol=0)
  # Translating the occluder's vertical boundary changes area at unit speed.
  torch.testing.assert_close(grad[3:6,0].sum(),rgb.new_tensor(1.),atol=1e-12,rtol=0)
  torch.testing.assert_close(grad[6:],torch.zeros_like(grad[6:]),atol=1e-12,rtol=0)
  rows.append(sum(parts))
 assert rows[0]==rows[1],rows


@pytest.mark.parametrize('apex',[.2,.25,.25000000000000006,.5])
def test_separation_keeps_overlap_contact_and_near_contact(apex):
 from fractions import Fraction as F
 v,fs,c=fixture(True)
 with torch.no_grad():v[6,0]=apex
 value=v.new_zeros(())
 for ids,q,r in batches(v,fs,c):
  n=r[2];value=value+q[0,:n,3][ids[0,:n]==0].sum()
 grad,=torch.autograd.grad(value,v)
 a=F(apex);edge=F(.25);right=F(.9);half=(F(.9)-F(.1))/2
 width=max(F(0),edge-a);cut=half*width*width/(right-a)
 expected=edge-cut
 derivative=half*(2*width*(right-a)-width*width)/(right-a)**2 if width else F(0)
 torch.testing.assert_close(value,value.new_tensor(float(expected)),atol=1e-13,rtol=0)
 torch.testing.assert_close(grad[6,0],value.new_tensor(float(derivative)),atol=1e-12,rtol=0)
 assert torch.isfinite(grad).all()


@pytest.mark.parametrize('apex',[.2,.25,.25000000000000006,.5])
def test_separation_cpu_cuda_bytes_and_first_vjp(apex):
 if os.environ.get('GFFX_CAPACITY_DEVICE')!='cuda':
  pytest.skip('device comparison belongs to CUDA worker')
 saved=[]
 try:
  for dev in ('cpu','cuda'):
   os.environ['GFFX_CAPACITY_DEVICE']=dev;v,f,c=fixture(True)
   with torch.no_grad():v[6,0]=apex
   rows=[]
   for ids,q,r in batches(v,f,c):
    upstream=torch.arange(q.numel(),dtype=q.dtype,device=dev).reshape(q.shape)/4096
    g,=torch.autograd.grad(q,v,upstream)
    rows.append((ids.cpu().numpy().tobytes(),q.detach().cpu().numpy().tobytes(),g.cpu().numpy().tobytes(),tuple(r)))
   saved.append(rows)
 finally:os.environ['GFFX_CAPACITY_DEVICE']='cuda'
 assert saved[0]==saved[1]
