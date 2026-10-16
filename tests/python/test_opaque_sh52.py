"""Installed-wheel controls for explicit SH52 continuation; defaults stay16."""
import os,torch,pytest

def fixture(count=20):
 dev=os.environ.get('GFFX_BATCH_DEVICE','cpu');v=[]
 for i in range(count):
  x=(i%5)/8;y=(i//5)/8;v.extend([[x,y,0.],[x+1/16,y,0.],[x,y+1/16,0.]])
 vertices=torch.tensor(v,dtype=torch.float64,device=dev);faces=torch.arange(count*3,dtype=torch.int32,device=dev).reshape(count,3)
 cuts=vertices.new_tensor([[-k/8,1-k/8,-k/8] for k in range(1,8)]+[[-k/8,-k/8,1-k/8] for k in range(1,8)]).T.expand(count,3,14).clone()
 return vertices,faces,cuts

def test_explicit_profile_exceeds16_batches_and_preserves_area():
 from gffx.torch.render import opaque_material_quadrature_batch as fn
 v,f,c=fixture();cursor=(0,0,0,0);area=v.new_zeros(());calls=0
 for i in range(52):
  ids,q,r=fn(v,f,c,cursor=cursor,rule=2);assert r[2]<=800 and r[3]<=32 and r[4]<=41600 and r[5]<=1664
  area+=q[0,:r[2],3].sum();calls+=1
  if r[1]:break
  cursor=(r[0],i+1,r[4],r[5])
 else:pytest.fail('SH52 did not complete')
 assert calls>16 and abs(area.item()-20/512)<1e-13
 with pytest.raises(ValueError):fn(v,f,c,cursor=(512,16,12800,512),rule=1)

def test_small_profile_preserves_exact_samples_receipts_and_first_vjp():
 from gffx.torch.render import opaque_material_quadrature_batch as fn
 records=[]
 for rule in (1,2):
  v,f,c=fixture(1);v.requires_grad_();c.requires_grad_();ids,q,r=fn(v,f,c,rule=rule)
  g=torch.arange(q.numel(),device=q.device,dtype=q.dtype).reshape(q.shape)/4096
  gradients=torch.autograd.grad(q,(v,c),g)
  records.append([t.detach().cpu().numpy().tobytes() for t in (ids,q,*gradients)]+[r])
 assert records[0]==records[1]

def test_profile_receipts_stop_at52_and_defaults_stop_at16():
 from gffx.torch._opaque_pixel import _advance
 cursor=((1632,51,40800,1632),);last=((1664,1,800,32,41600,1664,32,0),)
 assert _advance(cursor,last,rule=2)==(None,)
 with pytest.raises((OverflowError,RuntimeError,ValueError)):_advance(cursor,last)
 with pytest.raises(OverflowError):_advance(cursor,((1664,0,800,32,41600,1664,32,0),),rule=2)
 with pytest.raises((OverflowError,RuntimeError,ValueError)):_advance(((1664,52,41600,1664),),((1665,1,25,1,41625,1665,1,0),),rule=2)

@pytest.mark.parametrize('profile',[0,17,53,True,52.0])
def test_private_profile_rejects_unapproved_limits(profile):
 from gffx.torch._opaque_sh import _profile
 with pytest.raises(ValueError):_profile({'sh_batch_profile':profile})

def test_profile_selection_is_explicit():
 from gffx.torch._opaque_sh import _profile
 assert _profile({})==(16,1)
 assert _profile({'sh_batch_profile':52})==(52,2)
