import pytest,torch
from test_opaque_sh52 import fixture

@pytest.mark.parametrize('mask',[1,2,3])
def test_prefix_exact_serial_gradients(mask):
 from gffx.torch._opaque_partition import Pool
 from gffx.torch.render import opaque_material_quadrature_batch as ordinary
 pool=Pool(8*1024**2);v,f,c=fixture(2);v=v.contiguous();f=f.contiguous();c=c.contiguous();cursor=(0,0,0,0)
 for batch in range(52):
  results=[]
  for cached in (False,True):
   a=v.detach().requires_grad_(bool(mask&1));b=c.detach().requires_grad_(bool(mask&2))
   ids,q,r=pool.emit(a,f,b,(0.,0.),2,cursor,'p') if cached else ordinary(a,f,b,cursor=cursor,rule=2)
   cotangent=torch.sin(torch.arange(q.numel(),dtype=q.dtype)).reshape(q.shape)
   gs=torch.autograd.grad(q,[x for x in (a,b) if x.requires_grad],cotangent)
   results.append(([x.detach().numpy().tobytes() for x in (ids,q,*gs)],tuple(r[:7])))
  assert results[0]==results[1]
  if r[1]:break
  cursor=(r[0],batch+1,r[4],r[5])
 assert batch>0 and pool.entries['p'].prefix[0]>0

@pytest.mark.parametrize('change',['vertices','faces','cuts','order','mask','storage'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_prefix_owner_refuses_changed_binding(change,device):
 from gffx.torch._opaque_partition import Partition
 v,f,c=[x.to(device).contiguous() for x in fixture(2)];p=Partition(v,f,c,(0.,0.),2,8*1024**2)
 ids,q,r=p.emit(v,f,c,(0,0,0,0),2,(0.,0.));p.backward(v,f,c,torch.ones_like(q),(0,0,0,0),r,1,4*1024**2)
 cursor=(r[0],1,r[4],r[5]);ids,q,r=p.emit(v,f,c,cursor,2,(0.,0.));mask=1
 if change=='vertices':v=v.clone();v[0,0]+=2**-30
 if change=='faces':f=f.flip(0).contiguous()
 if change=='cuts':c=c.clone();c[0,0,0]+=2**-30
 if change=='order':cursor=(0,0,0,0)
 if change=='mask':mask=3
 if change=='storage':p.rows.add_(1.)
 with pytest.raises(ValueError):p.backward(v,f,c,torch.ones_like(q),cursor,r,mask,4*1024**2)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_prefix_failed_attempt_poisoned(monkeypatch,device):
 from gffx.torch._opaque_partition import Partition
 v,f,c=[x.to(device).contiguous() for x in fixture(1)];p=Partition(v,f,c,(0.,0.),2,8*1024**2);_,q,r=p.emit(v,f,c,(0,0,0,0),2,(0.,0.))
 def refuse(*a):raise RuntimeError('injected failure')
 monkeypatch.setattr(torch.ops.gffx,'opaque_material_prefix_backward',refuse)
 with pytest.raises(RuntimeError):p.backward(v,f,c,torch.ones_like(q),(0,0,0,0),r,1,4*1024**2)
 with pytest.raises(ValueError,match='failed'):p.check_inputs(v,f,c)

@pytest.mark.parametrize('case',['analytical','saved','empty'])
def test_cuda_persistent_partition_exact(case):
 import numpy as np,hashlib
 from pathlib import Path
 from gffx.torch._opaque_partition import Partition
 origin=(0.,0.)
 if case=='analytical':v,f,c=[x.contiguous() for x in fixture(2)]
 elif case=='empty':v=torch.empty((0,3),dtype=torch.float64);f=torch.empty((0,3),dtype=torch.int32);c=torch.empty((0,3,0),dtype=torch.float64)
 else:
  path=Path('D:/devspace/gffit/qualification-20261004/boundary-sh12/full-partition1.npz')
  assert hashlib.sha256(path.read_bytes()).hexdigest()=='a72226e0c94a008d22c24efbb6bfb088477ddbeb633cac5205fd66ad3a32f21a'
  with np.load(path) as z:v,f,c=[torch.from_numpy(z[k].copy()) for k in ('vertices','faces','cuts')];origin=tuple(z['origin'].tolist())
 caches=[Partition(v.to(device),f.to(device),c.to(device),origin,2,8*1024**2) for device in ('cpu','cuda')]
 assert caches[0].count==caches[1].count and caches[0].build_work==caches[1].build_work
 assert caches[0].rows[:caches[0].count].numpy().tobytes()==caches[1].rows[:caches[1].count].cpu().numpy().tobytes()
 cursor=(0,0,0,0)
 for batch in range(52):
  results=[cache.emit(*cache.inputs,cursor,2,origin) for cache in caches]
  for i in (0,1):assert results[0][i].numpy().tobytes()==results[1][i].cpu().numpy().tobytes()
  assert results[0][2]==results[1][2];r=results[0][2]
  if r[1]:break
  cursor=(r[0],batch+1,r[4],r[5])
 if case=='saved':assert batch==51 and r[4:6]==(40925,1637)
 with pytest.raises(ValueError):caches[1].emit(*caches[1].inputs,cursor,2,origin)

@pytest.mark.parametrize('mask',[1,2,3])
def test_cuda_persistent_prefix_gradient_bytes(mask):
 from gffx.torch._opaque_partition import Partition
 from gffx.torch.render import opaque_material_quadrature_batch as ordinary
 v,f,c=[x.contiguous() for x in fixture(2)]
 owners=[Partition(v.to(d),f.to(d),c.to(d),(0.,0.),2,8*1024**2) for d in ('cpu','cuda')]
 cursor=(0,0,0,0)
 for batch in range(52):
  results=[]
  g=torch.sin(torch.arange(3200,dtype=torch.float64)).reshape(1,800,4)
  for owner in owners:
   v,f,c=owner.inputs;ids,q,r=owner.emit(v,f,c,cursor,2,(0.,0.))
   gv,gc=owner.backward(v,f,c,g.to(v.device),cursor,r,mask,4*1024**2)
   results.append(([x.cpu().numpy().tobytes() for x in (ids,q,gv,gc)],owner.prefix))
  assert results[0]==results[1]
  if r[1]:break
  cursor=(r[0],batch+1,r[4],r[5])
 assert batch>0 and owners[0].prefix[0]>0

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_prefix_empty_and_invalid_metadata(device):
 from gffx.torch._opaque_partition import Partition
 v=torch.empty((0,3),dtype=torch.float64,device=device);f=torch.empty((0,3),dtype=torch.int32,device=device);c=torch.empty((0,3,0),dtype=torch.float64,device=device)
 p=Partition(v,f,c,(0.,0.),2,8*1024**2);_,q,r=p.emit(v,f,c,(0,0,0,0),2,(0.,0.))
 gv,gc=p.backward(v,f,c,torch.ones_like(q),(0,0,0,0),r,3,4*1024**2)
 assert gv.numel()==gc.numel()==0 and p.prefix==(0,0)
 with pytest.raises(RuntimeError):torch.ops.gffx.opaque_material_prefix_backward(v,f,c,torch.ones_like(q),0.,0.,0,0,0,0,4*1024**2,3,1,0)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_prefix_coincident_refusal_and_native_failure_poisoned(device):
 from gffx.torch._opaque_partition import Partition
 v=torch.tensor([[.05,.07,0.],[.9,.12,0.],[.13,.91,0.],[.96,.95,.1]],dtype=torch.float64,device=device)
 f=torch.tensor([[0,1,2],[0,1,2]],dtype=torch.int32,device=device);c=v.new_empty((2,3,0));g=v.new_ones((1,800,4))
 with pytest.raises(RuntimeError):torch.ops.gffx.opaque_material_prefix_backward(v,f,c,g,0.,0.,0,0,0,0,4*1024**2,3,0,0)
 with pytest.raises(OverflowError):Partition(v,f,c,(0.,0.),2,8*1024**2)
 v,f,c=[x.to(device).contiguous() for x in fixture(1)]
 p=Partition(v,f,c,(0.,0.),2,8*1024**2);_,q,r=p.emit(v,f,c,(0,0,0,0),2,(0.,0.))
 g=torch.ones_like(q);g[0,0,0]=float('nan')
 with pytest.raises(RuntimeError):p.backward(v,f,c,g,(0,0,0,0),r,3,4*1024**2)
 assert p.poisoned and p.prefix==(0,0)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_hidden_cut_reuse_exact_and_less_work(device):
 v=torch.tensor([[.05,.05,.1],[.95,.05,.1],[.05,.95,.1],[.15,.15,0.],[.55,.15,0.],[.15,.55,0.]],dtype=torch.float64,device=device)
 f=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32,device=device)
 c=torch.tensor([[[-.5,-.2],[.5,-.2],[-.5,.8]],[[-.4,-.3],[.6,-.3],[-.4,.7]]],dtype=torch.float64,device=device)
 g=torch.arange(3200,dtype=torch.float64).sin().reshape(1,800,4).to(device)
 rows=[torch.ops.gffx.opaque_material_batch_rule_backward(v,f,c,g,0.,0.,0,0,0,0,8*1024**2,rule,3) for rule in (1,2)]
 for j in (0,1):assert rows[0][j].cpu().numpy().tobytes()==rows[1][j].cpu().numpy().tobytes()
 assert rows[0][2][:7]==rows[1][2][:7] and rows[1][2][7]<rows[0][2][7]
 assert rows[1][0].abs().sum()>0
