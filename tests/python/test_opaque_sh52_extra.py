"""Additional native overflow and exact device controls after SH52 installation."""
import os,torch,pytest
from test_opaque_sh52 import fixture

def test_native_caps_and_default_exhaustion():
 from gffx.torch.render import opaque_material_quadrature_batch as fn
 for rule,cap,count in ((1,16,20),(2,52,27)):
  v,f,c=fixture(count);cursor=(0,0,0,0)
  with pytest.raises(OverflowError):
   for i in range(cap):
    ids,q,r=fn(v,f,c,cursor=cursor,rule=rule)
    assert not r[1]
    cursor=(r[0],i+1,r[4],r[5])
 v,f,c=fixture(1)
 for cursor in ((0,52,0,0),(1665,51,41600,1664),(2**40,0,0,0),(0,0,2**40,0)):
  with pytest.raises(RuntimeError):torch.ops.gffx.opaque_material_batch_rule(v,f,c,0.,0.,*cursor,67108864,2)

def test_cross_device_profile_samples_receipts_and_late_first_vjp():
 if os.environ.get('GFFX_BATCH_DEVICE')!='cuda':pytest.skip('CUDA worker only')
 from gffx.torch.render import opaque_material_quadrature_batch as fn
 saved=[]
 try:
  for dev in ('cpu','cuda'):
   os.environ['GFFX_BATCH_DEVICE']=dev;v,f,c=fixture(9);v.requires_grad_();cursor=(0,0,0,0);rows=[]
   for i in range(52):
    ids,q,r=fn(v,f,c,cursor=cursor,rule=2)
    entry=[t.detach().cpu().numpy().tobytes() for t in (ids,q)]+[tuple(r)]
    if i==16:
     upstream=torch.arange(q.numel(),device=dev,dtype=q.dtype).reshape(q.shape)/4096
     grad,=torch.autograd.grad(q,v,upstream);entry.append(grad.cpu().numpy().tobytes())
    rows.append(entry)
    if r[1]:break
    cursor=(r[0],i+1,r[4],r[5])
   assert len(rows)>16 and r[1];saved.append(rows)
 finally:os.environ['GFFX_BATCH_DEVICE']='cuda'
 assert saved[0]==saved[1]
