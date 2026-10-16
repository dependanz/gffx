"""Installed-wheel SH64 opt-in and unchanged legacy-profile controls."""
import torch,pytest,os
from test_opaque_sh52 import fixture as source_fixture
def fixture(count=20):return tuple(t.contiguous() for t in source_fixture(count))

def test_profile64_explicit_defaults_unchanged():
 from gffx.torch._opaque_sh import _profile
 assert _profile({})==(16,1)
 assert _profile({'sh_batch_profile':52})==(52,2)
 assert _profile({'sh_batch_profile':64})==(64,3)
 for value in (53,65,True,64.):
  with pytest.raises(ValueError):_profile({'sh_batch_profile':value})

def test_partition64_admits_over52_and_preserves_area():
 from gffx.torch._opaque_partition import Partition
 v,f,c=fixture(27)
 with pytest.raises(OverflowError):Partition(v,f,c,(0.,0.),2,67108864)
 p=Partition(v,f,c,(0.,0.),3,67108864);assert 1664<p.count<=2048
 assert p.rows.shape==(2048,17)
 cursor=(0,0,0,0);area=v.new_zeros(())
 for i in range(64):
  ids,q,receipt=p.emit(v,f,c,cursor,3,(0.,0.));area+=q[0,:receipt[2],3].sum()
  if receipt[1]:break
  cursor=(receipt[0],i+1,receipt[4],receipt[5])
 else:pytest.fail('SH64 failed to finish')
 assert i>=52 and abs(area.item()-27/512)<1e-13
 with pytest.raises(OverflowError):Partition(*fixture(33),(0.,0.),3,67108864)

def test_partition64_profile_storage_and_replay_binding():
 from gffx.torch._opaque_partition import Partition
 v,f,c=fixture(1);old=Partition(v,f,c,(0.,0.),2,67108864);new=Partition(v,f,c,(0.,0.),3,67108864)
 assert new.used_bytes-old.used_bytes==(2048-1664)*17*8
 for p in (old,new):
  with pytest.raises(ValueError):p.emit(v,f,c,(0,0,0,0),5-p.rule,(0.,0.))
 with pytest.raises(OverflowError):Partition(v,f,c,(0.,0.),3,new.used_bytes+1048576-1)
 with pytest.raises(RuntimeError):torch.ops.gffx.opaque_partition_emit(new.rows,new.count,0)
 with pytest.raises(RuntimeError):torch.ops.gffx.opaque_partition_emit64(old.rows,old.count,0)

def test_rule64_matches52_samples_and_complete_vjp():
 from gffx.torch.render import opaque_material_quadrature_batch as fn
 results=[]
 for rule in (2,3):
  v,f,c=fixture(1);v.requires_grad_();c.requires_grad_();ids,q,receipt=fn(v,f,c,rule=rule)
  gradient=torch.arange(q.numel(),device=q.device,dtype=q.dtype).reshape(q.shape)/4096
  grads=torch.autograd.grad(q,(v,c),gradient)
  results.append([x.detach().cpu().numpy().tobytes() for x in (ids,q,*grads)]+[receipt])
 assert results[0]==results[1]

def test_rule64_receipt_boundary_and_same_aggregate_work_guard():
 from gffx.torch._opaque_pixel import _advance,_Work
 assert _advance(((2016,63,50400,2016),),((2048,1,800,32,51200,2048,32,0),),rule=3)==(None,)
 with pytest.raises(OverflowError):_advance(((2016,63,50400,2016),),((2048,0,800,32,51200,2048,32,0),),rule=3)
 with pytest.raises((ValueError,OverflowError,RuntimeError)):_advance(((2048,64,51200,2048),),((2049,1,25,1,51225,2049,1,0),),rule=3)

 work=_Work();work(1999999);work(1)
 with pytest.raises(OverflowError):work(1)
 assert work.value==2000001

def test_scene64_reused_forward_and_prefix_backward_matches52():
 from test_opaque_texture_boundary_repair import rectangle
 from gffx.torch import _opaque_sh
 from gffx.torch._opaque_stream import _configuration
 results=[]
 for profile in (52,64):
  scene,camera,ps=rectangle(os.environ.get('GFFX_BATCH_DEVICE','cpu'),texture=True)
  config,tensors=_configuration(scene,camera,outputs=('colour','mask'),cull_mode=2,background=None,workspace_limit_bytes=64*1024**2,region=(0,0,1,1))
  config.update(sh_batch_profile=profile,sh_reuse_partition=True)
  result=_opaque_sh._render_candidate(config,tensors,('colour','mask'))
  config['sh_batch_profile']=16
  gradients=torch.autograd.grad(result['colour'].sum(),ps)
  results.append([t.detach().cpu().numpy().tobytes() for t in (result['colour'],result['mask'],*gradients)])
 assert results[0]==results[1]
