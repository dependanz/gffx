"""SH80 resource guards, analytical geometry and compact texture controls."""
import os,torch,pytest
from test_opaque_sh64 import fixture

def test_profile80_is_explicit():
 from gffx.torch._opaque_sh import _profile
 for calls,rule in [(16,1),(52,2),(64,3),(80,4)]:assert _profile({'sh_batch_profile':calls})==(calls,rule)
 assert _profile({})==(16,1)
 for v in (79,81,True,80.):
  with pytest.raises(ValueError):_profile({'sh_batch_profile':v})

def test_partition80_exceeds64_but_preserves_area_and_storage_bound():
 from gffx.torch._opaque_partition import Partition
 v,f,c=fixture(33)
 with pytest.raises(OverflowError):Partition(v,f,c,(0.,0.),3,67108864)
 p=Partition(v,f,c,(0.,0.),4,67108864);assert 2048<p.count<=2560
 assert p.rows.shape==(2560,17)
 cursor=(0,0,0,0);area=v.new_zeros(())
 for i in range(80):
  ids,q,r=p.emit(v,f,c,cursor,4,(0.,0.));area+=q[0,:r[2],3].sum()
  assert r[2]<=800 and r[3]<=32
  if r[1]:break
  cursor=(r[0],i+1,r[4],r[5])
 else:pytest.fail('SH80 failed to finish')
 assert i>=64 and abs(area.item()-33/512)<1e-13
 with pytest.raises(OverflowError):Partition(v,f,c,(0.,0.),4,p.used_bytes+1048576-1)
 with pytest.raises(RuntimeError):torch.ops.gffx.opaque_partition_emit64(p.rows,p.count,0)

def test_80_receipt_and_shared_edge_budget_guards():
 from gffx.torch._opaque_pixel import _advance,_Work
 from gffx.torch._opaque_quadrature import _batch_work,_charge_boundary64
 assert _advance(((2528,79,63200,2528),),((2560,1,800,32,64000,2560,32,0),),rule=4)==(None,)
 with pytest.raises(OverflowError):_advance(((2528,79,63200,2528),),((2560,0,800,32,64000,2560,32,0),),rule=4)
 work=_Work();work.sh_volume=(79,63200,2528);token=_batch_work.set(work)
 try:
  _charge_boundary64((32,1,800,32,800,32,32,0),rule=4)
  with pytest.raises(OverflowError):_charge_boundary64((0,1,0,0,0,0,0,0),rule=4)
 finally:_batch_work.reset(token)

def test_40_cut_fields_keep_area_and_reject41():
 from gffx.torch._opaque_partition import Partition
 device=os.environ.get('GFFX_BATCH_DEVICE','cpu')
 v=torch.tensor([[.1,.1,0],[.9,.1,0],[.1,.9,0]],dtype=torch.float64,device=device);f=torch.tensor([[0,1,2]],dtype=torch.int32,device=device)
 cuts=torch.stack([v[:,0]-(.1+.8*k/41) for k in range(1,41)],-1)[None].contiguous()
 with pytest.raises((ValueError,RuntimeError)):Partition(v,f,cuts,(0.,0.),3,67108864)
 p=Partition(v,f,cuts,(0.,0.),4,67108864);area=v.new_zeros(());cursor=(0,0,0,0)
 for i in range(80):
  _,q,r=p.emit(v,f,cuts,cursor,4,(0.,0.));area+=q[...,3].sum()
  if r[1]:break
  cursor=(r[0],i+1,r[4],r[5])
 assert r[1];torch.testing.assert_close(area,area.new_tensor(.32),atol=1e-13,rtol=0)
 with pytest.raises((ValueError,RuntimeError)):Partition(v,f,torch.cat((cuts,cuts[...,:1]),-1),(0.,0.),4,67108864)

@pytest.mark.parametrize('diagonal',[0,1])
def test_shared_edge_analytical_area_and_moving_geometry(diagonal):
 from test_opaque_texture_boundary_repair import rectangle
 from gffx.torch._opaque_stream import _configuration
 from gffx.torch._opaque_sh import _render_candidate
 device=os.environ.get('GFFX_BATCH_DEVICE','cpu');scene,camera,ps=rectangle(device,diagonal=diagonal)
 cfg,ts=_configuration(scene,camera,outputs=('colour','mask'),cull_mode=2,background=None,workspace_limit_bytes=64*1024**2,region=(0,0,1,1),sh_batch_profile=80);cfg.update(sh_batch_profile=80,sh_reuse_partition=True)
 out=_render_candidate(cfg,ts,('colour','mask'));g=torch.autograd.grad(out['colour'].sum(),ps,retain_graph=True);mg=torch.autograd.grad(out['mask'].sum(),ps[:2])
 torch.testing.assert_close(out['mask'],out['mask'].new_full((1,1,1),.48),atol=1e-12,rtol=0)
 torch.testing.assert_close(out['colour'].reshape(3),ps[2]*.48,atol=1e-12,rtol=0)
 for actual,want in zip(g[:2],[ps[2].sum()*.8,ps[2].sum()*.6]):torch.testing.assert_close(actual,want,atol=1e-11,rtol=0)
 for actual,want in zip(mg,[.8,.6]):torch.testing.assert_close(actual,actual.new_tensor(want),atol=1e-11,rtol=0)

def test_compact_texture_fields_equal_dense_selected_rows():
 from test_opaque_texture_boundary_repair import rectangle
 from gffx.torch._opaque_scene import _project
 from gffx.torch._opaque_sh import texture_cuts
 scene,camera,_=rectangle(os.environ.get('GFFX_BATCH_DEVICE','cpu'),texture=True)
 packed,_,px,keep,_=_project(scene,camera,2);selected=torch.zeros_like(keep);selected[1]=True
 dense=texture_cuts(packed,px,selected,(0,0,1,1));small=texture_cuts(packed,px,selected,(0,0,1,1),compact=True,cut_limit=40)
 assert small.shape==(1,3,dense.shape[-1]);assert torch.equal(small,dense[selected]);assert small.numel()*2==dense.numel()


def test_profile80_paired_tangents_do_not_repeat_scalar_work():
 from gffx.torch.render import opaque_material_quadrature_batch
 v,f,c=fixture(33);cursor=(512,16,12800,512);results=[]
 for rule in (3,4):
  _,q,r=opaque_material_quadrature_batch(v,f,c,cursor=cursor,rule=rule)
  result=torch.ops.gffx.opaque_material_batch_rule_backward(v,f,c,torch.ones_like(q),0.,0.,*cursor,67108864,rule,1)
  results.append(result)
 assert torch.equal(results[0][0],results[1][0]) and torch.equal(results[0][1],results[1][1])
 assert results[1][2][7]<=results[0][2][7], 'new profile lost paired replay and repeated scalar work'


def test_pooled_dependencies_preserve_saved_batch_bytes_without_reclipping_every_seed():
 import hashlib,numpy as np
 from pathlib import Path
 root=Path('D:/devspace/gffit/qualification-20261005/boundary-sh24')
 fixture=root/'geometry-fitted-11-16-1.npz';reference=root/'pair-probe-paired1.npz'
 if not fixture.exists() or not reference.exists():pytest.skip('caller-owned saved FLAME qualification fixture')
 assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='aedc716746efb3032b9e82b0cf75fbb55be74f21bd853e6b8d2a734ea03c6b01'
 assert hashlib.sha256(reference.read_bytes()).hexdigest()=='eece8db5c39bddb8c5e24a2c509f9f438dabccbb437de215aa3d821845c0aec7'
 device=os.environ.get('GFFX_BATCH_DEVICE','cpu')
 with np.load(fixture) as z:v=torch.from_numpy(z['vertices'].copy()).to(device);f=torch.from_numpy(z['faces'].copy()).to(device);c=torch.from_numpy(z['cuts'].copy()).to(device)
 import gffx.torch
 g=torch.ones((1,800,4),dtype=v.dtype,device=device)
 gv,gc,receipt,prefix=torch.ops.gffx.opaque_material_prefix_backward80(v,f,c,g,11.,16.,256,8,6400,256,67108864,1,55,255)
 assert receipt[:7]==[288,0,800,32,7200,288,32] and prefix==[65,283]
 with np.load(reference) as z:
  assert gv.cpu().numpy().tobytes()==z['gv'].tobytes()
  assert gc.cpu().numpy().tobytes()==z['gc'].tobytes()
 assert receipt[7]<1900498,'per-face dependency overflow forces redundant seeded clipping'
