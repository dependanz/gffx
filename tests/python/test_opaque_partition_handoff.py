"""Bounded forward-to-backward ownership, charges and invalidation."""
import os,torch,pytest
from test_opaque_partition_reuse import fixture


def completed_pool():
 from gffx.torch._opaque_partition import Pool
 v,f,c=fixture(1);pool=Pool(4*1024**2);cursor=(0,0,0,0);saved=[]
 while True:
  ids,q,r=pool.emit(v,f,c,(0.,0.),2,cursor,'cell');saved.append((ids.clone(),q.clone(),r))
  if r[1]:break
  cursor=(r[0],cursor[1]+1,r[4],r[5])
 return pool,(v,f,c),saved


def test_handoff_has_no_second_build_and_preserves_all_bytes():
 pool,(v,f,c),saved=completed_pool();entry=pool.entries['cell'];rows=entry.rows;used=pool.used_bytes;work=entry.build_work
 pool.begin_replay();cursor=(0,0,0,0)
 for oi,oq,old in saved:
  ids,q,r=pool.emit(v,f,c,(0.,0.),2,cursor,'cell')
  assert torch.equal(ids,oi) and torch.equal(q.view(torch.uint8),oq.view(torch.uint8))
  assert r[:7]==old[:7] and r[7]==0
  cursor=(r[0],cursor[1]+1,r[4],r[5])
 assert entry.rows is rows and pool.used_bytes==used and entry.build_work==work
 with pytest.raises(ValueError):pool.begin_replay()


@pytest.mark.parametrize('fault',['incomplete','rows','poison','derivative','new_key','inputs'])
def test_handoff_refuses_invalid_ownership(fault):
 pool,(v,f,c),saved=completed_pool();entry=pool.entries['cell']
 if fault=='incomplete':entry.finished=False
 if fault=='rows':entry.rows.add_(1)
 if fault=='poison':entry.poisoned=True
 if fault=='derivative':entry.derivatives=1
 if fault in ('new_key','inputs'):
  pool.begin_replay()
  if fault=='inputs':v[0,0]+=2**-40
  with pytest.raises(ValueError):pool.emit(v,f,c,(0.,0.),2,(0,0,0,0),'other' if fault=='new_key' else 'cell')
 else:
  with pytest.raises(ValueError):pool.begin_replay()
  with pytest.raises(ValueError):pool.begin_replay()


def scene_result():
 from test_opaque_texture_boundary_repair import rectangle
 from gffx.torch import _opaque_sh
 from gffx.torch._opaque_stream import _configuration
 scene,camera,ps=rectangle(os.environ.get('GFFX_BATCH_DEVICE','cpu'),diagonal=0,texture=True)
 config,tensors=_configuration(scene,camera,outputs=('colour','mask'),cull_mode=2,background=None,workspace_limit_bytes=64*1024**2,region=(0,0,1,1))
 config.update(sh_batch_profile=52,sh_reuse_partition=True)
 return _opaque_sh._render_candidate(config,tensors,('colour','mask')),ps


def test_scene_consumes_retained_pool_once_and_repeated_backward_is_fresh(monkeypatch):
 from gffx.torch._opaque_partition import Partition
 builds=[];original=Partition.__init__
 def init(self,*a,**kw):original(self,*a,**kw);builds.append(self)
 monkeypatch.setattr(Partition,'__init__',init)
 result,ps=scene_result();count=len(builds);assert count>0
 g1=torch.autograd.grad(result['colour'].sum(),ps,retain_graph=True)
 assert len(builds)==count
 g2=torch.autograd.grad(result['colour'].sum(),ps)
 assert len(builds)==2*count
 assert all(torch.equal(a.reshape(-1).contiguous().view(torch.uint8),b.reshape(-1).contiguous().view(torch.uint8)) for a,b in zip(g1,g2))


def test_failed_replay_is_not_retried(monkeypatch):
 from gffx.torch._opaque_partition import Partition
 original=Partition.backward;calls=[]
 def fail(self,*a,**kw):calls.append(1);raise RuntimeError('injected replay failure')
 result,ps=scene_result();monkeypatch.setattr(Partition,'backward',fail)
 with pytest.raises(RuntimeError,match='injected'):torch.autograd.grad(result['colour'].sum(),ps,retain_graph=True)
 monkeypatch.setattr(Partition,'backward',original)
 with pytest.raises(RuntimeError,match='failed'):torch.autograd.grad(result['colour'].sum(),ps)
 assert len(calls)==1


def test_handoff_budget_is_still_checked_before_any_preparation():
 from gffx.torch._opaque_partition import Pool
 v,f,c=fixture(1);pool=Pool(1048576)
 with pytest.raises(OverflowError,match='storage budget'):pool.emit(v,f,c,(0.,0.),2,(0,0,0,0),'cell')
 assert pool.used_bytes==0 and not pool.entries


def test_larger_frame_does_not_retain_forward_pools(monkeypatch):
 from test_opaque_texture_boundary_repair import rectangle
 from gffx.torch import _opaque_sh
 from gffx.torch._opaque_stream import _configuration
 from gffx.torch._opaque_partition import Pool
 claims=[];original=Pool.begin_replay
 def claim(self):claims.append(self);return original(self)
 monkeypatch.setattr(Pool,'begin_replay',claim)
 s,c,ps=rectangle(os.environ.get('GFFX_BATCH_DEVICE','cpu'),texture=True)
 cfg,ts=_configuration(s,c,outputs=('colour','mask'),cull_mode=2,background=None,workspace_limit_bytes=64*1024**2,region=(0,0,1,1))
 cfg.update(sh_batch_profile=52,sh_reuse_partition=True);cfg['shape']=(1,1,2)
 result=_opaque_sh._render_candidate(cfg,ts,('colour','mask'))
 torch.autograd.grad(result['colour'].sum(),ps)
 assert claims==[]


@pytest.mark.parametrize('offset',[-2.,2.])
def test_handoff_with_offwindow_faces_preserves_full_gradients(offset):
 from test_opaque_texture_boundary_repair import rectangle
 from gffx.torch import _opaque_sh,scene
 from gffx.torch._opaque_stream import _configuration
 device=os.environ.get('GFFX_BATCH_DEVICE','cpu');results=[]
 for reuse in (False,True):
  s,c,ps=rectangle(device,texture=True);mesh=s.prims[0]
  far=mesh.vertices.detach().clone();far[:,0]+=offset;far.requires_grad_()
  s=scene.Scene([*s.prims,scene.Mesh('far',far,mesh.faces,material='mat',attributes=mesh.attributes)])
  cfg,ts=_configuration(s,c,outputs=('colour','mask'),cull_mode=2,background=None,workspace_limit_bytes=64*1024**2,region=(0,0,1,1))
  cfg.update(sh_batch_profile=52,sh_reuse_partition=reuse)
  out=_opaque_sh._render_candidate(cfg,ts,('colour','mask'))
  gs=torch.autograd.grad(out['colour'].sum(),(*ps,far))
  assert torch.count_nonzero(gs[-1])==0
  results.append([t.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes() for t in (out['colour'],out['mask'],*gs)])
 assert results[0]==results[1]


def test_closed_face_filter_keeps_contacts_and_validates_excluded_inputs():
 from gffx.torch._opaque_scene import _forward_selected_faces,_face_bounds,_select_bounds
 v=torch.tensor([[1.,.2,0],[1.2,.2,0],[1.,.4,0],[2.,.2,0],[2.2,.2,0],[2.,.4,0]],dtype=torch.float64,device=os.environ.get('GFFX_BATCH_DEVICE','cpu'),requires_grad=True)
 f=torch.tensor([[0,1,2],[3,4,5]],device=v.device,dtype=torch.int32);keep=torch.ones(2,dtype=torch.bool,device=v.device)
 cuts=v.new_zeros((2,3,1));selected=_select_bounds(keep,_face_bounds(v,f),0,0,1,1)
 assert selected.tolist()==[True,False]
 assert _forward_selected_faces(v,keep,cuts,None,selected).tolist()==[True,False]
 cuts[1,0,0]=float('nan')
 with pytest.raises(ValueError):_forward_selected_faces(v,keep,cuts,None,selected)
