import torch,pytest
from test_opaque_sh52 import fixture as source_fixture

def fixture(count=20):return tuple(t.contiguous() for t in source_fixture(count))

def test_reused_split_samples_match_legacy_and_charge_build_once():
 from gffx.torch._opaque_partition import Partition
 from gffx.torch.render import opaque_material_quadrature_batch as fn
 v,f,c=fixture(2);cache=Partition(v,f,c,(0.,0.),2,4*1024**2);cursor=(0,0,0,0);work=0
 for i in range(52):
  ids,q,r=cache.emit(v,f,c,cursor,2,(0.,0.));oi,oq,old=fn(v,f,c,cursor=cursor,rule=2)
  assert ids.numpy().tobytes()==oi.numpy().tobytes() and q.numpy().tobytes()==oq.numpy().tobytes()
  assert tuple(r[:7])==tuple(old[:7]);work+=r[7]
  if r[1]:break
  cursor=(r[0],i+1,r[4],r[5])
 assert work==cache.build_work and cache.used_bytes<=4*1024**2

def test_cache_refuses_stale_inputs_profile_origin_and_order():
 from gffx.torch._opaque_partition import Partition
 v,f,c=fixture(1);cache=Partition(v,f,c,(0.,0.),2,4*1024**2)
 for rule,origin,cursor in [(1,(0.,0.),(0,0,0,0)),(2,(1.,0.),(0,0,0,0)),(2,(0.,0.),(32,1,800,32))]:
  with pytest.raises(ValueError):cache.emit(v,f,c,cursor,rule,origin)
 changed=v.clone();changed[0,0]+=2**-40
 with pytest.raises(ValueError):cache.emit(changed,f,c,(0,0,0,0),2,(0.,0.))
 cache.emit(v,f,c,(0,0,0,0),2,(0.,0.))
 with pytest.raises(ValueError):cache.emit(v,f,c,(0,0,0,0),2,(0.,0.))

def test_cache_budget_refuses_before_build():
 from gffx.torch._opaque_partition import Partition
 v,f,c=fixture(1)
 with pytest.raises(OverflowError):Partition(v,f,c,(0.,0.),2,1)

@pytest.mark.parametrize('diagonal',[0,1])
@pytest.mark.parametrize('grid',[(1,1),(1,2),(2,1)])
def test_scene_reuse_preserves_exact_forward_and_checked_replay(diagonal,grid,monkeypatch):
 from test_opaque_texture_boundary_repair import rectangle
 from gffx.torch import _opaque_sh
 from gffx.torch._opaque_stream import _configuration
 results=[]
 monkeypatch.setattr(_opaque_sh,'_whole_pixel',lambda *args:grid)
 for reuse in (False,True):
  scene,camera,ps=rectangle('cpu',diagonal=diagonal,texture=True)
  config,tensors=_configuration(scene,camera,outputs=('colour','mask'),cull_mode=2,background=None,workspace_limit_bytes=64*1024**2,region=(0,0,1,1))
  config.update(sh_batch_profile=52,sh_reuse_partition=reuse)
  result=_opaque_sh._render_candidate(config,tensors,('colour','mask'))
  config['sh_batch_profile']=16 # saved context must bind the original profile
  gradients=torch.autograd.grad(result['colour'].sum(),ps)
  results.append([t.detach().numpy().tobytes() for t in (result['colour'],result['mask'],*gradients)])
 assert results[0]==results[1]

def test_cache_capacity_and_storage_mutation_refuse():
 from gffx.torch._opaque_partition import Partition
 v,f,c=fixture(27)
 with pytest.raises(OverflowError):Partition(v,f,c,(0.,0.),2,4*1024**2)
 v,f,c=fixture(1);cache=Partition(v,f,c,(0.,0.),2,4*1024**2);cache.rows.add_(1.)
 with pytest.raises(ValueError):cache.emit(v,f,c,(0,0,0,0),2,(0.,0.))


def test_native_prepare_refuses_wrong_cut_middle_dimension():
 v,f,c=fixture(1)
 with pytest.raises(RuntimeError):torch.ops.gffx.opaque_partition_prepare(v,f,v.new_empty((1,4,0)),0.,0.,4*1024**2)


def test_saved_flame_partition_matches_all_preserved_prefix_bytes():
 import numpy as np,hashlib
 from pathlib import Path
 from gffx.torch._opaque_partition import Partition
 root=Path('D:/devspace/gffit/qualification-20261003/boundary-sh8')
 hashes={'0': 'e0ca785a05cd44b86bd9187887ef2aa9f697833955ddf89b78ada8e7132184a2', '1': 'd36f9f2300718621f5cc61ebe41dc18f531286bf19a62b4d49c6f96f81a95a70', '2': '5d2c2122798e80dac2ed52e588ed54eeed3904318725b012f6af1a7343ac13b2', '3': '7c989159c2b28534a49c8985d467effd3e3f3f70bd19e00557146a52e94fa575', '4': '5951d41a7fce841eaba6f5c7c5c60fd85bbf847a2bc3cba8f795dec9d885719a', '5': '1b5240457a8f0323ff4f61164adb9f9b9f64747e9e598664dc9cc1e91492e3f2', '6': 'bb16f67a1c2cbc1d958694b0d6933352edd64bbc1a16481d4f9bcb897899d476', '7': '0a0794f680bde935a0003cd88972cca6acfc1c586aa4f3b88ffb8fdc3e61b00e', '8': '85d63edffaa0b528edcb7c3cd94192a5d7e75e164e2f2a9431abaea45cc4848a', '9': 'e09827b366b9576d1bc462daaaf48b07a5a102a7686cdc8d42ab3f5537286901', '10': '43829b1eff97083dd0d4510453cabdfd5d7cb2797cc67d16391f13fc9bcfbd88', '11': '0f63bd16116bab2eeae77e16d956ca87c14cdb03e7f37c9c86100029f22653be', '12': 'a6718ec884fd581add5a4e5d21f5be86bbbca60605bd9c48bbf73492b138944f', '13': 'add7a0f35a66213233bdf34dcb868796688de55ac42de4917ac8ddf81ba4815d', '14': '8bed95a8cfa4564a37ebf3ce4373a9986fa3c9226e1b5fc016bc4f847b2e3b28'}
 arrays=[]
 for i in range(15):
  p=root/('prefix-capture1-batch%02d.npz'%i);assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(i)]
  with np.load(p,allow_pickle=False) as z:arrays.append({k:z[k].copy() for k in ('vertices','faces','cuts','origin','ids','q','receipt')})
 a=arrays[0];v,f,c=[torch.from_numpy(a[k]) for k in ('vertices','faces','cuts')];origin=tuple(a['origin'].tolist())
 cache=Partition(v,f,c,origin,2,4*1024**2);assert cache.count==1637;assert cache.build_work<=2000000
 cursor=(0,0,0,0)
 for i in range(52):
  ids,q,receipt=cache.emit(v,f,c,cursor,2,origin)
  if i<15:
   z=arrays[i];assert ids.numpy().tobytes()==z['ids'].tobytes();assert q.numpy().tobytes()==z['q'].tobytes();assert tuple(receipt[:7])==tuple(z['receipt'][:7])
  if receipt[1]:break
  cursor=(receipt[0],i+1,receipt[4],receipt[5])
 assert i==51 and receipt[1] and receipt[4:6]==(40925,1637)
