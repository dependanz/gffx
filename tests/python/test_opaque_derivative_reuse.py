"""Exact dual-input reuse: unchanged faces skip work; moving supports replay."""
import os,numpy as np,pytest,torch
import gffx.torch
@pytest.mark.parametrize('case',range(30))
def test_dual_reuse_preserves_gradients_or_expected_refusal(case):
 dev=os.environ.get('GFFX_BATCH_DEVICE','cpu');rng=np.random.default_rng(7131+case)
 v=np.array([[.05,.07,0.],[.9,.12,0.],[.13,.91,0.],[.96,.95,.1]],dtype=np.float64);f=np.array([[0,1,2],[1,3,2]],np.int32)
 if case%5==1:f=np.array([[0,1,2],[0,1,2]],np.int32)
 if case%5==2:v=np.concatenate([v[:3],v[:3]+[.1,-.03,.01]]);f=np.array([[0,1,2],[3,4,5]],np.int32)
 if case%5==3:v[3,2]=-.1
 if case%5==4:v[:,:2]+=rng.normal(0,.015,v[:,:2].shape)
 c=rng.normal(size=(len(f),3,(case//5)%3));g=rng.normal(size=(1,800,4))
 from gffx.torch._common import materialize
 v,f,c,g=[materialize(torch.tensor(x,device=dev)) for x in (v,f,c,g)]
 rows=[]
 for rule in (1,2):
  def run():return torch.ops.gffx.opaque_material_batch_rule_backward(v,f,c,g,0.,0.,0,0,0,0,67108864,rule,3)
  if case%5==1:
   with pytest.raises(RuntimeError):run()
  else:rows.append(run())
 if rows:
  for j in (0,1):assert rows[0][j].cpu().numpy().tobytes()==rows[1][j].cpu().numpy().tobytes()
  assert rows[0][2][:7]==rows[1][2][:7]
  assert rows[1][2][7]<=rows[0][2][7]

@pytest.mark.parametrize('batch',[0,1])
def test_many_face_dual_reuse_matches_original_rule(batch):
 from gffx.torch._common import materialize
 dev=os.environ.get('GFFX_BATCH_DEVICE','cpu');vertices=[]
 for y in range(7):
  for x in range(7):vertices.extend([[x/8,y/8,0.],[x/8+.09,y/8,0.],[x/8,y/8+.09,0.]])
 v=torch.tensor(vertices,dtype=torch.float64,device=dev);f=torch.arange(147,dtype=torch.int32,device=dev).reshape(49,3);c=materialize(v.new_empty((49,3,0)));g=torch.arange(3200,dtype=v.dtype,device=dev).reshape(1,800,4)/4096
 rows=[torch.ops.gffx.opaque_material_batch_rule_backward(v,f,c,g,0.,0.,32*batch,batch,800*batch,32*batch,67108864,rule,1) for rule in (1,2)]
 for j in (0,1):assert rows[0][j].cpu().numpy().tobytes()==rows[1][j].cpu().numpy().tobytes()
 assert rows[0][2][:7]==rows[1][2][:7]
 assert rows[1][2][7]<rows[0][2][7]/10

@pytest.mark.parametrize('crossing',[False,True])
def test_depth_dual_reuse_preserves_analytic_visibility_motion(crossing):
 # Depth-only occluder motion is irrelevant only for strictly ordered planes.
 # For a crossing, the visible boundary is x=.5 and moves by -dz.
 from gffx.torch._common import materialize
 dev=os.environ.get('GFFX_BATCH_DEVICE','cpu')
 v=torch.tensor([[-2.,-2.,.2],[6.,-2.,.2],[-2.,6.,.2],[-3.,-3.,-.2],[8.,-3.,-.2],[-3.,8.,-.2]],dtype=torch.float64,device=dev)
 if crossing:v[3:,2]=v[3:,0]-.3
 f=torch.arange(6,dtype=torch.int32,device=dev).reshape(2,3);c=materialize(v.new_empty((2,3,0)))
 rows=[]
 for rule in (1,2):
  ids,q,receipt=torch.ops.gffx.opaque_material_batch_rule(v,f,c,0.,0.,0,0,0,0,67108864,rule)
  assert receipt[1]
  colours=torch.where(ids==0,v.new_tensor(.3),v.new_tensor(.9));g=torch.zeros_like(q);g[:,:,3]=colours
  gv,gc,back=torch.ops.gffx.opaque_material_batch_rule_backward(v,f,c,g,0.,0.,0,0,0,0,67108864,rule,1)
  value=(q[:,:,3]*colours).sum()
  torch.testing.assert_close(value,value.new_tensor(.6 if crossing else .9),rtol=0,atol=1e-13)
  dz=gv[:,2].reshape(2,3).sum(1)
  torch.testing.assert_close(dz,dz.new_tensor([.6,-.6] if crossing else [0.,0.]),rtol=0,atol=1e-13)
  rows.append((q,gv,gc,back))
 for j in range(3):assert rows[0][j].cpu().numpy().tobytes()==rows[1][j].cpu().numpy().tobytes()
 assert rows[0][3][:7]==rows[1][3][:7]

@pytest.mark.parametrize('apex_x',[-2.,-3.])
def test_edge_dual_reuse_keeps_boundary_motion_and_ignores_inactive_apex(apex_x):
 from gffx.torch._common import materialize
 dev=os.environ.get('GFFX_BATCH_DEVICE','cpu')
 # Foreground covers exactly x<.5. Its left apex never touches this pixel.
 v=torch.tensor([[-2.,-2.,.2],[6.,-2.,.2],[-2.,6.,.2],[apex_x,.5,-.2],[.5,-4.,-.2],[.5,4.,-.2]],dtype=torch.float64,device=dev)
 f=torch.arange(6,dtype=torch.int32,device=dev).reshape(2,3);c=materialize(v.new_empty((2,3,0)));rows=[]
 for rule in (1,2):
  ids,q,receipt=torch.ops.gffx.opaque_material_batch_rule(v,f,c,0.,0.,0,0,0,0,67108864,rule)
  assert receipt[1]
  colours=torch.where(ids==0,v.new_tensor(.3),v.new_tensor(.9));g=torch.zeros_like(q);g[:,:,3]=colours
  gv,gc,back=torch.ops.gffx.opaque_material_batch_rule_backward(v,f,c,g,0.,0.,0,0,0,0,67108864,rule,1)
  value=(q[:,:,3]*colours).sum();torch.testing.assert_close(value,value.new_tensor(.6),rtol=0,atol=1e-13)
  torch.testing.assert_close(gv[3],torch.zeros_like(gv[3]),rtol=0,atol=1e-13)
  torch.testing.assert_close(gv[4:,0].sum(),value.new_tensor(.6),rtol=0,atol=1e-13)
  rows.append((q,gv,gc,back))
 for j in range(3):assert rows[0][j].cpu().numpy().tobytes()==rows[1][j].cpu().numpy().tobytes()
 assert rows[0][3][:7]==rows[1][3][:7]
 assert rows[1][3][7]<rows[0][3][7]


def test_dual_reuse_late_cursor_preserves_area_gradient(monkeypatch):
 # Exercise real ordered cursor counts beyond signed16 range, with independent
 # area and dilation derivatives; no invented native prefix token.
 from test_opaque_sh52 import fixture
 from gffx.torch._opaque_partition import Partition
 v,f,c=[x.contiguous() for x in fixture(22)]
 # Keep triangles strictly inside the pixel; boundary contact has a different
 # translation derivative and is not this analytical fixture.
 v[:,:2]+=1/32
 owner=Partition(v,f,c,(0.,0.),2,8*1024**2)
 cursor=(0,0,0,0);area=v.new_zeros(());gv_total=torch.zeros_like(v);work=owner.build_work
 from gffx.torch import _opaque_quadrature as quad
 original=quad._charge_batch
 def charge(receipt):
  nonlocal work
  work+=receipt[7];assert work<=2000000
  original(receipt)
 monkeypatch.setattr(quad,'_charge_batch',charge)
 for batch in range(52):
  _,q,r=owner.emit(v,f,c,cursor,2,(0.,0.))
  g=torch.zeros_like(q);g[:,:,3]=1.
  gv,_=owner.backward(v,f,c,g,cursor,r,1,4*1024**2)
  area+=q[0,:r[2],3].sum();gv_total+=gv
  if r[1]:break
  cursor=(r[0],batch+1,r[4],r[5])
 else:pytest.fail('late cursor fixture did not complete')
 assert r[4]>32767 and r[4]<=41600
 expected=22/512
 torch.testing.assert_close(area,area.new_tensor(expected),rtol=0,atol=1e-13)
 torch.testing.assert_close((gv_total[:,:2]*v[:,:2]).sum(),area.new_tensor(2*expected),rtol=0,atol=1e-12)
 torch.testing.assert_close(gv_total.sum(0),v.new_zeros(3),rtol=0,atol=1e-12)


@pytest.mark.parametrize('offset',[-2.,2.,0.,2**-40,-2**-40,-.5])
def test_dual_reuse_complement_keeps_texture_boundary_gradients(offset):
 from gffx.torch._common import materialize
 dev=os.environ.get('GFFX_BATCH_DEVICE','cpu')
 v=torch.tensor([[.1,.1,.2],[.8,.1,.2],[.1,.8,.2]],dtype=torch.float64,device=dev)
 f=torch.tensor([[0,1,2]],dtype=torch.int32,device=dev)
 c=materialize(v.new_tensor([[[offset],[1+offset],[offset]]]))
 g=torch.sin(torch.arange(3200,dtype=v.dtype,device=dev)).reshape(1,800,4)
 rows=[torch.ops.gffx.opaque_material_batch_rule_backward(v,f,c,g,0.,0.,0,0,0,0,67108864,rule,3) for rule in (1,2)]
 for i in (0,1):assert rows[0][i].cpu().numpy().tobytes()==rows[1][i].cpu().numpy().tobytes()
 assert rows[0][2][:7]==rows[1][2][:7]
 assert rows[1][2][7]<=rows[0][2][7]
