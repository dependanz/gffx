"""Bounded SH64 camera flux: legacy identity, streaming and shared accounting."""
import torch,pytest,os

def fixture(strips=20):
 device=os.environ.get('GFFX_BATCH_DEVICE','cpu');v=[];f=[]
 for i in range(strips):
  y=i/strips;z=(i+1)/strips;k=len(v)
  v.extend([[-1.,y,0.],[2.,y,0.],[2.,z,0.],[-1.,z,0.]])
  f.extend([[k,k+1,k+2],[k,k+2,k+3]])
 return torch.tensor(v,dtype=torch.float64,device=device),torch.tensor(f,dtype=torch.int32,device=device),torch.empty((len(f),3,0),dtype=torch.float64,device=device)

def test_boundary64_matches_small_legacy_bytes():
 from gffx.torch._opaque_quadrature import opaque_boundary_batch64,opaque_boundary_quadrature
 v,f,c=fixture(2);old=opaque_boundary_quadrature(v,f,c,1,1,edge=0)
 ids,q,r=opaque_boundary_batch64(v,f,c,edge=0)
 assert r[1] and r[3]==4 and r[2]==48 and r[7]>0
 assert all(a.cpu().numpy().tobytes()==b.cpu().numpy().tobytes() for a,b in zip(old,(ids,q)))

def test_boundary64_streams_whole_segments_and_exact_integrals():
 from gffx.torch._opaque_quadrature import opaque_boundary_batch64,opaque_boundary_quadrature
 from gffx.torch._opaque_pixel import _advance
 v,f,c=fixture();cursor=(0,0,0,0);area=0.;moment=0.;rs=[]
 with pytest.raises(OverflowError):opaque_boundary_quadrature(v,f,c,1,1,edge=0)
 for index in range(64):
  ids,q,r=opaque_boundary_batch64(v,f,c,edge=0,cursor=cursor);rs.append(r)
  assert r[2]==12*r[3] and r[3]<=32 and r[7]>0
  n=r[2];weights=q[0,:n,3];ys=(q[0,:n,:3]*v[f[ids[0,:n].long()].long(),1]).sum(-1)
  area+=weights.sum().item();moment+=(weights*ys).sum().item()
  nxt=_advance((cursor,),(r,),rule=3)[0]
  if nxt is None:break
  cursor=nxt
 assert len(rs)==2 and sum(r[3] for r in rs)==40
 assert abs(area-1.)<2e-14 and abs(moment-.5)<2e-14

@pytest.mark.parametrize('edge',[-1,4,True])
def test_boundary64_refuses_invalid_edges(edge):
 from gffx.torch._opaque_quadrature import opaque_boundary_batch64
 with pytest.raises(ValueError):opaque_boundary_batch64(*fixture(1),edge=edge)

def test_boundary64_refuses_forged_prefix_and_gradients():
 from gffx.torch._opaque_quadrature import opaque_boundary_batch64
 v,f,c=fixture()
 with pytest.raises((ValueError,OverflowError,RuntimeError)):opaque_boundary_batch64(v,f,c,edge=0,cursor=(32,1,1,32))
 v.requires_grad_()
 with pytest.raises(NotImplementedError):opaque_boundary_batch64(v,f,c,edge=0)

def test_boundary64_shared_budget_cannot_reset_volume_or_clipping():
 from gffx.torch._opaque_quadrature import _charge_boundary64,_batch_work
 from gffx.torch._opaque_pixel import _Work
 work=_Work();work.sh_volume=(63,50400,2016);work.sh_edges=(0,0,0);token=_batch_work.set(work)
 try:
  _charge_boundary64((1,1,12,1,12,1,1,3));assert work.value==3
  with pytest.raises(OverflowError):_charge_boundary64((1,1,12,1,12,1,1,3))
 finally:_batch_work.reset(token)
 work=_Work();work.value=1999999;work.sh_volume=(0,0,0);work.sh_edges=(0,0,0);token=_batch_work.set(work)
 try:
  with pytest.raises(OverflowError):_charge_boundary64((1,1,12,1,12,1,1,2))
 finally:_batch_work.reset(token)
