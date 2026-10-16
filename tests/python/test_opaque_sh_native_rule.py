import torch,pytest

def call(v,f,c,rule=1,**kwargs):
 from gffx.torch._opaque_quadrature import opaque_material_quadrature_batch
 return opaque_material_quadrature_batch(v,f,c,rule=rule,**kwargs)

def tri():
 v=torch.tensor([[.1,.1,0],[.8,.1,0],[.1,.7,0]],dtype=torch.float64,requires_grad=True)
 return v,torch.tensor([[0,1,2]],dtype=torch.int32),v.new_empty((1,3,0))

def test_sh25_native_samples_area_and_geometry_vjp():
 v,f,c=tri();ids,q,r=call(v,f,c);assert r[1] and r[2]==25 and r[3]==1
 area=q[0,:r[2],3].sum();g,=torch.autograd.grad(area,v)
 torch.testing.assert_close(area,area.new_tensor(.21),atol=1e-14,rtol=0)
 torch.testing.assert_close(g,g.new_tensor([[-.3,-.35,0],[.3,0,0],[0,.35,0]]),atol=1e-13,rtol=0)
 assert torch.equal(ids[0,r[2]:],torch.full_like(ids[0,r[2]:],-1))
 assert torch.equal(q[0,r[2]:],torch.zeros_like(q[0,r[2]:]))

def test_default_rule_retains_44_slots():
 v,f,c=tri();assert call(v,f,c,0)[2][2]==44

@pytest.mark.parametrize('rule',[True,-1,5,6,1.0,'sh'])
def test_rule_validation(rule):
 v,f,c=tri()
 with pytest.raises((TypeError,ValueError)):call(v,f,c,rule)

def test_sh25_refuses_curves_and_second_gradients():
 v,f,c=tri()
 with pytest.raises(ValueError):call(v,f,c,curves=v.new_zeros((1,6)))
 _,q,_=call(v,f,c)
 with pytest.raises(RuntimeError,match='first'):torch.autograd.grad(q.sum(),v,create_graph=True)

def test_cut_motion_cancellation_at_texture_boundary():
 v,f,c=tri();cut=torch.tensor([[[-.2],[.8],[-.2]]],dtype=v.dtype,requires_grad=True)
 _,q,r=call(v,f,cut);assert r[1] and r[2]%25==0
 # Globally smooth quadratic: split location cannot change its exact integral.
 beta=q[0,:r[2],:3];w=q[0,:r[2],3];loss=((beta[:,1]+2*beta[:,2])**2*w).sum()
 gv,gc=torch.autograd.grad(loss,(v,cut));torch.testing.assert_close(gc,torch.zeros_like(gc),atol=2e-13,rtol=0)

def test_native_requested_derivatives_preserve_requested_bytes_and_reduce_work():
 v,f,c=tri();c=v.new_tensor([[[-.2],[.8],[-.2]]]);_,q,r=call(v,f,c)
 g=torch.arange(q.numel(),dtype=v.dtype).reshape(q.shape)/4096
 fn=torch.ops.gffx.opaque_material_batch_rule_backward
 both=fn(v,f,c,g,0.,0.,0,0,0,0,67108864,1,3)
 only=fn(v,f,c,g,0.,0.,0,0,0,0,67108864,1,1)
 assert torch.equal(both[0],only[0]) and torch.equal(only[1],torch.zeros_like(only[1]))
 assert both[2][:7]==only[2][:7] and only[2][7]<both[2][7]
 for invalid in (-1,4):
  with pytest.raises(RuntimeError):fn(v,f,c,g,0.,0.,0,0,0,0,67108864,1,invalid)

@pytest.mark.parametrize('piece_index',[39,38,44])
def test_native_sh25_composed_shader_partials_against_independent_bound(piece_index):
 import json,copy,sys
 from pathlib import Path
 sys.path.insert(0,'C:/dev/gffx')
 from tools.opaque_sh_texture_reference import certify,field,I
 piece=json.loads(Path('D:/devspace/gffit/qualification-20261003/boundary-sh3/pieces1.json').read_bytes())['pieces'][piece_index]
 v,f,c=tri();ids,q,r=call(v,f,c);assert r[2]==25
 beta=q[0,:25,:3];w=q[0,:25,3]
 arrays={key:torch.tensor(piece[key],dtype=v.dtype,requires_grad=True) for key in ('normal','uv','tex','sh')}
 n=beta@arrays['normal'];n=n/torch.linalg.vector_norm(n,dim=-1,keepdim=True)
 s,t=(beta@arrays['uv']).unbind(-1);tex=arrays['tex']
 colour=tex[0]+s[:,None]*(tex[1]-tex[0])+t[:,None]*(tex[2]-tex[0])+(s*t)[:,None]*(tex[3]-tex[1]-tex[2]+tex[0])
 x,y,z=n.unbind(-1);basis=torch.stack([torch.ones_like(x),x,y,z,x*y,y*z,3*z*z-1,x*z,x*x-y*y],-1)
 output=((colour*(basis@arrays['sh']))*w[:,None]).sum(0)
 gs=torch.autograd.grad(output.sum(),[*arrays.values(),v]);piece=dict(piece,det=.42)
 value=certify(lambda u,v:field(piece,u,v))
 for actual,interval in zip(output.detach().tolist(),value['value']):assert (I(actual)-interval).bound<=I('1e-10').lo
 for key,g in zip(arrays,gs):
  for i,row in enumerate(piece[key]):
   for j in range(len(row)):
    a=copy.deepcopy(piece);a['d_'+key]=[[0. for _ in r] for r in piece[key]];a['d_'+key][i][j]=1.
    c=certify(lambda u,v:field(a,u,v));expected=sum(c['tangent'],I(0))
    assert (I(g[i,j].item())-expected).bound<=I('1e-10').lo
 # Moving top vertex changes determinant by width=.7, with fixed local fields.
 a=dict(piece,d_det=.7);c=certify(lambda u,v:field(a,u,v))
 assert (I(gs[-1][2,1].item())-sum(c['tangent'],I(0))).bound<=I('1e-10').lo
