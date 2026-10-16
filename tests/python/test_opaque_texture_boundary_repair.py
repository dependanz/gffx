"""Analytical area/texture derivatives at nearly coincident pixel and UV cuts."""
import math,torch,pytest
def rectangle(device,right=.8,diagonal=0,texture=False):
 import torch
 from gffx.torch import scene,material,light,render
 from gffx.torch.camera import Camera
 t=lambda a:torch.tensor(a,device=device,dtype=torch.float64)
 x=t(right).requires_grad_();y=t(.9).requires_grad_();a=t([.05,.1,.15] if texture else [.2,.4,.6]).requires_grad_();c=t([[math.sqrt(4*math.pi)]*3]+[[0.]*3]*8).requires_grad_()
 v=torch.stack([torch.stack([xx,-yy,xx*0-2]) for xx,yy in [(x*0+.2,y*0+.1),(x,y*0+.1),(x,y),(x*0+.2,y)]])
 f=torch.tensor([[0,2,1],[0,3,2]] if not diagonal else [[0,3,1],[1,3,2]],dtype=torch.int32,device=device)
 attrs={'normals':scene.Attribute(t([0,0,1]).expand(4,3).contiguous(),'vertex')}
 albedo=a
 if texture:
  tex=torch.stack([torch.stack([a,a*2]),torch.stack([a*3,a*4])]).contiguous();pyr,off=render.texture_pyramid(tex,levels=1)
  albedo=render.TextureSource(pyr,off,2,2,3,wrap_u=render.WRAP_BORDER,wrap_v=render.WRAP_BORDER,border=t([0,0,0]))
  attrs['uv']=scene.Attribute(t([[.1,.1],[.9,.1],[.9,.9],[.1,.9]]),'vertex')
 s=scene.Scene([scene.Mesh('mesh',v,f,material='mat',attributes=attrs),scene.Material('mat',material.classic,dict(albedo=albedo,specular=t(0),exponent=t(4))),scene.Light('sh',light.spherical_harmonics,dict(coefficients=c))])
 camera=Camera.from_orthographic(t(1),t(1),t(0),t(0),1,1,.1,10)
 return s,camera,(x,y,a,c)

@pytest.mark.parametrize('diagonal',[0,1])
def test_texture_boundary_keeps_tiny_strip_shape_derivatives(diagonal):
 from gffx.torch._opaque_scene import shade
 from gffx.torch._deferred import _camera
 s,c,params=rectangle('cpu',texture=True,diagonal=diagonal);c=_camera(c,1,factor=4)
 total=0.;mask=0.
 for y in range(4):
  for x in range(4):
   r=shade(s,c,outputs=('colour','mask'),cull_mode=2,background=None,workspace_limit_bytes=64*1024**2,_region=(y,x,1,1),_batch_cursors=((0,0,0,0),))
   assert r['_receipts'][0][1]
   total=total+r['colour'].sum()/16;mask=mask+r['mask'].sum()/16
 g=torch.autograd.grad(total,params[:2],retain_graph=True);mg=torch.autograd.grad(mask,params[:2]);expected=params[2].sum().detach()*2.5*(.755/.8)**2
 torch.testing.assert_close(total,expected*.48,atol=1e-12,rtol=0)
 torch.testing.assert_close(mask,mask.new_tensor(.48),atol=1e-12,rtol=0)
 for actual,want in zip(g,[expected*.8,expected*.6]):torch.testing.assert_close(actual,want,atol=1e-11,rtol=0)
 for actual,want in zip(mg,[.8,.6]):torch.testing.assert_close(actual,actual.new_tensor(want),atol=1e-11,rtol=0)

@pytest.mark.parametrize('depth',[1.,1e-20])
def test_proven_background_triangle_does_not_fragment_front_pixel(depth):
 from gffx.torch.render import opaque_material_quadrature_batch as batch
 v=torch.tensor([[-2.,-2.,0.],[6.,-2.,0.],[-2.,6.,0.],[.2,.2,depth],[.8,.3,depth],[.3,.8,depth]],dtype=torch.float64,requires_grad=True)
 front=torch.tensor([[0,1,2]],dtype=torch.int32);both=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32)
 results=[]
 for f in (front,both):
  owner,q,receipt=batch(v,f,v.new_empty((len(f),3,0)))
  assert receipt[1];alpha=q[...,3].sum();g=torch.autograd.grad(alpha,v)[0]
  torch.testing.assert_close(alpha,alpha.new_tensor(1.),atol=1e-12,rtol=0)
  torch.testing.assert_close(g,torch.zeros_like(g),atol=2e-12,rtol=0)
  results.append((receipt[2],receipt[3]))
 assert results[0]==results[1], 'behind surface introduced integration pieces'

@pytest.mark.parametrize('slope,offset',[(0.,-.1),(1.,-.5),(0.,0.),(1e-20,-.5e-20)])
def test_occlusion_guard_preserves_visible_and_uncertain_depth_crossings(slope,offset):
 from gffx.torch.render import opaque_material_quadrature_batch as batch
 from tools.opaque_pixel_oracle import integrate_affine
 rows=[[-2.,-2.,0.],[6.,-2.,0.],[-2.,6.,0.]]+[[x,y,slope*x+offset] for x,y in ((.2,.2),(.8,.3),(.3,.8))]
 v=torch.tensor(rows,dtype=torch.float64);f=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32)
 c=[[[.2,.3,.4]]*3,[[.7,.1,.8]]*3];ref=integrate_affine(rows,f.tolist(),c,height=1,width=1)
 owner,q,r=batch(v,f,v.new_empty((2,3,0)));assert r[1]
 colours=v.new_tensor(c)[owner.clamp_min(0).long()];rgb=(q[...,:3,None]*colours).sum(-2)
 out=(rgb*q[...,3,None]).sum((0,1));expected=v.new_tensor([float(x) for x in ref['colour'][0][0]])
 torch.testing.assert_close(out,expected,atol=1e-12,rtol=0)
 torch.testing.assert_close(q[...,3].sum(),v.new_tensor(float(ref['mask'][0][0])),atol=1e-12,rtol=0)

@pytest.mark.parametrize('grid',[(1,1),(1,2),(2,1),(2,2),(4,4)])
@pytest.mark.parametrize('diagonal',[0,1])
def test_private_rectangular_replay_preserves_integrals_and_background(grid,diagonal,monkeypatch):
 from gffx.torch import _opaque_sh
 from gffx.torch._opaque_stream import _configuration
 s,c,ps=rectangle('cpu',texture=True,diagonal=diagonal);bg=ps[0].new_tensor([.01,.02,.03],requires_grad=True)
 config,tensors=_configuration(s,c,outputs=('colour','mask'),cull_mode=2,background=bg,workspace_limit_bytes=64*1024**2,region=(0,0,1,1))
 monkeypatch.setattr(_opaque_sh,'_whole_pixel',lambda *args:grid)
 out=_opaque_sh._render_candidate(config,tensors,('colour','mask'))
 factor=2.5*(.755/.8)**2
 torch.testing.assert_close(out['colour'].reshape(3),ps[2]*.48*factor+bg*.52,atol=1e-12,rtol=0)
 torch.testing.assert_close(out['mask'],torch.full_like(out['mask'],.48),atol=1e-12,rtol=0)
 gx,gy,gb=torch.autograd.grad(out['colour'].sum(),(ps[0],ps[1],bg))
 e=ps[2].sum().detach()*factor-bg.sum().detach()
 torch.testing.assert_close(gx,e*.8,atol=1e-11,rtol=0);torch.testing.assert_close(gy,e*.6,atol=1e-11,rtol=0)
 torch.testing.assert_close(gb,torch.full_like(gb,.52),atol=1e-12,rtol=0)


def test_incomplete_profile_remains_publicly_unavailable():
 from gffx.torch import render
 from gffx.torch._opaque_stream import window
 s,c,_=rectangle('cpu',texture=True)
 for outputs in [('colour','mask'),('mask',)]:
  with pytest.raises(NotImplementedError,match='not qualified'):render(s,c,coverage='opaque',outputs=outputs)
 with pytest.raises(NotImplementedError,match='not qualified'):window(s,c,(0,0,1,1))
