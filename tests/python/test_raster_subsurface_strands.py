"""Portable CPU/CUDA scattering and explicit-strand controls; no studio assets."""
import pytest,torch,math
import gffx as g
from gffx.torch import _subsurface

@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_mean_free_path_conversion_is_scale_invariant_and_not_raw_radius(device):
 base=torch.tensor([[1.,1.,1.],[.8,.4,.2],[0.,0.,0.]],device=device)
 radius=torch.tensor([.006,.0012,.0006],device=device)
 actual=_subsurface.diffusion_length(radius,base)
 a=base.double()/base.double().amax(-1,keepdim=True).clamp_min(1e-8)
 reference=radius.double()/(1.9-a+3.5*(a-.8)**2)
 torch.testing.assert_close(actual.double(),reference,atol=1e-9,rtol=2e-6)
 torch.testing.assert_close(actual[0],radius/1.04,atol=1e-9,rtol=2e-6)
 torch.testing.assert_close(actual[:2],_subsurface.diffusion_length(radius,base[:2]*.1),atol=1e-9,rtol=2e-6)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_constant_and_boundaries(device):
 image=torch.full((9,11,3),.7,device=device);data=torch.zeros(9,11,6,device=device);data[...,0]=1;data[...,1:4]=.02;data[...,5]=1;ids=torch.zeros(9,11,dtype=torch.int32,device=device)
 output=_subsurface.diffuse(image,data,ids,100,100)
 torch.testing.assert_close(output,image,rtol=2e-6,atol=2e-7)
 ids[:,6:]=1;image[:,6:]=0
 output=_subsurface.diffuse(image,data,ids,100,100)
 torch.testing.assert_close(output[:,:6],image[:,:6],rtol=2e-6,atol=2e-7);assert output[:,6:].abs().max()==0


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_independent_diffusion(device):
 torch.manual_seed(49);image=torch.rand(13,17,3,device=device);data=torch.zeros(13,17,6,device=device);data[...,0]=1;data[...,1:4]=torch.tensor([.01,.007,.003],device=device);data[...,5]=1;ids=torch.zeros(13,17,dtype=torch.int32,device=device);data[:,9:,0]=1.5
 output=_subsurface.diffuse(image,data,ids,100,100);reference=_subsurface.reference_diffuse(image.double(),data.double(),ids,100,100)
 torch.testing.assert_close(output.double(),reference,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_spectral_spreading_and_depth_rejection(device):
 image=torch.zeros(25,25,3,device=device);image[12,12]=1;data=torch.zeros(25,25,6,device=device);data[...,0]=1;data[...,1:4]=torch.tensor([.03,.01,.003],device=device);data[...,5]=1;ids=torch.zeros(25,25,dtype=torch.int32,device=device)
 out=_subsurface.diffuse(image,data,ids,100,100)
 assert out[12,12,0]<out[12,12,1]<out[12,12,2]
 image[:,:12]=1;image[:,12:]=0;data[:,12:,0]=2
 out=_subsurface.diffuse(image,data,ids,100,100);assert out[:,13:].max()<1e-6


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_strand_geometry_and_gradient(device):
 points=torch.tensor([[[0.,0.,0.],[0.,0.,.1],[.02,0.,.2]]],device=device,requires_grad=True)
 mesh=g.Mesh.strands('hair',points,radius=.002,tip_radius=.0001,sides=6,device=device)
 assert mesh.vertices.shape==(18,3)and mesh.faces.shape==(24,3)
 v=mesh.vertices;tri=v[mesh.faces.long()];cross=torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);assert (cross.norm(dim=-1)>1e-9).all()
 normal=mesh.attributes['normals'].values[mesh.faces.long()].mean(1);assert ((cross*normal).sum(-1)>0).all()
 mesh.vertices.sum().backward();assert points.grad is not None and torch.isfinite(points.grad).all()


@pytest.mark.parametrize('bad',[dict(subsurface=-.1),dict(subsurface=1.1),dict(subsurface=.5,subsurface_radius=(-1,.1,.1)),dict(subsurface=.5,metallic=.3),dict(subsurface=.5,subsurface_transmission=2)])
def test_material_domains(bad):
 with pytest.raises((ValueError,TypeError)):g.Material.pbr('bad',(.5,.3,.2),**bad)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_zero_weight_preserves_image(device):
 scene=lambda weight:g.Scene([g.Material.pbr('skin',(.5,.3,.2),subsurface=weight,device=device),g.Mesh.sphere('sphere',radius=.3,material='skin',segments=16,rings=8,device=device),g.Light.point('key',position=(0,-1,1),power=10,device=device)])
 camera=g.Camera.look_at((0,-2,0),(0,0,0),up=(0,0,1),size=(32,32),device=device)
 a=g.render(scene(0),camera,integrator='raster',shadows=False)['colour'];b=g.render(scene(0.),camera,integrator='raster',shadows=False)['colour'];assert torch.equal(a,b)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_integrated_subsurface_render(device):
 scene=g.Scene([g.Material.pbr('skin',(.5,.3,.2),subsurface=.8,subsurface_radius=(.02,.01,.005),device=device),g.Mesh.sphere('sphere',radius=.3,material='skin',segments=16,rings=8,device=device),g.Light.point('key',position=(0,-1,1),power=10,device=device)])
 camera=g.Camera.look_at((0,-2,0),(0,0,0),up=(0,0,1),size=(32,32),device=device)
 image=g.render(scene,camera,integrator='raster',shadows=True,shadow_map_size=32)['colour']
 assert torch.isfinite(image).all()and image.max()>0


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_single_path_beer_lambert(device):
 from gffx.torch import _raster,_raster_accel
 # Analytic slab, light at z1, receiver atz0, entry atz.01/.02.
 points=torch.tensor([[0.,0.,0.]],device=device);normal=torch.tensor([[0.,0.,-1.]],device=device)
 source=points.new_zeros(1,40);source[0,0]=0;source[0,3]=1;source[0,4:7]=math.pi;source[0,17]=1
 camera=_raster._camera(points.new_tensor([0,0,1]),points.new_tensor([0,0,0]),32,near=.1,far=2)
 matrix=points.new_zeros(1,64);matrix[0,:16]=camera.world_to_clip().flatten();matrix[0,16:32]=torch.linalg.inv(camera.world_to_clip()).flatten();matrix[0,32:48]=camera.world_to_view().flatten();matrix[0,48]=camera.projection()[0,0,0];matrix[0,49]=camera.projection()[0,1,1];matrix[0,50]=.1;matrix[0,51]=1;matrix[0,52:55]=points.new_tensor([0,0,1])
 params={'base_color':points.new_ones(3),'subsurface_radius':points.new_tensor([.02,.01,.005]),'subsurface_transmission':1}
 values=[]
 for thickness in [.01,.02]:
  depth=points.new_full((1,32,32),-.1/(1-thickness));out=_subsurface.transmit(points,normal,params,(source,matrix,depth),32);values.append(out)
  torch.testing.assert_close(out[0],torch.exp(-thickness/params['subsurface_radius']),atol=1e-5,rtol=1e-5)
 assert (values[1]<values[0]).all()


def test_unused_subsurface_does_not_change_render():
 camera=g.Camera.look_at((0,0,2),(0,0,0),size=(16,16))
 prims=[g.Material.emissive('red',(1,0,0)),g.Mesh.sphere('ball',radius=.5,material='red',segments=12,rings=6)]
 base=g.render(g.Scene(prims),camera,integrator='raster')['colour']
 extra=g.Material.pbr('unused',(.5,.3,.2),subsurface=1.)
 result=g.render(g.Scene([extra,*prims]),camera,integrator='raster')['colour']
 assert torch.equal(base,result)


def test_world_normal_map_has_explicit_direction():
 from gffx.torch._deferred import TextureSource
 from gffx.torch import render
 color=torch.tensor([.85355339,.85355339,.5]).expand(2,2,3).contiguous();pyramid,offsets=render.texture_pyramid(color);tex=TextureSource(pyramid,offsets,2,2,3)
 scene=g.Scene([g.Material.pbr('mat',(1,1,1),ior=1,normal_map=tex,normal_space='world'),g.Mesh.quad('wall',u=(0,1,0),v=(0,0,1),material='mat'),g.Light.sun('sun',direction=(-1,0,0),irradiance=math.pi,angle=0)])
 cam=g.Camera.look_at((2,0,0),(0,0,0),up=(0,0,1),size=(16,16))
 out=g.render(scene,cam,integrator='raster')['colour'];torch.testing.assert_close(out[0,8,8],torch.full((3,),2**-.5),atol=2e-6,rtol=2e-6)


def test_strand_preserves_float64_input():
 p=torch.tensor([[[0.,0.,0.],[0.,0.,1.]]],dtype=torch.float64)
 assert g.Mesh.strands('hair',p).vertices.dtype==torch.float64


def test_strand_frame_gradient_matches_independent_finite_differences():
 p=torch.tensor([[[0.,0.,0.],[.03,.01,.1],[.07,-.02,.2]]],dtype=torch.float64,requires_grad=True)
 weights=torch.linspace(.2,1.7,54,dtype=torch.float64).reshape(18,3)
 def loss(q):return(g.Mesh.strands('hair',q,radius=.002,tip_radius=.0001,sides=6).vertices.square()*weights).sum()
 assert torch.autograd.gradcheck(loss,(p,),eps=1e-6,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_compact_rectangle_caster_selection_matches_legacy(device,monkeypatch):
 from gffx.torch import _raster
 torch.manual_seed(13);v=torch.rand(120,3,device=device)*2-1;f=torch.arange(120,dtype=torch.int32,device=device).reshape(-1,3)
 center=torch.tensor([.31,.28,.09],device=device);u=torch.tensor([1.,.2,.1],device=device);span=torch.tensor([-.1,1.,.3],device=device);normal=torch.linalg.cross(u,span);normal/=normal.norm()
 expected=f[((v[f.long()]-center)*normal).sum(-1).amax(-1)>-1e-5]
 scene=g.Scene([g.Material.diffuse('white',(1,1,1),device=device),g.Mesh('geometry',v,f,material='white'),g.Light.rectangle('area',center=center,u=u,v=span,radiance=(1,1,1),samples=1,visible=False,device=device)]);packed=scene.flatten();captured=[]
 def geometry(vertices,faces,camera):captured.append(faces.clone());return vertices,faces
 monkeypatch.setattr(_raster,'_shadow_geometry',geometry);monkeypatch.setattr(_raster,'_make_map',lambda vertices,faces,camera,prepared=False:{'depth':v.new_zeros(32,32)})
 _raster._light_sources(packed,v,f,32,True,None);assert len(captured)==1;assert torch.equal(captured[0],expected)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_orthographic_diffusion_scale_is_depth_invariant(device):
 torch.manual_seed(31);image=torch.rand(13,17,3,device=device);data=torch.zeros(13,17,6,device=device);data[...,0]=1;data[...,1:4]=torch.tensor([.02,.01,.005],device=device);data[...,5]=1;ids=torch.zeros(13,17,dtype=torch.int32,device=device)
 a=_subsurface.diffuse(image,data,ids,100,100,perspective=False);data[...,0]=3;b=_subsurface.diffuse(image,data,ids,100,100,perspective=False);assert torch.equal(a,b)
 reference=_subsurface.reference_diffuse(image.double(),data.double(),ids,100,100,perspective=False);torch.testing.assert_close(a.double(),reference,atol=2e-5,rtol=2e-5)
