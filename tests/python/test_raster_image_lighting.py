import math
from dataclasses import replace
import pytest
import torch
import gffx as g
from gffx.torch.scene import Attribute
from gffx.torch._raster_lighting import cube_directions,_sample_cube,_irradiance_coefficients,_sh,surface

DEVICES=['cpu']+(['cuda']if torch.cuda.is_available()else[])
@pytest.fixture(params=DEVICES)
def device(request):return request.param

def test_solid_angle_and_constant_irradiance(device):
    rgb=torch.tensor([.2,.4,.7],device=device);cube=rgb.expand(6,16,16,3).clone()
    coefficients=_irradiance_coefficients(cube);n=cube_directions(9,cube).reshape(-1,3)
    actual=(_sh(n)@coefficients).clamp_min(0)
    torch.testing.assert_close(actual,math.pi*rgb.expand_as(actual),atol=2e-5,rtol=2e-5)

def test_cube_axes_seams_and_filter_constant(device):
    d=cube_directions(32,torch.zeros(1,device=device));image=d*.4+.5
    query=torch.tensor([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,.999,0],[.999,1,0],[1,1,1]],device=device,dtype=torch.float32)
    query=query/query.norm(dim=-1,keepdim=True)
    torch.testing.assert_close(_sample_cube(image,query),query*.4+.5,atol=.013,rtol=0)
    env=g.Environment.from_cubemap(torch.ones(6,16,16,3,device=device)*.7,samples=32)
    for rough in [0,.15,.6,1]:torch.testing.assert_close(env.sample(query,rough),torch.ones_like(query)*.7,atol=2e-5,rtol=0)

def test_latlong_direction_orientation(device):
    h,w=64,128;phi=(torch.arange(w,device=device)+.5)/w*2*math.pi-math.pi;theta=(torch.arange(h,device=device)+.5)/h*math.pi
    t,p=torch.meshgrid(theta,phi,indexing='ij');image=torch.stack((t.sin()*p.cos(),t.sin()*p.sin(),t.cos()),-1)*.4+.5
    env=g.Environment.from_latlong(image,size=16,samples=16);d=cube_directions(8,image).reshape(-1,3)
    torch.testing.assert_close(env.sample(d),d*.4+.5,atol=.017,rtol=0)

def test_box_parallax_exact_room_intersection(device):
    like=torch.zeros(1,3,device=device);probe=g.ReflectionProbe((0,0,0),bounds=((-1,-1,-1),(1,1,1)))
    p=like+torch.tensor([.6,0,0],device=device);d=like+torch.tensor([0,1,0],device=device)
    expected=torch.tensor([[.6,1,0]],device=device);expected/=expected.norm(dim=-1,keepdim=True)
    torch.testing.assert_close(probe.direction(p,d),expected)

def test_diffuse_volume_interpolation_and_bounds(device):
    low=torch.zeros(3,device=device);high=torch.ones(3,device=device);coef=torch.zeros(2,2,2,9,3,device=device)
    coef[:,:,:,0]=1/.28209479177387814;coef[1,:,:,0]*=3
    volume=g.IrradianceVolume((low,high),coef,normal_bias=0)
    p=torch.tensor([[0,.5,.5],[.5,.5,.5],[1,.5,.5],[2,.5,.5]],device=device);n=torch.tensor([[0,0,1.]],device=device).expand_as(p)
    values,inside=volume.sample(p,n);torch.testing.assert_close(values[:3],torch.tensor([[1.]*3,[2.]*3,[3.]*3],device=device));assert inside.tolist()==[True,True,True,False]

def test_constant_environment_lambert_energy(device):
    env=g.Environment.constant([.2,.4,.6],device=device);p=torch.zeros(1,3,device=device);n=p+torch.tensor([0,0,1.],device=device)
    value=surface(p,n,n,dict(albedo=torch.tensor([.3,.5,.7],device=device),specular=torch.zeros(3,device=device)),environment=env,probes=(),irradiance=None,classic=True)
    torch.testing.assert_close(value,torch.tensor([[.06,.2,.42]],device=device),atol=2e-5,rtol=0)

def test_primary_clipping_preserves_linear_attributes(device):
    from gffx.torch._raster_clip import clip
    v=torch.tensor([[-1.,-.5,-.5],[1.,-.5,-2],[0,1,-2]],device=device);f=torch.tensor([[0,1,2]],device=device,dtype=torch.int32)
    prim=g.Mesh('tri',v,f,{'colour':Attribute(v.clone(),'vertex')},'diffuse');scene=g.Scene([g.Material.diffuse('diffuse',(.5,.5,.5),device=device),prim]);packed=scene.flatten()
    camera=g.Camera.look_at((0,0,0),(0,0,-1),size=(16,16),near=1,device=device)
    result,ids=clip(packed,camera);assert len(result.faces)==2 and ids.tolist()==[0,0]
    torch.testing.assert_close(result.vertices[result.faces.long()],result.attributes['colour'].values)
    assert bool((-result.vertices[result.faces.long()][...,2]>=1).all())

def test_planar_camera_obeys_reflection_law(device):
    from gffx.torch.scene import Camera as CameraPrim
    # Emission geometry makes this an independent visibility control, no BRDF/light tuning.
    mat=g.Material.diffuse('red',(0,0,0),device=device);mat=replace(mat,emission=torch.tensor([1.,0,0],device=device))
    obj=g.Mesh.quad('red-object',center=(0,-.5,.5),u=(.4,0,0),v=(0,0,.4),material='red',device=device)
    # Face needs to point toward the reflected camera, behind the mirror plane.
    obj=replace(obj,faces=obj.faces[:,[0,2,1]])
    scene=g.Scene([mat,obj]);camera=g.Camera.look_at((0,-2,1),(0,0,.5),up=(0,0,1),size=(40,40),device=device)
    mirror=g.PlanarReflection('mirror',(0,0,0),(0,-1,0))
    snap=mirror.capture(scene,camera,supersample=1)
    torch.testing.assert_close(snap['camera'].pose[0,:3,3],torch.tensor([0.,2,1],device=device))
    assert float(snap['image'][...,0].max())>.99

def test_descriptor_and_snapshot_updates(device):
    mat=g.Material.diffuse('red',(0,0,0),device=device);mat=replace(mat,emission=torch.tensor([1.,0,0],device=device))
    obj=g.Mesh.sphere('mover',center=(1,0,0),radius=.2,segments=16,rings=8,material='red',device=device)
    scene=g.Scene([mat,obj]);probe=g.ReflectionProbe((0,0,0),resolution=8,samples=16)
    snapshot=probe.capture(scene,shadows=False);d=torch.tensor([[1.,0,0]],device=device)
    assert float(snapshot.environment.sample(d)[0,0])>.5
    moved=g.Scene([mat,replace(obj,vertices=obj.vertices+obj.vertices.new_tensor([0,1,0]))]);updated=probe.capture(moved,shadows=False)
    assert float(updated.environment.sample(d)[0,0])<.05
    assert float(snapshot.environment.sample(d)[0,0])>.5

def test_invalid_inputs():
    with pytest.raises(ValueError):g.Environment.from_cubemap(torch.full((6,4,4,3),-1.))
    with pytest.raises(ValueError):g.Environment.from_latlong(torch.ones(8,16,3),size=7)
    with pytest.raises(ValueError):g.Environment.from_cubemap(torch.ones(6,4,4,3,requires_grad=True))

def test_atlas_matches_independent_seam_taps(device):
    from gffx.torch._raster_lighting import _atlas,_sample_atlas
    generator=torch.Generator().manual_seed(91);image=torch.rand(6,16,16,3,generator=generator).to(device)
    directions=torch.randn(2048,3,generator=generator).to(device)
    actual=_sample_atlas(_atlas(image),directions);reference=_sample_cube(image,directions)
    torch.testing.assert_close(actual,reference,atol=1e-5,rtol=2e-5)

def test_split_sum_constant_furnace_no_energy_creation(device):
    env=g.Environment.constant([1.,1.,1.],device=device)
    n=torch.tensor([[0.,0.,1.]],device=device).expand(32,3);p=torch.zeros_like(n)
    nv=torch.linspace(.05,1,32,device=device);v=torch.stack(((1-nv*nv).sqrt(),nv*0,nv),-1)
    for rough in (.05,.3,.8):
        value=surface(p,n,v,dict(base_color=torch.ones(3,device=device),roughness=torch.tensor(rough,device=device),metallic=torch.tensor(1.,device=device)),environment=env,probes=(),irradiance=None)
        assert bool((value>=0).all()) and float(value.max())<1.03

@pytest.mark.parametrize('classic',[False,True])
def test_native_image_lighting_against_tensor_fallback(device,classic):
    from gffx.torch import _raster_accel as accel
    generator=torch.Generator().manual_seed(74)
    def rand(*shape):return torch.rand(*shape,generator=generator).to(device)
    env=g.Environment.from_cubemap(rand(6,16,16,3),samples=32)
    probe=g.ReflectionProbe((.1,0,0),bounds=((-1,-1,-1),(1,1,1)),environment=g.Environment.from_cubemap(rand(6,16,16,3),samples=32))
    coef=(rand(3,3,2,9,3)-.2)*.3;coef[:,:,:,0]+=1
    volume=g.IrradianceVolume((torch.full((3,),-.8,device=device),torch.full((3,),.8,device=device)),coef)
    points=rand(513,3)*2-1;normal=rand(513,3)*2-1;normal/=normal.norm(dim=-1,keepdim=True);view=rand(513,3)*2-1;view/=view.norm(dim=-1,keepdim=True)
    camera=g.Camera.look_at((0,0,3),(0,0,0),size=(17,23),device=device)
    capture=dict(image=rand(17,23,3),camera=camera);descriptor=g.PlanarReflection('test',(0,0,0),(0,0,1))
    ids=torch.full((513,),-1,device=device,dtype=torch.int32);ids[::3]=0
    data=accel.pack_indirect(env,[probe],volume,[(descriptor,capture)],points)
    params=dict(albedo=rand(513,3),specular=torch.zeros(3,device=device))if classic else dict(base_color=rand(513,3),roughness=rand(513)*.95+.04,metallic=rand(513),ior=rand(513)+1.1,coat=rand(513),coat_roughness=rand(513)*.95+.04)
    actual=accel.indirect(points,normal,view,params,data,ids,classic=classic);assert actual is not None
    expected=surface(points,normal,view,params,environment=env,probes=[probe],irradiance=volume,mirror=[dict(selected=ids==0,capture=capture)],classic=classic)
    torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-4)

def test_directional_diffuse_integral_analytic(device):
    cube=cube_directions(64,torch.zeros(1,device=device));image=cube*0
    image[...,0]=.5+.2*cube[...,2];image[...,1]=.4+.15*cube[...,0];image[...,2]=.6+.1*cube[...,1]
    coefficients=_irradiance_coefficients(image);normal=cube_directions(7,image).reshape(-1,3)
    expected=normal.new_tensor([.5,.4,.6])*math.pi+torch.stack((.2*normal[:,2],.15*normal[:,0],.1*normal[:,1]),-1)*(2*math.pi/3)
    torch.testing.assert_close(_sh(normal)@coefficients,expected,atol=3e-4,rtol=0)

def test_rough_filter_against_dense_polar_quadrature(device):
    # Rotationally symmetric analytic signal lets an independent 1-D polar
    # integral check the normalized GGX prefilter without reusing its sampler.
    cube=cube_directions(64,torch.zeros(1,device=device));image=(.5+.4*cube[...,2:3]).expand(-1,-1,-1,3).contiguous()
    env=g.Environment.from_cubemap(image,samples=256);direction=torch.tensor([[0.,0,1]],device=device)
    t=(torch.arange(32768,dtype=torch.float64)+.5)/32768*(1-math.sqrt(.5))+math.sqrt(.5);nl=2*t*t-1
    for rough in (2/6,3/6):
        a2=rough**4;density=a2/(math.pi*(t*t*(a2-1)+1)**2);weight=nl*density*t
        expected=float(((.5+.4*nl)*weight).sum()/weight.sum());actual=float(env.sample(direction,rough)[0,0])
        assert abs(actual-expected)<.018,(rough,actual,expected)

def test_float64_fallback_and_lighting_validation():
    world=g.Environment.constant((.2,.4,.6),dtype=torch.float64)
    material=g.Material.diffuse('surface',(.3,.5,.7),dtype=torch.float64)
    mesh=g.Mesh.quad('quad',center=(0,0,-2),u=(2,0,0),v=(0,2,0),material='surface',dtype=torch.float64)
    scene=g.Scene([material,mesh]);camera=g.Camera.look_at((0,0,0),(0,0,-1),size=(8,8),dtype=torch.float64)
    out=g.render(scene,camera,integrator='raster',environment=world)
    torch.testing.assert_close(out['colour'][0,4,4],torch.tensor([.06,.2,.42],dtype=torch.float64),atol=2e-5,rtol=0)
    with pytest.raises(ValueError,match='dtype/device'):g.render(scene,camera,integrator='raster',environment=g.Environment.constant((1,1,1)))
