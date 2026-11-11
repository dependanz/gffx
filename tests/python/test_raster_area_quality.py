import math
import pytest
import torch
import gffx as g
from gffx.torch import _raster_accel as accel
from gffx.torch._ltc import evaluate
from gffx.torch._raster_material import pbr,unit
from gffx.torch._raster import _light_sources

@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    return request.param

def fixture(device,rough=.3):
    p=torch.tensor([[-.2,.1,0],[0,0,0],[.2,-.1,0]],device=device)
    n=p.new_tensor([[0,0,1]]).expand_as(p);v=unit(p.new_tensor([.4,-.2,2])-p)
    lamp=g.Light.rectangle('area',center=(0,0,1.2),u=(.8,0,0),v=(0,-.6,0),radiance=3,samples=8,device=device)
    mesh=g.Mesh.quad('plane',center=(0,0,0),u=(2,0,0),v=(0,2,0),material='surface',device=device)
    mat=g.Material.pbr('surface',(.2,.4,.7),roughness=rough,coat=.4,device=device)
    packed=g.Scene([mat,mesh,lamp]).flatten();sources=_light_sources(packed,packed.vertices,packed.faces,32,False,None)
    return p,n,v,mat.parameters,sources

def test_continuous_area_native_matches_tensor_clipping_and_fit(device):
    p,n,v,params,sources=fixture(device)
    expected=evaluate(p,n,n,v,params,sources[0]['area'])
    actual=accel.shade(p,n,n,v,params,accel.pack_sources(sources,p,32),32)
    assert torch.allclose(actual,expected,atol=3e-6,rtol=1e-4),(actual,expected)

@pytest.mark.parametrize('rough',[.12,.3,.6])
def test_ltc_area_against_independent_dense_ggx_quadrature(device,rough):
    p,n,v,params,sources=fixture(device,rough)
    # A single metallic lobe isolates the LTC fit from layered-coat approximations.
    params={**params,'metallic':p.new_tensor(1.),'coat':p.new_tensor(0.)}
    expected=[];axis=(torch.arange(256,device=device,dtype=torch.float64)+.5)/256-.5
    x,y=torch.meshgrid(axis,axis,indexing='ij');area=sources[0]['area']
    q=area['center'].double()+x.flatten()[:,None]*area['u'].double()+y.flatten()[:,None]*area['v'].double()
    normal=unit(torch.linalg.cross(area['u'].double(),area['v'].double()));size=torch.linalg.vector_norm(torch.linalg.cross(area['u'].double(),area['v'].double()))
    for point,norm,view in zip(p,n,v):
        delta=q-point.double();distance=delta.square().sum(-1);direction=unit(delta)
        weight=(direction*norm.double()).sum(-1).clamp_min(0)*(-(direction*normal).sum(-1)).clamp_min(0)/distance
        brdf=pbr(norm.double()[None],view.double()[None],direction[None],**{k:a.double() for k,a in params.items()})[0]
        expected.append((brdf*weight[:,None]).mean(0)*size*area['radiance'].double())
    expected=torch.stack(expected);actual=evaluate(p,n,n,v,params,area).double()
    assert torch.allclose(actual,expected,atol=4e-4,rtol=.12),(rough,actual,expected)

def test_rectangle_horizon_clipping_and_backside(device):
    p,n,v,params,sources=fixture(device)
    area={**sources[0]['area'],'center':p.new_tensor([1,0,.1])}
    tilted=unit(p.new_tensor([[1,0,.03]]).expand_as(n));value=evaluate(p,tilted,tilted,v,params,area)
    assert torch.isfinite(value).all() and (value>=0).all()
    back={**area,'v':-area['v']}
    assert torch.equal(evaluate(p,n,n,v,params,back),torch.zeros_like(p))

def test_cosine_domain_spot_blend_native_and_public(device):
    outer=math.cos(math.radians(45)/2);cosine=outer+(1-outer)*.15*.5
    p=torch.tensor([[math.sqrt(1-cosine*cosine),0,-cosine]],device=device)
    n=p.new_tensor([[0,0,1]]);source=g.Light.spot('spot',position=(0,0,0),direction=(0,0,-1),power=4*math.pi,angle=45,blend=.15,device=device)
    packet=source.evaluator(p,**source.parameters)
    assert torch.allclose(packet['irradiance'][0,0],p.new_full((3,),.5),atol=2e-5)
    row=dict(kind='spot',position=p.new_zeros(3),intensity=p.new_ones(3),direction=p.new_tensor([0,0,-1]),angle=math.radians(45),blend=.15,maps=[])
    actual=accel.shade(p,n,n,n,dict(albedo=p.new_ones(3)),accel.pack_sources([row],p,32),32,classic=True)
    assert torch.allclose(actual,p.new_full((1,3),cosine*.5/math.pi),atol=1e-5)

def test_shadow_maps_share_one_storage_without_changing_lookup(device):
    p,n,v,params,sources=fixture(device)
    mat=g.Material.diffuse('surface',(.6,.6,.6),device=device)
    floor=g.Mesh.quad('floor',center=(0,0,0),u=(2,0,0),v=(0,2,0),material='surface',device=device)
    lamp=g.Light.rectangle('area',center=(0,0,1),u=(.3,0,0),v=(0,-.3,0),samples=2,radiance=1,device=device)
    model=g.Scene([mat,floor,lamp]).flatten();sources=_light_sources(model,model.vertices,model.faces,32,True,None)
    maps=[entry['depth'] for s in sources for entry in s['maps']]
    assert len({x.untyped_storage().data_ptr()for x in maps})==1
    data=accel.pack_sources(sources,p,32)
    assert data[2].untyped_storage().data_ptr()==maps[0].untyped_storage().data_ptr()
    assert torch.equal(data[2],torch.stack(maps))

@pytest.mark.parametrize('size',[(19,23),(128,128),(511,509)])
def test_depth_only_shadow_matches_independent_native_raster(size):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    generator=torch.Generator().manual_seed(1167)
    ndc=(torch.rand((1200,3),generator=generator)*2.6-1.3).cuda();ndc[:,2]*=.5
    faces=torch.arange(1200,device='cuda',dtype=torch.int32).reshape(-1,3)
    faces=torch.cat((faces,faces[:3].flip(1),faces[:3]))
    expected=g.render.rasterize(ndc,faces,*size,cull_mode=1)[2].reshape(size)
    actual=accel.shadow(ndc,faces,*size)
    assert torch.equal(actual,expected),(actual-expected).abs().nan_to_num().max()

def test_continuous_highlight_stays_stable_between_old_grid_samples(device):
    p,n,v,params,sources=fixture(device,.08)
    p=torch.zeros((17,3),device=device);p[:,0]=torch.linspace(-.02,.02,17,device=device)
    n=p.new_tensor([[0,0,1]]).expand_as(p);v=n
    params={**params,'metallic':p.new_tensor(1.),'coat':p.new_tensor(0.)}
    actual=evaluate(p,n,n,v,params,sources[0]['area'])
    # Reflection stays well inside the emitter: moving across a point-grid cell
    # must not make a nearly mirror-like full-area highlight disappear.
    assert bool((actual.amin(0)>.5*actual.amax(0)).all())
    assert (actual[1:]-actual[:-1]).abs().max()<.01*actual.amax()
