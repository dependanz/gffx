import math
from dataclasses import replace
import pytest
import torch
import gffx as g
from gffx.torch import scene as s,light
from gffx.torch._raster_material import pbr
from gffx.torch._raster import _maps,_shadow,_shadow_group,_texture


@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    return request.param


def plane(device,mat=None):
    mat=mat or g.Material.diffuse('surface',(.6,.6,.6),device=device)
    mesh=g.Mesh.quad('plane',center=(0,0,0),u=(2,0,0),v=(0,2,0),material='surface',device=device)
    camera=g.Camera.look_at((0,0,3),(0,0,0),size=(17,17),fov=30,device=device)
    return mat,mesh,camera


def test_raster_point_matches_independent_lambert_inverse_square(device):
    mat,mesh,cam=plane(device)
    lamp=g.Light.point('source',position=(0,0,2),power=4*math.pi,device=device)
    image=g.render(g.Scene([mat,mesh,lamp]),cam,integrator='raster')['colour']
    assert torch.allclose(image[0,8,8],torch.full((3,),.6/(4*math.pi),device=device),atol=2e-6,rtol=1e-5)


def test_pbr_normal_incidence_matches_independent_ggx_formula(device):
    n=torch.tensor([[0.,0.,1.]],device=device,dtype=torch.float64)
    base=n.new_tensor([.2,.4,.7]);rough=.4;f0=.04
    result=pbr(n,n,n[:,None],base_color=base,roughness=rough)
    expected=(1-f0)*base/math.pi+f0/(4*math.pi*rough**4)
    assert torch.allclose(result[0,0],expected,atol=1e-12,rtol=1e-10)
    metal=pbr(n,n,n[:,None],base_color=base,roughness=rough,metallic=1.)
    assert torch.allclose(metal[0,0],base/(4*math.pi*rough**4),atol=1e-12,rtol=1e-10)


def test_pbr_material_derivative_matches_central_difference(device):
    n=torch.tensor([[0.,0.,1.]],device=device,dtype=torch.float64)
    l=n.new_tensor([[[.3,.2,1.]]]);rough=n.new_tensor(.37,requires_grad=True)
    def value(r):return pbr(n,n,l,base_color=n.new_tensor([.2,.4,.7]),roughness=r,coat=.3).sum()
    grad=torch.autograd.grad(value(rough),rough)[0];step=1e-5
    numerical=(value(rough.detach()+step)-value(rough.detach()-step))/(2*step)
    assert torch.allclose(grad,numerical,atol=1e-7,rtol=1e-5)


def test_spot_direction_and_sun_units(device):
    spot=g.Light.spot('spot',position=(0,0,2),direction=(0,0,-1),power=4*math.pi,angle=40,device=device)
    points=torch.tensor([[0.,0.,0.],[2.,0.,2.]],device=device)
    result=spot.evaluator(points,**spot.parameters)
    assert torch.allclose(result['irradiance'][0,0],points.new_full((3,),.25))
    assert torch.equal(result['irradiance'][1,0],points.new_zeros(3))
    sun=g.Light.sun('sun',direction=(0,0,-1),irradiance=2.,angle=0,device=device)
    assert torch.equal(sun.parameters['directions'],points.new_tensor([[0,0,1]]))
    assert torch.equal(sun.parameters['irradiance'],points.new_full((1,3),2.))


def test_sun_shadows_and_shared_map_lookup(device):
    mat,floor,cam=plane(device)
    blocker=g.Mesh.quad('blocker',center=(0,0,.5),u=(.5,0,0),v=(0,.5,0),material='surface',device=device)
    packed=g.Scene([mat,floor,blocker]).flatten()
    entries=_maps(packed.vertices,packed.faces,packed.vertices.new_tensor([0,0,1]),directional=True,size=128)
    points=packed.vertices.new_tensor([[0,0,0],[.7,0,0]])
    normals=points.new_tensor([[0,0,1],[0,0,1]])
    visibility=_shadow(entries,points,normals)
    assert visibility[0]<.01 and visibility[1]>.99
    grouped=_shadow_group(entries,points,normals)[:,0]
    assert torch.allclose(grouped,visibility,atol=1e-6)


def test_point_shadows_cover_opposite_hemispheres(device):
    v=torch.tensor([[-.5,-.5,1],[.5,-.5,1],[0,.5,1],[-.5,-.5,-1],[0,.5,-1],[.5,-.5,-1]],device=device)
    f=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32,device=device)
    entries=_maps(v,f,v.new_zeros(3),size=96)
    points=v.new_tensor([[0,0,2],[0,0,-2],[2,0,0]])
    normals=v.new_tensor([[0,0,1],[0,0,-1],[1,0,0]])
    weights=_shadow(entries,points,normals)
    assert weights[0]<.01 and weights[1]<.01 and weights[2]>.99


def test_texture_footprint_uses_mips(device):
    from gffx.torch._deferred import TextureSource
    image=torch.tensor([[(x+y)%2 for x in range(8)] for y in range(8)],device=device,dtype=torch.float32)[...,None]
    pyramid,offsets=g.render.texture_pyramid(image)
    source=TextureSource(pyramid,offsets,8,8,1,scalar=True)
    uv=image.new_tensor([[.01,.01]]);dx=image.new_tensor([[.5,0]]);dy=image.new_tensor([[0,.5]])
    assert torch.allclose(_texture(source,uv,dx,dy),image.new_tensor([.5]),atol=1e-6)


def test_pbr_render_and_explicit_gradient_refusal(device):
    mat=g.Material.pbr('surface',(.2,.4,.7),roughness=.35,coat=.3,device=device)
    _,mesh,cam=plane(device,mat)
    lamp=g.Light.sun('sun',direction=(0,0,-1),irradiance=1,angle=0,device=device)
    image=g.render(g.Scene([mat,mesh,lamp]),cam,integrator='raster',supersample=2)['colour']
    assert torch.isfinite(image).all() and image.max()>0
    bad=replace(mat,parameters={**mat.parameters,'roughness':mat.parameters['roughness'].clone().requires_grad_()})
    with pytest.raises(ValueError,match='forward-only'):g.render(g.Scene([bad,mesh,lamp]),cam,integrator='raster')


@pytest.mark.parametrize('key,value',[('roughness',0.),('metallic',1.1),('ior',.5),('coat',-1.)])
def test_pbr_bad_parameters_fail_at_authoring(key,value):
    with pytest.raises(ValueError):g.Material.pbr('bad',(.2,.3,.4),**{key:value})


@pytest.mark.parametrize('classic',[False,True])
def test_fused_native_matches_unfused_radiometric_material_math(device,classic):
    from gffx.torch import _raster_accel as accel,material
    from gffx.torch._raster_material import unit
    generator=torch.Generator().manual_seed(527)
    p=(torch.rand((103,3),generator=generator)-.5).to(device);p[:,2]=0
    n=unit(torch.cat((p[:,:2]*.2,torch.ones_like(p[:,:1])),1));geo=torch.zeros_like(n);geo[:,2]=1
    v=unit(p.new_tensor([0,0,2])-p)
    sources=[dict(kind='point',position=p.new_tensor([-.5,.2,2]),intensity=p.new_tensor([.2,.3,.4]),maps=[]),dict(kind='sun',direction=unit(p.new_tensor([.3,-.2,1])),irradiance=p.new_tensor([.3,.2,.1]),maps=[])]
    params=dict(albedo=p.new_tensor([.2,.4,.7]),specular=p.new_tensor(.2),exponent=p.new_tensor(23.)) if classic else dict(base_color=p.new_tensor([.2,.4,.7]),roughness=p.new_tensor(.3),metallic=p.new_tensor(.4),ior=p.new_tensor(1.7),coat=p.new_tensor(.6),coat_roughness=p.new_tensor(.2))
    data=accel.pack_sources(sources,p,32);actual=accel.shade(p,n,geo,v,params,data,32,classic=classic)
    assert actual is not None
    expected=torch.zeros_like(p)
    for source in sources:
        if source['kind']=='sun':l=source['direction'].expand_as(p);incoming=source['irradiance']
        else:offset=source['position']-p;l=unit(offset);incoming=source['intensity']/offset.square().sum(-1)[:,None]
        brdf=(material.classic if classic else pbr)(n,v,l[:,None],_geometric_normals=geo,**params)[:,0]
        expected+=brdf*incoming*(n*l).sum(-1).clamp_min(0)[:,None]
    assert torch.allclose(actual,expected,atol=3e-6,rtol=3e-4), (actual-expected).abs().max()


def test_fused_receiver_plane_shadow_matches_independent_unfused_pcf(device):
    from gffx.torch import _raster_accel as accel
    mat,floor,cam=plane(device)
    blocker=g.Mesh.quad('blocker',center=(0,0,.5),u=(.5,0,0),v=(0,.5,0),material='surface',device=device)
    packed=g.Scene([mat,floor,blocker]).flatten();p=packed.vertices.new_tensor([[0,0,0],[.7,0,0],[.25,.1,0]])
    n=p.new_tensor([[0,0,1]]).expand_as(p);entries=_maps(packed.vertices,packed.faces,p.new_tensor([0,0,1]),directional=True,size=128)
    source=dict(kind='sun',direction=p.new_tensor([0,0,1]),irradiance=p.new_ones(3),maps=entries)
    data=accel.pack_sources([source],p,128)
    actual=accel.shade(p,n,n,n,dict(albedo=p.new_ones(3),specular=p.new_tensor(0.),exponent=p.new_tensor(1.)),data,128,classic=True)
    expected=_shadow(entries,p,n)[:,None].expand_as(p)/math.pi
    assert torch.allclose(actual,expected,atol=2e-5,rtol=2e-4),(actual,expected)


def test_point_shadow_preserves_one_centimetre_contact_separation(device):
    mat,floor,cam=plane(device)
    blocker=g.Mesh.quad('nearby',center=(0,0,.01),u=(.4,0,0),v=(0,.4,0),material='surface',device=device)
    packed=g.Scene([mat,floor,blocker]).flatten()
    entries=_maps(packed.vertices,packed.faces,packed.vertices.new_tensor([0,0,2]),size=128)
    points=packed.vertices.new_tensor([[0,0,0],[.7,0,0]]);n=points.new_tensor([[0,0,1]]).expand_as(points)
    value=_shadow(entries,points,n)
    assert value[0]<.01 and value[1]>.99,value


def test_fragment_compaction_is_independent_of_block_size(device):
    mat,mesh,cam=plane(device)
    lamp=g.Light.point('source',position=(0,0,2),power=4*math.pi,device=device)
    scene=g.Scene([mat,mesh,lamp])
    a=g.render(scene,cam,integrator='raster',fragment_block=17,supersample=2)
    b=g.render(scene,cam,integrator='raster',fragment_block=4096,supersample=2)
    assert torch.equal(a['mask'],b['mask'])
    assert torch.allclose(a['colour'],b['colour'],atol=2e-7,rtol=1e-6)


def test_rectangle_caster_filter_keeps_emitting_hemisphere(device):
    from gffx.torch._raster import _light_sources
    from gffx.torch import _raster_accel as accel
    mat,floor,cam=plane(device)
    blocker=g.Mesh.quad('nearby',center=(0,0,.01),u=(.4,0,0),v=(0,.4,0),material='surface',device=device)
    lamp=g.Light.rectangle('area',center=(0,0,2),u=(.1,0,0),v=(0,-.1,0),radiance=10,samples=1,device=device)
    packed=g.Scene([mat,floor,blocker,lamp]).flatten()
    sources=_light_sources(packed,packed.vertices,packed.faces,128,True,None)
    p=packed.vertices.new_tensor([[0,0,0],[.7,0,0]]);n=p.new_tensor([[0,0,1]]).expand_as(p)
    visibility=_shadow(sources[0]['maps'],p,n)
    assert visibility[0]<.01 and visibility[1]>.99,visibility
    assert torch.isfinite(sources[0]['maps'][0]['depth']).any()


def test_cube_shadow_clips_crossing_caster_instead_of_dropping_it(device):
    v=torch.tensor([[-1.,-1.,-1.],[1.,-1.,1.],[0.,1.,1.]],device=device)
    f=torch.tensor([[0,1,2]],device=device,dtype=torch.int32)
    maps=_maps(v,f,v.new_zeros(3),size=128)
    p=v.new_tensor([[0,0,2.]])
    assert _shadow(maps,p,v.new_tensor([[0,0,1.]]))[0]<.01


@pytest.mark.parametrize('size',[(17,19),(128,128)])
@pytest.mark.parametrize('cull',[1,2,3])
def test_binned_cuda_raster_matches_native_face_order_edges_and_depth(size,cull):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from gffx.torch import _raster_accel as accel
    generator=torch.Generator().manual_seed(916)
    vertices=(torch.rand((360,3),generator=generator)*2.4-1.2).cuda();vertices[:,2]*=.5
    # Include exactly coplanar duplicate triangles and thin/subpixel faces.
    faces=torch.arange(360,device='cuda',dtype=torch.int32).reshape(-1,3)
    faces=torch.cat((faces,faces[:3],faces[3:6].flip(1)))
    expected=g.render.rasterize(vertices,faces,*size,cull_mode=cull)
    actual=accel.raster(vertices,faces,*size,cull)
    assert actual is not None
    assert torch.equal(actual[0],expected[0].reshape(-1))
    assert torch.equal(actual[1],expected[1].reshape(-1,3))
    assert torch.equal(actual[2],expected[2].reshape(-1))


def test_requested_texture_gradients_are_not_silently_discarded():
    from gffx.torch._deferred import TextureSource
    image=torch.ones((2,2,3));pyramid,offsets=g.render.texture_pyramid(image)
    source=TextureSource(pyramid.requires_grad_(),offsets,2,2,3)
    mat=g.Material.pbr('surface',source);_,mesh,cam=plane('cpu')
    with pytest.raises(ValueError,match='forward-only'):
        g.render(g.Scene([mat,mesh]),cam,integrator='raster')


@pytest.mark.parametrize('origin,expected',[('top-left',[0.,0.,1.]),('bottom-left',[1.,0.,0.])])
def test_portable_bundle_declares_image_origin_and_lower_left_uvs(tmp_path,origin,expected):
    import json,struct,zipfile
    image=[1.,0.,0.,0.,1.,0.,0.,0.,1.,1.,1.,1.]
    identity=[1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.]
    def record(file,shape):return dict(file=file,dtype='float32',shape=shape)
    spec=dict(format='gffx-scene-1',textures={'control':{**record('texture.bin',[2,2,3]),'origin':origin}},materials=[dict(name='surface',kind='pbr',parameters={'base_color':{'texture':'control'}})],meshes=[],lights=[],camera=dict(fx=10,fy=10,cx=1,cy=1,width=2,height=2,near=.025,far=10,pose=record('pose.bin',[4,4])))
    path=tmp_path/'fixture.gffx'
    with zipfile.ZipFile(path,'w')as z:
        z.writestr('scene.json',json.dumps(spec));z.writestr('texture.bin',struct.pack('<12f',*image));z.writestr('pose.bin',struct.pack('<16f',*identity))
    model=g.Scene.load(path,device='cpu');source=model.prims[0].parameters['base_color']
    uv=torch.tensor([[.25,.25]]);zero=torch.zeros_like(uv)
    assert torch.equal(_texture(source,uv,zero,zero),torch.tensor([expected]))


def test_geometry_outputs_remain_pixel_centred_under_shaded_antialiasing(device):
    mat,mesh,cam=plane(device);model=g.Scene([mat,mesh]);outputs=('depth','face_index')
    plain=g.render(model,cam,integrator='raster',outputs=outputs)
    aa=g.render(model,cam,integrator='raster',supersample=4,outputs=outputs)
    assert torch.equal(plain['face_index'],aa['face_index'])
    assert torch.equal(plain['depth'],aa['depth'])
