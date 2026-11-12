"""Independent pixel-area/segment references and meaningful AA regressions."""
from dataclasses import replace
import pytest
import torch
import gffx as g
from gffx.torch import _deferred as d,_area,light
from segment_reference import visibility

@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    return request.param

def clipped_area(polygon,x,y):
    # Sutherland-Hodgman against an axis-aligned pixel, then shoelace area.
    for axis,bound,sign in [(0,x,1),(0,x+1,-1),(1,y,1),(1,y+1,-1)]:
        output=[]
        if not polygon:return 0.
        previous=polygon[-1];pa=sign*(previous[axis]-bound)>=0
        for current in polygon:
            ca=sign*(current[axis]-bound)>=0
            if ca!=pa:
                ratio=(bound-previous[axis])/(current[axis]-previous[axis])
                output.append(tuple(previous[k]+ratio*(current[k]-previous[k]) for k in range(2)))
            if ca:output.append(current)
            previous,pa=current,ca
        polygon=output
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(polygon,polygon[1:]+polygon[:1])))/2

def triangle_scene(device):
    camera=g.Camera.look_at((0,0,3),(0,0,0),size=(8,8),device=device,dtype=torch.float64)
    polygon=[(1.2,1.1),(5.4,2.3),(2.3,5.8)]
    pixels=torch.tensor(polygon,device=device,dtype=torch.float64)
    v=torch.stack(((pixels[:,0]-4)*3/camera.fx[0],(4-pixels[:,1])*3/camera.fy[0],pixels[:,0]*0),-1)
    model=g.Scene([g.Material.emissive('paint',(.7,.2,.1),device=device,dtype=torch.float64),g.Mesh('t',v,torch.tensor([[0,2,1]],device=device,dtype=torch.int32),material='paint')])
    return model,camera,polygon

def test_subpixel_coverage_matches_independent_polygon_area(device):
    model,camera,polygon=triangle_scene(device)
    hard=g.render(model,camera);aa=g.render(model,camera,supersample=4)
    exact=torch.tensor([[clipped_area(polygon,x,y) for x in range(8)] for y in range(8)],device=device,dtype=torch.float64)[None]
    error=(aa['mask']-exact).abs();old=(hard['mask']-exact).abs()
    assert float(error.mean())<.015 and float(error.max())<.18
    assert error.mean()<old.mean()/2
    assert ((aa['mask']>0)&(aa['mask']<1)).sum()>5
    torch.testing.assert_close(aa['colour'],aa['mask'][...,None]*aa['colour'].new_tensor([.7,.2,.1]),rtol=1e-12,atol=1e-12)
    bg=aa['colour'].new_tensor([.1,.3,.8]);composite=g.render(model,camera,supersample=4,background=bg)
    torch.testing.assert_close(composite['colour'],aa['colour']+(1-aa['mask'][...,None])*bg,rtol=1e-12,atol=1e-12)

def test_streamed_samples_match_dense_reconstruction_and_keep_geometric_ids(device):
    model,camera,_=triangle_scene(device)
    fine=d._camera(camera,1,factor=4)
    dense=g.render(model,fine);actual=g.render(model,camera,supersample=4,outputs=('colour','mask','face_index','depth'))
    from gffx.torch.image import reconstruct
    expected=reconstruct(torch.cat((dense['colour'],dense['mask'][...,None]),-1),4)
    torch.testing.assert_close(actual['colour'],expected[...,:3],rtol=1e-12,atol=1e-12)
    assert torch.equal(actual['mask'],expected[...,3])
    centre=g.render(model,camera,outputs=('face_index','depth'))
    for field in centre:assert torch.equal(centre[field],actual[field])

def test_opaque_material_boundary_is_antialiased_without_alpha_holes(device):
    camera=g.Camera.look_at((0,0,3),(0,0,0),size=(8,8),device=device,dtype=torch.float64)
    xy=torch.tensor([[-3,-3],[3,-3],[3,3],[-3,3]],device=device,dtype=torch.float64)
    v=torch.cat((xy,xy[:,:1]*0),-1)
    model=g.Scene([g.Material.emissive('red',(1,0,0)),g.Material.emissive('blue',(0,0,1)),
        g.Mesh('a',v,torch.tensor([[0,1,2]],device=device,dtype=torch.int32),material='red'),
        g.Mesh('b',v,torch.tensor([[0,2,3]],device=device,dtype=torch.int32),material='blue')]).to(device,dtype=torch.float64)
    image=g.render(model,camera,supersample=4)
    assert torch.equal(image['mask'],torch.ones_like(image['mask']))
    assert ((image['colour'][...,0]>0)&(image['colour'][...,2]>0)).sum()>0
    torch.testing.assert_close(image['colour'].sum(-1),torch.ones_like(image['mask']),atol=1e-12,rtol=1e-12)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_close_contact_occluder_is_not_erased_by_depth_bias(device,dtype):
    scene=g.Scene([g.Mesh.quad('receiver',u=(2,0,0),v=(0,2,0)),g.Mesh.quad('occluder',center=(0,0,.001),u=(.8,0,0),v=(0,.8,0))]).to(device,dtype=dtype).flatten()
    p=torch.tensor([[0.,0.,0.]],device=device,dtype=dtype);source=p.new_tensor([0.,0.,2.]);n=p.new_tensor([[0.,0.,1.]])
    exact=visibility(p.to(torch.float64),source[None].to(torch.float64),scene.vertices[scene.faces.long()].to(torch.float64))
    actual=_area.visibility(scene,0,p,n,source,512,{},'close')
    assert exact.item()==0 and actual.item()==0

def test_independent_segment_reference_matches_analytic_triangle_and_endpoint(device):
    tri=torch.tensor([[[-1.,-1.,1.],[1.,-1.,1.],[0.,1.,1.]]],device=device,dtype=torch.float64)
    points=tri.new_tensor([[0,0,0],[2,0,0]]);sources=tri.new_tensor([[0,0,2],[0,0,.5],[0,0,1]])
    exact=visibility(points,sources,tri)
    # First point intersects at t=.5; triangles beyond/bordering the light
    # endpoint do not obstruct a finite segment. Far lateral point misses.
    assert exact[0].tolist()==[0.,1.,1.]
    assert exact[1,0]==1

def test_subpixels_reuse_maps_only_within_current_render(device,monkeypatch):
    model=g.Scene([g.Material.diffuse('m',.4),g.Mesh.quad('q',u=(2,0,0),v=(0,2,0),material='m'),g.Light.rectangle('l',center=(0,0,2),u=(.6,0,0),v=(0,-.6,0),radiance=8,samples=2,visible=False)]).to(device)
    camera=g.Camera.look_at((0,0,3),(0,0,0),size=(8,8),device=device)
    calls=[];original=_area._map
    def counted(*args):calls.append(1);return original(*args)
    monkeypatch.setattr(_area,'_map',counted)
    first=g.render(model,camera,supersample=4,shadows=True,shadow_map_size=32)['colour']
    assert len(calls)==4
    model.prims[-1].parameters['center'].add_(model.prims[-1].parameters['center'].new_tensor([.5,0,0]))
    second=g.render(model,camera,supersample=4,shadows=True,shadow_map_size=32)['colour']
    assert len(calls)==8 and not torch.equal(first,second)

def test_antialiased_render_has_checked_first_material_and_light_derivatives(device):
    a=torch.tensor([.3,.4,.5],device=device,dtype=torch.float64,requires_grad=True)
    intensity=torch.tensor(5.,device=device,dtype=torch.float64,requires_grad=True)
    camera=g.Camera.look_at((0,0,3),(0,0,0),size=(8,8),device=device,dtype=torch.float64)
    def objective(aa,ii):
        scene=g.Scene([g.Material.diffuse('m',aa),g.Mesh.sphere('s',radius=.7,material='m'),g.Light.rectangle('l',center=(.7,.8,2),u=(.6,0,0),v=(0,-.6,0),radiance=ii,samples=1,visible=False)]).to(device,dtype=torch.float64)
        return g.render(scene,camera,supersample=4)['colour'].mean()
    da,di=torch.autograd.grad(objective(a,intensity),(a,intensity));h=1e-5
    for index in range(3):
        plus=a.detach().clone();minus=plus.clone();plus[index]+=h;minus[index]-=h
        fd=(objective(plus,intensity.detach())-objective(minus,intensity.detach()))/(2*h)
        torch.testing.assert_close(da[index],fd,rtol=2e-5,atol=1e-8)
    fd=(objective(a.detach(),intensity.detach()+h)-objective(a.detach(),intensity.detach()-h))/(2*h)
    torch.testing.assert_close(di,fd,rtol=2e-5,atol=1e-8)


def test_segment_reference_matches_independent_sphere_quadratic_away_from_tangency(device):
    sphere=g.Mesh.sphere('s',radius=.5,center=(0,.5,0),dtype=torch.float64,device=device)
    points=torch.tensor([[0.,0.,0.],[.3,0.,0.],[1.2,0.,0.]],device=device,dtype=torch.float64)
    source=points.new_tensor([[0.,2.,0.]])
    direction=source-points;offset=points-points.new_tensor([0.,.5,0.])
    a=(direction*direction).sum(-1);b=2*(offset*direction).sum(-1);c=(offset*offset).sum(-1)-.25
    discriminant=b*b-4*a*c;root=discriminant.clamp_min(0).sqrt()
    near=(-b-root)/(2*a);far=(-b+root)/(2*a)
    hit=(discriminant>=0)&(((near>1e-8)&(near<1-1e-8))|((far>1e-8)&(far<1-1e-8)))
    expected=(~hit).to(torch.float64)
    assert expected.tolist()==[0.,0.,1.]
    actual=visibility(points,source,sphere.vertices[sphere.faces.long()])[:,0]
    assert torch.equal(actual,expected)

def test_full512px_forward_aa_is_admitted_without_dense_image_or_workspace_override(device,monkeypatch):
    model=g.Scene([g.Material.diffuse('m',.5),g.Mesh.sphere('s',radius=.5,material='m'),g.Light.rectangle('l',center=(0,2,0),u=(.7,0,0),v=(0,0,.6),radiance=12,samples=8)]).to(device)
    camera=g.Camera.look_at((0,1,4),(0,.5,0),size=(512,512),device=device)
    class Admitted(Exception):pass
    def inspect(packed,sample,*args,**kwargs):
        assert sample.image_height==512 and sample.image_width==512
        raise Admitted
    monkeypatch.setattr(d,'_buffers',inspect)
    with pytest.raises(Admitted):g.render(model,camera,supersample=4,shadows=True,shadow_map_size=512)

def test_antialiased_geometry_and_camera_first_derivatives_inside_coverage(device):
    radius=torch.tensor(.7,device=device,dtype=torch.float64,requires_grad=True)
    x=torch.tensor(.07,device=device,dtype=torch.float64,requires_grad=True)
    def objective(r,xx):
        eye=torch.stack((xx,xx.new_tensor(.13),xx.new_tensor(3.)))
        camera=g.Camera.look_at(eye,(0,0,0),size=(8,8))
        model=g.Scene([g.Material.glossy('m',(.3,.4,.5),roughness=.4),g.Mesh.sphere('s',radius=r,material='m'),g.Light.rectangle('l',center=(.8,.8,2),u=(.8,0,0),v=(0,-.8,0),radiance=8,samples=1,visible=False)]).to(device,dtype=torch.float64)
        return g.render(model,camera,supersample=4)['colour'][0,3:5,3:5].mean()
    grads=torch.autograd.grad(objective(radius,x),(radius,x));h=1e-5
    for index,gradient in enumerate(grads):
        plus=[radius.detach().clone(),x.detach().clone()];minus=[v.clone() for v in plus]
        plus[index]+=h;minus[index]-=h
        fd=(objective(*plus)-objective(*minus))/(2*h)
        assert abs(float(gradient))>1e-8
        torch.testing.assert_close(gradient,fd,rtol=2e-4,atol=1e-7)


def test_aa_storage_precision_follows_scene_when_camera_precision_differs(device):
    model=g.Scene([g.Material.emissive('m',(.5,.4,.3)),g.Mesh.quad('q',material='m')]).to(device,dtype=torch.float64)
    camera=g.Camera.look_at((0,0,3),(0,0,0),size=(8,8),device=device,dtype=torch.float32)
    plan=d._plan_workload(model,camera,1,2**26,('colour','mask'))
    assert plan['element_size']==8
    image=g.render(model,camera,supersample=4)
    assert image['colour'].dtype==torch.float64
