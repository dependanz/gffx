"""Physical visibility and quadrature regressions; no Einar fixture dependency."""
import math
import pytest
import torch
import gffx as g
from gffx.torch import _area,_raster as rs,_raster_accel as accel

@pytest.fixture(params=['cpu','cuda'])
def device(request):
 if request.param=='cuda'and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
 return request.param

@pytest.mark.parametrize('count',[1,3,8,16])
def test_raster_strata_cover_2d_cells_and_1d_projections_without_temporal_seed(device,count):
 c=torch.tensor([.1,.2,2.],device=device,dtype=torch.float64,requires_grad=True);u=c.new_tensor([2,0,0]);v=c.new_tensor([0,-2,0]);r=c.new_ones(3)
 p,n,power=rs._rectangle_samples(c,u,v,r,count);again=rs._rectangle_samples(c,u,v,r,count)[0]
 assert p.dtype==torch.float64 and torch.equal(p,again)
 q=torch.stack(((p[:,0]-c[0])/2+.5,-(p[:,1]-c[1])/2+.5),-1)
 for axis in [0,1]:assert torch.equal(torch.sort((q[:,axis]*count*count).floor().long()).values,torch.arange(count*count,device=device))
 cells=(q*count).floor().long();assert len(torch.unique(cells[:,0]*count+cells[:,1]))==count*count
 torch.testing.assert_close(p.mean(0),c,atol=2e-15,rtol=1e-14)
 torch.testing.assert_close(torch.autograd.grad(p.sum(),c)[0],c.new_full((3,),count*count),atol=1e-12,rtol=0)
 # Existing deferred rectangle sampling retains its original regular centres.
 old=_area._samples(c,u,v,r,count)[0];grid=(torch.arange(count,device=device,dtype=c.dtype)+.5)/count-.5;a,b=torch.meshgrid(grid,grid,indexing='ij')
 assert torch.equal(old,c+a.flatten()[:,None]*u+b.flatten()[:,None]*v)

def one_map(vertices,faces,source,size):
 camera=rs._camera(source,source+source.new_tensor([0,0,-1]),size,fov=140,near=.0001,far=20)
 return rs._make_map(vertices,faces,camera)

@pytest.mark.parametrize('gap',[.001,.01])
def test_raster_receiver_correction_retains_real_parallel_close_occlusion(device,gap):
 floor=g.Mesh.quad('floor',u=(4,0,0),v=(0,4,0),device=device)
 blocker=g.Mesh.quad('blocker',center=(0,0,gap),u=(.6,0,0),v=(0,.6,0),device=device)
 v=torch.cat((floor.vertices,blocker.vertices));f=torch.cat((floor.faces,blocker.faces+len(floor.vertices)))
 p=v.new_tensor([[0,0,0],[1,0,0]]);n=v.new_tensor([[0,0,1]]).expand_as(p);entry=one_map(v,f,v.new_tensor([0,0,2]),256)
 actual=rs._shadow_group([entry],p,n)[:,0]
 assert actual[0]<.001 and actual[1]>.999

def test_raster_detailed_receiver_does_not_lose_half_its_unobstructed_light(device):
 axis=torch.linspace(-.4,.4,65,device=device);x,y=torch.meshgrid(axis,axis,indexing='xy');z=.025*torch.cos(20*x)*torch.cos(15*y);v=torch.stack((x,y,z),-1).reshape(-1,3)
 i=torch.arange(64,device=device);a,b=torch.meshgrid(i,i,indexing='xy');base=(b*65+a).flatten();f=torch.cat((torch.stack((base,base+1,base+66),1),torch.stack((base,base+66,base+65),1))).to(torch.int32)
 t=v[f.long()];p=t.mean(1);n=torch.linalg.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);n/=n.norm(dim=-1,keepdim=True);keep=(p[:,:2].abs()<.30).all(-1);p=p[keep][::16];n=n[keep][::16]
 source=v.new_tensor([-.15,-.15,2]);delta=source-p
 # Height field slope is bounded by sqrt(.5^2+.375^2)=.625.
 # Every finite source ray rises faster than that bound, hence cannot hit it.
 assert bool((delta[:,2]>.625*delta[:,:2].norm(dim=-1)).all())
 entry=one_map(v,f,source,2048)
 # Analytic upper bound on ||dN/dx|| for this height field: Hessian norm<20/m.
 # Production obtains local curvature from geometric vertex normals, before maps.
 curvature=p.new_full((len(p),),20.)
 value=rs._shadow_group([entry],p,n,curvature)[:,0]
 assert float(value.mean())>.98 and float(value.min())>.9
 source_record=dict(kind='point',position=source,intensity=p.new_ones(3),maps=[entry])
 packet=accel.pack_sources([source_record],p,2048)
 shadow=accel.shade(p,n,n,n,dict(albedo=1.,specular=0.,exponent=1.),packet,2048,classic=True,curvature=curvature)
 clear=accel.shade(p,n,n,n,dict(albedo=1.,specular=0.,exponent=1.),accel.pack_sources([dict(source_record,maps=[])],p,2048),2048,classic=True,curvature=curvature)
 torch.testing.assert_close(shadow[:,0]/clear[:,0],value,atol=2e-4,rtol=2e-4)

@pytest.mark.parametrize('angle',[0.,30.,60.])
def test_flat_oblique_receivers_keep_millimetre_contact_without_curvature_bias(device,angle):
 a=math.radians(angle);u=(math.cos(a)*2,0,math.sin(a)*2);v=(0,2,0)
 n=torch.tensor([-math.sin(a),0,math.cos(a)],device=device)
 floor=g.Mesh.quad('floor',u=u,v=v,device=device)
 blocker=g.Mesh.quad('blocker',center=tuple((n*.001).tolist()),u=tuple(x*.2 for x in u),v=tuple(x*.2 for x in v),device=device)
 vertices=torch.cat((floor.vertices,blocker.vertices));faces=torch.cat((floor.faces,blocker.faces+len(floor.vertices)))
 points=vertices.new_zeros((1,3));normal=n[None];entry=one_map(vertices,faces,vertices.new_tensor([0,0,2]),2048)
 zero=rs._receiver_curvature(vertices[faces[:2].long()],normal[:,None].expand(2,3,3))
 assert torch.equal(zero,torch.zeros_like(zero))
 actual=rs._shadow_group([entry],points,normal,points.new_zeros(1))[0,0]
 assert float(actual)<.001

def test_native_area_visibility_uses_the_actual_stratified_map_origins(device):
 floor=g.Mesh.quad('floor',u=(5,0,0),v=(0,5,0),device=device)
 blocker=g.Mesh.quad('edge',center=(0,0,.8),u=(.5,0,0),v=(0,5,0),device=device)
 vertices=torch.cat((floor.vertices,blocker.vertices));faces=torch.cat((floor.faces,blocker.faces+len(floor.vertices)))
 center=vertices.new_tensor([0,0,3]);u=vertices.new_tensor([1.6,0,0]);v=vertices.new_tensor([0,-.8,0]);radiance=vertices.new_tensor([8,8,8]);count=4
 positions,normal,intensity=rs._rectangle_samples(center,u,v,radiance,count)
 area=dict(center=center,u=u,v=v,radiance=radiance,samples=count);sources=[]
 for i,source in enumerate(positions):sources.append(dict(kind='point',position=source,intensity=intensity,normal=normal,maps=[one_map(vertices,faces,source,256)],area=area,sample_index=i))
 x=torch.linspace(-.6,.6,29,device=device);points=torch.stack((x,x*0,x*0),-1);norm=points.new_tensor([0,0,1]).expand_as(points)
 params=dict(base_color=points.new_tensor([.4,.2,.1]),roughness=.4,metallic=0.,ior=1.5,coat=0.,coat_roughness=.1)
 actual=accel.shade(points,norm,norm,norm,params,accel.pack_sources(sources,points,256),256)
 numerator=x*0;denominator=x*0
 for s in sources:
  delta=s['position']-points;d2=delta.square().sum(-1);direction=delta/d2.sqrt()[:,None];weight=direction[:,2].square()/d2
  denominator+=weight;numerator+=weight*rs._shadow(s['maps'],points,norm)
 from gffx.torch._ltc import evaluate
 expected=evaluate(points,norm,norm,norm,params,area,numerator/denominator)
 torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-4)


"""Check the actual vertex-normal estimate, not only an analytic curvature bound."""
import pytest,torch
from gffx.torch import _raster as rs,_raster_accel as accel

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('classic',[False,True])
def test_actual_geometric_normal_curvature_preserves_unobstructed_detailed_receiver(device,classic):
 if device=='cuda'and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
 axis=torch.linspace(-.4,.4,65,device=device);x,y=torch.meshgrid(axis,axis,indexing='xy');z=.025*torch.cos(20*x)*torch.cos(15*y);v=torch.stack((x,y,z),-1).reshape(-1,3)
 vn=torch.stack((.5*torch.sin(20*x)*torch.cos(15*y),.375*torch.cos(20*x)*torch.sin(15*y),torch.ones_like(x)),-1).reshape(-1,3);vn/=vn.norm(dim=-1,keepdim=True)
 i=torch.arange(64,device=device);a,b=torch.meshgrid(i,i,indexing='xy');base=(b*65+a).flatten();f=torch.cat((torch.stack((base,base+1,base+66),1),torch.stack((base,base+66,base+65),1))).to(torch.int32)
 t=v[f.long()];p=t.mean(1);n=torch.linalg.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);n/=n.norm(dim=-1,keepdim=True);curvature=rs._receiver_curvature(t,vn[f.long()]);keep=(p[:,:2].abs()<.30).all(-1);p=p[keep][::16];n=n[keep][::16];curvature=curvature[keep][::16]
 source=v.new_tensor([-.15,-.15,2]);camera=rs._camera(source,source+v.new_tensor([0,0,-1]),2048,fov=140,near=.0001,far=20);entry=rs._make_map(v,f,camera)
 visibility=rs._shadow_group([entry],p,n,curvature)[:,0]
 assert float(visibility.mean())>.98 and float(visibility.min())>.9
 record=dict(kind='point',position=source,intensity=p.new_ones(3),maps=[entry]);data=accel.pack_sources([record],p,2048);clear=accel.pack_sources([dict(record,maps=[])],p,2048)
 params=dict(albedo=1.,specular=0.,exponent=1.)if classic else dict(base_color=p.new_ones(3),roughness=.4,metallic=0.,ior=1.5,coat=0.,coat_roughness=.1)
 actual=accel.shade(p,n,n,n,params,data,2048,classic=classic,curvature=curvature);direct=accel.shade(p,n,n,n,params,clear,2048,classic=classic,curvature=curvature)
 torch.testing.assert_close(actual[:,0]/direct[:,0],visibility,atol=2e-4,rtol=2e-4)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_flat_receiver_distant_blocker_stratum_penumbra_removes_staircase_steps(device):
 if device=='cuda'and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
 floor=g.Mesh.quad('floor',u=(5,0,0),v=(0,5,0),device=device)
 blocker=g.Mesh.quad('edge',center=(0,0,.8),u=(.5,0,0),v=(0,5,0),device=device)
 vertices=torch.cat((floor.vertices,blocker.vertices));faces=torch.cat((floor.faces,blocker.faces+len(floor.vertices)))
 center=vertices.new_tensor([0,0,3]);u=vertices.new_tensor([1.6,0,0]);v=vertices.new_tensor([0,-.8,0]);radiance=vertices.new_tensor([8,8,8]);count=8
 positions,normal,intensity=rs._rectangle_samples(center,u,v,radiance,count)
 area=dict(center=center,u=u,v=v,radiance=radiance,samples=count);sources=[]
 for i,source in enumerate(positions):sources.append(dict(kind='point',position=source,intensity=intensity,normal=normal,maps=[one_map(vertices,faces,source,1024)],area=area,sample_index=i))
 x=torch.linspace(.1,.55,181,device=device);points=torch.stack((x,x*0,x*0),-1);norm=points.new_tensor([0,0,1]).expand_as(points)
 params=dict(albedo=1.,specular=0.,exponent=1.)
 data=accel.pack_sources(sources,points,1024);clear=accel.pack_sources([dict(s,maps=[])for s in sources],points,1024)
 shadow=accel.shade(points,norm,norm,norm,params,data,1024,classic=True);direct=accel.shade(points,norm,norm,norm,params,clear,1024,classic=True)
 vis=shadow[:,0]/direct[:,0]
 py_vis=rs._shadow_group([s['maps'][0]for s in sources],points,norm)
 direction=positions[None]-points[:,None];d2=direction.square().sum(-1);direction=direction/d2.sqrt()[...,None]
 w=direction[...,2].clamp_min(0).square()/d2
 py_combined=(py_vis*w).sum(-1)/w.sum(-1)
 torch.testing.assert_close(vis,py_combined,atol=1e-5,rtol=1e-5)
 diffs=vis[1:]-vis[:-1]
 assert float(diffs.min())>=-1e-4 and float((diffs[1:]-diffs[:-1]).abs().max())<.012



def _analytic_stratum_entry(device, near, blocker_distance):
    """Exact half-plane blocker and planar receiver; no fitted near plane."""
    from types import SimpleNamespace
    size, receiver_distance, far = 128, 3., 10.
    texels = torch.tensor([61.5, 63.25, 63.5, 63.75, 64., 64.25, 64.5, 64.75, 66.5],
                          dtype=torch.float32, device=device)
    x = ((texels + .5) * 2 / size - 1) * receiver_distance
    points = torch.stack((x, torch.zeros_like(x), torch.full_like(x, -receiver_distance)), -1)
    normals = points.new_tensor([0., 0., 1.]).expand_as(points)
    projection = points.new_tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
        [0., 0., -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0., 0., -1., 0.]])
    depth = points.new_full((size, size), -near / receiver_distance)
    depth[:, :size // 2] = -near / blocker_distance
    entry = dict(camera=SimpleNamespace(near=near), depth=depth,
        matrix=projection, inverse=torch.linalg.inv(projection), inverse_depth=True,
        view=torch.eye(4, device=device, dtype=points.dtype), projection=projection,
        origin=points.new_zeros(3), stratum=points.new_tensor([12.8, 0., 0., 12.8, 0., 0.]))
    return entry, points, normals


def _analytic_stratum_visibility(entry, points, normals):
    """Compare the tensor implementation with the independently compiled shader."""
    tensor = rs._shadow_group([entry], points, normals)[:, 0]
    size = entry['depth'].shape[-1]
    source = dict(kind='point', position=points.new_zeros(3),
                  intensity=points.new_ones(3), maps=[entry])
    params = dict(albedo=1., specular=0., exponent=1.)
    shadow = accel.shade(points, normals, normals, normals, params,
                        accel.pack_sources([source], points, size), size, classic=True)
    clear = accel.shade(points, normals, normals, normals, params,
                       accel.pack_sources([dict(source, maps=[])], points, size), size, classic=True)
    assert shadow is not None and clear is not None, 'Native raster shader is required'
    native = shadow[:, 0] / clear[:, 0]
    torch.testing.assert_close(native, tensor, atol=1e-5, rtol=1e-5)
    return tensor


def test_stratum_penumbra_is_invariant_to_shadow_near_plane(device):
    profiles = []
    for near in [1e-4, .001, .01, .025, .1, .5]:
        entry, points, normals = _analytic_stratum_entry(device, near, blocker_distance=2.)
        visibility = _analytic_stratum_visibility(entry, points, normals)
        # At this point ordinary bilinear PCF is already fully lit; an active
        # finite stratum still overlaps the distant blocker. This rules out
        # making the invariance test pass simply by disabling the new filter.
        assert .65 < float(visibility[4]) < .8
        assert float(visibility[0]) < 1e-6 and float(visibility[-1]) > 1 - 1e-6
        assert bool((visibility[1:] >= visibility[:-1] - 1e-6).all())
        profiles.append(visibility)
    for visibility in profiles[1:]:
        torch.testing.assert_close(visibility, profiles[0], atol=1e-5, rtol=1e-5)


def test_stratum_filter_preserves_millimetre_contact_at_every_near_plane(device):
    # The blocker lies one millimetre in front of the same receiver. It must
    # remain occluding, with the ordinary subtexel transition at the edge.
    for near in [1e-4, .001, .01, .025, .1, .5]:
        entry, points, normals = _analytic_stratum_entry(device, near, blocker_distance=2.999)
        visibility = _analytic_stratum_visibility(entry, points, normals)
        expected = points.new_tensor([0., .25, .5, .75, 1., 1., 1., 1., 1.])
        torch.testing.assert_close(visibility, expected, atol=1e-5, rtol=1e-5)
