"""Production horizon cuts,independent pixels and conservative curved refusal."""
import torch,pytest,json,os
from decimal import Decimal as D,localcontext

def fixture(device='cpu',varying=False):
    from gffx.torch import scene,material,light
    from gffx.torch.camera import Camera
    t=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
    theta=t(0).requires_grad_();v=t([[-1,.01,-2],[1,.01,-2],[1,-.01,-2],[-1,-.01,-2]])
    f=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32,device=device)
    normal=torch.stack((theta*0+1,theta*0,theta))
    attr=scene.Attribute(normal.repeat(4,1) if varying else normal,'vertex' if varying else 'constant')
    budget=material.PrincipledBudget.bundled('ggx-smith-loss-'+device+'-f64-v1',device=device,dtype=torch.float64)
    mat=scene.Material('m',material.principled,dict(albedo=t([0,0,0]),f0=t([1,1,1]),roughness=t(1),budget=budget))
    lamp=scene.Light('l',light.directional,dict(directions=t([[1,0,0]]),irradiance=t([[1,1,1]])))
    s=scene.Scene([scene.Mesh('msh',v,f,attributes={'normals':attr},material='m'),mat,lamp]);c=Camera.from_intrinsics(t(1),t(100),t(.5),t(.5),1,1,.1,10)
    return s,c,theta

def test_production_view_horizon_matches_independent_pixel_and_motion(record_property):
    from tools.opaque_scene_horizon_reference import view_pixel
    from gffx.torch import render
    ref=view_pixel();s,c,theta=fixture(os.environ.get('GFFX_KNOT_DEVICE','cpu'))
    out=render.render(s,c,coverage='opaque',cull_mode=render.CULL_NONE);g=torch.autograd.grad(out['colour'][0,0,0,0],theta)[0]
    actual={'value':out['colour'][0,0,0].tolist(),'derivative':[g.item()]*3}
    for key,values in actual.items():
        for value,want in zip(values,ref[key]):
            with localcontext() as ctx:
                ctx.prec=90;assert want.hi-want.lo<=D('2e-12');d=D.from_float(value);assert max(want.lo-d,d-want.hi,D(0))<=D('1e-10')+D('1e-9')*abs((want.lo+want.hi)/2)
    assert abs(out['mask'].item()-1)<=1e-12 and g.item()>0
    record_property('scene_horizon',json.dumps(dict(reference=ref,actual=actual,device=c.fx.device.type),default=str))

def test_production_equal_vertex_normals_keep_quadratic_motion():
    from gffx.torch import render,scene
    from dataclasses import replace
    s,c,theta=fixture(os.environ.get('GFFX_KNOT_DEVICE','cpu'));p=s.prims[0];n=p.attributes['normals'].values.detach().repeat(4,1).requires_grad_();s=scene.Scene([replace(p,attributes={'normals':scene.Attribute(n,'vertex')}),*s.prims[1:]])
    from gffx.torch._opaque_scene import _project,_horizon_cuts,_normals
    from tools.opaque_scene_horizon_reference import view_pixel
    packed,c,px,keep,clip=_project(s,c,render.CULL_NONE);cuts,curves=_horizon_cuts(packed,c,clip,_normals(packed),keep)
    assert curves is not None and cuts.shape[-1]==0
    result=render.render(s,c,coverage='opaque',cull_mode=render.CULL_NONE);g,=torch.autograd.grad(result['colour'][0,0,0,0],n)
    ref=view_pixel()
    for value,bound in ((result['colour'][0,0,0,0].item(),ref['value'][0]),(g[:,2].sum().item(),ref['derivative'][0])):
        actual=D.from_float(value);assert max(bound.lo-actual,actual-bound.hi,D(0))<=D('1e-10')+D('1e-9')*abs((bound.lo+bound.hi)/2)


def test_production_light_horizon_cut_and_constant_view_motion_are_connected():
    from gffx.torch import scene,render
    from gffx.torch._opaque_scene import _project,_horizon_cuts,_normals
    from dataclasses import replace
    s,c,theta=fixture();p=s.prims[0];t=p.vertices.new_tensor
    # Front view throughout; changing vertex normal crosses one directional light horizon.
    n=t([[-.1,0,1],[.1,0,1],[.1,0,1],[-.1,0,1]])+theta*t([1,0,0]);s=scene.Scene([replace(p,attributes={'normals':scene.Attribute(n,'vertex')}),*s.prims[1:]])
    packed,c,px,keep,clip=_project(s,c,render.CULL_NONE);cuts,curves=_horizon_cuts(packed,c,clip,_normals(packed),keep)
    assert cuts.shape==(2,3,1);assert cuts[0,0,0]<0<cuts[0,1,0]
    location=-cuts[0,0,0]/(cuts[0,1,0]-cuts[0,0,0]);g=torch.autograd.grad(location,theta)[0]
    assert abs(location.item()-.5)<1e-15 and g.item()<0

def test_production_horizon_cut_capacity_is_shared_and_grazing_refuses():
    from gffx.torch import scene,render
    from gffx.torch._opaque_scene import _project,_horizon_cuts,_normals
    from dataclasses import replace
    s,c,theta=fixture();packed,c,px,keep,clip=_project(s,c,render.CULL_NONE)
    with pytest.raises(OverflowError,match='capacity'): _horizon_cuts(packed,c,clip,_normals(packed),keep,remaining=0)
    p=s.prims[0];s=scene.Scene([replace(p,attributes={'normals':scene.Attribute(p.vertices.new_tensor([0,1,0]),'constant')}),*s.prims[1:]])
    packed,c,px,keep,clip=_project(s,c,render.CULL_NONE)
    with pytest.raises(NotImplementedError,match='grazing'): _horizon_cuts(packed,c,clip,_normals(packed),keep)
