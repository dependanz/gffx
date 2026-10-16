"""Independent scene alpha and explicit current capability boundaries."""
import pytest,torch
from gffx.torch import scene
from gffx.torch.camera import Camera
from test_deferred import fixture,api

def test_opaque_scene_projection_has_independent_camera_and_vertex_gradients():
    model,camera,vertices,faces,_=fixture()
    x=torch.tensor(.125,dtype=torch.float64,requires_grad=True);fx=torch.tensor(4.,dtype=torch.float64,requires_grad=True)
    z=x.new_tensor(-2.);left=x.new_tensor(-2.);top=x.new_tensor(2.);bottom=-top
    v=torch.stack([torch.stack([left,bottom,z]),torch.stack([x,bottom,z]),torch.stack([x,top,z]),torch.stack([left,top,z])])
    s=scene.Scene([scene.Mesh('quad',v,faces)])
    c=Camera.from_intrinsics(fx,fx.new_tensor(4.),fx.new_tensor(2.),fx.new_tensor(2.),4,4,.1,10.)
    out=api().render(s,c,coverage='opaque',outputs=('mask',))['mask']
    assert torch.equal(out,x.new_tensor([1.,1.,.25,0.]).expand(1,4,4))
    gx,gf=torch.autograd.grad(out.sum(),(x,fx));assert gx==8 and gf==.25

def test_opaque_scene_refuses_unqualified_colour_and_options():
    s,c,*_=fixture()
    with pytest.raises(NotImplementedError,match='shaded'):api().render(s,c,coverage='opaque')
    for kw in ({'soft':.2},{'supersample':2},{'background':c.fx.new_zeros(3)}):
        with pytest.raises(ValueError):api().render(s,c,coverage='opaque',outputs=('mask',),**kw)
    for mode in ('unknown',True):
        with pytest.raises(ValueError):api().render(s,c,coverage=mode,outputs=('mask',))

def test_opaque_scene_partial_projection_clipping_is_refused():
    _,c,v,f,_=fixture();v=v.clone();v[0,2]=-.05
    with pytest.raises(NotImplementedError,match='clipping'):api().render(scene.Scene([scene.Mesh('quad',v,f)]),c,coverage='opaque',outputs=('mask',))

def test_opaque_scene_empty_and_culling_are_exact():
    s,c,*_=fixture()
    assert torch.equal(api().render(scene.Scene([]),c,coverage='opaque',outputs=('mask',))['mask'],torch.zeros((1,4,4),dtype=torch.float64))
    assert torch.equal(api().render(s,c,coverage='opaque',cull_mode=3,outputs=('mask',))['mask'],torch.zeros((1,4,4),dtype=torch.float64))


def test_opaque_scene_camera_plane_crossing_is_not_mistaken_for_far_rejection():
    _,c,v,f,_=fixture();v=v.clone();v[:,2]=-100.;v[0,2]=100.
    # All divided z values can lie beyond far even though a camera-plane crossing
    # includes geometry inside the clip volume. Reject this unsupported topology.
    with pytest.raises(NotImplementedError,match='clipping'):
        api().render(scene.Scene([scene.Mesh('crossing',v,f)]),c,coverage='opaque',outputs=('mask',))
