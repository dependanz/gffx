"""Installed bounded opaque windows and replayed first derivatives."""
import torch,pytest,importlib.util
from pathlib import Path
from gffx.torch import render,scene
from gffx.torch.camera import Camera
spec=importlib.util.spec_from_file_location('opaque_fixture',Path(__file__).with_name('test_opaque_nonlinear_scene.py'));fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)

def setup(size=8,grad=False):
    s,c,v,f,b=fixtures.fixture(texture=True,rough=True);t=v.new_tensor
    if grad:s.prims[0].vertices.requires_grad_()
    return s,Camera.from_intrinsics(t(128.*size),t(128.*size),t(size/2-.03),t(size/2-.02),size,size,.1,10.)

def call(s,c,cap=None):return render.render(s,c,coverage='opaque',cull_mode=render.CULL_NONE,opaque_tile_pixels=cap)

def test_opaque_stream_admits_formerly_refused_eight_square_and_matches_windows(record_property):
    import json
    s,c=setup();a=call(s,c);b=call(s,c,1);d=call(s,c,3)
    for key in a:
        assert torch.equal(a[key],b[key]) and torch.equal(a[key],d[key])
    assert a['colour'].shape==(1,8,8,3)
    record_property('streamed8',json.dumps({k:v.tolist() for k,v in a.items()}))

def test_opaque_stream_replayed_gradient_is_tile_partition_independent():
    s,c=setup(3,True);values=[]
    for cap in (1,2,3):
        out=call(s,c,cap);loss=(out['colour']*out['colour'].new_tensor([.2,-.3,.7])).sum()+out['mask'].sum()*.1
        values.append(torch.autograd.grad(loss,s.prims[0].vertices)[0])
    assert torch.equal(values[0],values[1]) and torch.equal(values[0],values[2])
    assert torch.isfinite(values[0]).all() and values[0].abs().max()>0

def test_opaque_stream_retains_inputs_not_per_window_graphs_and_checks_mutation():
    s,c=setup(8,True);saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda x:(saved.append(tuple(x.shape)) or x),lambda x:x):out=call(s,c,2)
    assert len(saved)<40 and not any(800 in shape for shape in saved)
    out=call(s,c,2)
    with torch.no_grad():s.prims[0].vertices.add_(.0001)
    with pytest.raises(RuntimeError,match='modified by an inplace'):out['colour'].sum().backward()

def test_opaque_stream_validates_tile_and_total_output_budget():
    s,c=setup()
    for cap in (0,-1,True,1.5):
        with pytest.raises((TypeError,ValueError)):call(s,c,cap)
    with pytest.raises(ValueError,match='budget|reserve'):render.render(s,c,coverage='opaque',workspace_limit_bytes=1024)
    huge=Camera.from_intrinsics(c.fx,c.fy,c.cx,c.cy,4096,4096,.1,10.)
    with pytest.raises(ValueError,match='budget|reserve'):call(s,huge)
    with pytest.raises(ValueError,match='opaque'):render.render(s,c,opaque_tile_pixels=2)


def test_opaque_stream_batched_depth_material_and_subpixel_windows_match_resident():
    from gffx.torch._opaque_scene import shade
    from gffx.torch import material
    from dataclasses import replace
    s,c=setup(2);t=c.fx.new_tensor;base=s.prims[0]
    foreground=replace(base,name='foreground',vertices=base.vertices*.43+t([.001,-.001,.5]),material='front')
    back=replace(base,vertices=torch.stack([base.vertices,base.vertices+t([.001,0,0])]))
    front=replace(s.prims[1],name='front',parameters={**s.prims[1].parameters,'albedo':t([.8,.1,.2]),'roughness':t(.45)})
    s=scene.Scene([back,foreground,front,*s.prims[1:]])
    resident=shade(s,c,outputs=('colour','mask'),cull_mode=render.CULL_NONE,background=None,workspace_limit_bytes=67108864)
    for cap in (1,2):
        streamed=call(s,c,cap)
        for k in resident:assert torch.equal(resident[k],streamed[k])
    assert resident['colour'].shape==(2,2,2,3)


def test_opaque_stream_replayed_geometry_gradient_agrees_with_two_differences():
    from dataclasses import replace
    s,c=setup(2);base=s.prims[0];theta=c.fx.new_tensor(0).requires_grad_()
    def objective(value):
        shifted=replace(base,vertices=base.vertices+value*base.vertices.new_tensor([.0001,-.0002,0]))
        out=call(scene.Scene([shifted,*s.prims[1:]]),c,1)
        return (out['colour']*out['colour'].new_tensor([.2,-.3,.7])).sum()+out['mask'].sum()*.1
    actual=torch.autograd.grad(objective(theta),theta)[0]
    for eps in (1e-4,5e-5):
        expected=(objective(theta.detach()+eps)-objective(theta.detach()-eps))/(2*eps)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-4)


def test_opaque_stream_refuses_mask_only_tile_argument():
    s,c=setup(1)
    with pytest.raises(ValueError,match='opaque_tile_pixels'):render.render(s,c,coverage='opaque',outputs=('mask',),opaque_tile_pixels=1)


def test_opaque_perspective_normalization_explicit_vjp_gradcheck():
    from gffx.torch._opaque_scene import _PerspectiveWeights
    raw=torch.tensor([[.17,.29,.54],[.31,.24,.45]],dtype=torch.float64,requires_grad=True)
    cw=torch.tensor([[1.7,2.3,3.1],[2.1,1.3,2.9]],dtype=torch.float64,requires_grad=True)
    active=torch.ones(2,dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda a,b:_PerspectiveWeights.apply(a,b,active),(raw,cw),eps=1e-6,atol=1e-6,rtol=1e-4)
    u=raw/cw;expected=u/((u[:,0]+u[:,1])+u[:,2])[:,None]
    assert torch.equal(_PerspectiveWeights.apply(raw,cw,active),expected)
