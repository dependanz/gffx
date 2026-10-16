"""Public global-coordinate opaque windows: bounded storage composition contract."""
import torch,pytest,importlib.util
from pathlib import Path
from gffx.torch import render
from gffx.torch.camera import Camera

def fixture(size=3,device='cpu'):
    spec=importlib.util.spec_from_file_location('window_fixture',Path(__file__).with_name('test_opaque_nonlinear_scene.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    s,c,v,_,_=m.fixture(texture=True,rough=True,device=device);v.requires_grad_();t=v.new_tensor
    return s,Camera.from_intrinsics(t(128.*size),t(128.*size),t(size/2-.03),t(size/2-.02),size,size,.1,10),v

def test_public_windows_equal_resident_pixels_and_canonical_vjp():
    import gffx.render as facade
    s,c,v=fixture();r=render.render(s,c,coverage='opaque',cull_mode=render.CULL_NONE,opaque_tile_pixels=2)
    gc=v.new_tensor([.2,-.3,.7]);expected=torch.autograd.grad((r['colour']*gc).sum()+r['mask'].sum()*.1,v)[0];acc=torch.zeros_like(v)
    for y in range(3):
        for x in range(3):
            w=facade.opaque_window(s,c,(y,x,1,1),cull_mode=render.CULL_NONE)
            for key in w:assert torch.equal(w[key],r[key][:,y:y+1,x:x+1])
            g,=torch.autograd.grad((w['colour']*gc).sum()+w['mask'].sum()*.1,v);acc.add_(g)
    assert acc.numpy().tobytes()==expected.numpy().tobytes()

def test_public_window_validates_extent_before_allocation_and_input_mutation():
    s,c,v=fixture()
    for region in ((-1,0,1,1),(0,0,0,1),(0,2,1,2),(True,0,1,1),(0,0,1),[0,0,1,1]):
        with pytest.raises((TypeError,ValueError)):render.opaque_window(s,c,region)
    with pytest.raises(ValueError,match='budget|reserve'):render.opaque_window(s,c,(0,0,1,1),workspace_limit_bytes=1024)
    w=render.opaque_window(s,c,(0,0,1,1),cull_mode=render.CULL_NONE)
    with torch.no_grad():v.add_(.00001)
    with pytest.raises(RuntimeError,match='modified by an inplace'):w['colour'].sum().backward()

def test_public_window_full_frame_reserve_is_replaced_by_explicit_window_reserve():
    s,c,v=fixture();c.image_height=1080;c.image_width=1920
    with pytest.raises(ValueError,match='reserve'):render.render(s,c,coverage='opaque')
    out=render.opaque_window(s,c,(100,100,1,1),cull_mode=render.CULL_NONE)
    assert out['colour'].shape==(1,1,1,3) and out['mask'].shape==(1,1,1)
    assert torch.isfinite(out['colour']).all()
