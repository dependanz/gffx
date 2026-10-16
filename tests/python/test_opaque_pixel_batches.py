"""Installed scene continuation: complete material integral and bounded replay."""
import math,os,torch,pytest
from dataclasses import replace
from gffx.torch import scene,render
from gffx.torch.camera import Camera
from test_opaque_nonlinear_scene import fixture

def crowded(device=None,grad=False):
    device=device or os.environ.get('GFFX_BATCH_DEVICE','cpu')
    model,camera,_,_,_=fixture(device=device);t=camera.fx.new_tensor;n=32
    pixels=t([[.5,.5]]+[[.5+2*math.cos(2*math.pi*i/n),.5+2*math.sin(2*math.pi*i/n)] for i in range(n)])
    v=torch.stack(((pixels[:,0]-.5)/64,(.5-pixels[:,1])/64,torch.full_like(pixels[:,0],-2)),-1)
    f=torch.tensor([[0,i+1,(i+1)%n+1] for i in range(n)],dtype=torch.int32,device=device)
    albedo=t([.2,.4,.6]).requires_grad_(grad)
    mesh=scene.Mesh('fan',v,f,attributes={'normals':scene.Attribute(t([.01,.01,1.]).expand(n+1,3).contiguous(),'vertex')},material='m')
    mat=replace(model.prims[1],parameters={**model.prims[1].parameters,'albedo':albedo})
    return scene.Scene([mesh,mat,model.prims[2]]),camera,albedo

def call(s,c,**kw):return render.render(s,c,coverage='opaque',cull_mode=render.CULL_NONE,**kw)

def test_scene_batch_completes_overflow_and_matches_equivalent_surface():
    s,c,_=crowded();simple,d,*_=fixture(device=c.fx.device.type)
    from gffx.torch._opaque_scene import shade
    with pytest.raises(OverflowError):shade(s,c,outputs=('colour','mask'),cull_mode=render.CULL_NONE,background=None,workspace_limit_bytes=67108864)
    a=call(s,c);b=call(simple,d)
    torch.testing.assert_close(a['colour'],b['colour'],atol=1e-10,rtol=1e-9)
    assert abs(a['mask'].item()-1)<1e-12

def test_scene_batch_first_vjp_replays_without_retaining_sample_graphs():
    s,c,a=crowded(grad=True);saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda x:(saved.append(tuple(x.shape)) or x),lambda x:x):out=call(s,c)
    assert not any(800 in shape for shape in saved)
    g,=torch.autograd.grad(out['colour'].sum(),a)
    simple,d,*_=fixture(device=c.fx.device.type);other=a.detach().clone().requires_grad_();simple=scene.Scene([simple.prims[0],replace(simple.prims[1],parameters={**simple.prims[1].parameters,'albedo':other}),*simple.prims[2:]])
    expected,=torch.autograd.grad(call(simple,d)['colour'].sum(),other)
    torch.testing.assert_close(g,expected,atol=1e-10,rtol=1e-9)

def test_scene_batch_checks_input_versions_and_rejects_second_derivatives():
    s,c,a=crowded(grad=True);out=call(s,c)
    with torch.no_grad():a.add_(.001)
    with pytest.raises(RuntimeError,match='modified'):out['colour'].sum().backward()
    s,c,a=crowded(grad=True)
    with pytest.raises(RuntimeError,match='first'):torch.autograd.grad(call(s,c)['colour'].sum(),a,create_graph=True)


def test_scene_batch_background_is_composited_once():
    s,c,_=crowded();bg=c.fx.new_tensor([.12,.23,.34]).requires_grad_()
    a=call(s,c,background=bg);b=call(s,c)
    torch.testing.assert_close(a['colour'],b['colour'],atol=1e-12,rtol=0)
    g,=torch.autograd.grad(a['colour'].sum(),bg)
    torch.testing.assert_close(g,torch.zeros_like(g),atol=1e-12,rtol=0)

def test_scene_batch_coloured_material_motion_matches_two_differences():
    s,c,_=crowded();base=s.prims[0];t=c.fx.new_tensor
    albedo=t([.2,.4,.6])+((base.vertices[:,0]*64+.5)*.1)[:,None]
    attrs={**base.attributes,'colour':scene.Attribute(albedo,'vertex')}
    mat=replace(s.prims[1],parameters={**s.prims[1].parameters,'albedo':render.AttributeSource('colour')})
    theta=t(0.).requires_grad_()
    def value(q):
        moved=replace(base,vertices=base.vertices+q*t([.001,0,0]),attributes=attrs)
        return call(scene.Scene([moved,mat,*s.prims[2:]]),c)['colour'].sum()
    g,=torch.autograd.grad(value(theta),theta)
    assert abs(g.item())>1e-5
    for eps in (1e-4,5e-5):
        expected=(value(theta.detach()+eps)-value(theta.detach()-eps))/(2*eps)
        torch.testing.assert_close(g,expected,atol=1e-6,rtol=1e-4)
