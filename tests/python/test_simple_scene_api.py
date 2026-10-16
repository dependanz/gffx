"""User-facing scene construction, real rendering and gradient regressions."""
import ast
import importlib
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import torch
import gffx as g


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_quad_sphere_orientation_and_geometry_gradient(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual GPU required')
    length=torch.tensor(2.,dtype=torch.float64,device=device,requires_grad=True)
    zero=length*0
    quad=g.Mesh.quad('q',u=torch.stack((length,zero,zero)),v=(0,1,0))
    normals,areas,valid=g.mesh.face_geometry(quad.vertices,quad.faces)
    assert valid.all() and normals.device.type==device
    torch.testing.assert_close(normals,quad.vertices.new_tensor([[0,0,1],[0,0,1]]))
    areas.sum().backward();torch.testing.assert_close(length.grad,length.new_tensor(1.))
    radius=torch.tensor(.6,dtype=torch.float64,device=device,requires_grad=True)
    ball=g.Mesh.sphere('s',radius=radius,segments=12,rings=8)
    n,a,valid=g.mesh.face_geometry(ball.vertices,ball.faces)
    assert valid.all()
    centroids=ball.vertices[ball.faces.long()].mean(1)
    assert bool(((n*centroids).sum(-1)>0).all())
    total=a.sum();derivative,=torch.autograd.grad(total,radius)
    torch.testing.assert_close(derivative,2*total/radius)


def test_scene_transfer_and_look_at_keep_tensor_gradients():
    center=torch.tensor([0.,0.,0.],dtype=torch.float64,requires_grad=True)
    original=g.Scene([g.Material.diffuse('m',(.4,.5,.6),dtype=torch.float64),
                      g.Mesh.sphere('s',center=center,material='m',segments=8,rings=4)])
    moved=original.to(dtype=torch.float32)
    assert moved is not original and original.prims[1].vertices.dtype==torch.float64
    moved.prims[1].vertices.sum().backward()
    torch.testing.assert_close(center.grad,center.new_full((3,),moved.prims[1].vertices.shape[0]))
    eye=torch.tensor([0.,0.,3.],dtype=torch.float64,requires_grad=True)
    cam=g.Camera.look_at(eye,(0,0,0),size=(20,30),fov=50.)
    projected=cam.project(eye.new_tensor([[0.,0.,0.]]))
    torch.testing.assert_close(projected.pixels,eye.new_tensor([[15.,10.]]))
    grad,=torch.autograd.grad(projected.view_depth.sum(),eye)
    torch.testing.assert_close(grad,eye.new_tensor([0.,0.,1.]))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_rectangle_render_has_correct_albedo_and_radiance_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual GPU required')
    albedo=torch.tensor([.3,.4,.5],dtype=torch.float64,device=device,requires_grad=True)
    emission=torch.tensor([5.,6.,7.],dtype=torch.float64,device=device,requires_grad=True)
    model=g.Scene([g.Material.diffuse('m',albedo),
                   g.Mesh.quad('q',u=(2,0,0),v=(0,2,0),material='m',dtype=torch.float64),
                   g.Light.rectangle('l',center=(0,1,2),u=(.5,0,0),v=(0,-.5,0),
                                     radiance=emission,samples=2,visible=False)]).to(device)
    cam=g.Camera.look_at((0,0,3),(0,0,0),size=(12,12),fov=45.,dtype=torch.float64,device=device)
    rgb=g.render(model,cam,shadows=True,shadow_map_size=32)['colour']
    assert rgb.device.type==device and torch.isfinite(rgb).all() and float(rgb.detach().sum())>0
    da,de=torch.autograd.grad(rgb.sum(),(albedo,emission))
    sums=rgb.sum((0,1,2))
    torch.testing.assert_close(da,sums/albedo,atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(de,sums/emission,atol=1e-9,rtol=1e-9)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_generic_shadow_occluder_and_visible_emitter(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual GPU required')
    model=g.Scene([g.Material.diffuse('m',(.6,.6,.6)),
                   g.Mesh.quad('floor',center=(0,0,0),u=(3,0,0),v=(0,0,-3),material='m'),
                   g.Mesh.sphere('occluder',center=(0,.35,0),radius=.35,segments=12,rings=8,material='m'),
                   g.Light.rectangle('l',center=(0,1.8,0),u=(.4,0,0),v=(0,0,.4),radiance=(8,8,8),samples=2)]).to(device)
    cam=g.Camera.look_at((1.8,1.5,2.8),(0,.25,0),size=(32,32),device=device)
    direct=g.render(model,cam,ambient=.05)['colour']
    shaded=g.render(model,cam,ambient=.05,shadows=True,shadow_map_size=64)['colour']
    assert bool((shaded<=direct+1e-6).all())
    assert float((direct-shaded).sum())>.01
    # Emission is independent of lighting; no separate caller-authored light mesh.
    lamp=g.Scene([g.Light.rectangle('panel',center=(0,0,0),u=(1,0,0),v=(0,1,0),radiance=(2,3,4),samples=1)]).to(device)
    view=g.Camera.look_at((0,0,2),(0,0,0),size=(8,8),device=device)
    result=g.render(lamp,view)['colour'][0]
    torch.testing.assert_close(result[4,4],result.new_tensor([2.,3.,4.]))


def test_png_export_preserves_input_gradient_and_needs_no_numpy(tmp_path):
    image=torch.full((1,2,3,3),.2,dtype=torch.float64,requires_grad=True)
    path=g.io.write_png(tmp_path/'image.png',image)
    assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert struct.unpack('>II',path.read_bytes()[16:24])==(3,2)
    with pytest.raises(FileExistsError):g.io.write_png(path,image)
    image.sum().backward();assert torch.equal(image.grad,torch.ones_like(image))


def test_constructor_refusals():
    for action in [lambda:g.Mesh.quad('q',u=(1,0,0),v=(2,0,0)),
                   lambda:g.Mesh.sphere('s',radius=-1),
                   lambda:g.Material.diffuse('m',(1.1,0,0)),
                   lambda:g.Light.rectangle('l',center=(0,0,0),u=(1,0,0),v=(0,1,0),radiance=(-1,0,0)),
                   lambda:g.Camera.look_at((0,0,0),(0,0,0)),
                   lambda:g.Camera.look_at((0,0,1),(0,0,0),up=(0,0,1))]:
        with pytest.raises((ValueError,TypeError)):action()


def test_top_level_api_and_png_namespace_are_lazy():
    script='import sys,gffx; assert "Scene" in dir(gffx); import gffx.io; assert "torch" not in sys.modules; assert "numpy" not in sys.modules; assert "PIL" not in sys.modules'
    subprocess.run([sys.executable,'-c',script],check=True,env=os.environ.copy(),timeout=20)


def test_cornell_example_is_scene_declaration_not_a_renderer():
    path=Path(__file__).resolve().parents[2]/'examples/cornell_box.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    assert not any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for node in ast.walk(tree))
    imports=[n.name for node in ast.walk(tree) if isinstance(node,ast.Import) for n in node.names]
    assert imports==['gffx']
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
    assert any(node.func.attr=='to' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='auto' for node in calls)
    assert any(node.func.attr=='look_at' and any(k.arg=='device' and isinstance(k.value,ast.Constant) and k.value.value=='auto' for k in node.keywords) for node in calls)
    assert len(path.read_text().splitlines())<=30


def test_scene_transfer_recurses_through_texture_sources():
    values=torch.tensor([.2,.3,.4],dtype=torch.float64,requires_grad=True)
    source=g.render.TextureSource(values,torch.tensor([0,3],dtype=torch.int32),1,1,3)
    from gffx.torch import material
    mat=g.Material('m',material.classic,dict(albedo=source,specular=values.new_tensor(0.),exponent=values.new_tensor(1.)))
    model=g.Scene([mat,g.Mesh.quad('q',material='m',dtype=torch.float64)])
    moved=model.to(dtype=torch.float32)
    texture=moved.prims[0].parameters['albedo']
    assert texture.pyramid.dtype==torch.float32 and texture.level_offsets.dtype==torch.int32
    texture.pyramid.sum().backward();torch.testing.assert_close(values.grad,torch.ones_like(values))
    with pytest.raises(TypeError):model.to(dtype=torch.int32)
    with pytest.raises(ValueError):model.to('meta')


def test_emission_respects_front_side_when_culling_is_disabled():
    model=g.Scene([g.Light.rectangle('panel',center=(0,0,0),u=(1,0,0),v=(0,1,0),radiance=3,samples=1)])
    cam=g.Camera.look_at((0,0,-2),(0,0,0),size=(8,8))
    rgb=g.render(model,cam,cull_mode=g.render.CULL_NONE)['colour']
    assert torch.equal(rgb,torch.zeros_like(rgb))


def test_render_and_png_without_numpy_or_pillow(tmp_path):
    script = r"""
import importlib.abc, sys
class RejectOptionalImages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'PIL'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, RejectOptionalImages())
import gffx as g
scene = g.Scene([
    g.Material.diffuse('m', (.5,.4,.3)),
    g.Mesh.quad('q', material='m'),
    g.Light.rectangle('l', center=(0,0,1), u=(.5,0,0), v=(0,-.5,0), radiance=4, samples=1),
])
camera = g.Camera.look_at((0,0,3), (0,0,0), size=(16,16))
image = g.render(scene, camera, shadows=True, shadow_map_size=16, ambient=.1)
path = g.io.write_png(sys.argv[1], image['colour'])
assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
assert 'numpy' not in sys.modules and 'PIL' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', script, str(tmp_path/'standalone.png')],
                   check=True, env=os.environ.copy(), timeout=60)


def test_auto_device_requires_both_framework_and_loaded_provider(monkeypatch):
    from gffx.torch._scene_builders import resolve_device
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert str(resolve_device('auto')) == 'cpu'
    scene = g.Scene([g.Material.diffuse('m', .5), g.Mesh.quad('q', material='m')]).to('auto')
    camera = g.Camera.look_at((0,0,3), (0,0,0), size=(8,8), device='auto')
    assert scene.prims[1].vertices.device.type == camera.fx.device.type == 'cpu'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(g.cuda, 'capabilities', lambda: dict(status='unavailable'))
    assert str(resolve_device('auto')) == 'cpu'
    monkeypatch.setattr(g.cuda, 'capabilities', lambda: dict(status='loaded'))
    assert str(resolve_device('auto')) == 'cuda'
    assert resolve_device('cpu') == 'cpu' and resolve_device(None) is None


def test_cpu_shadow_default_and_explicit_workspace_refusal():
    scene=g.Scene([g.Material.diffuse('m',.5),g.Mesh.quad('q',material='m'),
                   g.Light.rectangle('l',center=(0,0,1),u=(.3,0,0),v=(0,-.3,0),radiance=4,samples=1)])
    camera=g.Camera.look_at((0,0,3),(0,0,0),size=(256,256))
    with pytest.raises(ValueError,match='workspace budget exceeded'):
        g.render(scene,camera,shadows=True,workspace_limit_bytes=67108864)
    with pytest.raises(ValueError,match='workspace budget exceeded'):
        g.render(scene,camera,shadows=True,cpu_workspace_allowance_bytes=67108864)
    result=g.render(scene,camera,shadows=True)
    assert result['colour'].shape==(1,256,256,3) and bool(torch.isfinite(result['colour']).all())
    # Existing unshadowed CPU calls keep their original 64 MiB allowance.
    with pytest.raises(ValueError,match='workspace budget exceeded'):
        g.render(scene,camera)


def test_installed_example_is_exactly_the_public_short_source():
    spec=importlib.util.find_spec('gffx.examples.cornell_box')
    assert spec is not None
    public=Path(__file__).resolve().parents[2]/'examples/cornell_box.py'
    assert Path(spec.origin).read_text(encoding='utf-8')==public.read_text(encoding='utf-8')
