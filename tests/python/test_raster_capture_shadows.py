"""Excluded capture geometry must not survive as ghost shadow casters."""
import pytest
import torch
import gffx as g

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_inline_capture_matches_explicit_snapshot_with_excluded_caster(device):
    floor=g.Mesh.quad('floor',center=(0,0,0),u=(4,0,0),v=(0,4,0),material='white',device=device)
    sphere=g.Mesh.sphere('excluded reflector',center=(0,0,.5),radius=.5,segments=20,rings=12,material='metal',device=device)
    scene=g.Scene([g.Material.diffuse('white',(1,1,1),device=device),g.Material.pbr('metal',base_color=(.9,.9,.9),metallic=1,roughness=.18,device=device),floor,sphere,g.Light.point('key',position=(0,0,3),power=100,device=device)])
    camera=g.Camera.look_at((0,-2,1.2),(0,0,.4),up=(0,0,1),size=(32,32),device=device)
    probe=g.ReflectionProbe((0,-1,1),exclude=('excluded reflector',),resolution=16,samples=16)
    env=g.Environment.constant((.03,.03,.03),device=device)
    args=dict(integrator='raster',environment=env,shadows=True,shadow_map_size=64)
    snap=probe.capture(scene,environment=env,shadows=True,shadow_map_size=64)
    expected=g.render(scene,camera,probes=[snap],**args)['colour']
    actual=g.render(scene,camera,probes=[probe],**args)['colour']
    assert float(expected.max())>.1
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shadow_context_tracks_casters_and_shadow_configuration(device):
    from gffx.torch._raster import render
    floor=g.Mesh.quad('floor',center=(0,0,0),u=(4,0,0),v=(0,4,0),material='white',device=device)
    sphere=g.Mesh.sphere('caster',center=(0,0,.5),radius=.4,segments=16,rings=8,material='white',device=device)
    mat=g.Material.diffuse('white',(1,1,1),device=device);light=g.Light.point('key',position=(0,0,3),power=100,device=device)
    full=g.Scene([mat,floor,sphere,light]);filtered=g.Scene([mat,floor,light]);camera=g.Camera.look_at((0,-2,2),(0,0,0),up=(0,0,1),size=(24,24),device=device)
    context={};render(full,camera,shadows=True,shadow_map_size=64,_context=context)
    for shadows,size in [(True,64),(False,64),(True,32)]:
        actual=render(filtered,camera,shadows=shadows,shadow_map_size=size,_context=context)['colour']
        expected=render(filtered,camera,shadows=shadows,shadow_map_size=size)['colour']
        assert float(expected.max())>.1
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
