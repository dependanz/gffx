"""Public D01 authoring, hollow-cup transport and portable scene controls."""
import math

import pytest
import torch

import gffx as g
from gffx.examples.studio_d import build_scene, tumbler, write_bundle
from gffx.torch._dielectric import dielectric
from gffx.torch.scene import Camera as CameraPrim


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64], ids=['float32', 'float64'])
def dtype(request):
    return request.param


def tolerance(dtype):
    return 3e-6 if dtype == torch.float32 else 3e-13


def horizontal_camera(device, dtype):
    angle = math.pi / 24
    middle = (.5, .47, 0.)
    normal = (math.cos(angle), 0., math.sin(angle))
    return g.Camera.look_at(
        tuple(middle[i] + 1.5 * normal[i] for i in range(3)), middle,
        size=(1, 1), fov=20., near=.001, far=5., device=device, dtype=dtype)


def draw(prims, camera, background):
    return g.render(g.Scene(prims), camera, integrator='raster', shadows=False,
                    background=background, outputs=('colour', 'mask', 'transmission'))


@pytest.mark.parametrize('radius', [.31, .61], ids=['authored_cavity', 'wider_air_gap'])
def test_authored_cup_attenuates_two_walls_only(device, dtype, radius):
    # Changing the cavity diameter must not absorb over its additional air.
    sigma = (1., 2., 4.)
    result = draw([
        g.Material.dielectric('cup_glass', ior=1., absorption=sigma,
                              device=device, dtype=dtype),
        tumbler('cup', radius=radius, device=device, dtype=dtype),
    ], horizontal_camera(device, dtype), (1., 1., 1.))
    # Polygon radii describe vertices; a facet-normal ray sees this thickness.
    distance = 2 * .055 * math.cos(math.pi / 24)
    expected = result['colour'].new_tensor([math.exp(-s * distance) for s in sigma])
    torch.testing.assert_close(result['colour'][0, 0, 0], expected,
                               atol=tolerance(dtype), rtol=tolerance(dtype))
    assert result['transmission']['camera_glass_interfaces'] == 4
    assert result['transmission']['pixels_with_discarded_weight'] == 0
    assert torch.equal(result['mask'], torch.ones_like(result['mask']))


def test_authored_cup_open_mouth_and_bottom(device, dtype):
    sigma = (1., 2., 4.)
    camera = g.Camera.look_at((.5, 1.4, 0.), (.5, .02, 0.), up=(0., 0., 1.),
                              size=(1, 1), fov=20., near=.001, far=5.,
                              device=device, dtype=dtype)
    result = draw([
        g.Material.dielectric('cup_glass', ior=1., absorption=sigma,
                              device=device, dtype=dtype),
        tumbler('cup', device=device, dtype=dtype),
    ], camera, (1., 1., 1.))
    expected = result['colour'].new_tensor([math.exp(-s * .09) for s in sigma])
    torch.testing.assert_close(result['colour'][0, 0, 0], expected,
                               atol=tolerance(dtype), rtol=tolerance(dtype))
    assert result['transmission']['camera_glass_interfaces'] == 2
    assert result['transmission']['pixels_with_discarded_weight'] == 0


def test_authored_glass_cup_sums_four_interface_reflections(device, dtype):
    angle = math.pi / 24
    normal = (math.cos(angle), 0., math.sin(angle))
    middle = (.5, .47, 0.)
    result = draw([
        g.Material.dielectric('cup_glass', ior=1.5, device=device, dtype=dtype),
        tumbler('cup', device=device, dtype=dtype),
        g.Material.emissive('back', (1., 1., 1.), device=device, dtype=dtype),
        g.Mesh.quad('rear_emitter',
            center=tuple(middle[i] - normal[i] for i in range(3)),
            u=(3 * math.sin(angle), 0., -3 * math.cos(angle)), v=(0., 3., 0.),
            material='back', device=device, dtype=dtype),
    ], horizontal_camera(device, dtype), (0., 0., 0.))
    # Incoherent normal-incidence series for two lossless parallel glass walls.
    reflectance = ((1.5 - 1) / (1.5 + 1)) ** 2
    expected = (1 - reflectance) / (1 + 3 * reflectance)
    observed = result['colour'][0, 0, 0]
    omitted = observed.new_tensor(result['transmission']['discarded_path_weight_max_rgb'])
    error = expected - observed
    # This scene's terminal radiance is bounded by one. Under that explicit
    # condition, omitted throughput bounds omitted radiance; not in general.
    assert bool((error >= -tolerance(dtype)).all())
    assert bool((error <= omitted + tolerance(dtype)).all())
    assert bool((observed > (1 - reflectance) ** 4).all())
    assert result['transmission']['capacity_limited_paths'] == 0


def test_d01_bundle_preserves_glass_and_authored_meshes(tmp_path, device, dtype):
    scene, camera = build_scene(device=device, dtype=dtype, size=(3, 5))
    path = tmp_path / 'd01.gffx-scene'
    write_bundle(path, scene, camera)
    loaded = g.Scene.load(path, device=device, dtype=dtype)
    before = {p.name: p for p in scene.prims}
    after = {p.name: p for p in loaded.prims}
    assert set(after) == set(before) | {'camera'}
    for name in ('sphere_glass', 'cup_glass'):
        assert after[name].evaluator is dielectric
        assert set(after[name].parameters) == {'ior', 'absorption'}
        for key in ('ior', 'absorption'):
            torch.testing.assert_close(after[name].parameters[key], before[name].parameters[key],
                                       atol=0., rtol=0.)
    meshes = [p for p in scene.prims if isinstance(p, g.Mesh)]
    assert len(meshes) == 90
    assert sum(len(p.vertices) for p in meshes) == 716
    assert sum(len(p.faces) for p in meshes) == 896
    for original in meshes:
        restored = after[original.name]
        assert isinstance(restored, g.Mesh)
        assert restored.material == original.material
        assert restored.vertices.device.type == device
        assert restored.vertices.dtype == dtype
        torch.testing.assert_close(restored.faces, original.faces, atol=0., rtol=0.)
        # Bundle geometry is deliberately serialized as float32, even when the
        # authoring/loaded scene uses float64; compare to that documented value.
        expected = original.vertices.to(torch.float32).to(dtype)
        torch.testing.assert_close(restored.vertices, expected, atol=0., rtol=0.)
    assert isinstance(after['camera'], CameraPrim)
    restored_camera = after['camera'].camera
    assert (restored_camera.image_height, restored_camera.image_width) == (3, 5)
    torch.testing.assert_close(restored_camera.pose, camera.pose.to(torch.float32).to(dtype),
                               atol=0., rtol=0.)
