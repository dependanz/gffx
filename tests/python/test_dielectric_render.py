"""Analytic transport and admission controls for the Studio D glass path."""
from dataclasses import replace
import math

import pytest
import torch

import gffx as g
from gffx.torch._raster_lighting import Environment


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


def box(name, *, z0=0., z1=1., half=2., material='glass', device='cpu'):
    vertices = torch.tensor([
        [-half, -half, z0], [half, -half, z0], [half, half, z0], [-half, half, z0],
        [-half, -half, z1], [half, -half, z1], [half, half, z1], [-half, half, z1],
    ], device=device, dtype=torch.float32)
    faces = torch.tensor([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], device=device, dtype=torch.int32)
    return g.Mesh(name, vertices, faces, material=material)


def camera(device, *, size=3, eye=(0., 0., 3.), target=(0., 0., 0.), near=.05):
    posed = g.Camera.look_at(eye, target, size=(size, size), near=near, far=10., device=device)
    return g.Camera.from_orthographic(10., 10., size / 2, size / 2,
                                     size, size, near, 10., pose=posed.pose)


def back_plate(device, *, z=-1., emission=True):
    surface = (g.Material.emissive('plate', (1., 1., 1.), device=device) if emission
               else g.Material.diffuse('plate', (.25, .25, .25), device=device))
    mesh = g.Mesh.quad('back', center=(0., 0., z), u=(8., 0., 0.), v=(0., 8., 0.),
                       material='plate', device=device)
    return [surface, mesh]


def draw(prims, cam, **kwargs):
    return g.render(g.Scene(prims), cam, integrator='raster',
                    outputs=('colour', 'mask', 'transmission'), **kwargs)


def test_parallel_slab_sums_internal_fresnel_reflections(device):
    eta = 1.5
    glass = g.Material.dielectric('glass', ior=eta, device=device)
    result = draw([glass, box('slab', device=device), *back_plate(device)], camera(device),
                  background=(0., 0., 0.))
    reflectance = ((eta - 1) / (eta + 1)) ** 2
    expected = (1 - reflectance) / (1 + reflectance)
    torch.testing.assert_close(result['colour'], torch.full_like(result['colour'], expected),
                               atol=1e-5, rtol=0.)
    # The internal-reflection contribution is observable, not just (1-R)^2.
    assert float(result['colour'].min()) > (1 - reflectance) ** 2 + .001
    assert torch.equal(result['mask'], torch.ones_like(result['mask']))
    assert result['transmission']['capacity_limited_paths'] == 0
    assert result['transmission']['depth_limited_paths'] == 0


def test_absorbing_slab_matches_independent_geometric_series(device):
    eta, thickness = 1.5, .8
    sigma = (.2, .7, 1.3)
    glass = g.Material.dielectric('glass', ior=eta, absorption=sigma, device=device)
    result = draw([glass, box('slab', z1=thickness, device=device), *back_plate(device)],
                  camera(device), background=(0., 0., 0.))
    r = ((eta - 1) / (eta + 1)) ** 2
    q = result['colour'].new_tensor([math.exp(-s * thickness) for s in sigma])
    expected = (1 - r) ** 2 * q / (1 - r * r * q * q)
    torch.testing.assert_close(result['colour'], expected.expand_as(result['colour']), atol=2e-5, rtol=0.)


def test_lossless_white_environment_closes_completed_and_discarded_weight(device):
    glass = g.Material.dielectric('glass', ior=1.5, device=device)
    env = Environment.constant((1., 1., 1.), device=device)
    result = draw([glass, box('slab', device=device)], camera(device), environment=env)
    # All orthographic samples see the same interfaces; each has the same loss.
    discarded = result['colour'].new_tensor(result['transmission']['discarded_path_weight_max_rgb'])
    torch.testing.assert_close(result['colour'] + discarded,
                               torch.ones_like(result['colour']), atol=5e-6, rtol=0.)
    assert result['transmission']['pixels_with_discarded_weight'] == 9


def test_environment_lights_an_opaque_endpoint_through_glass(device):
    glass = g.Material.dielectric('glass', ior=1.5, device=device)
    env = Environment.constant((1., 1., 1.), device=device)
    result = draw([glass, box('slab', device=device), *back_plate(device, emission=False)],
                  camera(device), environment=env)
    r = .04
    transmittance = (1 - r) / (1 + r)
    expected = (1 - transmittance) + .25 * transmittance
    omitted = result['colour'].new_tensor(result['transmission']['discarded_path_weight_max_rgb'])
    # This specific control has radiance in [0,1], so omitted path weight
    # bounds its omitted radiance. That is not a bound for arbitrary scenes.
    error = expected - result['colour']
    assert bool((error >= -5e-6).all())
    assert bool((error <= omitted + 5e-6).all())
    assert float(result['colour'].min()) > .30


@pytest.mark.parametrize('gap', [.3, .9])
def test_closed_hollow_shell_absorbs_only_two_walls(device, gap):
    wall = .02
    sigma = (30., 10., 2.)
    outer = box('outer', z0=0., z1=gap + 2 * wall, half=2., device=device)
    inner = box('inner', z0=wall, z1=gap + wall, half=2. - wall, device=device)
    shell = g.Mesh('hollow', torch.cat((outer.vertices, inner.vertices)),
                   torch.cat((outer.faces, inner.faces.flip(-1) + len(outer.vertices))),
                   material='glass')
    glass = g.Material.dielectric('glass', ior=1., absorption=sigma, device=device)
    result = draw([glass, shell, *back_plate(device)], camera(device), background=(0., 0., 0.))
    expected = result['colour'].new_tensor([math.exp(-s * (2 * wall)) for s in sigma])
    torch.testing.assert_close(result['colour'], expected.expand_as(result['colour']), atol=3e-5, rtol=0.)
    assert result['transmission']['camera_glass_interfaces'] == 9 * 4
    assert result['transmission']['pixels_with_discarded_weight'] == 0


def test_direct_light_uses_declared_straight_segment_transmittance(device):
    sigma = (.2, .6, 1.)
    glass = g.Material.dielectric('glass', ior=1.5, absorption=sigma, device=device)
    slab = box('slab', z0=1., z1=2., half=.25, device=device)
    floor = g.Material.diffuse('floor', (.6, .6, .6), device=device)
    receiver = g.Mesh.quad('receiver', u=(8., 0., 0.), v=(0., 8., 0.), material='floor', device=device)
    lamp = g.Light.point('light', position=(0., 0., 3.), power=4 * math.pi, device=device)
    cam = camera(device, size=1, eye=(3., 0., 3.))
    prims = [glass, slab, floor, receiver, lamp]
    clear = draw(prims, cam, shadows=False)['colour']
    result = draw(prims, cam, shadows=True)
    expected = clear.new_tensor([.96 ** 2 * math.exp(-s) for s in sigma])
    torch.testing.assert_close(clear, torch.full_like(clear, .6 / (9 * math.pi)), atol=2e-6, rtol=0.)
    torch.testing.assert_close(result['colour'] / clear, expected.expand_as(clear), atol=2e-5, rtol=0.)
    assert result['transmission']['shadow_glass_interfaces'] == 2
    assert result['transmission']['shadow_paths_truncated'] == 0


def test_direct_light_endpoint_inside_glass_is_rejected(device):
    glass = g.Material.dielectric('glass', device=device)
    floor = g.Material.diffuse('floor', (.6, .6, .6), device=device)
    receiver = g.Mesh.quad('receiver', u=(8., 0., 0.), v=(0., 8., 0.), material='floor', device=device)
    lamp = g.Light.point('light', position=(0., 0., 1.5), device=device)
    prims = [glass, box('slab', z0=1., z1=2., half=.25, device=device), floor, receiver, lamp]
    with pytest.raises(NotImplementedError, match='air'):
        draw(prims, camera(device, size=1, eye=(3., 0., 3.)), shadows=True)


def test_opaque_endpoint_inside_glass_is_rejected(device):
    glass = g.Material.dielectric('glass', device=device)
    with pytest.raises(NotImplementedError, match='inside glass'):
        draw([glass, box('slab', device=device), *back_plate(device, z=.5)], camera(device))


def test_nested_glass_is_rejected(device):
    glass = g.Material.dielectric('glass', device=device)
    with pytest.raises(NotImplementedError, match='nested'):
        draw([glass, box('outer', device=device), box('inner', z0=.2, z1=.8, half=1., device=device)],
             camera(device))


def test_open_or_inward_glass_is_rejected(device):
    glass = g.Material.dielectric('glass', device=device)
    opened = g.Mesh.quad('sheet', u=(2., 0., 0.), v=(0., 2., 0.), material='glass', device=device)
    with pytest.raises(ValueError, match='closed'):
        draw([glass, opened], camera(device))
    inward = box('inward', device=device)
    inward = replace(inward, faces=inward.faces.flip(-1))
    with pytest.raises(ValueError, match='outward'):
        draw([glass, inward], camera(device))


def test_camera_inside_glass_is_rejected(device):
    glass = g.Material.dielectric('glass', device=device)
    with pytest.raises(NotImplementedError, match='unmatched exit'):
        draw([glass, box('slab', device=device)], camera(device, eye=(0., 0., .5), target=(0., 0., -1.)))


def test_raw_emissive_glass_and_implicit_gradient_are_rejected(device):
    glass = g.Material.dielectric('glass', device=device)
    emission = glass.parameters['absorption'].new_ones(3)
    with pytest.raises(NotImplementedError, match='emissive dielectric'):
        draw([replace(glass, emission=emission), box('slab', device=device)], camera(device))
    mesh = box('slab', device=device)
    mesh.vertices.requires_grad_(True)
    with pytest.raises(ValueError, match='forward-only'):
        draw([glass, mesh], camera(device))


def test_unused_glass_keeps_original_opaque_pixels(device):
    floor = g.Material.diffuse('floor', (.6, .6, .6), device=device)
    receiver = g.Mesh.quad('receiver', u=(8., 0., 0.), v=(0., 8., 0.), material='floor', device=device)
    lamp = g.Light.point('light', position=(0., 0., 3.), power=4 * math.pi, device=device)
    cam = camera(device, size=3)
    plain = g.render(g.Scene([floor, receiver, lamp]), cam, integrator='raster')
    unused = g.Material.dielectric('unused', device=device)
    other = g.render(g.Scene([floor, unused, receiver, lamp]), cam, integrator='raster')
    assert torch.equal(plain['colour'], other['colour'])
    assert torch.equal(plain['mask'], other['mask'])


def test_float64_camera_and_scene_keep_their_precision(device):
    glass = g.Material.dielectric('glass', ior=1.5, device=device)
    model = g.Scene([glass, box('slab', device=device), *back_plate(device)]).to(dtype=torch.float64)
    pose = camera(device).pose.to(torch.float64)
    cam = g.Camera.from_orthographic(10., 10., 1.5, 1.5, 3, 3, .05, 10., pose=pose)
    result = g.render(model, cam, integrator='raster', background=(0., 0., 0.))
    assert result['colour'].dtype == torch.float64
    torch.testing.assert_close(result['colour'], torch.full_like(result['colour'], .96 / 1.04),
                               atol=1e-5, rtol=0.)


def test_depth_and_face_index_remain_centre_primary_visibility(device):
    glass = g.Material.dielectric('glass', ior=1., device=device)
    model = g.Scene([glass, box('slab', device=device), *back_plate(device)])
    cam = camera(device)
    result = g.render(model, cam, integrator='raster', supersample=2,
                      outputs=('colour', 'mask', 'depth', 'face_index'))
    expected = cam.view_depth_to_ndc(torch.full_like(result['depth'], 2.))
    torch.testing.assert_close(result['depth'], expected, atol=2e-6, rtol=0.)
    assert bool(((result['face_index'] == 2) | (result['face_index'] == 3)).all())
    assert torch.equal(result['mask'], torch.ones_like(result['mask']))


def test_finite_long_tir_path_exits_the_tumbler_before_termination(device):
    from gffx.examples.studio_d import build_scene

    model, full = build_scene(device=device, size=(128, 128))
    # Preserve the ray through original pixel (99,66) in a one-pixel camera.
    # It enters the cup wall, undergoes 50 TIR events, then leaves the medium
    # and reaches a lit endpoint. A 48-interface cap lost almost 90% of its
    # blue path weight immediately before this finite exit.
    cam = g.Camera.from_intrinsics(full.fx, full.fy, full.cx - 99, full.cy - 66,
                                    1, 1, full.near, full.far, pose=full.pose)
    result = g.render(model, cam, integrator='raster', shadows=True,
                      background=(.035, .045, .065), outputs=('colour', 'transmission'))
    report = result['transmission']
    assert report['camera_glass_interfaces'] > 48
    assert report['total_internal_reflection_events'] >= 50
    assert report['depth_limited_paths'] == 0
    assert report['capacity_limited_paths'] == 0
    assert max(report['discarded_path_weight_max_rgb']) < 1e-3
    assert bool(torch.isfinite(result['colour']).all())
    assert float(result['colour'].amax()) > .1
