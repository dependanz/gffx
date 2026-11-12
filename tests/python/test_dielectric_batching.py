"""Batching controls for preserving the accepted D01 camera and transport."""
import pytest
import torch

import gffx as g
from gffx.torch._dielectric_render import _primary_ray_batch, _primary_rays


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


def _camera(device, projection, *, size=(31, 79), dtype=torch.float32):
    cam = g.Camera.look_at((3.137, -.219, 7.871), (.13, .48, -.2),
                           size=size, fov=32., near=.07, far=30.,
                           device=device, dtype=dtype)
    if projection == 'orthographic':
        return g.Camera.from_orthographic(cam.fx / 4, cam.fy / 4, cam.cx, cam.cy,
                                         *size, cam.near, cam.far, pose=cam.pose)
    return cam


@pytest.mark.parametrize('projection', ['perspective', 'orthographic'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_regrouped_transport_keeps_original_camera_rays_exact(device, projection, dtype):
    cam = _camera(device, projection, dtype=dtype)
    count = cam.image_height * cam.image_width
    # Freeze the original CUDA pixel groups independently of the new helper.
    # An oblique pose, non-square image, fractional sample and final partial
    # group make batch-dependent camera rounding visible to this comparison.
    for dx, dy in ((0., 0.), (.25, -.25)):
        original = [
            _primary_rays(cam, torch.arange(first, min(first + 256, count), device=device), dx, dy)
            for first in range(0, count, 256)
        ]
        expected = tuple(torch.cat(parts) for parts in zip(*original))
        for transport_size in (256, 512, 2048):
            regrouped = [
                _primary_ray_batch(cam, first, min(first + transport_size, count), dx, dy, 256)
                for first in range(0, count, transport_size)
            ]
            actual = tuple(torch.cat(parts) for parts in zip(*regrouped))
            for old, new in zip(expected, actual):
                assert old.dtype == new.dtype == dtype
                assert torch.equal(old, new)


def _slab_scene(device):
    vertices = torch.tensor([
        [-.65, -.65, 0.], [.65, -.65, 0.], [.65, .65, 0.], [-.65, .65, 0.],
        [-.65, -.65, .8], [.65, -.65, .8], [.65, .65, .8], [-.65, .65, .8],
    ], device=device)
    faces = torch.tensor([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=torch.int32, device=device)
    return g.Scene([
        g.Material.dielectric('glass', ior=1.5, absorption=(.2, .7, 1.3), device=device),
        g.Mesh('slab', vertices, faces, material='glass'),
        g.Material.emissive('plate', (.9, .7, .4), device=device),
        g.Mesh.quad('back', center=(0., 0., -1.), u=(8., 0., 0.), v=(0., 8., 0.),
                    material='plate', device=device),
    ])


@pytest.mark.parametrize('projection', ['perspective', 'orthographic'])
@pytest.mark.parametrize('supersample', [1, 2])
def test_public_render_preserves_batched_colour_and_primary_outputs(device, projection, supersample):
    cam = _camera(device, projection, size=(17, 33))
    scene = _slab_scene(device)
    options = dict(integrator='raster', supersample=supersample,
                   background=(.05, .1, .2),
                   outputs=('colour', 'mask', 'depth', 'face_index', 'transmission'))
    reference = g.render(scene, cam, fragment_block=256, **options)
    assert reference['transmission']['camera_glass_interfaces'] > 0
    assert reference['transmission']['camera_query_rays'] > reference['transmission']['primary_samples']
    # 600 exercises an upper bound that is not a multiple of the original
    # camera group; 2048 exercises the full new CUDA transport group.
    for fragment_block in (600, 2048):
        result = g.render(scene, cam, fragment_block=fragment_block, **options)
        for name in ('colour', 'depth'):
            torch.testing.assert_close(result[name], reference[name], atol=3e-5, rtol=3e-4)
        for name in ('mask', 'face_index'):
            assert torch.equal(result[name], reference[name])
        stats, old = result['transmission'], reference['transmission']
        assert 0 < stats['transport_ray_chunk'] <= fragment_block
        assert stats['transport_ray_chunk'] % stats['primary_ray_generation_chunk'] == 0
        assert stats['primary_ray_generation_chunk'] == (256 if device == 'cuda' else 64)
        assert stats['transport_ray_chunk'] == ((512 if fragment_block == 600 else 2048)
                                                 if device == 'cuda' else 64)
        assert stats['capacity_limited_paths'] == stats['shadow_paths_truncated'] == 0
        for name in ('camera_query_rays', 'camera_glass_interfaces',
                     'opaque_endpoint_paths', 'total_internal_reflection_events',
                     'depth_limited_paths', 'weight_limited_paths'):
            assert stats[name] == old[name]
        for value, baseline in zip(stats['discarded_path_weight_max_rgb'],
                                   old['discarded_path_weight_max_rgb']):
            assert value <= baseline + 1e-6 + 3e-4 * baseline
