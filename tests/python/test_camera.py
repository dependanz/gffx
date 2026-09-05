"""CAMERA_CONTRACT_V0_1.md section 9: the camera object.

Red baseline for the `camera` subproject's Phase 0. Every fixture here is specified in section
9.5 and fails on the absence of `gffx.camera`, not on a version gate and not on an assertion. CAM-01 through CAM-09 are held by the transforms fixtures and are not repeated.

Where a fixture claims the object reproduces the manual chain, the comparison is `torch.equal`
rather than a tolerance: the object is a composition over the same two operations, so any
difference at all is a second convention, which is the defect these tests exist to prevent.
"""

from __future__ import annotations

import math

import pytest
import torch

import gffx.torch as gt
try:
    from gffx import camera as gcam  # the facade of API_CONTRACT_V0_1.md section 2.1
    _RED = None
except ImportError as _absent:  # the red baseline: the module does not exist yet
    gcam = None
    _RED = str(_absent)

# While `gffx.camera` is absent every fixture here is an expected failure, strictly: the moment the
# module exists these run for real, and an unexpected pass is reported as a failure so the marker
# cannot outlive the baseline it guards.
pytestmark = pytest.mark.xfail(_RED is not None, reason="red baseline: " + str(_RED), strict=True)


def _require_module():
    if gcam is None:
        pytest.xfail("red baseline: " + _RED)


# The vocaset configuration of CAMERA_CONTRACT_V0_1.md section 6, carried as fixture data.
FX = 2377.489709675
W = H = 800
CX = CY = 400.0
NEAR, FAR = 0.01, 3.0


def vocaset(dtype=torch.float64, device="cpu", batch=1):
    pose = torch.eye(4, dtype=dtype, device=device).repeat(batch, 1, 1)
    pose[:, 2, 3] = 1.0  # camera at (0, 0, 1) looking down -z, section 6
    return gcam.Camera.from_intrinsics(
        fx=torch.full((batch,), FX, dtype=dtype, device=device),
        fy=torch.full((batch,), FX, dtype=dtype, device=device),
        cx=torch.full((batch,), CX, dtype=dtype, device=device),
        cy=torch.full((batch,), CY, dtype=dtype, device=device),
        image_height=H, image_width=W, near=NEAR, far=FAR, pose=pose,
    )


def manual_projection(fx, fy, cx, cy, w, h, near, far, dtype):
    """Section 3, assembled by hand."""
    P = torch.zeros(4, 4, dtype=dtype)
    P[0, 0] = 2 * fx / w
    P[0, 2] = 1 - 2 * cx / w
    P[1, 1] = 2 * fy / h
    P[1, 2] = 2 * cy / h - 1
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = (2 * far * near) / (near - far)
    P[3, 2] = -1
    return P


def manual_chain(points, M, w, h):
    """transform_points -> perspective_divide -> the pixel mapping of section 9.2."""
    homogeneous = gt.transforms.transform_points(points, M)
    ndc, valid = gt.transforms.perspective_divide(homogeneous)
    px = (ndc[..., 0] + 1) * w / 2
    py = (1 - ndc[..., 1]) * h / 2
    return torch.stack([px, py], dim=-1), ndc[..., 2], valid


def random_points_in_front(n, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(n, 3, generator=g, dtype=dtype)
    p[:, 0] = (p[:, 0] - 0.5) * 0.4
    p[:, 1] = (p[:, 1] - 0.5) * 0.4
    p[:, 2] = 0.2 + p[:, 2] * 0.6  # world z in (0.2, 0.8): in front of a camera at z = 1
    return p


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cam10_world_to_clip_is_bit_identical_to_the_hand_assembled_matrix(dtype):
    cam = vocaset(dtype)
    P = manual_projection(FX, FX, CX, CY, W, H, NEAR, FAR, dtype)
    V = torch.eye(4, dtype=dtype)
    V[2, 3] = -1.0
    assert torch.equal(cam.projection()[0], P)
    assert torch.equal(cam.world_to_view()[0], V)
    assert torch.equal(cam.world_to_clip()[0], P @ V)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cam11_project_is_bit_identical_to_the_manual_chain(dtype):
    cam = vocaset(dtype)
    M = cam.world_to_clip()
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-0.2, 0.2, 9, dtype=dtype), torch.linspace(-0.2, 0.2, 9, dtype=dtype),
        indexing="ij"), dim=-1).reshape(-1, 2)
    grid = torch.cat([grid, torch.full((grid.shape[0], 1), 0.5, dtype=dtype)], dim=-1)
    landmarks = random_points_in_front(68, dtype, seed=1)  # the unstructured landmark case
    for points in (grid, landmarks):
        want_px, want_z, want_valid = manual_chain(points, M, W, H)
        got = cam.project(points)
        assert torch.equal(got.pixels, want_px)
        assert torch.equal(got.ndc_depth, want_z)
        assert torch.equal(got.valid, want_valid)


def test_cam12_ndc_depth_matches_the_rasterizer_at_a_vertex_on_a_pixel_centre():
    dtype = torch.float64
    cam = vocaset(dtype)
    # Build a triangle whose first vertex projects exactly to the centre of pixel (row 300, col 500).
    target_px = torch.tensor([500.5, 300.5], dtype=dtype)
    z_view = -0.6
    x_view = (target_px[0] - CX) * (-z_view) / FX
    y_view = (CY - target_px[1]) * (-z_view) / FX
    v0 = torch.tensor([x_view, y_view, z_view + 1.0], dtype=dtype)  # world = view + (0,0,1)
    verts = torch.stack([v0, v0 + torch.tensor([0.05, 0.0, 0.0], dtype=dtype),
                         v0 + torch.tensor([0.0, 0.05, 0.0], dtype=dtype)])
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    proj = cam.project(verts)
    ndc = torch.cat([(proj.pixels[:, 0] * 2 / W - 1)[:, None],
                     (1 - proj.pixels[:, 1] * 2 / H)[:, None], proj.ndc_depth[:, None]], dim=-1)
    face_index, bary, depth, _ = gt.render.rasterize(
        ndc_vertices=ndc, faces=faces, image_height=H, image_width=W,
        faces_per_pixel=1, blur_radius_px=0.0, cull_mode=gt.render.CULL_NONE)
    assert face_index[0, 300, 500, 0].item() == 0
    assert abs(depth[0, 300, 500, 0].item() - proj.ndc_depth[0].item()) < 1e-9


def test_cam13_view_depth_and_ndc_depth_convert_through_the_projection_row():
    dtype = torch.float64
    cam = vocaset(dtype)
    depths = torch.cat([torch.tensor([NEAR, FAR], dtype=dtype), torch.linspace(0.02, 2.9, 10, dtype=dtype)])
    ndc = cam.view_depth_to_ndc(depths)
    back = cam.ndc_to_view_depth(ndc)
    assert torch.allclose(back, depths, rtol=0, atol=1e-9)
    assert abs(ndc[0].item() + 1.0) < 1e-12 and abs(ndc[1].item() - 1.0) < 1e-12


def test_cam14_unproject_recovers_interpolated_world_positions_from_a_rendered_depth_buffer():
    dtype = torch.float64
    cam = vocaset(dtype)
    verts = torch.tensor([[-0.1, -0.1, 0.4], [0.1, -0.1, 0.4], [0.0, 0.1, 0.6]], dtype=dtype)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    proj = cam.project(verts)
    ndc = torch.cat([(proj.pixels[:, 0] * 2 / W - 1)[:, None],
                     (1 - proj.pixels[:, 1] * 2 / H)[:, None], proj.ndc_depth[:, None]], dim=-1)
    face_index, bary, depth, _ = gt.render.rasterize(
        ndc_vertices=ndc, faces=faces, image_height=H, image_width=W,
        faces_per_pixel=1, blur_radius_px=0.0, cull_mode=gt.render.CULL_NONE)
    world = gt.render.interpolate(face_index, bary, verts[faces])  # [1,H,W,1,3]
    covered = face_index[..., 0] >= 0
    xyz, valid = cam.unproject(depth[..., 0], kind="ndc")
    assert torch.equal(valid, covered)
    assert torch.allclose(xyz[covered], world[..., 0, :][covered], rtol=0, atol=1e-7)
    assert torch.all(xyz[~covered] == 0)
    # The same scene through metric depth, as a sensor would report it.
    view_depth = cam.ndc_to_view_depth(depth[..., 0])
    xyz2, valid2 = cam.unproject(view_depth, kind="view", valid=covered)
    assert torch.allclose(xyz2[covered], world[..., 0, :][covered], rtol=0, atol=1e-7)


def test_cam15_unproject_points_inverts_project():
    dtype = torch.float64
    cam = vocaset(dtype)
    p = random_points_in_front(256, dtype, seed=2)
    proj = cam.project(p)
    back = cam.unproject_points(proj.pixels, proj.view_depth, kind="view")
    assert torch.allclose(back, p, rtol=0, atol=1e-9)
    back2 = cam.unproject_points(proj.pixels, proj.ndc_depth, kind="ndc")
    assert torch.allclose(back2, p, rtol=0, atol=1e-9)


def test_cam16_stack_builds_a_rig_that_equals_the_single_camera_runs():
    dtype = torch.float64
    a = vocaset(dtype)
    b = gcam.Camera.from_intrinsics(fx=torch.tensor([600.0], dtype=dtype), fy=torch.tensor([650.0], dtype=dtype),
                                    cx=torch.tensor([300.0], dtype=dtype), cy=torch.tensor([450.0], dtype=dtype),
                                    image_height=H, image_width=W, near=NEAR, far=FAR)
    rig = gcam.Camera.stack([a, b])
    assert rig.batch_size == 2
    p = random_points_in_front(32, dtype, seed=3)
    got = rig.project(p[None].expand(2, -1, -1))
    assert torch.equal(got.pixels[0], a.project(p).pixels)
    assert torch.equal(got.pixels[1], b.project(p).pixels)
    c = gcam.Camera.from_intrinsics(fx=torch.tensor([600.0], dtype=dtype), fy=torch.tensor([650.0], dtype=dtype),
                                    cx=torch.tensor([300.0], dtype=dtype), cy=torch.tensor([450.0], dtype=dtype),
                                    image_height=H // 2, image_width=W, near=NEAR, far=FAR)
    with pytest.raises(ValueError):
        gcam.Camera.stack([a, c])


def test_cam17_gradcheck_reaches_every_intrinsic_and_the_pose_of_each_rig_camera():
    dtype = torch.float64
    p = random_points_in_front(8, dtype, seed=4)
    fx = torch.tensor([FX, 900.0], dtype=dtype, requires_grad=True)
    fy = torch.tensor([FX, 950.0], dtype=dtype, requires_grad=True)
    cx = torch.tensor([CX, 380.0], dtype=dtype, requires_grad=True)
    cy = torch.tensor([CY, 410.0], dtype=dtype, requires_grad=True)
    pose = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
    pose[:, 2, 3] = 1.0
    pose = pose.clone().requires_grad_(True)

    def f(fx, fy, cx, cy, pose):
        cam = gcam.Camera.from_intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, image_height=H, image_width=W,
                                          near=NEAR, far=FAR, pose=pose)
        return cam.project(p[None].expand(2, -1, -1)).pixels

    assert torch.autograd.gradcheck(f, (fx, fy, cx, cy, pose), eps=1e-6, atol=1e-6)
    f(fx, fy, cx, cy, pose).sum().backward()
    for t in (fx, fy, cx, cy):
        assert t.grad is not None and torch.all(t.grad != 0)
    assert torch.any(pose.grad[0] != 0) and torch.any(pose.grad[1] != 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cam17_cuda_gradcheck_on_device():
    dtype = torch.float64
    p = random_points_in_front(8, dtype, seed=5).cuda()
    fx = torch.tensor([FX], dtype=dtype, device="cuda", requires_grad=True)
    cam = gcam.Camera.from_intrinsics(fx=fx, fy=fx.detach(), cx=torch.tensor([CX], dtype=dtype, device="cuda"),
                                      cy=torch.tensor([CY], dtype=dtype, device="cuda"),
                                      image_height=H, image_width=W, near=NEAR, far=FAR)
    assert torch.autograd.gradcheck(lambda fx: cam.project(p).pixels, (fx,), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize("bad", [
    dict(near=3.0, far=0.01),
    dict(fx=torch.tensor([-1.0], dtype=torch.float64)),
    dict(fx=torch.tensor([float("nan")], dtype=torch.float64)),
    dict(image_width=0),
])
def test_cam18_invalid_construction_is_rejected(bad):
    kwargs = dict(fx=torch.tensor([FX], dtype=torch.float64), fy=torch.tensor([FX], dtype=torch.float64),
                  cx=torch.tensor([CX], dtype=torch.float64), cy=torch.tensor([CY], dtype=torch.float64),
                  image_height=H, image_width=W, near=NEAR, far=FAR)
    kwargs.update(bad)
    with pytest.raises(ValueError):
        gcam.Camera.from_intrinsics(**kwargs)
    with pytest.raises(ValueError):
        gcam.Camera.from_intrinsics(fx=torch.tensor([FX, FX], dtype=torch.float64), fy=torch.tensor([FX], dtype=torch.float64),
                                    cx=torch.tensor([CX], dtype=torch.float64), cy=torch.tensor([CY], dtype=torch.float64),
                                    image_height=H, image_width=W, near=NEAR, far=FAR)


def test_cam19_unproject_of_an_infinite_background_is_zero_valid_false_and_free_of_nan():
    dtype = torch.float64
    cam = vocaset(dtype)
    depth = torch.full((1, 8, 8), float("inf"), dtype=dtype)
    depth[0, 2:5, 2:5] = 0.0  # a covered block at NDC depth zero
    depth.requires_grad_(True)
    xyz, valid = cam.unproject(depth, kind="ndc")
    assert not torch.isnan(xyz).any()
    assert torch.equal(valid, torch.isfinite(depth.detach()))
    assert torch.all(xyz[~valid] == 0)
    xyz.sum().backward()
    assert not torch.isnan(depth.grad).any()
    assert torch.all(depth.grad[~valid] == 0)


def test_cam20_the_facade_computes_nothing():
    """Every output through gffx.camera equals the same call through gffx.torch.camera exactly."""
    from gffx.torch import camera as bound
    dtype = torch.float64
    p = random_points_in_front(64, dtype, seed=6)
    a = vocaset(dtype)
    b = bound.Camera.from_intrinsics(fx=a.fx, fy=a.fy, cx=a.cx, cy=a.cy, image_height=H, image_width=W,
                                     near=NEAR, far=FAR, pose=a.pose)
    pa, pb = a.project(p), b.project(p)
    for x, y in zip(pa, pb):
        assert torch.equal(x, y)
    assert torch.equal(a.world_to_clip(), b.world_to_clip())
    xa, va = a.unproject(torch.zeros(1, 4, 4, dtype=dtype), kind="ndc")
    xb, vb = b.unproject(torch.zeros(1, 4, 4, dtype=dtype), kind="ndc")
    assert torch.equal(xa, xb) and torch.equal(va, vb)
