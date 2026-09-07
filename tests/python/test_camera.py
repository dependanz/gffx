"""CAMERA_CONTRACT_V0_1.md section 9: the camera object.

Red baseline for the `camera` subproject's Phase 0. Every fixture here is specified in section
9.5 and fails on the absence of `gffx.camera`, not on a version gate and not on an assertion. CAM-01 through CAM-09 are held by the transforms fixtures and are not repeated.

Where a fixture claims the object reproduces the manual chain, the comparison is `torch.equal`
rather than a tolerance: the object is a composition over the same two operations, so any
difference at all is a second convention, which is the defect these tests exist to prevent.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

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


def _check(request,name):
    plugin=request.config.pluginmanager.get_plugin('tools.testing_dashboard.pytest_plugin')
    return plugin.Evidence(request.node.nodeid).check(name) if plugin else nullcontext()


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
def test_cam10_world_to_clip_is_bit_identical_to_the_hand_assembled_matrix(dtype,request):
    cam = vocaset(dtype)
    P = manual_projection(FX, FX, CX, CY, W, H, NEAR, FAR, dtype)
    V = torch.eye(4, dtype=dtype)
    V[2, 3] = -1.0
    with _check(request,'matrix convention') as evidence:
        if evidence:evidence.record('expected and actual matrices',{'expected':(P@V).tolist(),'actual':cam.world_to_clip()[0].tolist(),'dtype':str(dtype)})
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
    # Rasterizer barycentrics are screen-space. World-space attributes require
    # perspective correction: interpolate attribute/w, then divide by interpolated 1/w.
    reciprocal_w=proj.view_depth.reciprocal()
    numerator=gt.render.interpolate(face_index,bary,(verts*reciprocal_w[:,None])[faces])
    denominator=gt.render.interpolate(face_index,bary,reciprocal_w[:,None][faces])
    world=numerator/torch.where(denominator!=0,denominator,torch.ones_like(denominator))
    covered = face_index[..., 0] >= 0
    xyz, valid = cam.unproject(depth[..., 0], kind="ndc")
    assert torch.equal(valid, covered)
    assert torch.allclose(xyz[covered], world[..., 0, :][covered], rtol=0, atol=1e-7)
    # Independent scene fact: the triangle lies on the world plane z = y + 0.5.
    assert torch.allclose(xyz[...,2][covered],xyz[...,1][covered]+.5,rtol=0,atol=1e-9)
    assert torch.all(xyz[~covered] == 0)
    # The same scene through metric depth, as a sensor would report it.
    view_depth = cam.ndc_to_view_depth(depth[..., 0])
    xyz2, valid2 = cam.unproject(view_depth, kind="view", valid=covered)
    assert torch.allclose(xyz2[covered], world[..., 0, :][covered], rtol=0, atol=1e-7)


def test_cam15_unproject_points_inverts_project(request):
    dtype = torch.float64
    cam = vocaset(dtype)
    p = random_points_in_front(256, dtype, seed=2)
    proj = cam.project(p)
    back = cam.unproject_points(proj.pixels, proj.view_depth, kind="view")
    with _check(request,'metric depth round trip') as evidence:
        if evidence:evidence.record('round trip error',{'points':len(p),'max_absolute_world_error':float((back-p).abs().max()),'tolerance':1e-9})
        assert torch.allclose(back, p, rtol=0, atol=1e-9)
    back2 = cam.unproject_points(proj.pixels, proj.ndc_depth, kind="ndc")
    with _check(request,'NDC depth round trip') as evidence:
        if evidence:evidence.record('round trip error',{'points':len(p),'max_absolute_world_error':float((back2-p).abs().max()),'tolerance':1e-9})
        assert torch.allclose(back2, p, rtol=0, atol=1e-9)


def test_cam16_stack_builds_a_rig_that_equals_the_single_camera_runs(request):
    dtype = torch.float64
    a = vocaset(dtype)
    b = gcam.Camera.from_intrinsics(fx=torch.tensor([600.0], dtype=dtype), fy=torch.tensor([650.0], dtype=dtype),
                                    cx=torch.tensor([300.0], dtype=dtype), cy=torch.tensor([450.0], dtype=dtype),
                                    image_height=H, image_width=W, near=NEAR, far=FAR)
    rig = gcam.Camera.stack([a, b])
    assert rig.batch_size == 2
    p = random_points_in_front(32, dtype, seed=3)
    got = rig.project(p[None].expand(2, -1, -1))
    with _check(request,'rig equals separate cameras') as evidence:
        expected=torch.stack([a.project(p).pixels,b.project(p).pixels])
        if evidence:evidence.record('two-camera comparison',{'cameras':2,'points_per_camera':len(p),'max_pixel_error':float((got.pixels-expected).abs().max()),'first_pixels_per_camera':got.pixels[:,:3].tolist()})
        assert torch.equal(got.pixels,expected)
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
    # Independent storage: numerical perturbations must not mutate another input.
    intrinsics = [torch.tensor([value], dtype=dtype, device="cuda", requires_grad=True)
                  for value in (FX, FX, CX, CY)]
    pose = torch.eye(4, dtype=dtype, device="cuda")[None].clone().requires_grad_()
    def project(fx, fy, cx, cy, pose):
        return gcam.Camera.from_intrinsics(fx, fy, cx, cy, H, W, NEAR, FAR, pose=pose).project(p).pixels
    assert torch.autograd.gradcheck(project, (*intrinsics, pose), eps=1e-6, atol=1e-6, rtol=1e-4)


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


def test_cam19_unproject_of_an_infinite_background_is_zero_valid_false_and_free_of_nan(request):
    dtype = torch.float64
    cam = vocaset(dtype)
    depth = torch.full((1, 8, 8), float("inf"), dtype=dtype)
    depth[0, 2:5, 2:5] = 0.0  # a covered block at NDC depth zero
    depth.requires_grad_(True)
    xyz, valid = cam.unproject(depth, kind="ndc")
    with _check(request,'background outputs and gradients') as evidence:
        assert not torch.isnan(xyz).any()
        assert torch.equal(valid, torch.isfinite(depth.detach()))
        assert torch.all(xyz[~valid] == 0)
        xyz.sum().backward()
        if evidence:evidence.record('background mask',{'pixels':depth.numel(),'valid_pixels':int(valid.sum()),'background_pixels':int((~valid).sum()),'max_background_depth_gradient':float(depth.grad[~valid].abs().max())})
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


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA device required'))])
@pytest.mark.parametrize('kind',['view','ndc'])
def test_camera_phase3_unprojection_gradients(device,kind,request):
    dtype=torch.float64
    params=[torch.tensor([v],dtype=dtype,device=device,requires_grad=True) for v in (90.,110.,16.,12.)]
    pose=torch.eye(4,dtype=dtype,device=device)[None].clone().requires_grad_()
    depth=torch.full((1,2,2),1.2 if kind=='view' else .4,dtype=dtype,device=device,requires_grad=True)
    def dense(fx,fy,cx,cy,pose,depth):
        cam=gcam.Camera.from_intrinsics(fx,fy,cx,cy,24,32,.1,10.,pose=pose)
        return cam.unproject(depth,kind=kind)[0]
    with _check(request,'dense inverse derivatives') as evidence:
        assert torch.autograd.gradcheck(dense,(*params,pose,depth),eps=1e-6,atol=1e-6,rtol=1e-4)
        if evidence:evidence.record('gradient contract',{'device':device,'depth_kind':kind,'inputs':['fx','fy','cx','cy','pose','depth'],'eps':1e-6,'atol':1e-6})
    pixels=torch.tensor([[[3.,4.],[12.,9.]]],dtype=dtype,device=device,requires_grad=True)
    point_depth=depth.reshape(1,4)[:,:2].detach().clone().requires_grad_()
    def scattered(fx,fy,cx,cy,pose,pixels,depth):
        return gcam.Camera.from_intrinsics(fx,fy,cx,cy,24,32,.1,10.,pose=pose).unproject_points(pixels,depth,kind=kind)
    with _check(request,'scattered inverse derivatives') as evidence:
        assert torch.autograd.gradcheck(scattered,(*params,pose,pixels,point_depth),eps=1e-6,atol=1e-6,rtol=1e-4)
        if evidence:evidence.record('pixel derivative contract',{'device':device,'depth_kind':kind,'includes_pixel_coordinates':True})


def test_camera_phase3_stacked_rig_gradients_are_independent(request):
    dtype=torch.float64;cameras=[];leaves=[]
    for i in range(2):
        intrinsics=[torch.tensor([v],dtype=dtype,requires_grad=True) for v in (90.+i*10,110.,16.,12.)]
        pose=torch.eye(4,dtype=dtype)[None];pose[:,0,3]=i*.2;pose=pose.requires_grad_()
        leaves.extend([*intrinsics,pose]);cameras.append(gcam.Camera.from_intrinsics(*intrinsics,24,32,.1,10.,pose=pose))
    points=torch.tensor([[.1,.2,-1.],[.3,-.1,-2.]],dtype=dtype)
    losses=gcam.Camera.stack(cameras).project(points[None].expand(2,-1,-1)).pixels.square().sum(dim=(1,2))
    with _check(request,'independent rig backward') as evidence:
        first=torch.autograd.grad(losses[0],leaves,retain_graph=True,allow_unused=True)
        assert all(g is None or torch.count_nonzero(g)==0 for g in first[5:])
        expected=[torch.autograd.grad(c.project(points).pixels.square().sum(),leaves[i*5:(i+1)*5]) for i,c in enumerate(cameras)]
        losses.sum().backward()
        for actual,reference in zip(leaves,[g for camera in expected for g in camera]):
            assert actual.grad is not None and torch.isfinite(actual.grad).all() and torch.any(actual.grad!=0)
            torch.testing.assert_close(actual.grad,reference,atol=1e-9,rtol=1e-9)
        if evidence:evidence.record('two cameras, one loss',{'fx_gradients':[float(leaves[i].grad.item()) for i in (0,5)],'cross_camera_gradient_max':0.,'compared_to':'independent single-camera backward'})


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA device required')
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_camera_phase2_cuda_project_unproject_preserves_device(dtype,request):
    cpu=vocaset(dtype=dtype);gpu=vocaset(dtype=dtype,device='cuda');points=random_points_in_front(7,dtype,seed=35)
    with _check(request,'inherited CUDA composition') as evidence:
        projected=gpu.project(points.cuda());baseline=cpu.project(points)
        for actual,reference in zip(projected,baseline):
            assert actual.device.type=='cuda'
            torch.testing.assert_close(actual.cpu(),reference,atol=2e-4 if dtype==torch.float32 else 1e-9,rtol=1e-5 if dtype==torch.float32 else 1e-9)
        depth=torch.ones(1,2,3,dtype=dtype,device='cuda')
        xyz,valid=gpu.unproject(depth,kind='view');expected,mask=cpu.unproject(depth.cpu(),kind='view')
        assert xyz.device.type=='cuda' and valid.device.type=='cuda'
        torch.testing.assert_close(xyz.cpu(),expected);assert torch.equal(valid.cpu(),mask)
        if evidence:evidence.record('device preservation',{'dtype':str(dtype),'project':'cuda','unproject':'cuda','implementation':'inherited transform composition; no new camera CUDA kernel'})


def test_camera_plain_numbers_and_inert_hooks():
    cam=gcam.Camera.from_intrinsics(100.,110.,16.,12.,24,32,.1,10.)
    assert cam.batch_size==1
    assert cam.distortion is None and cam.exposure==1 and cam.rolling_shutter is None and cam.lens_stack is None
    assert cam.project(torch.tensor([[0.,0.,-1.]])).pixels.tolist()==[[16.,12.]]


@pytest.mark.parametrize('bad',[
    {'image_height':True},{'image_width':2.5},{'near':0.},{'far':float('inf')},
    {'fy':0.},{'cx':float('nan')},{'fx':torch.ones(1,dtype=torch.int64)},
    {'pose':torch.zeros(1,4,4,dtype=torch.float64)},
    {'pose':torch.eye(4,dtype=torch.float64)},
])
def test_camera_additional_construction_errors(bad):
    values=dict(fx=torch.tensor([100.],dtype=torch.float64),fy=torch.tensor([100.],dtype=torch.float64),
        cx=torch.tensor([16.],dtype=torch.float64),cy=torch.tensor([12.],dtype=torch.float64),
        image_height=24,image_width=32,near=.1,far=10.)
    values.update(bad)
    with pytest.raises((ValueError,TypeError)):gcam.Camera.from_intrinsics(**values)


def test_camera_packed_rig_matches_single_cameras():
    a=vocaset();b=vocaset();b.pose=b.pose.clone();b.pose[:,0,3]=.02
    rig=gcam.Camera.stack([a,b]);p=random_points_in_front(7,torch.float64,seed=7)
    offsets=torch.tensor([0,2,7],dtype=torch.int32)
    got=rig.project(p,point_offsets=offsets)
    for actual,first,second in zip(got,a.project(p[:2]),b.project(p[2:])):
        assert torch.equal(actual,torch.cat([first,second]))
    with pytest.raises(ValueError):rig.project(p)
    with pytest.raises(ValueError):rig.project(p,point_offsets=torch.tensor([0,6,5],dtype=torch.int32))
    with pytest.raises(ValueError):rig.project(torch.zeros(2,3,4,dtype=torch.float64))
    with pytest.raises(ValueError):rig.project(torch.zeros(3,4,dtype=torch.float64))


def test_camera_empty_dense_and_packed_batches():
    rig=gcam.Camera.stack([vocaset(),vocaset()])
    got=rig.project(torch.empty(2,0,3,dtype=torch.float64))
    assert got.pixels.shape==(2,0,2) and got.valid.shape==(2,0)
    got=rig.project(torch.empty(0,3,dtype=torch.float64),point_offsets=torch.tensor([0,0,0],dtype=torch.int32))
    assert got.pixels.shape==(0,2)


def test_camera_unprojection_mask_and_invalid_kind():
    cam=vocaset();d=torch.ones(1,3,4,dtype=torch.float64,requires_grad=True)
    mask=torch.ones_like(d,dtype=torch.bool);mask[:,1,2]=False
    xyz,valid=cam.unproject(d,kind='view',valid=mask)
    assert torch.equal(valid,mask) and torch.equal(xyz[~mask],torch.zeros_like(xyz[~mask]))
    xyz.sum().backward();assert torch.isfinite(d.grad).all() and torch.all(d.grad[~mask]==0)
    with pytest.raises(ValueError):cam.unproject(d,kind='guess')
    with pytest.raises(ValueError):cam.unproject(d,kind='view',valid=mask[:,:1])
    with pytest.raises(TypeError):cam.unproject(d,kind='view',valid=d)
    with pytest.raises(ValueError):cam.unproject_points(torch.zeros(2,2,dtype=d.dtype),torch.ones(3,dtype=d.dtype),kind='view')


def test_camera_projection_dtype_cast_and_input_immutability():
    cam=vocaset();p=random_points_in_front(5,torch.float32,seed=8);before=p.clone();pose=cam.pose.clone()
    got=cam.project(p)
    assert got.pixels.dtype==p.dtype and got.pixels.device==p.device
    assert torch.equal(p,before) and torch.equal(cam.pose,pose) and cam.fx.dtype==torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA device required')
def test_camera_mixed_devices_are_rejected():
    cam=vocaset()
    with pytest.raises(ValueError,match='device'):cam.project(random_points_in_front(2,torch.float64).cuda())
    with pytest.raises(ValueError,match='device'):cam.project(random_points_in_front(2,torch.float64),point_offsets=torch.tensor([0,2],dtype=torch.int32,device='cuda'))


def test_camera_stack_rejects_incompatible_clipping_and_empty_list():
    a=vocaset();b=gcam.Camera.from_intrinsics(100.,100.,400.,400.,H,W,.1,4.)
    with pytest.raises(ValueError):gcam.Camera.stack([a,b])
    with pytest.raises(ValueError):gcam.Camera.stack([])


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device required'))])
def test_camera_front_filter_preserves_projection_validity_not_visibility(device,request):
    cam=vocaset(device=device)
    # Front, behind, on the camera plane, outside the viewport, beyond far.
    p=torch.tensor([[.01,.02,.5],[.01,.02,1.5],[0.,0.,1.],
                    [10.,0.,.5],[0.,0.,-10.]],dtype=torch.float64,device=device,requires_grad=True)
    ordinary=cam.project(p)
    expected=manual_chain(p,cam.world_to_clip(),W,H)
    with _check(request,'mathematical validity') as evidence:
        if evidence:evidence.record('point categories and default mask',{'categories':['front','behind','camera plane','off-screen','beyond far'],'valid':ordinary.valid.tolist()})
        assert torch.equal(ordinary.valid,expected[2])
        assert ordinary.valid.tolist()==[True,True,False,True,True]
    filtered=cam.project(p,in_front_only=True)
    with _check(request,'optional front filter') as evidence:
        if evidence:evidence.record('front-only mask',{'valid':filtered.valid.tolist(),'does_not_test':'occlusion or viewport visibility'})
        assert filtered.valid.tolist()==[True,False,False,True,True]
        for actual,baseline in zip(filtered[:-1],ordinary[:-1]):
            assert torch.equal(actual[filtered.valid],baseline[filtered.valid])
            assert torch.all(actual[~filtered.valid]==0)
    (filtered.pixels.sum()+filtered.ndc_depth.sum()+filtered.view_depth.sum()).backward()
    with _check(request,'rejected point gradients') as evidence:
        if evidence:evidence.record('gradients',{'device':device,'point_gradients':p.grad.tolist(),'rejected_max_abs':float(p.grad[~filtered.valid].abs().max())})
        assert torch.isfinite(p.grad).all() and torch.all(p.grad[~filtered.valid]==0)
    with pytest.raises(TypeError):cam.project(p,in_front_only='yes')
