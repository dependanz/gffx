"""Installed alpha GPU controls: exact matched VJPs and independent local differences.

These tiny fixtures validate the packaged public primitive route, not the paused
filtered-renderer campaign. Gradcheck uses its standard tolerances on smooth,
nondegenerate float64 cases; matching CPU/CUDA values and VJPs remain byte-exact.
"""
import pytest
import torch
import gffx.torch as gt


OPERATIONS = ('face_geometry', 'vertex_normals', 'gather_faces', 'transform_points',
              'perspective_divide', 'knn', 'closest_point_on_mesh', 'sample_surface',
              'rasterize', 'interpolate', 'texture_pyramid', 'texture')


def case(name, device, dtype):
    def t(value):
        return torch.tensor(value, dtype=dtype, device=device, requires_grad=True)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    vertices = lambda: t([[0., 0., 0.], [1.1, .05, .1], [.05, 1.2, -.1]])
    if name == 'face_geometry':
        return lambda v: gt.mesh.face_geometry(v, faces)[:2], (vertices(),)
    if name == 'vertex_normals':
        return lambda v: gt.mesh.vertex_normals(v, faces), (vertices(),)
    if name == 'gather_faces':
        return lambda v: gt.mesh.gather_faces(v, faces), (vertices(),)
    if name == 'transform_points':
        return gt.transforms.transform_points, (vertices(), t([[[1.1, .1, .2, .3], [.2, 1.2, .1, -.2], [.1, .2, .9, .5], [0., 0., 0., 1.]]]))
    if name == 'perspective_divide':
        return lambda x: gt.transforms.perspective_divide(x)[0], (t([[.1, .2, .3, 2.], [.7, -.3, .2, 3.]]),)
    if name == 'knn':
        return lambda q, r: gt.points.knn(q, r, neighbor_count=1)[0], (t([[.1, .2, .3], [2., 1.1, .4]]), t([[0., 0., 0.], [2., 1., .5], [4., 3., 2.]]))
    if name == 'closest_point_on_mesh':
        return lambda q, v: gt.points.closest_point_on_mesh(q, v, faces)[0], (t([[.2, .2, 1.], [.3, .15, 2.]]), vertices())
    if name == 'sample_surface':
        key = torch.tensor([0x12345678, 0x9ABCDEF0], dtype=torch.uint32, device=device)
        counter = torch.tensor([0, 0], dtype=torch.uint32, device=device)
        return lambda v: gt.mesh.sample_surface(v, faces, 4, key, counter)[0], (vertices(),)
    if name == 'rasterize':
        def fn(v):
            index, barycentric, depth, _ = gt.render.rasterize(v, faces, 4, 4, cull_mode=gt.render.CULL_NONE)
            valid = index >= 0
            return barycentric[valid], depth[valid]
        return fn, (t([[-.83, -.71, .2], [.91, -.67, .4], [.13, .87, .6]]),)
    if name == 'interpolate':
        index = torch.zeros((1, 1, 1, 1), dtype=torch.int32, device=device)
        return lambda b, a: gt.render.interpolate(index, b, a), (t([[[[[.2, .3, .5]]]]]), t([[[.1, .2], [.5, .6], [.8, .9]]]))
    if name == 'texture_pyramid':
        return lambda x: gt.render.texture_pyramid(x)[0], (t([[[.1, .2], [.3, .4]], [[.5, .6], [.7, .8]]]),)
    offsets = torch.tensor([0, 8], dtype=torch.int32, device=device)
    return lambda x, uv: gt.render.texture(x, offsets, 2, 2, uv, channel_count=2), (t([.1, .2, .3, .4, .5, .6, .7, .8]), t([[.4, .35], [.6, .65]]))


def outputs(value):
    return (value,) if isinstance(value, torch.Tensor) else tuple(value)


def exact(left, right):
    left, right = left.detach().cpu().contiguous(), right.detach().cpu().contiguous()
    assert left.shape == right.shape and left.dtype == right.dtype
    assert torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8))


@pytest.mark.parametrize('name', OPERATIONS)
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_gpu_alpha_primitive_forward_and_matched_vjp(name, dtype):
    assert torch.cuda.is_available(), 'GPU alpha validation requires an actual CUDA device'
    cpu_fn, cpu_inputs = case(name, 'cpu', dtype)
    gpu_fn, gpu_inputs = case(name, 'cuda', dtype)
    expected, actual = outputs(cpu_fn(*cpu_inputs)), outputs(gpu_fn(*gpu_inputs))
    assert all(value.device.type == 'cuda' for value in actual)
    for left, right in zip(expected, actual):
        exact(left, right)
    cotangents = tuple((torch.arange(1, x.numel()+1, dtype=dtype)/16).reshape(x.shape) for x in expected)
    cg = torch.autograd.grad(expected, cpu_inputs, cotangents)
    gg = torch.autograd.grad(actual, gpu_inputs, tuple(x.cuda() for x in cotangents))
    for left, right in zip(cg, gg):
        exact(left, right)


@pytest.mark.parametrize('name', OPERATIONS)
def test_gpu_alpha_primitive_independent_numerical_gradient(name):
    assert torch.cuda.is_available()
    fn, inputs = case(name, 'cuda', torch.float64)
    assert torch.autograd.gradcheck(fn, inputs, fast_mode=True)


@pytest.mark.parametrize('name', ('blend_shapes', 'rodrigues', 'forward_kinematics', 'skin_lbs', 'evaluate_barycentric'))
def test_gpu_alpha_animation_independent_numerical_gradient(name):
    from test_animation_cuda import _case
    assert torch.cuda.is_available()
    fn, original = _case(name, torch.float64)
    inputs = tuple(x.detach().cuda().requires_grad_(x.requires_grad) for x in original)
    assert torch.autograd.gradcheck(fn, inputs, fast_mode=True)


@pytest.mark.parametrize('field', ('key', 'counter'))
def test_gpu_alpha_sample_rejects_mixed_rng_before_dispatch(field, monkeypatch):
    assert torch.cuda.is_available()
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], device='cuda')
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')
    key = torch.tensor([1, 2], dtype=torch.uint32, device='cuda')
    counter = torch.tensor([0, 0], dtype=torch.uint32, device='cuda')
    if field == 'key': key = key.cpu()
    else: counter = counter.cpu()
    def forbidden(*args, **kwargs):
        raise AssertionError('mixed-device inputs reached native dispatch')
    monkeypatch.setattr(torch.ops.gffx, 'sample_surface', forbidden)
    with pytest.raises(ValueError, match='same device'):
        gt.mesh.sample_surface(vertices, faces, 4, key, counter)


@pytest.mark.parametrize('field', ('key', 'counter'))
def test_gpu_alpha_native_sample_rejects_host_rng(field):
    assert torch.cuda.is_available()
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], device='cuda')
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')
    vertex_offsets = torch.tensor([0, 3], dtype=torch.int32, device='cuda')
    face_offsets = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
    key = torch.tensor([1, 2], dtype=torch.uint32, device='cuda')
    counter = torch.tensor([0, 0], dtype=torch.uint32, device='cuda')
    if field == 'key': key = key.cpu()
    else: counter = counter.cpu()
    with pytest.raises(RuntimeError, match='rng_'+field+"|rng_counter"):
        torch.ops.gffx.sample_surface(vertices, faces, vertex_offsets, face_offsets, 4, key, counter, 2.**-20)
