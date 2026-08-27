"""Phase 3 Step 1b fixtures TB-01..TB-10 for the remaining ten PyTorch CPU operations.

Fixture numbers match TORCH_ADAPTER_ACCEPTANCE_V0_1.md section 10.

How these differ from TA-01, and why. The first operation was checked against a committed oracle
emitted by the C kernel, so its parity claim is bit-exact against the reference implementation.
Extending that to ten operations would mean ten more emitters, and it would mostly re-test the C
kernels the geometry suite already covers with 660 assertions.

These fixtures instead check the properties an adapter can actually get wrong, which the C suite
cannot see: argument order and count across the boundary, shape and dtype of every output, whether
the gradient is wired to the right input, and whether validation rejects what the record says it
rejects. Where an operation admits an independent reference computable in plain torch, that
reference is used, which is a stronger check than an oracle for adapter defects because it is
computed a different way rather than by the same code being tested.

Numerical correctness of the kernels themselves stays where it belongs: in tests/geometry.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

EPS = 2.0 ** -20


@pytest.fixture(scope="module")
def gffx_torch():
    import gffx.torch as adapter

    return adapter


def tetra(dtype=torch.float64, requires_grad=False):
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype, requires_grad=requires_grad,
    )
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.int32)
    return vertices, faces


# ---------------------------------------------------------------------------------- TB-01

def test_tb01_vertex_normals(gffx_torch):
    """Area and uniform weighting differ, both are unit length, and the gradient reaches vertices."""
    vertices, faces = tetra(requires_grad=True)
    area = gffx_torch.mesh.vertex_normals(vertices, faces)
    uniform = gffx_torch.mesh.vertex_normals(
        vertices, faces, weighting=gffx_torch.mesh.WEIGHTING_UNIFORM
    )
    assert area.shape == (4, 3)
    assert area.dtype == torch.float64
    # Every vertex of a tetrahedron has incident faces, so every normal is unit length.
    assert torch.allclose(area.norm(dim=1), torch.ones(4, dtype=torch.float64))
    # The two weightings must actually differ, or the argument is not reaching the kernel.
    assert not torch.equal(area, uniform)

    area.sum().backward()
    assert vertices.grad is not None and torch.isfinite(vertices.grad).all()

    with pytest.raises(ValueError):
        gffx_torch.mesh.vertex_normals(vertices.detach(), faces, weighting=99)


# ---------------------------------------------------------------------------------- TB-02

def test_tb02_gather_faces(gffx_torch):
    """A pure gather: compare against torch's own indexing, which is an independent computation."""
    vertices, faces = tetra(requires_grad=True)
    gathered = gffx_torch.mesh.gather_faces(vertices, faces)
    expected = vertices.detach()[faces.long()]
    assert gathered.shape == (4, 3, 3)
    assert torch.equal(gathered, expected)

    gathered.sum().backward()
    # Each vertex appears in exactly three of the four faces of a tetrahedron.
    assert torch.equal(vertices.grad, torch.full((4, 3), 3.0, dtype=torch.float64))


# ---------------------------------------------------------------------------------- TB-03

def test_tb03_transform_points(gffx_torch):
    """Compare against an explicit matrix-vector product in torch."""
    points = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], dtype=torch.float64, requires_grad=True)
    matrix = torch.tensor([[
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, -0.25],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ]], dtype=torch.float64)

    homogeneous = gffx_torch.transforms.transform_points(points, matrix)
    assert homogeneous.shape == (2, 4)
    padded = torch.cat([points.detach(), torch.ones(2, 1, dtype=torch.float64)], dim=1)
    expected = padded @ matrix[0].T
    assert torch.allclose(homogeneous, expected)

    homogeneous.sum().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()


# ---------------------------------------------------------------------------------- TB-04

def test_tb04_perspective_divide(gffx_torch):
    """Division by w, with a degenerate w reported rather than producing an infinity."""
    homogeneous = torch.tensor(
        [[2.0, 4.0, 6.0, 2.0], [1.0, 1.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    ndc, valid = gffx_torch.transforms.perspective_divide(homogeneous)
    assert ndc.shape == (2, 3)
    assert valid.dtype == torch.bool
    assert torch.equal(ndc[0], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    assert bool(valid[0]) and not bool(valid[1])
    # The invalid row is exactly zero, not inf or nan.
    assert torch.equal(ndc[1], torch.zeros(3, dtype=torch.float64))

    ndc.sum().backward()
    assert torch.isfinite(homogeneous.grad).all()


# ---------------------------------------------------------------------------------- TB-05

def test_tb05_build_edge_topology(gffx_torch):
    """A tetrahedron has exactly six edges, each shared by exactly two faces."""
    _, faces = tetra()
    edges, edge_face_offsets, edge_faces, mesh_edge_offsets = \
        gffx_torch.mesh.build_edge_topology(faces)

    edge_count = int(mesh_edge_offsets[-1])
    assert edge_count == 6, "a closed tetrahedron has six undirected edges"
    valid_edges = edges[:edge_count]
    assert (valid_edges[:, 0] < valid_edges[:, 1]).all(), "edges are canonical (min, max)"
    # Trailing rows are the documented (-1, -1) sentinel rather than stale memory.
    assert torch.equal(
        edges[edge_count:], torch.full((edges.shape[0] - edge_count, 2), -1, dtype=torch.int32)
    )
    for edge in range(edge_count):
        incident = edge_faces[int(edge_face_offsets[edge]):int(edge_face_offsets[edge + 1])]
        assert incident.numel() == 2, "every edge of a closed manifold has two incident faces"


# ---------------------------------------------------------------------------------- TB-06

def test_tb06_knn(gffx_torch):
    """Compare selection against a brute-force torch computation."""
    query = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=torch.float64,
                         requires_grad=True)
    reference = torch.tensor(
        [[1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64)

    distance_squared, index, valid = gffx_torch.points.knn(query, reference, 2)
    assert distance_squared.shape == (2, 2)
    assert index.dtype == torch.int32 and valid.dtype == torch.bool
    assert valid.all()

    brute = torch.cdist(query.detach(), reference) ** 2
    expected, expected_index = brute.sort(dim=1)
    assert torch.allclose(distance_squared, expected[:, :2])
    assert torch.equal(index.long(), expected_index[:, :2])

    distance_squared.sum().backward()
    assert torch.isfinite(query.grad).all()


# ---------------------------------------------------------------------------------- TB-07

def test_tb07_closest_point_on_mesh(gffx_torch):
    """A point above a face projects onto it, and only distance_squared carries a gradient."""
    vertices, faces = tetra()
    points = torch.tensor([[0.25, 0.25, -1.0]], dtype=torch.float64, requires_grad=True)

    distance_squared, face_index, barycentric, closest, valid = \
        gffx_torch.points.closest_point_on_mesh(points, vertices, faces)
    assert bool(valid[0])
    # The query sits below the z = 0 face, so the closest point is its vertical projection.
    assert torch.allclose(closest[0], torch.tensor([0.25, 0.25, 0.0], dtype=torch.float64))
    assert abs(float(distance_squared[0]) - 1.0) < 1e-12
    assert torch.allclose(barycentric.sum(dim=1), torch.ones(1, dtype=torch.float64))

    distance_squared.sum().backward()
    # d(dist^2)/dp = 2 (p - c) = 2 * (0, 0, -1)
    assert torch.allclose(points.grad[0], torch.tensor([0.0, 0.0, -2.0], dtype=torch.float64))


# ---------------------------------------------------------------------------------- TB-08

def test_tb08_sample_surface(gffx_torch):
    """Stateless sampling: same counter reproduces, advanced counter does not."""
    vertices, faces = tetra(requires_grad=True)
    key = torch.tensor([0x12345678, 0x9ABCDEF0], dtype=torch.uint32)
    counter = torch.tensor([0, 0], dtype=torch.uint32)

    points, face_index, barycentric, next_counter = \
        gffx_torch.mesh.sample_surface(vertices, faces, 32, key, counter)
    assert points.shape == (1, 32, 3)
    assert face_index.dtype == torch.int32
    assert (face_index >= 0).all() and (face_index < 4).all()
    assert torch.allclose(barycentric.sum(dim=2), torch.ones(1, 32, dtype=torch.float64))

    again, _, _, _ = gffx_torch.mesh.sample_surface(vertices, faces, 32, key, counter)
    assert torch.equal(points, again), "the same counter must reproduce the same samples"

    advanced, _, _, _ = gffx_torch.mesh.sample_surface(vertices, faces, 32, key, next_counter)
    assert not torch.equal(points, advanced), "an advanced counter must produce new samples"

    points.sum().backward()
    assert torch.isfinite(vertices.grad).all()


# ---------------------------------------------------------------------------------- TB-09

def test_tb09_rasterize(gffx_torch):
    """A triangle covering the image centre produces fragments; blur admits outside ones."""
    ndc = torch.tensor(
        [[-0.9, -0.9, 0.5], [0.9, -0.9, 0.5], [0.0, 0.9, 0.5]],
        dtype=torch.float64, requires_grad=True)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)

    face_index, barycentric, depth, signed_distance = gffx_torch.render.rasterize(
        ndc, faces, image_height=16, image_width=16, cull_mode=gffx_torch.render.CULL_NONE
    )
    assert face_index.shape == (1, 16, 16, 1)
    assert barycentric.shape == (1, 16, 16, 1, 3)
    covered = (face_index >= 0)
    assert covered.any(), "the triangle must cover part of the image"
    assert not covered.all(), "and must not cover all of it"
    # Inside a covered fragment the signed distance is negative, in squared pixel units.
    assert (signed_distance[covered] <= 0).all()

    blurred_index, _, _, _ = gffx_torch.render.rasterize(
        ndc, faces, image_height=16, image_width=16, blur_radius_px=6.0,
        cull_mode=gffx_torch.render.CULL_NONE,
    )
    assert int((blurred_index >= 0).sum()) > int(covered.sum()), \
        "a blur radius must admit fragments the hard rasterizer rejected"

    depth.sum().backward()
    assert torch.isfinite(ndc.grad).all()


# ---------------------------------------------------------------------------------- TB-10

def test_tb10_interpolate(gffx_torch):
    """Interpolating a constant attribute reproduces it wherever a fragment is covered."""
    ndc = torch.tensor(
        [[-0.9, -0.9, 0.5], [0.9, -0.9, 0.5], [0.0, 0.9, 0.5]], dtype=torch.float64)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    face_index, barycentric, _, _ = gffx_torch.render.rasterize(
        ndc, faces, image_height=8, image_width=8, cull_mode=gffx_torch.render.CULL_NONE
    )

    # Every corner carries the same value, so any barycentric combination returns it.
    attributes_in = torch.full((1, 3, 2), 0.75, dtype=torch.float64, requires_grad=True)
    attributes = gffx_torch.render.interpolate(face_index, barycentric, attributes_in)
    assert attributes.shape == face_index.shape + (2,)
    covered = (face_index >= 0)
    assert covered.any()
    assert torch.allclose(
        attributes[covered], torch.full((int(covered.sum()), 2), 0.75, dtype=torch.float64)
    )

    attributes.sum().backward()
    assert attributes_in.grad is not None and torch.isfinite(attributes_in.grad).all()
