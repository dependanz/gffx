"""Independent optical and geometric controls for the finite glass path."""
import math

import pytest
import torch

from gffx.torch import _dielectric as optics


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64])
def precision(request):
    return request.param


def tensor(data, device, precision):
    return torch.tensor(data, dtype=precision, device=device)


def close(actual, expected, *, tight=False):
    atol = (2e-6 if actual.dtype == torch.float32 else 2e-12)
    if not tight:
        atol *= 8
    assert torch.allclose(actual, torch.as_tensor(expected, dtype=actual.dtype, device=actual.device),
                          atol=atol, rtol=atol), (actual, expected)


def oriented_planes(z_values, outward_signs, device, precision):
    vertices, faces = [], []
    for z, sign in zip(z_values, outward_signs):
        offset = len(vertices)
        vertices.extend([[-10., -10., z], [10., -10., z], [10., 10., z], [-10., 10., z]])
        local = [[offset, offset + 1, offset + 2], [offset, offset + 2, offset + 3]]
        faces.extend(local if sign > 0 else [f[::-1] for f in local])
    return (tensor(vertices, device, precision),
            torch.tensor(faces, dtype=torch.int32, device=device))


def test_equal_ior_preserves_direction_even_at_grazing(device, precision):
    directions = tensor([[0, 0, -1], [.6, 0, -.8], [1, 0, 0]], device, precision)
    normal = directions.new_tensor([0, 0, 1])
    value = optics.interface(directions, normal, 1.5, 1.5)
    close(value.transmitted, directions, tight=True)
    assert torch.equal(value.fresnel, directions.new_zeros(3))
    assert not bool(value.tir.any())
    assert bool(torch.isfinite(value.reflected).all())


def test_normal_incidence_fresnel_matches_analytic_reflectance(device, precision):
    directions = tensor([[0, 0, -1]], device, precision)
    value = optics.interface(directions, -directions, 1., 1.5)
    expected = ((1.5 - 1.) / (1.5 + 1.)) ** 2
    close(value.fresnel, [expected], tight=True)
    close(value.reflected, [[0, 0, 1]], tight=True)
    close(value.transmitted, directions, tight=True)
    assert not bool(value.tir.any())


def test_snell_direction_and_exact_unpolarized_fresnel(device, precision):
    theta = math.radians(37)
    eta_i, eta_t = 1., 1.52
    transmitted_angle = math.asin(eta_i * math.sin(theta) / eta_t)
    directions = tensor([[math.sin(theta), 0, -math.cos(theta)]], device, precision)
    value = optics.interface(directions, directions.new_tensor([0, 0, 1]), eta_i, eta_t)
    close(value.transmitted, [[math.sin(transmitted_angle), 0, -math.cos(transmitted_angle)]], tight=True)
    rs = ((eta_i * math.cos(theta) - eta_t * math.cos(transmitted_angle)) /
          (eta_i * math.cos(theta) + eta_t * math.cos(transmitted_angle))) ** 2
    rp = ((eta_t * math.cos(theta) - eta_i * math.cos(transmitted_angle)) /
          (eta_t * math.cos(theta) + eta_i * math.cos(transmitted_angle))) ** 2
    close(value.fresnel, [(rs + rp) / 2], tight=True)
    close(torch.linalg.vector_norm(value.transmitted, dim=-1), [1.], tight=True)


def test_brewster_angle_has_only_the_perpendicular_reflection_term(device, precision):
    eta_i, eta_t = 1., 1.5
    theta = math.atan(eta_t / eta_i)
    direction = tensor([[math.sin(theta), 0, -math.cos(theta)]], device, precision)
    value = optics.interface(direction, direction.new_tensor([0, 0, 1]), eta_i, eta_t)
    expected = .5 * ((eta_i ** 2 - eta_t ** 2) / (eta_i ** 2 + eta_t ** 2)) ** 2
    close(value.fresnel, [expected], tight=True)


def test_total_internal_reflection_changes_at_the_critical_angle(device, precision):
    critical = math.asin(1 / 1.5)
    angles = [critical - 1e-3, critical + 1e-3]
    directions = tensor([[math.sin(a), 0, -math.cos(a)] for a in angles], device, precision)
    value = optics.interface(directions, directions.new_tensor([0, 0, 1]), 1.5, 1.)
    assert value.tir.tolist() == [False, True]
    assert float(value.fresnel[1]) == 1.
    assert torch.equal(value.transmitted[1], directions.new_zeros(3))
    close(torch.linalg.vector_norm(value.reflected, dim=-1), [1., 1.], tight=True)
    assert bool(torch.isfinite(torch.cat((value.reflected.flatten(), value.transmitted.flatten(),
                                         value.fresnel))).all())


def test_plane_parallel_slab_exit_and_lateral_displacement(device, precision):
    thickness = .6
    theta_i = math.radians(40)
    theta_t = math.asin(math.sin(theta_i) / 1.5)
    vertices, faces = oriented_planes([0., -thickness], [1, -1], device, precision)
    query = optics.TriangleQuery(vertices, faces, ray_chunk=1, face_chunk=1)
    direction = vertices.new_tensor([[math.sin(theta_i), 0, -math.cos(theta_i)]])
    origin = vertices.new_tensor([[-math.tan(theta_i), 0, 1.]])
    front = query.intersect(origin, direction)
    assert bool(front.hit.all())
    entry = origin + front.distance[:, None] * direction
    close(entry, [[0., 0., 0.]])
    first = optics.interface(direction, front.normal, 1., 1.5)
    back = query.intersect(entry, first.transmitted, t_min=1e-5)
    assert bool(back.hit.all())
    exit_point = entry + back.distance[:, None] * first.transmitted
    second = optics.interface(first.transmitted, -back.normal, 1.5, 1.)
    close(second.transmitted, direction, tight=True)
    close(back.distance, [thickness / math.cos(theta_t)])
    close(exit_point, [[thickness * math.tan(theta_t), 0., -thickness]])
    unbent_exit = vertices.new_tensor([[thickness * math.tan(theta_i), 0., -thickness]])
    perpendicular = vertices.new_tensor([[math.cos(theta_i), 0., math.sin(theta_i)]])
    lateral = ((unbent_exit - exit_point) * perpendicular).sum(dim=-1)
    expected_lateral = thickness * math.sin(theta_i - theta_t) / math.cos(theta_t)
    close(lateral, [expected_lateral])


def test_four_hollow_wall_crossings_absorb_only_the_glass_lengths(device, precision):
    wall = .2
    absorption = tensor([.4, .7, 1.1], device, precision)
    attenuations = []
    for gap in (.5, 3.):
        vertices, faces = oriented_planes([0., -wall, -wall - gap, -2 * wall - gap],
                                         [1, -1, 1, -1], device, precision)
        query = optics.TriangleQuery(vertices, faces, ray_chunk=1, face_chunk=2)
        origin = vertices.new_tensor([[.13, .27, 1.]])
        direction = vertices.new_tensor([[0, 0, -1.]])
        current_ior = 1.
        attenuation = absorption.new_ones((1, 3))
        glass_lengths = []
        crossed_normals = []
        for crossing in range(4):
            hit = query.intersect(origin, direction, t_min=1e-5)
            assert bool(hit.hit.all())
            crossed_normals.append(float(hit.normal[0, 2]))
            if current_ior > 1:
                attenuation *= optics.beer(absorption, hit.distance)
                glass_lengths.append(float(hit.distance[0]))
            entering = crossing % 2 == 0
            next_ior = 1.5 if entering else 1.
            opposed = hit.normal if entering else -hit.normal
            interface = optics.interface(direction, opposed, current_ior, next_ior)
            assert not bool(interface.tir.any())
            origin = origin + hit.distance[:, None] * direction
            direction = interface.transmitted
            current_ior = next_ior
        assert crossed_normals == [1., -1., 1., -1.]
        assert current_ior == 1.
        close(absorption.new_tensor(glass_lengths), [wall, wall])
        close(attenuation, [[math.exp(-.4 * 2 * wall), math.exp(-.7 * 2 * wall),
                             math.exp(-1.1 * 2 * wall)]])
        attenuations.append(attenuation)
        assert not bool(query.intersect(origin, direction, t_min=1e-5).hit.any())
    close(attenuations[0], attenuations[1])


def test_triangle_barycentric_front_back_parallel_and_miss(device, precision):
    vertices = tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], device, precision)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    query = optics.TriangleQuery(vertices, faces, ray_chunk=2, face_chunk=1)
    origins = vertices.new_tensor([[.25, .25, 1], [.25, .25, -1], [2, 2, 1], [.25, .25, 1]])
    directions = vertices.new_tensor([[0, 0, -1], [0, 0, 1], [0, 0, -1], [1, 0, 0]])
    hit = query.intersect(origins, directions)
    assert hit.face_index.tolist() == [0, 0, -1, -1]
    close(hit.distance[:2], [1, 1], tight=True)
    close(hit.barycentric[:2], [[.5, .25, .25], [.5, .25, .25]], tight=True)
    close(hit.normal[:2], [[0, 0, 1], [0, 0, 1]], tight=True)
    assert bool(torch.isinf(hit.distance[2:]).all())
    assert torch.equal(hit.barycentric[2:], vertices.new_zeros((2, 3)))


def test_nearest_ties_bounds_and_exclusion_are_chunk_independent(device, precision):
    vertices = tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                       [0, 0, -2], [1, 0, -2], [0, 1, -2]], device, precision)
    faces = torch.tensor([[0, 1, 2], [0, 2, 1], [3, 4, 5]], dtype=torch.int32, device=device)
    origin = vertices.new_tensor([[.2, .3, 1.]]).expand(4, 3)
    direction = vertices.new_tensor([[0, 0, -1.]]).expand_as(origin)
    for face_chunk in (1, 2, 8):
        query = optics.TriangleQuery(vertices, faces, ray_chunk=2, face_chunk=face_chunk)
        hit = query.intersect(origin, direction,
                              t_min=vertices.new_tensor([0., 1., 0., 0.]),
                              t_max=vertices.new_tensor([10., 3., .9, 10.]),
                              ignore_faces=torch.tensor([-1, -1, -1, 0], dtype=torch.int64, device=device))
        assert hit.face_index.tolist() == [0, 2, -1, 1]
        close(hit.distance[[0, 1, 3]], [1., 3., 1.], tight=True)


def test_shared_quad_diagonal_and_geometry_scale_do_not_lose_hits(device, precision):
    unit_vertices = tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], device, precision)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device=device)
    for scale in (1e-12, 1., 1e12):
        vertices = unit_vertices * scale
        query = optics.TriangleQuery(vertices, faces, ray_chunk=2, face_chunk=1)
        origins = unit_vertices.new_tensor([[.1, .1, 1.], [.5, .5, 1.], [.9, .9, 1.],
                                            [1.00001, 1.00001, 1.]]) * scale
        directions = unit_vertices.new_tensor([[0, 0, -1.]]).expand_as(origins)
        hit = query.intersect(origins, directions)
        assert hit.hit.tolist() == [True, True, True, False]
        close(hit.distance[:3] / scale, [1., 1., 1.], tight=True)
        close(hit.barycentric[:3].sum(dim=-1), [1., 1., 1.], tight=True)
        close(hit.normal[:3], [[0, 0, 1]] * 3, tight=True)


def test_empty_geometry_and_degenerate_face_return_explicit_misses(device, precision):
    vertices = tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], device, precision)
    origin = vertices.new_tensor([[0, 0, 1.]])
    direction = vertices.new_tensor([[0, 0, -1.]])
    for faces in (torch.empty((0, 3), dtype=torch.int32, device=device),
                  torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)):
        hit = optics.TriangleQuery(vertices, faces).intersect(origin, direction)
        assert hit.face_index.tolist() == [-1]
        assert bool(torch.isinf(hit.distance).all())
        assert torch.equal(hit.normal, vertices.new_zeros((1, 3)))


def test_geometry_mutation_requires_rebuilding_the_query(device, precision):
    vertices = tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], device, precision)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    query = optics.TriangleQuery(vertices, faces)
    vertices[0, 2] += 1
    with pytest.raises(ValueError, match="changed"):
        query.intersect(vertices.new_tensor([[0, 0, 2.]]), vertices.new_tensor([[0, 0, -1.]]))


@pytest.mark.parametrize("ior,absorption", [(.9, [0, 0, 0]), (4.1, [0, 0, 0]),
                                              (float("nan"), [0, 0, 0]),
                                              (1.5, [0, -.1, 0]), (1.5, .2),
                                              (1.5, [0, float("inf"), 0])])
def test_invalid_optical_parameters_are_rejected(ior, absorption):
    with pytest.raises((ValueError, TypeError)):
        optics.validate_parameters({"ior": torch.tensor(ior), "absorption": torch.tensor(absorption, dtype=torch.float32)})


def test_interface_and_distance_validation():
    direction = torch.tensor([[0., 0., -1.]])
    with pytest.raises(ValueError, match="oppose"):
        optics.interface(direction, direction, 1., 1.5)
    with pytest.raises(ValueError, match="nonzero"):
        optics.interface(torch.zeros_like(direction), -direction, 1., 1.5)
    with pytest.raises(ValueError, match="positive"):
        optics.interface(direction, -direction, 0., 1.5)
    # Very large but representable ratios at normal incidence are finite;
    # an unrepresentable ratio is rejected before it can create a NaN path.
    extreme = optics.interface(direction, -direction, 1e15, 1e-15)
    assert bool(torch.isfinite(extreme.transmitted).all())
    close(extreme.transmitted, direction, tight=True)
    with pytest.raises(ValueError, match="ratio exceeds"):
        optics.interface(direction, -direction, 1e30, 1e-30)
    with pytest.raises(ValueError, match="nonnegative"):
        optics.beer(torch.ones(3), -1.)
    with pytest.raises(ValueError, match="exactly"):
        optics.validate_parameters({"ior": torch.tensor(1.5), "absorption": torch.zeros(3), "opacity": torch.tensor(.5)})
    with pytest.raises(NotImplementedError, match="transmission renderer"):
        optics.dielectric()


def test_nonunit_rays_and_invalid_indices_are_rejected():
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    query = optics.TriangleQuery(vertices, faces)
    with pytest.raises(ValueError, match="unit length"):
        query.intersect(torch.tensor([[0., 0., 1.]]), torch.tensor([[0., 0., -2.]]))
    with pytest.raises(ValueError, match="out of bounds"):
        optics.TriangleQuery(vertices, torch.tensor([[0, 1, 3]], dtype=torch.int32))
    with pytest.raises(ValueError, match="ray bounds"):
        query.intersect(torch.tensor([[0., 0., 1.]]), torch.tensor([[0., 0., -1.]]), t_min=2., t_max=1.)
