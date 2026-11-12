"""Adversarial admission controls for experimental short-flight recovery."""

import math

import pytest
import torch

from gffx.torch._dielectric import TriangleQuery
from gffx.torch._dielectric_shortflight import EdgeShortFlight


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def fixture(device, *, disconnected=False, reverse=False,
            second_glass=True):
    # Face 0 is a horizontal surface with outward normal -Z.
    # Face 1 shares its edge and has outward normal -X.
    # A ray travelling -X,+Z can enter at face 0 and exit at face 1.
    vertices = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ]

    if disconnected:
        vertices.extend([
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
        ])
        second = [4, 5, 6]
    else:
        second = [0, 3, 2]

    if reverse:
        second = second[::-1]

    vertices = torch.tensor(
        vertices, dtype=torch.float32, device=device
    )
    faces = torch.tensor(
        [[0, 2, 1], second],
        dtype=torch.int32, device=device,
    )

    query = TriangleQuery(
        vertices, faces, ray_chunk=1, face_chunk=2
    )

    owners = torch.zeros(
        2, dtype=torch.long, device=device
    )
    glass = torch.tensor(
        [True, second_glass],
        dtype=torch.bool, device=device
    )

    recovery = EdgeShortFlight(
        vertices, faces, owners, glass, query.normal
    )

    return query, recovery


def ray(device, *, x=5e-7, parallel=False):
    origin = torch.tensor(
        [[x, .25, 0.]],
        dtype=torch.float32, device=device
    )

    direction = (
        [0., 0., 1.] if parallel
        else [-1 / math.sqrt(2), 0., 1 / math.sqrt(2)]
    )

    direction = torch.tensor(
        [direction],
        dtype=torch.float32, device=device
    )

    return origin, direction


LOWER = 1.9073486328125e-6


def evaluate(query, recovery, origin, direction, *,
             medium=0, previous=0):
    previous = torch.tensor(
        [previous], dtype=torch.long,
        device=origin.device,
    )
    medium = torch.tensor(
        [medium], dtype=torch.long,
        device=origin.device,
    )

    original = query.intersect(
        origin, direction,
        t_min=LOWER,
        ignore_faces=previous,
    )

    candidate = recovery.recover(
        original,
        origin,
        direction,
        LOWER,
        float("inf"),
        previous,
        medium,
    )

    return original, candidate


def assert_hit_unchanged(before, after):
    for field in (
        "face_index",
        "distance",
        "barycentric",
        "normal",
        "hit",
    ):
        assert torch.equal(
            getattr(before, field),
            getattr(after, field),
        ), f"Unexpected mutation of {field}"


def test_valid_adjacent_short_exit_is_recovered(device):
    query, recovery = fixture(device)
    origin, direction = ray(device)

    before, after = evaluate(
        query, recovery, origin, direction
    )

    assert not bool(before.hit[0])
    assert bool(after.hit[0])
    assert int(after.face_index[0]) == 1

    distance = float(after.distance[0])
    assert 0 < distance < LOWER
    assert abs(distance - math.sqrt(2) * 5e-7) < 1e-12

    normal = after.normal[0]
    assert float((direction[0] * normal).sum()) > 0
    torch.testing.assert_close(
        after.barycentric.sum(-1),
        torch.ones(1, device=device),
        atol=1e-6,
        rtol=0,
    )


@pytest.mark.parametrize("case", [
    "air_cannot_exit",
    "wrong_medium_owner",
    "reversed_winding",
    "zero_distance",
    "outside_short_window",
    "disconnected_topology",
    "no_previous_face",
    "parallel_ray",
    "neighbor_not_glass",
])
def test_invalid_short_flight_is_not_recovered(device, case):
    query, recovery = fixture(
        device,
        disconnected=case == "disconnected_topology",
        reverse=case == "reversed_winding",
        second_glass=case != "neighbor_not_glass",
    )

    x = (
        0. if case == "zero_distance"
        else 5e-5 if case == "outside_short_window"
        else 5e-7
    )

    origin, direction = ray(
        device,
        x=x,
        parallel=case == "parallel_ray",
    )

    medium = (
        -1 if case == "air_cannot_exit"
        else 1 if case == "wrong_medium_owner"
        else 0
    )

    previous = -1 if case == "no_previous_face" else 0

    before, after = evaluate(
        query, recovery, origin, direction,
        medium=medium, previous=previous,
    )

    assert_hit_unchanged(before, after)


def test_regular_distant_hit_is_preserved(device):
    query, recovery = fixture(device)
    origin, direction = ray(device, x=2e-5)

    before, after = evaluate(
        query, recovery, origin, direction
    )

    assert bool(before.hit[0])
    assert int(before.face_index[0]) == 1
    assert float(before.distance[0]) > LOWER

    assert_hit_unchanged(before, after)
