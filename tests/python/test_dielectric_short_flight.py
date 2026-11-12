"""Regression for the D01 tumbler-rim submicrometre optical flight."""

from collections import defaultdict

import pytest
import torch

from gffx.examples.studio_d import build_scene
from gffx.torch._area import expand_emitters
from gffx.torch._deferred import _prepare
from gffx.torch._dielectric_render import (
    _Transport,
    _primary_ray_batch,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Captured D01 failure uses CUDA float32 ray generation",
)
def test_d01_rim_short_flight_supersample_block():
    """A valid TIR event must not be skipped by the ray minimum distance."""

    with torch.no_grad():
        scene, camera = build_scene(
            device="cuda",
            size=(192, 192),
        )

        packed, camera = _prepare(
            expand_emitters(scene),
            camera,
        )

        transport = _Transport(
            packed,
            camera,
            sources=[],
            background=packed.vertices.new_tensor(
                (.035, .045, .065)
            ),
            environment=None,
            shadows=False,
            query_chunk=2048,
            diagnostics=defaultdict(int),
        )

        # Original failing case:
        # sample=2, block=11, pixel=(115,127).
        start = 11 * 2048
        stop = start + 2048

        origins, directions, near, far = _primary_ray_batch(
            camera,
            start,
            stop,
            dx=.25,
            dy=-.25,
            generation_chunk=256,
        )

        colour, face, distance, lost = transport.trace(
            origins, directions, near, far
        )

        assert colour.shape == (2048, 3)
        assert torch.isfinite(colour).all()
        assert torch.isfinite(lost).all()
        assert transport.stats["camera_glass_interfaces"] > 0
