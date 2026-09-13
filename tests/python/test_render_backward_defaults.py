"""Absent raster cotangents follow the input, never the ambient default device."""
from types import SimpleNamespace

import pytest
import torch

from gffx.torch.render import _Rasterize


@pytest.mark.parametrize("present", [0, 1, 2])
def test_raster_backward_missing_cotangent_device(monkeypatch, present):
    """A meta default simulates an unrelated device without initializing a GPU."""
    vertices = torch.zeros((3, 3), dtype=torch.float64, device="cpu")
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cpu")
    indices = torch.zeros((1, 2, 2, 1), dtype=torch.int32, device="cpu")
    shapes = [indices.shape + (3,), indices.shape, indices.shape]
    grads = [None, None, None]
    grads[present] = torch.ones(shapes[present], dtype=vertices.dtype, device="cpu")
    captured = []

    def backward(*args):
        captured.extend(args[-3:])
        return vertices.clone()

    monkeypatch.setattr(torch.ops.gffx, "rasterize_backward", backward)
    ctx = SimpleNamespace(saved_tensors=(vertices, faces, indices),
                          image_height=2, image_width=2)
    with torch.device("meta"):
        result = _Rasterize.backward(ctx, None, *grads)
    assert len(result) == 10
    for index, gradient in enumerate(captured):
        assert gradient.device == vertices.device
        assert gradient.dtype == vertices.dtype
        assert gradient.shape == shapes[index]
        assert torch.equal(gradient, torch.ones_like(gradient) if index == present
                           else torch.zeros_like(gradient))


def test_raster_backward_all_missing_does_not_call_native(monkeypatch):
    def forbidden(*args):
        raise AssertionError("no cotangents must not dispatch")
    monkeypatch.setattr(torch.ops.gffx, "rasterize_backward", forbidden)
    ctx = SimpleNamespace(saved_tensors=(None, None, None))
    assert _Rasterize.backward(ctx, None, None, None, None) == (None,) * 10
