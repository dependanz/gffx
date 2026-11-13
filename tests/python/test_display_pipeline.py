"""Scene-linear to display RGB contracts, shared between preview and PNG export.

Inspect the real PNG bytes with only Python's standard library; Pillow and NumPy
are not needed for these acceptance tests or for the exporter itself.
"""
from __future__ import annotations

import math
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import pytest
import torch
import gffx as g


def _read_png_rgb8(path: Path) -> torch.Tensor:
    payload = path.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    cursor = 8
    pixels = bytearray()
    width = height = None
    srgb = False
    ended = False
    while cursor < len(payload):
        size = struct.unpack(">I", payload[cursor:cursor + 4])[0]
        name = payload[cursor + 4:cursor + 8]
        data = payload[cursor + 8:cursor + 8 + size]
        crc = struct.unpack(">I", payload[cursor + 8 + size:cursor + 12 + size])[0]
        assert (zlib.crc32(name + data) & 0xffffffff) == crc
        cursor += 12 + size
        if name == b"IHDR":
            width, height, depth, kind, comp, filt, interlace = struct.unpack(">IIBBBBB", data)
            assert (depth, kind, comp, filt, interlace) == (8, 2, 0, 0, 0)
        elif name == b"sRGB":
            srgb = data == b"\0"
        elif name == b"IDAT":
            pixels.extend(data)
        elif name == b"IEND":
            ended = True
            break
    assert ended and srgb and width is not None and height is not None
    raw = zlib.decompress(pixels)
    stride = 3 * width
    assert len(raw) == height * (stride + 1)
    assert all(raw[row * (stride + 1)] == 0 for row in range(height))
    data = b"".join(raw[row * (stride + 1) + 1:(row + 1) * (stride + 1)] for row in range(height))
    return torch.tensor(list(data), dtype=torch.uint8).reshape(height, width, 3)


@pytest.mark.parametrize("tone_map", ["aces", "clip"])
@pytest.mark.parametrize("exposure", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched", [False, True])
def test_png_byte_parity_with_display_tensor(tmp_path, tone_map, exposure, dtype, batched):
    """Preview differs from PNG only by 8-bit quantization and PNG metadata."""
    base = torch.tensor([
        [[0.0, .0031308, .018], [.18, .5, 1.0], [2.0, 4.0, 10.0]],
        [[.04, .10, .9], [.99, 1.1, 1.2], [20.0, 200.0, 1000.0]],
    ], dtype=dtype)
    image = base.unsqueeze(0) if batched else base
    unchanged = image.clone()
    display = g.image.to_display(image, tone_map=tone_map, exposure=exposure)
    assert display.shape == image.shape
    assert display.dtype == image.dtype and display.device == image.device
    assert display.is_contiguous()
    assert bool(torch.isfinite(display).all())
    assert bool(((display >= 0) & (display <= 1)).all())
    assert torch.equal(image, unchanged)

    output = tmp_path / "preview.png"
    g.io.write_png(output, image, tone_map=tone_map, exposure=exposure)
    expected = (display[0] if batched else display).mul(255).round().to(torch.uint8)
    assert torch.equal(_read_png_rgb8(output), expected)


def _display_scalar(x: float, *, tone_map="aces", exposure=.5) -> float:
    x *= 2 ** exposure
    if tone_map == "aces":
        v = min(x, 1e6)
        x = max(0.0, min(1.0, v * (2.51 * v + .03) / (v * (2.43 * v + .59) + .14)))
    else:
        x = max(0.0, min(1.0, x))
    return 12.92 * x if x <= .0031308 else 1.055 * x ** (1 / 2.4) - .055


@pytest.mark.parametrize("tone_map", ["aces", "clip"])
def test_value_and_first_derivatives_match_independent_scalar_reference(tone_map):
    """The preview is an actual differentiable PyTorch operation, not an I/O convenience."""
    rgb = torch.tensor([[[.02, .12, .28], [.41, .63, .73]]], dtype=torch.float64,
                       requires_grad=True)
    weights = torch.tensor([[[1., .7, 2.], [.3, 1.2, 1.1]]], dtype=rgb.dtype)
    result = g.image.to_display(rgb, tone_map=tone_map, exposure=.5)
    reference = torch.tensor([[_display_scalar(v, tone_map=tone_map)
                               for v in row] for row in rgb.detach().reshape(-1, 3).tolist()],
                             dtype=rgb.dtype).reshape_as(rgb)
    torch.testing.assert_close(result, reference, rtol=3e-12, atol=3e-12)

    actual, = torch.autograd.grad(result, rgb, grad_outputs=weights)
    delta = 1e-6
    manual = torch.tensor([[
        (_display_scalar(v + delta, tone_map=tone_map) -
         _display_scalar(v - delta, tone_map=tone_map)) / (2 * delta)
        for v in row
    ] for row in rgb.detach().reshape(-1, 3).tolist()], dtype=rgb.dtype).reshape_as(rgb)
    torch.testing.assert_close(actual, manual * weights, rtol=3e-5, atol=3e-6)


def test_png_export_keeps_live_original_graph(tmp_path):
    """Export never detaches or mutates the user's original differentiable image."""
    image = torch.tensor([[[.05, .18, 2.0]]], dtype=torch.float64, requires_grad=True)
    preview = g.image.to_display(image)
    assert preview.requires_grad and preview.grad_fn is not None
    g.io.write_png(tmp_path / "display.png", image)
    assert image.grad is None and image.requires_grad
    preview.sum().backward()
    assert image.grad is not None and bool(torch.isfinite(image.grad).all())


@pytest.mark.parametrize("invalid, error", [
    (lambda: [[.1, .2, .3]], TypeError),
    (lambda: torch.ones((2, 2, 4)), ValueError),
    (lambda: torch.ones((2, 2)), ValueError),
    (lambda: torch.tensor([[[-1., .2, .3]]]), ValueError),
    (lambda: torch.tensor([[[float("nan"), .2, .3]]]), ValueError),
    (lambda: torch.ones((1, 1, 3), dtype=torch.int32), TypeError),
])
def test_invalid_display_input_is_refused(invalid, error):
    with pytest.raises(error):
        g.image.to_display(invalid())


@pytest.mark.parametrize("kwargs", [
    {"tone_map": "raw"},
    {"exposure": True},
    {"exposure": float("inf")},
    {"exposure": float("nan")},
])
def test_invalid_display_options_are_refused(kwargs):
    with pytest.raises(ValueError):
        g.image.to_display(torch.ones((1, 1, 3)), **kwargs)


def test_display_module_import_does_not_eagerly_load_torch():
    script = (
        "import sys, gffx, gffx.image, gffx.io; "
        "assert callable(gffx.image.to_display); "
        "assert 'torch' not in sys.modules; "
        "assert 'numpy' not in sys.modules; "
        "assert 'PIL' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
