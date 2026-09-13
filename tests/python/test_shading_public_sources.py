"""Native source -> public material -> public lighting red integration fixtures."""
import importlib
import math

import pytest
import torch

from test_material_public_contract import envelope


def t(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


def interfaces():
    # Resolve public modules first: absent API must not look like a native sampler error.
    material = importlib.import_module('gffx.torch.material')
    light = importlib.import_module('gffx.torch.light')
    render = importlib.import_module('gffx.torch.render')
    return material, light, render


def sample(render, texture, uv):
    pyramid, offsets = render.texture_pyramid(texture.contiguous(), levels=1)
    return render.texture(pyramid, offsets, 2, 2, uv.contiguous(), channel_count=3,
                          filter=render.FILTER_BILINEAR, mip_filter=render.MIP_NEAREST,
                          wrap_u=render.WRAP_REPEAT, wrap_v=render.WRAP_REPEAT)


def illuminate(material, light, colour, model):
    p = colour.shape[0]
    n = t([0., 0., 1.]).expand(p, 3)
    if model == 'classic':
        brdf = material.classic(n, n, t([[0., 0., 1.]]), albedo=colour, specular=t(0.), exponent=t(2.))
    else:
        resource = material.PrincipledBudget.from_envelope(envelope(), device='cpu', dtype=torch.float64, allow_unverified=True)
        brdf = material.principled(n, n, t([[0., 0., 1.]]), albedo=colour, roughness=t(1.), f0=t(.04),
                                  budget=resource, allow_experimental_budget=True)
    packet = dict(measure='delta', direction=n[:, None, :],
                  irradiance=torch.ones((p, 1, 3), dtype=torch.float64),
                  valid=torch.ones((p, 1), dtype=torch.bool))
    return light.integrate(n, brdf, packet)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try: yield
    finally: torch.set_num_threads(previous)


@pytest.mark.parametrize('model', ['classic', 'principled'])
def test_constant_corner_and_texture_sources_match_complete_lit_output(model):
    """Equivalent vertex/corner and texture colours must produce the same final light."""
    material, light, render = interfaces()
    colour = t([.2, .4, .6])
    attributes = colour.expand(1, 3, 3).clone().requires_grad_()
    texture = colour.expand(2, 2, 3).clone().requires_grad_()
    face_index = torch.zeros((1, 1, 2, 1), dtype=torch.int32)
    bary = t([[[[[.2, .3, .5]], [[.4, .2, .4]]]]], True)
    uv = t([[.35, .45], [.45, .55]], True)
    corner = render.interpolate(face_index, bary, attributes).reshape(2, 3)
    textured = sample(render, texture, uv)
    a = illuminate(material, light, corner, model)
    b = illuminate(material, light, textured, model)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-11)
    if model == 'classic': expected = colour / math.pi
    else:
        f0 = .04
        loss = .7 - .3 * f0
        mean = .9 * (1 - f0) / 3 + 2 * loss / 3
        expected = f0 / (4 * math.pi) + colour * loss**2 / (math.pi * mean)
    torch.testing.assert_close(a, expected.expand(2, 3), atol=1e-12, rtol=1e-11)
    ga = torch.autograd.grad(a.sum(), (attributes, bary))
    gb = torch.autograd.grad(b.sum(), (texture, uv))
    assert all(torch.isfinite(g).all() for g in (*ga, *gb))
    assert ga[0].abs().sum() > 0 and gb[0].abs().sum() > 0


@pytest.mark.parametrize('model', ['classic', 'principled'])
def test_nonconstant_texture_to_lit_output_gradcheck(model):
    """Fitting reaches source texels and UV coordinates through the whole shading chain."""
    material, light, render = interfaces()
    texture = t([[[.2, .3, .4], [.3, .4, .5]], [[.4, .5, .6], [.5, .6, .7]]], True)
    uv = t([[.35, .45]], True)
    def evaluate(tex, coords):
        return illuminate(material, light, sample(render, tex, coords), model)
    assert torch.autograd.gradcheck(evaluate, (texture, uv), eps=1e-6, atol=1e-5, rtol=1e-3)
    gradient, = torch.autograd.grad(evaluate(texture, uv).sum(), (uv,))
    assert (gradient.abs() > 1e-5).all()


def test_face_corner_uv_seams_survive_public_shading():
    """Separate face-corner UVs must not be merged into a shared vertex coordinate."""
    material, light, render = interfaces()
    texture = t([[[.1, .2, .3], [.7, .6, .5]], [[.1, .2, .3], [.7, .6, .5]]])
    corner_uv = t([[[.3, .5]] * 3, [[.7, .5]] * 3], True)
    faces = torch.tensor([[[[0], [1]]]], dtype=torch.int32)
    bary = t([[[[[.2, .3, .5]], [[.4, .2, .4]]]]])
    uv = render.interpolate(faces, bary, corner_uv).reshape(2, 2)
    result = illuminate(material, light, sample(render, texture, uv), 'classic')
    assert not torch.allclose(result[0], result[1])
    gradient, = torch.autograd.grad(result.sum(), (corner_uv,))
    assert torch.isfinite(gradient).all() and (gradient[..., 0].abs().sum(-1) > 0).all()


def test_learned_texture_rebuilds_graph_for_each_optimization_step():
    """Two fitting steps must use updated texels rather than a stale cached pyramid/graph."""
    material, light, render = interfaces()
    texture = t([[[.2, .3, .4], [.3, .4, .5]], [[.4, .5, .6], [.5, .6, .7]]], True)
    uv = t([[.35, .45]])
    values = []
    for _ in range(2):
        value = illuminate(material, light, sample(render, texture, uv), 'classic').sum()
        gradient, = torch.autograd.grad(value, (texture,))
        values.append(value.detach().item())
        with torch.no_grad(): texture.sub_(.01 * gradient)
    assert values[1] < values[0]


def test_coverage_selection_keeps_background_out_of_material_inputs():
    """Uncovered pixels are composited separately rather than treated as a zero material."""
    material, light, render = interfaces()
    faces = torch.tensor([[[[0], [-1]]]], dtype=torch.int32)
    bary = t([[[[[.2, .3, .5]], [[0., 0., 0.]]]]])
    attributes = t([[[.2, .4, .6]] * 3])
    colour = render.interpolate(faces, bary, attributes).reshape(2, 3)
    covered = faces.reshape(-1) >= 0
    output = t([.01, .02, .03]).expand(2, 3).clone()
    output[covered] = illuminate(material, light, colour[covered], 'classic')
    torch.testing.assert_close(output[0], t([.2, .4, .6]) / math.pi, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(output[1], t([.01, .02, .03]), atol=1e-12, rtol=1e-12)
