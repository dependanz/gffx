"""Phase-0 light recipes, not public gffx implementation acceptance."""
import importlib
import math

import pytest

torch = pytest.importorskip('torch')


def reference():
    return importlib.import_module('tools.light_contract_reference')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_sh_constant_environment_returns_pi_in_every_direction(dtype):
    normals = torch.tensor([[1., 0., 0.], [0., 2., 0.], [0., 0., -3.]], dtype=dtype)
    coefficients = torch.zeros(9, 3, dtype=dtype)
    coefficients[0] = math.sqrt(4*math.pi)
    result = reference().spherical_harmonics(normals, coefficients)
    torch.testing.assert_close(result, torch.full_like(normals, math.pi))
    assert result.dtype == dtype


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_sh_directional_projection_matches_truncated_kernel_not_exact_cosine(axis):
    direction = torch.eye(3, dtype=torch.float64)[axis]
    normals = torch.cat((torch.eye(3), -torch.eye(3))).double()
    coefficients = reference().basis(direction)[:, None].expand(9, 3)
    result = reference().spherical_harmonics(normals, coefficients)
    cosine = normals @ direction
    # Addition theorem: sum_m Y_lm(n)Y_lm(l)=(2l+1)/(4*pi)P_l(n.l).
    expected = .25 + .5*cosine + (5/32)*(3*cosine.square()-1)
    torch.testing.assert_close(result, expected[:, None].expand_as(result), atol=1e-12, rtol=1e-12)
    assert float(result[axis, 0]) == pytest.approx(17/16)
    assert float(result[axis+3, 0]) == pytest.approx(1/16)
    # Exact clamped-cosine light would instead give one at the pole and zero behind it.


def test_sh_basis_order_and_signed_output_are_preserved():
    basis = reference().basis(torch.eye(3, dtype=torch.float64))
    assert basis[1, 1] == pytest.approx(math.sqrt(3/(4*math.pi)))
    assert basis[2, 2] == pytest.approx(math.sqrt(3/(4*math.pi)))
    assert basis[0, 3] == pytest.approx(math.sqrt(3/(4*math.pi)))
    coefficients = torch.zeros(9, 3, dtype=torch.float64)
    coefficients[0] = -math.sqrt(4*math.pi)
    torch.testing.assert_close(reference().spherical_harmonics(torch.eye(3).double(), coefficients),
                               torch.full((3, 3), -math.pi, dtype=torch.float64))


def test_sh_batches_do_not_mix_and_shared_inputs_accumulate_gradients():
    normals = torch.tensor([[[0., 0., 1.]], [[1., 0., 0.]]], dtype=torch.float64)
    coefficients = torch.zeros(2, 9, 3, dtype=torch.float64)
    coefficients[:, 0] = torch.tensor([[1., 2., 3.], [4., 5., 6.]]) * math.sqrt(4*math.pi)
    result = reference().spherical_harmonics(normals, coefficients)
    torch.testing.assert_close(result[:, 0], coefficients[:, 0]*math.sqrt(math.pi)/2)
    shared = coefficients[:1].clone().requires_grad_()
    reference().spherical_harmonics(normals, shared).sum().backward()
    assert shared.grad[0, 0, 0] == pytest.approx(math.sqrt(math.pi))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_point_inverse_square_color_direction_and_translation(dtype):
    points = torch.tensor([[0., 0., 0.], [0., 0., -1.]], dtype=dtype)
    positions = torch.tensor([[0., 0., 1.]], dtype=dtype)
    intensity = torch.tensor([[4., 8., 12.]], dtype=dtype)
    sample = reference().point(points, positions, intensity)
    assert sample['direction'].shape == (2, 1, 3)
    torch.testing.assert_close(sample['irradiance'][:, 0], torch.tensor([[4., 8., 12.], [1., 2., 3.]], dtype=dtype))
    torch.testing.assert_close(sample['direction'], torch.tensor([[[0., 0., 1.]], [[0., 0., 1.]]], dtype=dtype))
    torch.testing.assert_close(sample['distance_squared'], torch.tensor([[1.], [4.]], dtype=dtype))
    translated = reference().point(points+7, positions+7, intensity)
    assert all(torch.equal(sample[key], translated[key]) for key in sample)


def test_point_coincidence_masks_only_that_pair_with_finite_zero_gradients():
    points = torch.tensor([[0., 0., 0.]], dtype=torch.float64, requires_grad=True)
    positions = torch.tensor([[0., 0., 0.], [0., 0., 2.]], dtype=torch.float64, requires_grad=True)
    intensity = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    sample = reference().point(points, positions, intensity)
    assert sample['valid'].tolist() == [[False, True]]
    assert torch.equal(sample['direction'][0, 0], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(sample['irradiance'][0, 0], torch.zeros(3, dtype=torch.float64))
    loss = sample['direction'][0, 0].sum()+sample['irradiance'][0, 0].sum()+sample['distance_squared'][0, 0]
    gradients = torch.autograd.grad(loss, (points, positions, intensity))
    assert all(torch.equal(g, torch.zeros_like(g)) for g in gradients)


def test_point_shared_and_batched_lights_keep_explicit_axes():
    points = torch.zeros(2, 4, 3, dtype=torch.float64)
    positions = torch.tensor([[[0., 0., 1.]], [[0., 0., 2.]]], dtype=torch.float64)
    result = reference().point(points, positions, torch.ones(1, 3, dtype=torch.float64))
    assert result['irradiance'].shape == (2, 4, 1, 3)
    assert result['valid'].shape == (2, 4, 1)
    assert torch.equal(result['irradiance'][0], torch.ones(4, 1, 3).double())
    assert torch.equal(result['irradiance'][1], torch.full((4, 1, 3), .25).double())


def test_directional_normalizes_explicitly_without_distance_or_cosine():
    directions = torch.tensor([[0., 0., 7.], [2., 0., 0.]], dtype=torch.float64)
    irradiance = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64)
    sample = reference().directional(directions, irradiance)
    assert torch.equal(sample['direction'], torch.tensor([[0., 0., 1.], [1., 0., 0.]], dtype=torch.float64))
    assert torch.equal(sample['irradiance'], irradiance)
    assert 'pdf' not in sample  # A delta light has no ordinary solid-angle density.


@pytest.mark.parametrize('operation', ['sh', 'point', 'directional'])
def test_light_interior_gradients_match_finite_differences(operation):
    ref = reference()
    direction = torch.tensor([[.3, .5, .8]], dtype=torch.float64, requires_grad=True)
    if operation == 'sh':
        coeff = torch.arange(27, dtype=torch.float64).reshape(9, 3).div(27).requires_grad_()
        assert torch.autograd.gradcheck(ref.spherical_harmonics, (direction, coeff))
    elif operation == 'point':
        position = torch.tensor([[1., 2., 3.]], dtype=torch.float64, requires_grad=True)
        intensity = torch.tensor([[.4, .7, .9]], dtype=torch.float64, requires_grad=True)
        def fn(*args):
            sample = ref.point(*args)
            return sample['direction'], sample['irradiance'], sample['distance_squared']
        assert torch.autograd.gradcheck(fn, (direction, position, intensity))
    else:
        intensity = torch.ones(1, 3, dtype=torch.float64, requires_grad=True)
        def fn(*args):
            sample = ref.directional(*args)
            return sample['direction'], sample['irradiance']
        assert torch.autograd.gradcheck(fn, (direction, intensity))


@pytest.mark.parametrize('case', ['shape', 'batch', 'dtype', 'integer', 'nan', 'zero', 'negative'])
def test_light_invalid_inputs_fail_explicitly(case):
    ref = reference()
    p = torch.ones(2, 3, dtype=torch.float64)
    c = torch.ones(9, 3, dtype=torch.float64)
    if case == 'shape':
        with pytest.raises(ValueError, match='shape'): ref.spherical_harmonics(p, c[:8])
    elif case == 'batch':
        with pytest.raises(ValueError, match='batch'): ref.spherical_harmonics(p, c.expand(2, 9, 3))
    elif case == 'dtype':
        with pytest.raises(TypeError, match='dtype'): ref.spherical_harmonics(p, c.float())
    elif case == 'integer':
        with pytest.raises(TypeError, match='dtype'): ref.spherical_harmonics(p.long(), c)
    elif case == 'nan':
        p[0, 0] = float('nan')
        with pytest.raises(ValueError, match='finite'): ref.spherical_harmonics(p, c)
    elif case == 'zero':
        with pytest.raises(ValueError, match='zero'): ref.spherical_harmonics(p*0, c)
    else:
        with pytest.raises(ValueError, match='nonnegative'): ref.point(p, p[:1]+1, -p[:1])


def test_light_noncontiguous_empty_and_repeat_execution_contract():
    ref = reference()
    points = torch.arange(24, dtype=torch.float64).reshape(4, 6)[:, ::2]
    assert not points.is_contiguous()
    positions = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
    sample = ref.point(points, positions, torch.ones_like(positions))
    again = ref.point(points, positions, torch.ones_like(positions))
    assert all(torch.equal(sample[k], again[k]) for k in sample)
    assert ref.point(points[:0], positions, torch.ones_like(positions))['direction'].shape == (0, 1, 3)
    assert ref.point(points, positions[:0], positions[:0])['valid'].shape == (4, 0)
    assert ref.spherical_harmonics(points[:0], torch.zeros(9, 3).double()).shape == (0, 3)


def test_point_unrepresentable_arithmetic_is_not_silently_clamped():
    ref = reference()
    for distance in (1e-200, 1e200):
        with pytest.raises(ValueError, match='representable'):
            ref.point(torch.zeros(1, 3).double(), torch.tensor([[distance, 0., 0.]], dtype=torch.float64),
                      torch.ones(1, 3).double())


def test_light_units_and_batch_errors_cover_every_recipe():
    ref = reference()
    p = torch.ones(2, 3, dtype=torch.float64)
    with pytest.raises(TypeError, match='tensors'): ref.directional(p.tolist(), p)
    with pytest.raises(ValueError, match='device'):
        ref.directional(p, torch.empty(2, 3, dtype=torch.float64, device='meta'))
    with pytest.raises(ValueError, match='zero'): ref.directional(p*0, p)
    with pytest.raises(ValueError, match='nonnegative'): ref.directional(p, -p)
    with pytest.raises(ValueError, match='shape'): ref.point(p, p, p[:1])
    with pytest.raises(ValueError, match='batch'): ref.point(p, p.expand(3, 2, 3), p)
    with pytest.raises(ValueError, match='batch'): ref.spherical_harmonics(p.expand(0, 2, 3), torch.zeros(9, 3).double())
    with pytest.raises(ValueError, match='shape'): ref.directional(p[0], p)
    with pytest.raises(ValueError, match='representable'):
        ref.point(p*0, p*.01, torch.full_like(p, 1e308))
    with pytest.raises(ValueError, match='representable'):
        ref.spherical_harmonics(p, torch.full((9, 3), 1e308, dtype=torch.float64))


def test_lighting_demo_is_finite_and_keeps_negative_sh_values():
    demo = importlib.import_module('tools.testing_dashboard.lighting_demo')
    data = demo.examples()
    assert len(data['distance']) == 16
    assert len(data['angle']) == 181
    assert data['distance'][0] == pytest.approx((.5, 16., 4.))
    assert data['distance'][3] == pytest.approx((2., 1., 4.))
    assert data['angle'][0][1:] == pytest.approx((1., 17/16))
    assert data['angle'][-1][1:] == pytest.approx((0., 1/16))
    assert min(row[2] for row in data['angle']) < 0
    assert data['coincident_valid'] is False
    assert data['image'].shape == (128, 256, 3)
    assert data['image'].dtype == torch.uint8


def test_lighting_textbook_examples_and_evidence_links_are_present():
    build = importlib.import_module('tools.testing_dashboard.build_lessons')
    body = build.article()['html']
    for phrase in ('Point lights', 'Spherical harmonics', '17/16', '1/16',
                   'shading/light-contract', 'shading/lights-demo', 'normal-incidence', 'zero distance'):
        assert phrase in body
    assert 'not a public' in body


def test_lighting_demo_publishes_finite_artifacts_with_honest_scope(tmp_path):
    server = importlib.import_module('tools.testing_dashboard.server')
    demo = importlib.import_module('tools.testing_dashboard.lighting_demo')
    with server.TestingServer(tmp_path, port=0, allow_loopback_for_tests=True) as service:
        demo.publish('http://'+service.address, 'lighting-fixture')
        page = service.index.page('shading/lights-demo')
        assert page['kind'] == 'demo' and page['counts'] == {}
        assert len(page['artifacts']) == 5
        assert page['evidence']['run'] == 'lighting-fixture'
