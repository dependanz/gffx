"""Prospective CI-07 full-effect recovery; see docs/IMAGE_RECOVERY_RECIPE.md.

Independent scalar targets and differences never call a gffx operation.
Only a bounded synthetic gffx recipe is tested, not a consumer application.
"""
import math
import struct
import time

import torch

SIZE = 32
RADIUS = 2
NAMES = ("tx", "ty", "rotation", "sx", "sy", "opacity", "exposure", "brightness", "sigma")
TARGET = (.37, -.29, .043, 1.035, .965, .77, .18, .012, .85)
INITIAL = (.62, -.48, .015, 1.012, .988, .68, -.04, -.006, 1.05)
CENTRE = (0., 0., 0., 1., 1., .725, 0., 0., 1.)
RANGE = (1., 1., .15, .1, .1, .225, .5, .05, .4)
PIVOT = (14.3, 15.7)
MAX_ITERATIONS = 60
MAX_EVALUATIONS = 100


def sources():
    """Deterministic straight-sRGB textures, source-space mask, opaque canvas."""
    moving, locked, mask = [], [], []
    for y in range(SIZE):
        mr, lr, ar = [], [], []
        for x in range(SIZE):
            ellipse = ((x - 15) / 8) ** 2 + ((y - 14) / 10) ** 2
            alpha = max(0., min(.85, (1 - ellipse) * 2))
            if x > 15 and 11 < y < 17:
                alpha *= .35
            mr.append([.2 + .55*x/31, .25 + .3*y/31, .72 - .35*x/31, alpha])
            diamond = abs(x - 8)/7 + abs(y - 23)/5
            la = max(0., min(.7, 1 - diamond))
            lr.append([.75 - .25*y/31, .2 + .5*x/31, .18 + .25*y/31, la])
            ar.append([.65 + .3*(x + 2*y)/(3*31)])
        moving.append(mr); locked.append(lr); mask.append(ar)
    tensors = tuple(torch.tensor(x, dtype=torch.float64) for x in (moving, locked, mask))
    canvas = torch.tensor([.025, .045, .07, 1.], dtype=torch.float64).expand(SIZE, SIZE, 4).clone()
    return (*tensors, canvas)


def _decode(x):
    return x/12.92 if x <= .04045 else ((x + .055)/1.055)**2.4


def _intake(pixels):
    return [[[ *(_decode(p[c])*p[3] for c in range(3)), p[3]] for p in row] for row in pixels]


def scalar_reference(fixed, parameters, *, return_sampling=False, return_stages=False, pivot=PIVOT):
    """Pure Python/math equations: transfer, effect, mask, blur, inverse warp, over."""
    raw, lower, masks, canvas = (x.tolist() for x in fixed)
    height, width = len(raw), len(raw[0])
    tx, ty, angle, sx, sy, opacity, exposure, brightness, sigma = parameters
    upper = _intake(raw); lower = _intake(lower)
    stages = {"intake": [[p[:] for p in row] for row in upper], "locked_intake": lower}
    for y in range(height):
        for x in range(width):
            p = upper[y][x]
            upper[y][x] = [(p[c]*2**exposure + brightness*p[3]) for c in range(3)] + [p[3]]
    stages["adjust"] = [[p[:] for p in row] for row in upper]
    for y in range(height):
        for x in range(width):
            q = masks[y][x][0]*opacity
            upper[y][x] = [c*q for c in upper[y][x]]
    stages["mask"] = [[p[:] for p in row] for row in upper]
    weights = [math.exp(-.5*(i/sigma)**2) for i in range(-RADIUS, RADIUS+1)]
    total = sum(weights); weights = [w/total for w in weights]
    for axis in (0, 1):  # horizontal, then vertical; transparent support.
        filtered = [[[0.]*4 for _ in range(width)] for _ in range(height)]
        for y in range(height):
            for x in range(width):
                for offset, weight in zip(range(-RADIUS, RADIUS+1), weights):
                    yy, xx = (y, x+offset) if axis == 0 else (y+offset, x)
                    if 0 <= yy < height and 0 <= xx < width:
                        for c in range(4):
                            filtered[y][x][c] += upper[yy][xx][c]*weight
        upper = filtered
    stages["blur"] = upper
    cosine, sine = math.cos(angle), math.sin(angle)
    a, b, d, e = cosine*sx, -sine*sy, sine*sx, cosine*sy
    ox = pivot[0]+tx-a*pivot[0]-b*pivot[1]
    oy = pivot[1]+ty-d*pivot[0]-e*pivot[1]
    stages["matrix"] = [[a, b, ox], [d, e, oy]]
    determinant = a*e-b*d
    result, warped, sampling = [], [], []
    for y in range(height):
        row, warped_row = [], []
        for x in range(width):
            px, py = (e*(x-ox)-b*(y-oy))/determinant, (a*(y-oy)-d*(x-ox))/determinant
            ix, iy = math.floor(px), math.floor(py)
            fx, fy = px-ix, py-iy
            sample = [0.]*4
            active = False
            for yy, wy in ((iy, 1-fy), (iy+1, fy)):
                for xx, wx in ((ix, 1-fx), (ix+1, fx)):
                    if 0 <= yy < height and 0 <= xx < width:
                        active = active or upper[yy][xx][3] != 0.
                        for c in range(4):
                            sample[c] += upper[yy][xx][c]*wx*wy
            sampling.append((ix, iy, px, py, active, min(fx, 1-fx, fy, 1-fy)))
            warped_row.append(sample)
            beneath = [lower[y][x][c]+(1-lower[y][x][3])*canvas[y][x][c] for c in range(4)]
            row.append([sample[c]+(1-sample[3])*beneath[c] for c in range(4)])
        result.append(row)
        warped.append(warped_row)
    if return_stages:
        return dict(stages, warp=warped, composite=result)
    return (result, sampling) if return_sampling else result


def evaluate(fixed, parameters, *, return_stages=False, pivot=PIVOT):
    """Fresh public graph; explicit linear-premultiplied RGBA output metadata."""
    from gffx.torch import image
    raw, locked, mask, canvas = fixed
    def intake(source):
        linear = torch.cat((image.srgb_to_linear(source[..., :3]), source[..., 3:]), -1)
        return image.convert_alpha(linear, source="straight", target="premultiplied")
    layer = intake(raw); lower = intake(locked)
    stages = {"intake": layer, "locked_intake": lower}
    layer = image.adjust(layer, exposure=parameters[6], brightness=parameters[7])
    stages["adjust"] = layer
    layer = image.apply_mask(layer, mask, opacity=parameters[5])
    stages["mask"] = layer
    layer = image.gaussian_blur(layer, parameters[8], radius=RADIUS, border="transparent")
    stages["blur"] = layer
    matrix = image.affine_matrix(parameters[:2], parameters[2], parameters[3:5], parameters.new_tensor(pivot))
    stages["matrix"] = matrix
    layer = image.warp_affine(layer, matrix, tuple(raw.shape[:2]), filter="bilinear", border="transparent")
    stages["warp"] = layer
    result = image.composite([lower, layer], canvas)
    return dict(stages, composite=result) if return_stages else result


def _bytes(tensor):
    return struct.pack("<" + "d"*tensor.numel(), *tensor.detach().reshape(-1).tolist())


def _fixed_identity(fixed):
    return tuple((x._version, _bytes(x)) for x in fixed)


def _scalar_mse(values, target):
    total = 0.
    for y in range(SIZE):
        for x in range(SIZE):
            for c in range(4):
                total += (values[y][x][c]-target[y][x][c])**2
    return total/(SIZE*SIZE*4)


def _check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("full recovery cooperative deadline exceeded")


def _recover(fixed, target, deadline):
    _check_deadline(deadline)
    centre = target.new_tensor(CENTRE); scale = target.new_tensor(RANGE)
    initial = target.new_tensor(INITIAL)
    raw = torch.atanh((initial-centre)/scale).requires_grad_()
    versions = tuple(x._version for x in (*fixed, target))
    initial_loss = float((evaluate(fixed, initial)-target).square().mean())
    optimizer = torch.optim.LBFGS([raw], lr=1., max_iter=MAX_ITERATIONS,
        max_eval=MAX_EVALUATIONS, tolerance_grad=1e-12, tolerance_change=1e-15,
        history_size=9, line_search_fn="strong_wolfe")
    calls = 0
    def closure():
        nonlocal calls
        _check_deadline(deadline)
        calls += 1
        if calls > MAX_EVALUATIONS:
            raise RuntimeError("prospective full-recovery evaluation ceiling exceeded")
        assert tuple(x._version for x in (*fixed, target)) == versions
        optimizer.zero_grad(set_to_none=True)
        physical = centre+scale*raw.tanh()
        output = evaluate(fixed, physical)
        loss = (output-target).square().mean()
        assert loss.grad_fn is not None and bool(torch.isfinite(loss))
        loss.backward()
        _check_deadline(deadline)
        assert raw.grad is not None and bool(torch.isfinite(raw.grad).all())
        return loss
    optimizer.step(closure)
    physical = (centre+scale*raw.detach().tanh()).detach()
    # This evaluation cannot reuse an optimizer graph or stale objective.
    fresh = evaluate(fixed, physical)
    final_loss = float((fresh-target).square().mean())
    _check_deadline(deadline)
    assert initial_loss > 0 and final_loss <= .01*initial_loss
    assert tuple(x._version for x in (*fixed, target)) == versions
    return physical, fresh, initial_loss, final_loss, calls


def test_full_image_recipe_stages_match_tiny_independent_scalar_oracle(record_property):
    """Find stage mismatches before an optimizer can mask them through compensation."""
    moving, locked, mask, canvas = sources()
    fixed = (moving[8:13, 10:17].clone(), locked[20:25, 5:12].clone(),
             mask[8:13, 10:17].clone(), canvas[:5, :7].clone())
    before = _fixed_identity(fixed)
    expected = scalar_reference(fixed, INITIAL, return_stages=True, pivot=(3.1, 2.2))
    actual = evaluate(fixed, fixed[0].new_tensor(INITIAL), return_stages=True, pivot=(3.1, 2.2))
    errors = {}
    assert actual.keys() == expected.keys()
    for stage, values in expected.items():
        reference = torch.tensor(values, dtype=torch.float64)
        torch.testing.assert_close(actual[stage], reference, atol=2e-12, rtol=2e-12)
        errors[stage] = float((actual[stage]-reference).abs().max())
    assert _fixed_identity(fixed) == before
    record_property("per_stage_max_absolute_errors", str(errors))


def test_full_image_recipe_matches_independent_scalar_target_and_derivatives(record_property):
    """Check all nine physical derivatives against scalar differences before recovery."""
    deadline = time.monotonic()+120
    fixed = sources(); original = _fixed_identity(fixed)
    target_values = scalar_reference(fixed, TARGET)
    target = torch.tensor(target_values, dtype=torch.float64)
    actual_target = evaluate(fixed, target.new_tensor(TARGET))
    torch.testing.assert_close(actual_target, target, atol=2e-12, rtol=2e-12)
    parameters = target.new_tensor(INITIAL).requires_grad_()
    loss = (evaluate(fixed, parameters)-target).square().mean()
    analytical = torch.autograd.grad(loss, parameters)[0]
    assert bool(torch.isfinite(analytical).all()) and bool((analytical != 0).all())
    _, base_sampling = scalar_reference(fixed, INITIAL, return_sampling=True)
    margin = min(row[5] for row in base_sampling if row[4])
    maximum_displacement = 0.
    errors = []
    for epsilon in (1e-6, 5e-7):
        differences = []
        for i in range(len(NAMES)):
            _check_deadline(deadline)
            plus, minus = list(INITIAL), list(INITIAL)
            plus[i] += epsilon; minus[i] -= epsilon
            plus_values, plus_sampling = scalar_reference(fixed, plus, return_sampling=True)
            minus_values, minus_sampling = scalar_reference(fixed, minus, return_sampling=True)
            for perturbed in (plus_sampling, minus_sampling):
                for base, probe in zip(base_sampling, perturbed):
                    if base[4] or probe[4]:
                        assert base[:2] == probe[:2], (NAMES[i], epsilon, base, probe)
                        maximum_displacement = max(maximum_displacement,
                            abs(base[2]-probe[2]), abs(base[3]-probe[3]))
            differences.append((_scalar_mse(plus_values, target_values)-
                                _scalar_mse(minus_values, target_values))/(2*epsilon))
        expected = target.new_tensor(differences)
        assert bool(torch.isfinite(analytical).all()) and bool(torch.isfinite(expected).all())
        torch.testing.assert_close(analytical, expected, atol=5e-5, rtol=5e-4)
        errors.append(float((analytical-expected).abs().max()))
    _check_deadline(deadline)
    assert _fixed_identity(fixed) == original
    record_property("oracle", "scalar-transfer-adjust-mask-separable-blur-inverse-bilinear-over-v1")
    record_property("minimum_initial_sampling_knot_distance_pixels", margin)
    record_property("maximum_finite_difference_coordinate_displacement_pixels", maximum_displacement)
    record_property("finite_difference_active_support_floor_cells_unchanged", True)
    record_property("physical_parameter_names", str(NAMES))
    record_property("gradient_values", str(analytical.tolist()))
    record_property("two_scale_max_gradient_absolute_errors", str(errors))


def test_full_image_recipe_recovers_and_repeats_with_locked_inputs(record_property):
    """Recover bounded transform/effect changes twice without changing fixed data."""
    deadline = time.monotonic()+120
    fixed = sources(); target = torch.tensor(scalar_reference(fixed, TARGET), dtype=torch.float64)
    original = _fixed_identity((*fixed, target))
    first = _recover(fixed, target, deadline)
    assert _fixed_identity((*fixed, target)) == original
    second = _recover(fixed, target, deadline)
    assert _fixed_identity((*fixed, target)) == original
    assert _bytes(first[0]) == _bytes(second[0])
    assert _bytes(first[1]) == _bytes(second[1])
    assert first[2:] == second[2:]
    record_property("initial_mse", first[2]); record_property("final_fresh_mse", first[3])
    record_property("closure_evaluations_each_repeat", first[4])
    record_property("target_parameters", str(TARGET))
    record_property("initial_parameters", str(INITIAL))
    record_property("recovered_parameters", str(first[0].tolist()))
    record_property("absolute_parameter_errors", str((first[0]-target.new_tensor(TARGET)).abs().tolist()))
    record_property("recovery_boundary", "one fixed local basin; image improvement is not unique parameter recovery")
