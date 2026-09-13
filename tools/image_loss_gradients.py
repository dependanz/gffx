"""Create-only supervised direct-light texture-loss qualification (contract8.36)."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from tools import image_loss_reference as reference
from tools import shading_independent_evaluator as scalar
from tools.material_loss_cost import check_deadline, supervise, write_report
from tools.principled_direct_images import PreparedCurve, RESOURCE, MANIFEST, MANIFEST_HASH, SCALAR_HASH
from tools.shading_image_bundle import load_bundle
from tools.shading_image_probe import producer_identity

WHEEL_HASH = '1c97f8dfa1f565078131a5e1192492680dbecd8d4ed8620c56d2ab4fa825c4c9'
PASSED = 'direct image-loss adjoint passed; fixed CPU colour fitting only'
PROFILE = dict(name='direct-image-loss-v1', dtype='float64', device='cpu', threads=1,
               points_per_call=4, texture_shape=[32, 32, 3], image_shape=[128, 128],
               cooperative_seconds=120, watchdog_seconds=180, sampled_working_set_limit_bytes=1024**3,
               aggregate_seconds=900, step_size=5., target_scale=.85, passes=2)


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    root = Path(__file__).parent
    names = ('image_loss_gradients.py', 'image_loss_reference.py', 'shading_independent_evaluator.py',
             'principled_direct_images.py', 'shading_image_bundle.py', 'shading_image_probe.py',
             'material_loss_cost.py', 'corpus_registry.py')
    result = {name: digest(root/name) for name in names}
    if result['shading_independent_evaluator.py'] != SCALAR_HASH: raise ValueError('scalar source identity changed')
    return result


def texture():
    y, x = torch.meshgrid(torch.arange(32, dtype=torch.float64), torch.arange(32, dtype=torch.float64), indexing='ij')
    return torch.stack((.2+.5*(x+.5)/32, .2+.5*(y+.5)/32, torch.full_like(x, .25)), -1)


def schedule(): return [('classic', 64), ('principled', 64), ('classic', 6536), ('principled', 6536)]


def validate(n, v, uv, model, deadline):
    check_deadline(deadline)
    if model not in ('classic', 'principled'): raise ValueError('unknown material')
    for field, channels in ((n, 3), (v, 3), (uv, 2)):
        if (not isinstance(field, torch.Tensor) or field.ndim != 2 or field.shape[1] != channels
                or not 1 <= len(field) <= 6536 or field.dtype != torch.float64
                or field.device.type != 'cpu' or field.requires_grad or not bool(torch.isfinite(field).all())):
            raise ValueError('finite nonempty CPU float64 fixture required')
    if len(n) != len(v) or len(n) != len(uv): raise ValueError('fixture lengths differ')
    if bool((n.norm(dim=-1) == 0).any()) or bool((v.norm(dim=-1) == 0).any()):
        raise ValueError('zero direction')


def coefficients(n, v, model, budget, deadline):
    curve = PreparedCurve(scalar.LossCurve(budget.to_envelope()['payload']), roughness=1.) if model == 'principled' else None
    unit = scalar._normal([1., 1., 2.]); intercepts = []; slopes = []
    for nn, vv in zip(n.tolist(), v.tolist()):
        check_deadline(deadline)
        cosine = max(0., scalar._dot(scalar._normal(nn), unit))
        pair = []
        for a in (0., 1.):
            rgb = (scalar.classic(nn, vv, unit, albedo=[a]*3, specular=.1, exponent=2.)
                   if model == 'classic' else scalar.principled(nn, vv, unit, albedo=[a]*3, roughness=1., f0=.04, budget=curve))
            pair.append([x*cosine for x in rgb])
        intercepts.append(pair[0]); slopes.append([x-y for x, y in zip(pair[1], pair[0])])
    return intercepts, slopes


def _metric(actual, expected, atol, rtol):
    a, e = torch.as_tensor(actual, dtype=torch.float64), torch.as_tensor(expected, dtype=torch.float64)
    if a.shape != e.shape or not bool(torch.isfinite(a).all()): raise ValueError('measurement shape/finiteness')
    error = (a-e).abs(); tolerance = atol+rtol*e.abs()
    return dict(passed=bool((error <= tolerance).all()), max_error=float(error.max()), max_normalized_error=float((error/tolerance).max()))


def inspect_pass(tex, weights, b, s, target, actual, gradient, loss):
    expected = reference.evaluate(tex, weights, b, s, target)
    metrics = dict(rgb=_metric(actual, expected['values'], 1e-10, 1e-8),
                   loss=_metric(loss, expected['loss'], 1e-12, 1e-8),
                   gradient=_metric(gradient, expected['gradient'], 1e-11, 1e-7))
    touched = {(y, x) for taps in weights for y, x, w in taps if w != 0}
    zero = all(gradient[y][x] == [0., 0., 0.] for y in range(32) for x in range(32) if (y, x) not in touched)
    nonzero = any(value != 0 for row in gradient for rgb in row for value in rgb)
    passed = all(row['passed'] for row in metrics.values()) and zero and nonzero
    if not passed: raise ValueError('image loss/adjoint comparison failed: '+str(metrics))
    return dict(metrics=metrics, untouched_zero=zero, nonzero=nonzero, reference_loss=expected['loss'], passed=passed)


def evaluate(n, v, uv, model, *, deadline):
    validate(n, v, uv, model, deadline)
    from gffx.torch import material, light, render
    budget = material.PrincipledBudget.bundled(RESOURCE, device='cpu', dtype=torch.float64)
    if budget.metadata['admission'] != 'accepted': raise ValueError('accepted resource required')
    b, s = coefficients(n, v, model, budget, deadline)
    weights = [reference.stencil(32, 32, u) for u in uv.tolist()]
    initial = texture(); tex = initial.clone()
    target = reference.evaluate((initial*.85).tolist(), weights, b, s, [[0.]*3 for _ in weights])['values']
    target_tensor = torch.tensor(target, dtype=torch.float64)
    direction = torch.tensor([[1., 1., 2.]], dtype=torch.float64)
    passes = []; calls = 0; fd = None
    for iteration in range(2):
        check_deadline(deadline)
        tex = tex.detach().requires_grad_(True); gradient = torch.zeros_like(tex); values = []; losses = []
        for start in range(0, len(n), 4):
            check_deadline(deadline); end = min(start+4, len(n)); count = end-start
            pyramid, offsets = render.texture_pyramid(tex, levels=1)
            a = render.texture(pyramid, offsets, 32, 32, uv[start:end].contiguous(), channel_count=3,
                               filter=render.FILTER_BILINEAR, mip_filter=render.MIP_NEAREST,
                               wrap_u=render.WRAP_REPEAT, wrap_v=render.WRAP_REPEAT)
            if model == 'classic':
                brdf = material.classic(n[start:end], v[start:end], direction, albedo=a,
                                        specular=torch.tensor(.1, dtype=torch.float64), exponent=torch.tensor(2., dtype=torch.float64))
            else:
                brdf = material.principled(n[start:end], v[start:end], direction, albedo=a,
                                           roughness=torch.tensor(1., dtype=torch.float64), f0=torch.tensor(.04, dtype=torch.float64), budget=budget)
            packet = dict(measure='delta', direction=direction.expand(count, 1, 3),
                          irradiance=torch.ones((count, 1, 3), dtype=torch.float64), valid=torch.ones((count, 1), dtype=torch.bool))
            rgb = light.integrate(n[start:end], brdf, packet)
            objective = (rgb-target_tensor[start:end]).square().sum()/(3*len(n))
            grad, = torch.autograd.grad(objective, tex)
            gradient += grad.detach(); values.extend(rgb.detach().tolist()); losses.append(float(objective.detach()))
            calls += 1
            del grad, objective, rgb, brdf, a, pyramid, offsets, packet
        tex_list, grad_list, loss = tex.detach().tolist(), gradient.tolist(), math.fsum(losses)
        checked = inspect_pass(tex_list, weights, b, s, target, values, grad_list, loss)
        passes.append(dict(texture=tex_list, values=values, gradient=grad_list, loss=loss, **checked))
        if iteration == 0:
            adjoint = reference.evaluate(tex_list, weights, b, s, target)['gradient']
            fd = reference.finite_differences(tex_list, weights, b, s, target, adjoint)
            if not all(row['passed'] for row in fd): raise ValueError('independent finite difference disagreement')
            tex = (tex.detach()-5*gradient).contiguous()
            if not bool(((tex >= 0) & (tex <= 1)).all()): raise ValueError('fitting step outside texture domain')
    check_deadline(deadline)
    if not passes[1]['loss'] < passes[0]['loss']: raise ValueError('fresh fitting step did not decrease image loss')
    return dict(passed=True, pixels=len(n), calls=calls, max_points=min(4, len(n)),
                coefficient_evaluations=2*len(n), passes=passes, finite_differences=fd,
                intercepts=b, slopes=s, target=target, budget=budget.metadata)


def selected(root, count):
    buffers = load_bundle(root, path=MANIFEST, sha256=MANIFEST_HASH)['buffers']
    ids = buffers['covered'].flatten().nonzero().flatten()
    if len(ids) != 6536 or count not in (64, 6536): raise ValueError('fixed mask/pixel count required')
    if count == 64: ids = ids[torch.tensor([i*6535//63 for i in range(64)])]
    fields = [buffers[name].reshape(-1, channels)[ids].contiguous() for name, channels in (('normal', 3), ('view', 3), ('uv', 2))]
    return ids.tolist(), fields


def worker(root, output, wheel, model, count):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    start = time.perf_counter(); deadline = start+120
    report = dict(status='working', profile=PROFILE, worker_pid=os.getpid(), model=model, pixels=count,
                  corpus_root=str(root.resolve()), reference_manifest=dict(path=MANIFEST, sha256=MANIFEST_HASH), sources=sources())
    try:
        report['producer'] = producer_identity(wheel)
        if report['producer']['wheel']['sha256'] != WHEEL_HASH: raise ValueError('wrong installed wheel')
        report['indices'], (n, v, uv) = selected(root, count)
        report['result'] = evaluate(n, v, uv, model, deadline=deadline)
        report['elapsed_seconds'] = time.perf_counter()-start
        if len(json.dumps(report, allow_nan=False).encode('utf-8')) > 16*1024**2:
            raise ValueError('worker report exceeds 16 MiB')
        report['status'] = PASSED
    except Exception as error:
        report.update(status='stopped', error=type(error).__name__+': '+str(error))
        write_report(output, report); raise
    write_report(output, report)


def run(root, output, wheel):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    cohort = dict(status='working', profile=PROFILE, sources=sources(), workers=[])
    write_report(output/'cohort.json', cohort)
    for index, (model, count) in enumerate(schedule()):
        if time.perf_counter()-start > 720:
            cohort.update(status='stopped', error='aggregate launch reserve exhausted', elapsed_seconds=time.perf_counter()-start)
            write_report(output/'cohort.json', cohort)
            raise TimeoutError('aggregate launch reserve exhausted')
        path = output/f'worker-{index}.json'
        code = supervise(root, path, worker_module='tools.image_loss_gradients',
                         worker_args=('--wheel', str(wheel), '--model', model, '--pixels', str(count)))
        cohort['workers'].append(dict(path=path.name, sha256=digest(path), model=model, pixels=count))
        cohort['elapsed_seconds'] = time.perf_counter()-start
        cohort['status'] = 'stopped' if code else 'working'
        write_report(output/'cohort.json', cohort)
        if code: raise RuntimeError('worker stopped; no automatic retry')
    cohort['status'] = PASSED
    write_report(output/'cohort.json', cohort)
    return cohort


def replay(root, sha256, *, blob=None):
    if blob is None and (root/'cohort.json').stat().st_size > 65536: raise ValueError('cohort byte limit')
    raw = (root/'cohort.json').read_bytes() if blob is None else blob
    if len(raw) > 65536: raise ValueError('cohort byte limit')
    if hashlib.sha256(raw).hexdigest() != sha256: raise ValueError('cohort hash mismatch')
    record = json.loads(raw)
    if record['status'] != PASSED or record['profile'] != PROFILE or record['sources'] != sources():
        raise ValueError('cohort identity/status mismatch')
    if not 0 < record['elapsed_seconds'] <= 900 or len(record['workers']) != 4: raise ValueError('cohort limits')
    for index, (entry, (model, count)) in enumerate(zip(record['workers'], schedule())):
        if entry['path'] != f'worker-{index}.json' or (entry['model'], entry['pixels']) != (model, count):
            raise ValueError('schedule mismatch')
        path = root/entry['path']
        if path.stat().st_size > 16*1024**2 or digest(path) != entry['sha256']: raise ValueError('worker bytes changed')
        report = json.loads(path.read_text(encoding='utf-8')); sup = report['supervisor']
        if (report['status'] != PASSED or report['profile'] != PROFILE or report['sources'] != record['sources']
                or report['model'] != model or report['pixels'] != count
                or report['reference_manifest'] != dict(path=MANIFEST, sha256=MANIFEST_HASH)
                or report['producer']['wheel']['sha256'] != WHEEL_HASH
                or report['producer'] != producer_identity(Path(report['producer']['wheel']['path']))):
            raise ValueError('worker identity mismatch')
        if (sup['failure'] is not None or sup['exit_code'] != 0 or sup['monitored_pid'] != report['worker_pid']
                or not 0 < sup['sampled_peak_working_set_bytes'] <= 1024**3
                or not 0 < sup['elapsed_seconds'] <= 180 or not 0 < report['elapsed_seconds'] <= 120):
            raise ValueError('worker limits/identity failed')
        ids, (n, v, uv) = selected(Path(report['corpus_root']), count)
        if report['indices'] != ids: raise ValueError('covered-pixel identity mismatch')
        from gffx.torch.material import PrincipledBudget
        budget = PrincipledBudget.bundled(RESOURCE, device='cpu', dtype=torch.float64)
        b, s = coefficients(n, v, model, budget, time.perf_counter()+120)
        r = report['result']; weights = [reference.stencil(32, 32, u) for u in uv.tolist()]
        target = reference.evaluate((texture()*.85).tolist(), weights, b, s, [[0.]*3 for _ in weights])['values']
        if (r['intercepts'] != b or r['slopes'] != s or r['target'] != target or r['budget'] != budget.metadata
                or r['pixels'] != count or r['calls'] != 2*math.ceil(count/4) or r['max_points'] != 4
                or r['coefficient_evaluations'] != 2*count or r['passed'] is not True or len(r['passes']) != 2):
            raise ValueError('result identity/counts failed')
        expected_tex = texture()
        for iteration, row in enumerate(r['passes']):
            if row['texture'] != expected_tex.tolist(): raise ValueError('texture recipe/step mismatch')
            checked = inspect_pass(row['texture'], weights, b, s, target, row['values'], row['gradient'], row['loss'])
            if any(row[key] != value for key, value in checked.items()): raise ValueError('replayed metrics differ')
            if iteration == 0:
                gradient = reference.evaluate(row['texture'], weights, b, s, target)['gradient']
                fd = reference.finite_differences(row['texture'], weights, b, s, target, gradient)
                if fd != r['finite_differences'] or not all(x['passed'] for x in fd): raise ValueError('FD evidence differs')
                expected_tex -= 5*torch.tensor(row['gradient'], dtype=torch.float64)
        if not r['passes'][1]['loss'] < r['passes'][0]['loss']: raise ValueError('no improvement')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--model', choices=('classic', 'principled')); parser.add_argument('--pixels', type=int)
    args = parser.parse_args()
    if args.worker: worker(args.fixtures, args.output, args.wheel, args.model, args.pixels)
    else: run(args.fixtures, args.output, args.wheel)
    return 0


if __name__ == '__main__': raise SystemExit(main())
