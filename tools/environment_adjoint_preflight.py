"""Identity-bound qualification of bounded colour Jacobians, before full images."""
import argparse
import json
import os
from pathlib import Path
import time
import torch

from tools import environment_image_adjoint as method
from tools import environment_loss as old
from tools.shading_cuda_closeout import identity

PASSED = 'colour-Jacobian environment adjoint preflight passed; not full images'
PROFILE = dict(points=2, profiles=4, offsets=[.5, .25], iterations=2,
    maximum_calls=1088, maximum_directions=1114112, maximum_colour_backwards=3264,
    maximum_texture_backwards=32, threads=1, device='cpu', dtype='float64',
    cooperative_seconds=120, watchdog_seconds=180, host_bytes=1024**3,
    aggregate_seconds=900, step_size=5.)


def sources():
    result = old.sources()
    for module in (method,):
        result[Path(module.__file__).name] = old.direct.digest(Path(module.__file__))
    result[Path(__file__).name] = old.direct.digest(Path(__file__))
    return result


def evaluate(n, v, uv, model, environment, deadline):
    from gffx.torch.material import PrincipledBudget
    budget = PrincipledBudget.bundled(old.direct.RESOURCE, device='cpu', dtype=torch.float64)
    inputs = old.reference_inputs(n, v, uv, model, environment, budget, deadline)
    orders = [r['history'][-1]['n_mu'] for r in inputs['references']]
    if orders != [128, 32]:
        raise ValueError('predeclared selected-point work changed')
    weights = [old.ref.stencil(32, 32, u) for u in uv.tolist()]
    tex = old.direct.texture(); target = torch.tensor(inputs['target'], dtype=torch.float64)
    passes = []; counts = dict(calls=0, directions=0, colour_backwards=0, texture_backwards=0)
    for iteration in range(2):
        average = torch.zeros_like(tex); rows = []
        for split in (.5, .25):
            result = method.angular_pass(tex, n, v, uv, target, model=model, environment=environment,
                budget=budget, orders=orders, split=split, deadline=deadline)
            for key in counts: counts[key] += result[key]
            checked = old.inspect(tex.tolist(), weights, inputs, result['values'].tolist(),
                                  result['gradient'].tolist(), result['loss'])
            rows.append(dict(split=split, values=result['values'].tolist(),
                gradient=result['gradient'].tolist(), loss=result['loss'], **checked))
            average += result['gradient']/2
        passes.append(dict(texture=tex.tolist(), offsets=rows))
        if iteration == 0:
            expected = old.ref.evaluate(tex.tolist(), weights, inputs['intercepts'], inputs['slopes'], inputs['target'])
            differences = old.ref.finite_differences(tex.tolist(), weights, inputs['intercepts'], inputs['slopes'],
                                                   inputs['target'], expected['gradient'])
            if not all(x['passed'] for x in differences): raise ValueError('independent finite differences')
            tex = (tex-5*average).contiguous()
            if not bool(((tex >= 0) & (tex <= 1)).all()): raise ValueError('update outside texture domain')
    if any(b['loss'] >= a['loss'] for a, b in zip(passes[0]['offsets'], passes[1]['offsets'])):
        raise ValueError('loss did not decrease')
    if counts != dict(calls=272, directions=278528, colour_backwards=816, texture_backwards=8):
        raise ValueError('work count mismatch')
    return dict(inputs=inputs, passes=passes, finite_differences=differences, orders=orders, counts=counts)


def worker(root, output, wheel, model, environment):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    start = time.perf_counter()
    report = dict(status='working', worker_pid=os.getpid(), profile=PROFILE, sources=sources(),
                  model=model, environment=environment, manifest_sha256=old.direct.MANIFEST_HASH)
    try:
        if (model, environment) not in old.schedule(): raise ValueError('unknown profile')
        report['producer'] = identity(wheel, old.direct.WHEEL_HASH)
        report['indices'], (n, v, uv) = old.selected(root)
        report['result'] = evaluate(n, v, uv, model, environment, start+120)
        old.check_deadline(start+120)
        report.update(status=PASSED, elapsed_seconds=time.perf_counter()-start)
    except Exception as error:
        report.update(status='stopped', error=type(error).__name__+': '+str(error))
        old.write_report(output, report); raise
    old.write_report(output, report)


def run(root, output, wheel):
    output.mkdir(parents=True, exist_ok=False); start = time.perf_counter()
    cohort = dict(status='working', profile=PROFILE, sources=sources(), workers=[])
    old.write_report(output/'cohort.json', cohort)
    for index, (model, environment) in enumerate(old.schedule()):
        old.check_deadline(start+720)
        path = output/f'worker-{index}.json'
        code = old.supervise(root, path, worker_module='tools.environment_adjoint_preflight',
            worker_args=('--wheel', str(wheel), '--model', model, '--environment', environment))
        cohort['workers'].append(dict(path=path.name, sha256=old.direct.digest(path)))
        cohort.update(status='stopped' if code else 'working', elapsed_seconds=time.perf_counter()-start)
        old.write_report(output/'cohort.json', cohort)
        if code: raise RuntimeError('stopped preflight; no automatic retry')
    cohort['status'] = PASSED; old.write_report(output/'cohort.json', cohort)
    return cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures', 'output', 'wheel'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--model'); parser.add_argument('--environment')
    args = parser.parse_args()
    if args.worker: worker(args.fixtures, args.output, args.wheel, args.model, args.environment)
    else: run(args.fixtures, args.output, args.wheel)
    return 0


if __name__ == '__main__': raise SystemExit(main())
