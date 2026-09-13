"""Finite all-input fitting controls, not a renderer or optimizer benchmark."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from tools import material_loss_cost as cost

PARAMETERS = ('texture', 'uv', 'normal', 'view', 'directions', 'shape', 'specular', 'radiance')
WHEEL = Path('build/light-batches-20260908/final-wheels/gffx-0.2.0.dev0-cp310-abi3-win_amd64.whl')
WHEEL_SHA = '1c97f8dfa1f565078131a5e1192492680dbecd8d4ed8620c56d2ab4fa825c4c9'
PROFILE = dict(name='shading-fit-closure-v1', calls_limit=160, cases=17,
               cooperative_seconds=120, watchdog_seconds=180, host_bytes=1024**3,
               step=1e-4, epsilons=[1e-5, 5e-6], atol=1e-10, rtol=1e-5)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule():
    return [(model, parameter) for model in ('classic', 'principled') for parameter in PARAMETERS] + [('sh', 'coefficients')]


def fields():
    import torch
    def t(x): return torch.tensor(x, dtype=torch.float64)
    return [t([[[.2,.3,.4],[.3,.4,.5]],[[.4,.5,.6],[.5,.6,.7]]]),
            t([[.35,.45]]), t([[.12,.18,1.]]), t([[.21,.09,1.]]),
            t([[.3,.4,1.],[-.2,.1,1.]]), t(.63), t(.09), t([[.7,.8,.9],[.4,.5,.6]])]


def chain(values, model, budget):
    from gffx.torch import material, render, light
    import torch
    tex,uv,n,v,d,r,f,incoming = values
    pyramid,offsets = render.texture_pyramid(tex.contiguous(), levels=2)
    colour = render.texture(pyramid, offsets, 2, 2, uv.contiguous(), channel_count=3,
                            lod=uv.new_tensor([.35]), filter=render.FILTER_BILINEAR, mip_filter=render.MIP_LINEAR)
    if model == 'classic':
        brdf = material.classic(n,v,d,albedo=colour,specular=f,exponent=r*4)
    elif model == 'principled':
        brdf = material.principled(n,v,d,albedo=colour,roughness=r,f0=f,budget=budget)
    else: raise ValueError('model')
    packet = dict(measure='solid_angle', direction=d[None], radiance=incoming[None],
                  valid=torch.ones((1,2),dtype=torch.bool), solid_angle=n.new_tensor([[.6,.4]]))
    return light.integrate(n,brdf,packet)


def validate_result(row):
    numbers = [row['before'],row['after'],row['analytic'],*row['differences']]
    if len(row['differences']) != 2 or not all(math.isfinite(x) for x in numbers):
        raise ValueError('finite two-scale derivative evidence required')
    if any(abs(x-row['analytic']) > PROFILE['atol']+PROFILE['rtol']*abs(row['analytic']) for x in row['differences']):
        raise ValueError('loss derivative mismatch')
    if not (0 <= row['after'] < row['before'] and row['analytic'] < 0 and row['fresh_finite'] is True):
        raise ValueError('fresh finite descent required')


def evaluate(model, parameter, *, deadline):
    cost.check_deadline(deadline)
    if (model,parameter) not in schedule(): raise ValueError('undeclared parameter/model')
    import torch
    from gffx.torch import material, light
    budget = material.PrincipledBudget.bundled('ggx-smith-loss-cpu-f64-v1',device='cpu',dtype=torch.float64) if model=='principled' else None
    if model == 'sh':
        base = torch.arange(27,dtype=torch.float64).reshape(9,3)/100
        base[0] = math.sqrt(4*math.pi)
        normals = torch.tensor([[.12,.18,1.]],dtype=torch.float64)
        function = lambda x: light.spherical_harmonics(normals,x)
    else:
        args = fields(); index = PARAMETERS.index(parameter); base = args[index]
        def function(x):
            values = list(args); values[index] = x
            return chain(values,model,budget)
    calls = 0
    def forward(x):
        nonlocal calls
        cost.check_deadline(deadline); calls += 1
        return function(x)
    target = .9*forward(base).detach()
    leaf = base.clone().requires_grad_()
    before = (forward(leaf)-target).square().mean()
    gradient, = torch.autograd.grad(before,leaf)
    norm = gradient.norm()
    if not bool(torch.isfinite(gradient).all()) or not bool(norm>0):
        raise ValueError('finite nonzero gradient required')
    direction = -gradient.detach()/norm.clamp_min(1.)
    analytic = float((gradient*direction).sum())
    differences = []
    with torch.no_grad():
        for epsilon in PROFILE['epsilons']:
            plus = (forward(base+epsilon*direction)-target).square().mean()
            minus = (forward(base-epsilon*direction)-target).square().mean()
            differences.append(float((plus-minus)/(2*epsilon)))
    updated = (base+PROFILE['step']*direction).detach().requires_grad_()
    after = (forward(updated)-target).square().mean()
    fresh, = torch.autograd.grad(after,updated)
    row = dict(model=model,parameter=parameter,before=float(before.detach()),after=float(after.detach()),
               analytic=analytic,differences=differences,fresh_finite=bool(torch.isfinite(fresh).all()),calls=calls)
    validate_result(row)
    return row


def identity():
    import gffx
    installed = Path(gffx.__file__).resolve().parent
    expected = Path('build/light-batches-20260908/final-installed/gffx').resolve()
    if installed != expected or digest(WHEEL) != WHEEL_SHA:
        raise ValueError('installed wheel identity')
    return dict(wheel_sha256=WHEEL_SHA, installed=str(installed),
                runtime={str(p.relative_to(installed)):digest(p) for p in installed.rglob('*') if p.is_file() and p.suffix in ('.py','.pyd','.dll','.json')})


def worker(output):
    row = dict(status='working',worker_pid=os.getpid(),profile=PROFILE,cases=[],calls=0)
    cost.write_report(output,row)
    deadline = time.perf_counter()+PROFILE['cooperative_seconds']
    try:
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        row['identity'] = identity(); row['source_sha256'] = digest(__file__)
        for model,parameter in schedule():
            result = evaluate(model,parameter,deadline=deadline)
            row['cases'].append(result); row['calls'] += result['calls']
            if row['calls'] > PROFILE['calls_limit']: raise ValueError('aggregate call limit')
            cost.write_report(output,row)
        cost.check_deadline(deadline); row['status'] = 'passed'
    except Exception as error:
        row.update(status='stopped',error=f'{type(error).__name__}: {error}'); raise
    finally: cost.write_report(output,row)


def replay(path, sha):
    if digest(path) != sha: raise ValueError('report identity')
    row = json.loads(path.read_text())
    if (row['status'] != 'passed' or row['profile'] != PROFILE or row['source_sha256'] != digest(__file__)
            or row['identity'] != identity() or [(c['model'],c['parameter']) for c in row['cases']] != schedule()):
        raise ValueError('qualification identity/schedule')
    supervisor = row['supervisor']
    if (supervisor['exit_code'] != 0 or supervisor['failure'] is not None
            or supervisor['monitored_pid'] != row['worker_pid']
            or not 0 < supervisor['sampled_peak_working_set_bytes'] <= PROFILE['host_bytes']
            or not 0 < supervisor['elapsed_seconds'] <= PROFILE['watchdog_seconds']):
        raise ValueError('supervision')
    if row['calls'] != 119 or any(case['calls'] != 7 for case in row['cases']):
        raise ValueError('call accounting')
    for case in row['cases']: validate_result(case)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.worker: worker(args.output); return 0
    if args.output.exists(): raise FileExistsError('preserve existing qualification')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    return cost.supervise(args.fixtures,args.output,worker_module='tools.shading_fit_closure')


if __name__ == '__main__': raise SystemExit(main())
