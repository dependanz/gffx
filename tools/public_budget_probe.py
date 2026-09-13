"""Prospective public-budget reference guard. Never generates or admits a table."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile
from tools import material_loss_cost as cost

ROUGHNESSES = (.00179, .0193, .213, .827)
VIEWS = (.019, .463, .947, .9997)
FRESNELS = (.07, .47, .93, 1.)
WHEEL_SHA = 'e22255c1582fd961cf9674c30b372b984d3ed0e679350bdfcb95d743d5727927'
FROZEN = {
    'material_loss_smooth.py': cost.SMOOTH_HASH,
    'material_loss_resource.py': '62df9a500bca26e9eeb79fef05275e42567de45eef301a369ba7d6fa838839fd',
    'material_loss_reference.py': 'e5179a64182ebdcae97acad31a8da3c5746e412eefa0a891c335df704160971f',
    'material_mean_validation.py': 'c3c0f324459c21f3775186a61a38d2a1d8249df4491da7b071607f76c2d0de7e',
    'shading_energy_probe.py': '084f5cafc9e226291a26d4e20eae1ac09e154da207dbfea817b991004d850da7',
}


def directional_points():
    return [(r, c, f) for r in ROUGHNESSES for c in (*VIEWS, .0027*r*r) for f in FRESNELS]


def guard_points():
    for r in ROUGHNESSES:
        for c in (*VIEWS, .0027*r*r, 0., 1., *(r*r/16*k for k in (.999, 1., 1.001))):
            yield r, c


def verify_references(root):
    actual = {}
    for name, expected in FROZEN.items():
        actual[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual[name] != expected:
            raise ValueError('reference SHA mismatch: '+name)
    return actual


def verify_wheel(wheel):
    import gffx
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA:
        raise ValueError('wheel SHA mismatch')
    root = Path(gffx.__file__).resolve().parent.parent
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.startswith('gffx/') and not name.endswith('/')]
        for name in names:
            if (root/name).read_bytes() != archive.read(name):
                raise ValueError('installed wheel member mismatch: '+name)
    return dict(sha256=WHEEL_SHA, installed_root=str(root), matched_members=len(names))


def run_guard(pair, check_deadline, save):
    rows = []
    for r, c in guard_points():
        check_deadline()
        a = pair(c, r, resolution=280, azimuth_resolution=32)
        check_deadline()
        b = pair(c, r, resolution=512, azimuth_resolution=64)
        check_deadline()
        for f in (0, 1):
            low, high = float(a[f]), float(b[f])
            valid = all(math.isfinite(v) and 0 <= v <= 1 for v in (low, high))
            rows.append(dict(roughness=r, cosine=c, f0=f, valid=valid,
                             lower=low if math.isfinite(low) else None,
                             reference=high if math.isfinite(high) else None,
                             error=abs(low-high) if valid else None,
                             limit=(1e-12 if high == 0 else .0001*high) if valid else None))
        save(rows)
    return rows


def guard_passed(rows):
    return len(rows) == 80 and all(row['valid'] and row['error'] <= row['limit'] for row in rows)


class PublicComparison:
    """Test-only bridge: public queries, separate scalar quadrature oracle."""
    def __init__(self, public, oracle, deadline):
        self.public, self.oracle, self.deadline = public, oracle, deadline

    def evaluate(self, c, r, f):
        self.deadline()
        return dict(loss=self.public.directional_loss(c, r, f),
                    mean_loss=self.public.mean_loss(r, f))

    def integrate_for_test(self, r, f, *, order):
        self.deadline()
        return self.oracle.integrate_for_test(r, f, order=order)


def envelope(data, name):
    payload_hash = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dict(format_version=1, model='ggx-correlated-smith-schlick-reciprocal-matte-v1',
                representation='bounded-cubic-loss-v1', dtype='float64',
                domain=dict(roughness=[data['roughness_nodes'][0], data['roughness_nodes'][-1]], cosine=[0, 1], f0=[0, 1]),
                generator=dict(source_sha256=FROZEN['material_loss_resource.py'], policy='conditional-allocation-v1'),
                evidence_id='public-budget-v0.1-prospective-'+name, payload=data, payload_sha256=payload_hash)


def verify_predecessor(report, mode):
    previous = {'directional': 'guard', 'mean': 'directional', 'furnace': 'mean'}[mode]
    supervisor = report.get('supervisor', {})
    if (report.get('mode') != previous or report.get('passed') is not True
            or report.get('status') != previous+' passed; not public admission'
            or report.get('fixture_hashes') != cost.HASHES or report.get('references') != FROZEN
            or report.get('wheel', {}).get('sha256') != WHEEL_SHA
            or not report.get('worker_pid') or supervisor.get('monitored_pid') != report.get('worker_pid')
            or supervisor.get('exit_code') != 0 or supervisor.get('failure') is not None
            or not 0 < supervisor.get('sampled_peak_working_set_bytes', 0) <= cost.PROFILE['sampled_working_set_limit_bytes']
            or not 0 < supervisor.get('elapsed_seconds', 0) <= cost.PROFILE['watchdog_seconds']):
        raise ValueError('prerequisite stage missing, failed, mismatched or unsupervised')


def run_physical(fixtures, output, report, mode, deadline):
    import pytest
    os.environ['GFFX_PUBLIC_BUDGET_FIXTURES'] = str(fixtures.resolve())
    os.environ['GFFX_PUBLIC_BUDGET_DEADLINE'] = str(deadline)
    rows_path = output.with_suffix('.rows.json')
    os.environ['GFFX_PUBLIC_BUDGET_ROWS'] = str(rows_path.resolve())
    result = pytest.main(['-q', '-p', 'tools.testing_dashboard.pytest_plugin',
                         'tests/python/test_public_budget_physical.py', '-k', mode,
                         '--junitxml='+str(output.with_suffix('.xml'))])
    report['pytest_exit_code'] = int(result)
    if rows_path.exists():
        report['rows'] = json.loads(rows_path.read_text(encoding='utf-8'))
    expected = dict(directional=80, mean=16, furnace=192)[mode]
    if result or len(report['rows']) != expected:
        raise ValueError('physical stage failed or incomplete')
    report['passed'] = True


def worker(fixtures, output, wheel, mode='guard', prior=None):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    deadline = time.perf_counter()+cost.PROFILE['cooperative_seconds']
    report = dict(status='working', worker_pid=os.getpid(), mode=mode,
                  profile=dict(cost.PROFILE, profile_id='public-budget-v0.1', expected_rows=dict(guard=80,directional=80,mean=16,furnace=192)[mode]),
                  environment=dict(python=sys.version, torch=torch.__version__), rows=[])
    cost.write_report(output, report)
    try:
        report['references'] = verify_references(Path(__file__).parent)
        report['wheel'] = verify_wheel(wheel)
        if mode != 'guard':
            if prior is None:
                raise ValueError('prerequisite report required')
            prior_blob = prior.read_bytes()
            verify_predecessor(json.loads(prior_blob), mode)
            report['prerequisite_sha256'] = hashlib.sha256(prior_blob).hexdigest()
        for name, digest in cost.HASHES.items():
            cost.check_deadline(deadline)
            cost.load_json(fixtures/(name+'.json'), digest)
        report['fixture_hashes'] = cost.HASHES
        report['payload_hashes'] = {name: envelope(cost.load_json(fixtures/(name+'.json'), digest), name)['payload_sha256'] for name, digest in cost.HASHES.items()}
        if mode != 'guard':
            run_physical(fixtures, output, report, mode, deadline)
            cost.check_deadline(deadline)
            report['status'] = mode+' passed; not public admission'
            cost.write_report(output, report)
            return
        from tools.material_loss_resource import generation_pair
        def pair(c, r, **kwargs):
            return generation_pair(c, torch.tensor(r, dtype=torch.float64), **kwargs)
        def save(rows):
            report['rows'] = rows
            cost.write_report(output, report)
        with torch.no_grad():
            rows = run_guard(pair, lambda: cost.check_deadline(deadline), save)
        report['passed'] = guard_passed(rows)
        if not report['passed']:
            raise ValueError('prospective reference convergence guard failed; later stages prohibited')
        report['status'] = 'guard passed; not public admission'
    except Exception as error:
        report.update(status='stopped', error=f'{type(error).__name__}: {error}')
        cost.write_report(output, report)
        raise
    cost.write_report(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--mode', choices=('guard','directional','mean','furnace'), default='guard')
    parser.add_argument('--prior', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.fixtures, args.output, args.wheel, args.mode, args.prior)
        return 0
    return cost.supervise(args.fixtures, args.output, worker_module='tools.public_budget_probe',
                          worker_args=('--wheel', str(args.wheel), '--mode', args.mode,
                                       *(('--prior', str(args.prior)) if args.prior else ())))


if __name__ == '__main__':
    raise SystemExit(main())
