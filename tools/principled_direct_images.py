"""Bounded declared-model direct images; not an environment or training renderer."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from tools import shading_independent_evaluator as scalar
from tools.material_loss_cost import check_deadline, supervise, write_report
from tools.shading_image_bundle import load_bundle, tensor_bytes, tensor_from_bytes
from tools.shading_image_probe import producer_identity
from tools.shading_image_reference import compare_images

RESOURCE = 'ggx-smith-loss-cpu-f64-v1'
MANIFEST_HASH = 'a6d19536f1708f720e85bf514a6e9bcafa33b50fa1f8a7a3000fe42ce4cfa45e'
MANIFEST = 'profiles/synthetic_face_m1/classic-images-v1/'+MANIFEST_HASH+'/manifest.json'
SCALAR_HASH = '8e1dd0a473034264e79617930b433e1d0ef33aeab98da32154215271034d641c'
WHEEL_HASH = '6ab5ccb2335281b8c05e6d1466c791cc94dfb56fbd425b77af11546516c56285'
PROFILE = dict(name='principled-direct-images-v1', width=128, height=128,
    dtype='float64', device='cpu', roughness=1., f0=[.04]*3, direction=[1., 1., 2.],
    irradiance=[1., 1., 1.], albedos=['constant', 'vertex', 'gradient', 'checker'],
    points_per_tile=8, max_tiles=2048, gradients=False, threads=1,
    channel_atol=1e-10, channel_rtol=1e-8, cooperative_seconds=120,
    watchdog_seconds=180, sampled_working_set_limit_bytes=1024**3)


class PreparedCurve:
    """Explicit fixed-roughness reference setup, never a public query cache.

    Original independent evaluator owns Hermite polynomials and adaptive means.
    Preparing once avoids repeating the entire angular construction per pixel.
    """
    def __init__(self, curve, *, roughness):
        self.r = curve.r
        self.roughness = scalar._range(roughness, self.r[0], self.r[-1])
        self.c = curve.c
        self.rows, self.logs, self.slopes = curve._angular(roughness)
        self.means = (curve.mean(roughness, 0.), curve.mean(roughness, 1.))

    def _check(self, roughness, f0):
        if roughness != self.roughness:
            raise ValueError('prepared reference roughness mismatch')
        return scalar._range(f0)

    def loss(self, cosine, roughness, f0):
        f = self._check(roughness, f0); c = scalar._range(cosine)
        if c < self.c[1] or len(self.logs) == 1:
            u = c/self.c[1]
            pair = [(1-u)*self.rows[0][k]+u*self.rows[1][k] for k in range(2)]
        else:
            pair = scalar._hermite(self.logs, self.rows[1:], self.slopes, math.log(c))
        return (1-f)*pair[0]+f*pair[1]

    def mean(self, roughness, f0):
        f = self._check(roughness, f0)
        return (1-f)*self.means[0]+f*self.means[1]


def validate_buffers(buffers, colour):
    if colour not in PROFILE['albedos']:
        raise ValueError('unknown profile albedo')
    mask = buffers['covered']
    if (not isinstance(mask, torch.Tensor) or mask.shape != (128, 128)
            or mask.dtype != torch.bool or mask.device.type != 'cpu' or not bool(mask.any())):
        raise ValueError('nonempty CPU image mask required')
    for name in ('normal', 'view', 'albedo/'+colour):
        field = buffers[name]
        if (not isinstance(field, torch.Tensor) or field.shape != (128, 128, 3)
                or field.device.type != 'cpu' or field.dtype != torch.float64):
            raise ValueError('invalid CPU float64 image field: '+name)
        if field.requires_grad:
            raise ValueError('forward-only evidence refuses training graphs')
        if not bool(torch.isfinite(field).all()):
            raise ValueError('nonfinite image field')
        if name != 'albedo/'+colour and bool((field[mask].norm(dim=-1) == 0).any()):
            raise ValueError('zero image direction')
    albedo = buffers['albedo/'+colour]
    if bool(((albedo < 0) | (albedo > 1)).any()):
        raise ValueError('image albedo outside domain')
    return mask


def evaluate(buffers, colour, budget, *, deadline):
    from gffx.torch import material, light
    check_deadline(deadline)
    mask = validate_buffers(buffers, colour)
    if budget.metadata['resource_id'] != RESOURCE or budget.metadata['admission'] != 'accepted':
        raise ValueError('exact accepted image budget required')
    curve = PreparedCurve(scalar.LossCurve(budget.to_envelope()['payload']), roughness=1.)
    check_deadline(deadline)
    n, v, a = (buffers[name][mask] for name in ('normal', 'view', 'albedo/'+colour))
    direction = torch.tensor([[1., 1., 2.]], dtype=torch.float64)
    reference = torch.zeros((128, 128, 3), dtype=torch.float64)
    values = []
    unit = scalar._normal([1., 1., 2.])
    for index, (nn, vv, aa) in enumerate(zip(n.tolist(), v.tolist(), a.tolist())):
        if index % 8 == 0: check_deadline(deadline)
        brdf = scalar.principled(nn, vv, unit, albedo=aa, roughness=1., f0=.04, budget=curve)
        cosine = max(0., scalar._dot(scalar._normal(nn), unit))
        values.append([x*cosine for x in brdf])
    reference[mask] = torch.tensor(values, dtype=torch.float64)
    actual = torch.zeros_like(reference)
    pixels = torch.zeros_like(n)
    # Deliberate application-side forward tiling, not a hidden runtime batcher.
    with torch.no_grad():
        for start in range(0, len(n), 8):
            check_deadline(deadline)
            end = start+8; count = len(n[start:end])
            brdf = material.principled(n[start:end], v[start:end], direction,
                albedo=a[start:end], roughness=torch.tensor(1., dtype=torch.float64),
                f0=torch.tensor(.04, dtype=torch.float64), budget=budget)
            packet = dict(measure='delta', direction=direction.expand(count, 1, 3),
                irradiance=torch.ones((count, 1, 3), dtype=torch.float64),
                valid=torch.ones((count, 1), dtype=torch.bool))
            pixels[start:end] = light.integrate(n[start:end], brdf, packet)
        actual[mask] = pixels
    check_deadline(deadline)
    return dict(actual=actual, reference=reference, metrics=compare_images(actual, reference, mask),
                tiles=(len(n)+7)//8, covered_pixels=len(n))


def worker(root, output, wheel, colour):
    from gffx.torch.material import PrincipledBudget
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    deadline = time.perf_counter()+120
    report = dict(status='working', worker_pid=os.getpid(), profile=PROFILE, albedo=colour,
                  reference_manifest=dict(path=MANIFEST, sha256=MANIFEST_HASH))
    try:
        report['producer'] = producer_identity(wheel)
        if report['producer']['wheel']['sha256'] != WHEEL_HASH:
            raise ValueError('unqualified candidate wheel identity')
        digest = hashlib.sha256(Path(scalar.__file__).read_bytes()).hexdigest()
        if digest != SCALAR_HASH: raise ValueError('frozen scalar evaluator identity changed')
        report['sources'] = {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             Path(scalar.__file__).name: digest}
        bundle = load_bundle(root, path=MANIFEST, sha256=MANIFEST_HASH)
        budget = PrincipledBudget.bundled(RESOURCE, device='cpu', dtype=torch.float64)
        report['budget'] = budget.metadata
        result = evaluate(bundle['buffers'], colour, budget, deadline=deadline)
        report.update(measurements=result['metrics'], tiles=result['tiles'], arrays={})
        for name in ('actual', 'reference'):
            blob = tensor_bytes(result[name]); path = output.with_suffix('.'+name+'.bin')
            with path.open('xb') as stream: stream.write(blob)
            report['arrays'][name] = dict(path=path.name, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
                                          shape=[128, 128, 3], dtype='float64')
        if not result['metrics']['passed']: raise ValueError('principled direct image tolerance failed')
        check_deadline(deadline)
        report['status'] = 'passed declared-model direct image; environment and M1 not qualified'
    except Exception as error:
        report.update(status='stopped', error=f'{type(error).__name__}: {error}')
        write_report(output, report)
        raise
    write_report(output, report)


def launch(root, output, wheel, colour):
    for path in (output, output.with_suffix('.actual.bin'), output.with_suffix('.reference.bin')):
        if path.exists(): raise FileExistsError('refusing to replace image evidence: '+str(path))
    return supervise(root, output, worker_module='tools.principled_direct_images',
                     worker_args=('--wheel', str(wheel), '--albedo', colour))


def read_measured(path, expected_hash, mask):
    """Recheck bounded saved evidence; no shading, generator or unpickling."""
    from tools.corpus_registry import _read, _safe
    path = Path(path)
    raw = _read(path, 65536)
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('report identity mismatch')
    record = json.loads(raw)
    if (record['profile'] != PROFILE or record['reference_manifest']['sha256'] != MANIFEST_HASH
            or record['reference_manifest']['path'] != MANIFEST
            or record['producer']['wheel']['sha256'] != WHEEL_HASH
            or record['sources']['shading_independent_evaluator.py'] != SCALAR_HASH
            or record['budget']['resource_id'] != RESOURCE or record['budget']['admission'] != 'accepted'
            or record['albedo'] not in PROFILE['albedos']
            or record['status'] != 'passed declared-model direct image; environment and M1 not qualified'):
        raise ValueError('image profile/identity/status mismatch')
    supervision = record['supervisor']
    if (supervision['exit_code'] != 0 or supervision['failure'] is not None
            or supervision['monitored_pid'] != record['worker_pid']
            or not 0 < supervision['sampled_peak_working_set_bytes'] <= 1024**3
            or not 0 < supervision['elapsed_seconds'] <= 180):
        raise ValueError('image supervision failed')
    result = dict(record=record, report_sha256=expected_hash)
    for name in ('actual', 'reference'):
        descriptor = record['arrays'][name]
        if (descriptor['path'] != path.with_suffix('.'+name+'.bin').name
                or descriptor['bytes'] != 393216 or descriptor['shape'] != [128, 128, 3]
                or descriptor['dtype'] != 'float64'):
            raise ValueError('image array descriptor mismatch')
        blob = _read(_safe(path.parent.resolve(), descriptor['path']), 393216)
        if hashlib.sha256(blob).hexdigest() != descriptor['sha256']:
            raise ValueError('image array hash mismatch')
        result[name] = tensor_from_bytes(blob, descriptor['shape'], descriptor['dtype'])
    result['metrics'] = compare_images(result['actual'], result['reference'], mask)
    if result['metrics'] != record['measurements'] or not result['metrics']['passed']:
        raise ValueError('saved image metrics do not reproduce')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--albedo', choices=PROFILE['albedos'], required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker: worker(args.fixtures, args.output, args.wheel, args.albedo); return 0
    return launch(args.fixtures, args.output, args.wheel, args.albedo)


if __name__ == '__main__': raise SystemExit(main())
