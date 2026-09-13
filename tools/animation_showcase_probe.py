"""Supervised finite actual-rig showcase; no unbounded dashboard producer."""
import argparse
import os
from pathlib import Path
import time

from tools import material_loss_cost as cost, shading_cuda_closeout as artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    for name in ('fixtures', 'output', 'wheel'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--mode', choices=('tests', 'render'), default='tests')
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--address', default='http://127.0.0.1:8791')
    parser.add_argument('--run', default='animation-bloom-20260909')
    args = parser.parse_args()
    if not args.worker:
        extra = ['--wheel', str(args.wheel), '--sha256', args.sha256, '--mode', args.mode,
                 '--address', args.address, '--run', args.run]
        if args.diagnostic:
            extra.append('--diagnostic')
        return cost.supervise(args.fixtures, args.output, worker_module='tools.animation_showcase_probe', worker_args=extra)
    if args.output.exists():
        raise ValueError('preserve previous worker report')
    row = dict(status='working', worker_pid=os.getpid(), mode=args.mode)
    cost.write_report(args.output, row)
    deadline = time.perf_counter() + 120
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        row['wheel'] = artifact.identity(args.wheel, args.sha256)
        os.environ['GFFX_TEST_ARTIFACT_SHA256'] = args.sha256
        row['sources'] = {path: artifact.digest(path) for path in (
            'tools/testing_dashboard/animation_showcase.py', 'tools/animation_showcase_probe.py',
            'tools/testing_dashboard/scene_showcase.py', 'tools/testing_dashboard/cinematic.py',
            'tests/python/test_animation_showcase.py', 'tools/animation_showcase_cached.py')}
        if args.mode == 'tests':
            import pytest
            code = pytest.main(['-q', '-x', '-p', 'tools.testing_dashboard.pytest_plugin',
                'tests/python/test_animation_showcase.py', '--junitxml=' + str(args.output.with_suffix('.xml'))])
            row['pytest_exit_code'] = int(code)
            if code:
                raise ValueError('animation showcase tests failed')
        else:
            from tools.testing_dashboard import animation_showcase as demo
            from tools.testing_dashboard.cinematic import encoded
            from tools.testing_dashboard.presentation import AssetWriter
            data = demo.payload(lambda: cost.check_deadline(deadline), diagnostic=args.diagnostic)
            from tools.animation_showcase_cached import cache_payload
            row['cache'] = cache_payload(data, args.output.parent / 'arrays')
            writer = AssetWriter(args.output.parent / 'images')
            row['images'] = [writer.put(encoded(value)) for value in data['images']]
            row['resolution'] = list(data['images'][0].shape[:2])
            row['frames'] = len(data['clip'])
            row['vertices'] = len(data['clip'][0]['vertices'])
            row['faces'] = len(data['clip'][0]['faces'])
            row['maximum_native_reference_error'] = data['maximum_error']
            row['logical_clip_bytes'] = sum(frame[key].numel() * frame[key].element_size()
                for frame in data['clip'] for key in ('vertices', 'faces', 'colors', 'normals'))
        cost.check_deadline(deadline)
        row['status'] = 'passed'
    except Exception as error:
        row.update(status='stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        cost.write_report(args.output, row)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
