"""Bounded instrumented startup frames; never an acceptance timing anchor.

The original installed-wheel stream worker enforces its existing memory/time caps.
Profiling changes execution overhead, so these results are diagnostic only.
"""
import argparse
import cProfile
import functools
import json
from pathlib import Path
import pstats
import sys
import time


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    if args.width != args.height or args.width not in (32, 64):
        raise ValueError('diagnostic is limited to startup frames, never a time-admission bypass')
    from tools import deferred_stream_probe as driver
    from tools import deferred_stream as stream
    from gffx.torch import _deferred as deferred
    original_supervise = driver.cost.supervise
    original_write = driver.cost.write_report
    original_stream = stream.stream_into
    stages = {}
    current_phase = ['setup']
    patched = []

    def wrap(owner, name, label, phase=None):
        original = getattr(owner, name)
        @functools.wraps(original)
        def measured(*a, **kw):
            previous = current_phase[0]
            if phase is not None:
                current_phase[0] = phase
            key = current_phase[0] + '/' + label
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                row = stages.setdefault(key, dict(calls=0, seconds=0.0))
                row['calls'] += 1
                row['seconds'] += time.perf_counter() - start
                current_phase[0] = previous
        setattr(owner, name, measured)
        patched.append((owner, name, original))

    def profiled(*a, **kw):
        profiler = cProfile.Profile()
        start = time.perf_counter()
        profiler.enable()
        try:
            return original_stream(*a, **kw)
        finally:
            profiler.disable()
            elapsed = time.perf_counter() - start
            stats = pstats.Stats(profiler)
            rows = []
            for (file, line, name), (primitive, calls, self_s, cumulative_s, callers) in stats.stats.items():
                rows.append(dict(file=file, line=line, function=name, primitive_calls=primitive,
                                 calls=calls, self_seconds=self_s, cumulative_seconds=cumulative_s))
            rows.sort(key=lambda r: r['cumulative_seconds'], reverse=True)
            report = dict(diagnostic_only=True, elapsed_seconds=elapsed, stages=stages,
                          functions=rows, scope='Instrumented host-wall times include profiler overhead. Nested times overlap; CUDA execution is asynchronous, so these are not independent kernel timings or acceptance anchors.')
            with args.output.with_suffix('.stages.json').open('x', encoding='utf-8') as handle:
                json.dump(report, handle, indent=2)

    def supervise(fixtures, output, **kw):
        kw['worker_module'] = 'tools.deferred_stage_diagnostic'
        return original_supervise(fixtures, output, **kw)

    def write(path, row):
        if '--worker' in sys.argv:
            row['diagnostic_only'] = True
            row['profile']['instrumentation'] = 'stage48_cprofile_host_wall'
            row['diagnostic_driver_sha256'] = driver.artifact.digest(__file__)
            row['scope'] = 'Instrumented startup diagnostic only; never an acceptance anchor or scale-matrix credit.'
        return original_write(path, row)

    try:
        if '--worker' in sys.argv:
            wrap(deferred, '_prepare', 'prepare')
            wrap(deferred, '_buffers', 'buffers')
            wrap(deferred, '_shade', 'shade', 'forward')
            wrap(stream, '_cohort_partials', 'cohort_partials', 'backward')
            wrap(deferred.geometry, 'face_geometry', 'face_geometry')
            wrap(deferred.transforms, 'transform_points', 'transform_points')
            wrap(deferred.transforms, 'perspective_divide', 'perspective_divide')
            stream.stream_into = profiled
        driver.cost.supervise = supervise
        driver.cost.write_report = write
        return driver.main()
    finally:
        stream.stream_into = original_stream
        driver.cost.supervise = original_supervise
        driver.cost.write_report = original_write
        for owner, name, original in reversed(patched):
            setattr(owner, name, original)


if __name__ == '__main__':
    raise SystemExit(main())
