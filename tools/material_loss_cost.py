"""Finite costs of the frozen scalar cubic resource, never a public budget API.

Run from the repository: python -m tools.material_loss_cost --fixtures DIR --output FILE.
The supervisor bounds the child and writes a partial failure report if it stops.
No server, generator, network call, unpickling or public query implementation.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

MAX_FILE_BYTES = 8*1024*1024
PROFILE = dict(batch_sizes=[1, 8, 32], warmups=1, repetitions=3,
               batch_implementation='scalar loop with retained batch graph',
               generation='not measured; hash-verified saved fixtures',
               dtype='float64', device='cpu', threads=1,
               cooperative_seconds=120, watchdog_seconds=180,
               sampled_working_set_limit_bytes=1024**3)
HASHES = dict(coarse='404ef671cca98bb95c636e7167cc8fb796fbd220f43f89b69fc5489382cf8628',
              fine='c85381a9e78538d2367a13134c895dbc0c7c4180516e504ae8a5b0b4cfce9046')
SMOOTH_HASH = 'cfea6ae94274eebf5d5af6cc2894b9acd4943de3cee5bb0965d34067178123ae'


def load_json(path, expected_hash):
    """Read at most the declared bytes and validate shape before tensor allocation."""
    with Path(path).open('rb') as stream:
        blob = stream.read(MAX_FILE_BYTES+1)
    if len(blob) > MAX_FILE_BYTES:
        raise ValueError('resource file byte limit')
    if hashlib.sha256(blob).hexdigest() != expected_hash:
        raise ValueError('resource SHA-256 mismatch')
    value = json.loads(blob)
    if not isinstance(value, dict) or set(value) != {'roughness_nodes', 'cosine_nodes', 'losses'}:
        raise ValueError('resource keys')
    r, c, v = (value[key] for key in ('roughness_nodes', 'cosine_nodes', 'losses'))
    if (not all(isinstance(x, list) for x in (r, c, v))
            or not 2 <= len(r) <= 129 or not 2 <= len(c) <= 513 or len(v) != len(r)):
        raise ValueError('resource node limits or shape')
    if any(not isinstance(row, list) or len(row) != len(c) for row in v):
        raise ValueError('resource loss shape')
    if any(not isinstance(pair, list) or len(pair) != 2 for row in v for pair in row):
        raise ValueError('resource endpoint shape')
    numbers = r+c+[x for row in v for pair in row for x in pair]
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in numbers):
        raise ValueError('resource finite numeric data')
    return value


def construct(value):
    import torch
    from tools.material_loss_smooth import SmoothLossResource
    return SmoothLossResource(*(torch.tensor(value[key], dtype=torch.float64)
                                for key in ('roughness_nodes', 'cosine_nodes', 'losses')))


def storage(resource):
    """Account for tuple constants; do not pretend they are persistent tensors."""
    seen = set()

    def python_bytes(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        size = sys.getsizeof(value)
        if isinstance(value, dict):
            size += sum(python_bytes(k)+python_bytes(v) for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            size += sum(python_bytes(x) for x in value)
        return size

    def leaves(value):
        return sum(map(leaves, value)) if isinstance(value, (list, tuple)) else 1

    constants = {field.name: getattr(resource, field.name) for field in fields(resource)}
    nr, nc = len(resource.roughness_nodes), len(resource.cosine_nodes)
    return dict(roughness_nodes=nr, cosine_nodes=nc,
                raw_grid_float64_bytes=8*(nr+nc+2*nr*nc),
                all_constants_float64_equivalent_bytes=8*sum(map(leaves, constants.values())),
                python_object_bytes=sys.getsizeof(resource)+python_bytes(vars(resource)),
                persistent_tensor_bytes=0,
                qualification='Python object accounting, not RSS; float64 equivalent is not allocated tensor storage')


def inputs(size, *, gradients):
    import torch
    if type(size) is not int or size not in PROFILE['batch_sizes']:
        raise ValueError('batch size outside predeclared profile')
    if size == 1:
        values = ([.337], [.0813], [.29])
    else:
        u = [i/(size-1) for i in range(size)]
        values = ([.003+x*(.995-.003) for x in u],
                  [math.exp(math.log(.00473)+x*math.log(.963/.00473)) for x in u],
                  [.29+x*(.61-.29) for x in u])
    return tuple(torch.tensor(v, dtype=torch.float64, requires_grad=gradients) for v in values)


def run_batch(resource, arguments, *, backward):
    import torch
    with torch.set_grad_enabled(backward):
        rows = [resource.evaluate(c, r, f) for c, r, f in zip(*arguments)]
        objective = torch.stack([row['loss']+row['mean_loss'] for row in rows]).sum()
        if not bool(torch.isfinite(objective)):
            raise ValueError('nonfinite forward measurement')
        gradients = torch.autograd.grad(objective, arguments) if backward else None
        if gradients is not None and not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError('nonfinite gradient measurement')
        return dict(objective=objective.item(),
                    gradient_sums=[g.sum().item() for g in gradients] if gradients else None)


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError('cooperative measurement deadline reached')


def samples(call, deadline):
    times = []
    result = None
    for i in range(PROFILE['warmups']+PROFILE['repetitions']):
        check_deadline(deadline)
        start = time.perf_counter()
        result = call()
        elapsed = time.perf_counter()-start
        check_deadline(deadline)
        if i >= PROFILE['warmups']:
            times.append(elapsed)
    return result, dict(seconds=times, median_seconds=statistics.median(times))


def write_report(path, report):
    # Only the new, explicitly selected experiment output is replaced.
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    # Windows readers/scanners can briefly deny replacement even though the
    # existing report is valid. Retry the SAME bytes, not the computation.
    # All waiting remains charged to the enclosing worker/watchdog clock.
    for attempt in range(7):
        try:
            temporary.replace(path)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 6:
                raise
            time.sleep(.01 * (2 ** attempt))  # At most .63 s total waiting.


def worker(fixtures, output):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    deadline = time.perf_counter()+PROFILE['cooperative_seconds']
    smooth = Path(__file__).with_name('material_loss_smooth.py')
    actual_hash = hashlib.sha256(smooth.read_bytes()).hexdigest()
    report = dict(status='working', profile=PROFILE, worker_pid=os.getpid(),
                  environment=dict(python=sys.version, executable=sys.executable,
                                   torch=torch.__version__, platform=platform.platform(),
                                   processor=platform.processor(), host=platform.node()),
                  smooth_sha256=actual_hash,
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), resources={})
    try:
        if actual_hash != SMOOTH_HASH:
            raise ValueError('frozen smooth implementation changed')
        for name, digest in HASHES.items():
            path = fixtures/(name+'.json')
            data, loading = samples(lambda: load_json(path, digest), deadline)
            resource, construction = samples(lambda: construct(data), deadline)
            row = dict(source=str(path.resolve()), sha256=digest,
                       file_bytes=path.stat().st_size, storage=storage(resource),
                       warm_filesystem_read_hash_parse=loading,
                       validated_cubic_construction=construction, queries=[])
            report['resources'][name] = row
            write_report(output, report)
            for size in PROFILE['batch_sizes']:
                for backward in (False, True):
                    arguments = inputs(size, gradients=backward)
                    result, timing = samples(lambda: run_batch(resource, arguments, backward=backward), deadline)
                    row['queries'].append(dict(batch_size=size,
                        mode='forward_plus_backward' if backward else 'forward_no_grad',
                        **timing, **result))
                    write_report(output, report)
            del resource, data
        report['status'] = 'measured; not public admission'
    except Exception as error:
        report.update(status='stopped', error=f'{type(error).__name__}: {error}')
        write_report(output, report)
        raise
    write_report(output, report)


def working_set_reader(pid):
    """Return a sampled child working-set reader plus handle cleanup."""
    if os.name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)]+[
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                    'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not handle:
            raise OSError(ctypes.get_last_error(), 'cannot monitor child working set')
        def read():
            info = Counters()
            info.cb = ctypes.sizeof(info)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(info), info.cb):
                raise OSError(ctypes.get_last_error(), 'cannot sample child working set')
            return int(info.WorkingSetSize)
        return read, lambda: kernel.CloseHandle(handle)
    if sys.platform.startswith('linux'):
        return lambda: int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'), lambda: None
    raise RuntimeError('working-set supervisor is supported only on Windows/Linux')


def worker_command(fixtures, output, *, worker_module='tools.material_loss_cost', worker_args=()):
    # Windows venv python.exe may be a redirector spawning another process.
    # Launch the real interpreter directly, retaining this environment's module
    # search paths (including its Torch installation) without copying packages.
    bootstrap = ('import sys; sys.path = '+repr(sys.path)+
                 '; sys.executable = '+repr(sys.executable)+
                 '; sys.prefix = '+repr(sys.prefix)+'; sys.exec_prefix = '+repr(sys.exec_prefix)+
                 '; from '+worker_module+' import main; raise SystemExit(main())')
    return [getattr(sys, '_base_executable', sys.executable), '-c', bootstrap,
            '--worker', '--fixtures', str(fixtures), '--output', str(output), *worker_args]


def supervise(fixtures, output, *, worker_module='tools.material_loss_cost', worker_args=()):
    if output.exists():
        raise ValueError('refusing to replace a previous experiment report')
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(worker_command(fixtures, output, worker_module=worker_module, worker_args=worker_args))
    start, peak, failure = time.perf_counter(), 0, None
    close = lambda: None
    try:
        read, close = working_set_reader(process.pid)
        while process.poll() is None:
            check_deadline(start+PROFILE['watchdog_seconds'])
            try:
                peak = max(peak, read())
            except OSError:
                if process.poll() is None:
                    raise
                break
            if peak > PROFILE['sampled_working_set_limit_bytes']:
                raise MemoryError('sampled process working-set limit exceeded')
            time.sleep(.05)
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        if process.poll() is None:
            process.kill()
        process.wait()
    finally:
        close()
    report = json.loads(output.read_text(encoding='utf-8')) if output.exists() else dict(profile=PROFILE, resources={})
    if not failure and not process.returncode and report.get('worker_pid') != process.pid:
        failure = 'worker identity mismatch: monitored PID was not the measured worker'
    report['supervisor'] = dict(elapsed_seconds=time.perf_counter()-start,
        monitored_pid=process.pid,
        sampled_peak_working_set_bytes=peak, sample_interval_seconds=.05,
        includes='whole worker including Python/PyTorch imports; not resource incremental memory',
        exit_code=process.returncode, failure=failure)
    if failure or process.returncode:
        report['status'] = 'stopped'
    write_report(output, report)
    print(json.dumps(dict(status=report['status'], output=str(output), supervisor=report['supervisor'])))
    return 1 if report['status'] == 'stopped' else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.fixtures, args.output)
        return 0
    return supervise(args.fixtures, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
