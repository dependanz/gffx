"""Run only the finite Windows 0.2.0 installed-wheel release cohort. Never publishes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import zipfile

CPU = [
    'pytorch/test_face_geometry.py', 'pytorch/test_all_operations.py',
    'pytorch/test_animation.py', 'pytorch/test_animation_completion.py',
    'pytorch/test_animation_edges.py',
    'pytorch/test_animation_empty_reductions.py', 'pytorch/test_animation_finite_chunks.py',
    'pytorch/test_animation_gradient_selection.py', 'pytorch/test_animation_shared_gradients.py',
    'pytorch/test_animation_stream.py', 'pytorch/test_animation_validation.py',
    'python/test_camera.py', 'python/test_orthographic_camera.py',
    'packaging/test_stable_camera_order.py', 'python/test_image.py',
    'packaging/test_stable_operations.py',
]
GPU = ['packaging/test_gpu_alpha.py', 'pytorch/test_all_operations.py',
       'pytorch/test_face_geometry.py', 'pytorch/test_animation_cuda.py',
       'pytorch/test_animation_completion_cuda.py', 'python/test_camera.py',
       'python/test_orthographic_camera.py']
GPU_IMAGES = ['test_image_cpu_cuda_reference_bytes_and_repeat',
    'test_affine_independent_argument_vjps_cpu_cuda',
    'test_texture_nondefault_stream_preserves_caller_and_exact_vjps',
    'test_warp_odd_grid_and_batch_cpu_cuda_exact', 'test_blur_odd_batch_cpu_cuda_exact',
    'test_odd_batch_mask_and_adjust_shared_parameter_vjps',
    'test_odd_batch_multilayer_independent_leaf_vjps']
CPU_EXCLUSIONS = 'not cuda and not test_tb14 and not test_ta05_device_policy and not test_camera_mixed_devices_are_rejected'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', required=True, type=Path)
    parser.add_argument('--profile', choices=['cpu', 'gpu', 'cuda-torch-cpu'], required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    import gffx
    import torch
    import pytest
    assert platform.system() == 'Windows' and platform.machine().lower() in ['amd64', 'x86_64']
    assert gffx.__version__ == '0.2.0'
    assert Path(gffx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), gffx.__file__
    assert torch.__version__.split('+')[0] == '2.10.0', torch.__version__
    expected_cuda = None if args.profile == 'cpu' else '12.6'
    assert torch.version.cuda == expected_cuda, torch.version.cuda
    torch.set_num_threads(1)
    os.environ["GFFX_KNOT_DEVICE"] = "cuda" if args.profile == "gpu" else "cpu"
    os.environ["GFFIT_TEST_DEVICE"] = "cuda" if args.profile == "gpu" else "cpu"
    package_root = Path(gffx.__file__).resolve().parent.parent
    payload = []
    with zipfile.ZipFile(args.wheel) as archive:
        for member in archive.namelist():
            if member.startswith(('gffx/', 'gffx.libs/')) and not member.endswith('/'):
                actual = (package_root/member).read_bytes()
                assert actual == archive.read(member), member
                payload.append(dict(path=member, sha256=hashlib.sha256(actual).hexdigest()))
    sys.path[:0] = [str(root/'tests'/name) for name in ['python', 'pytorch', 'packaging']]
    info = dict(profile=args.profile, python=platform.python_version(), executable=sys.executable,
                torch=str(torch.__version__), torch_path=torch.__file__, gffx_path=gffx.__file__,
                wheel=args.wheel.name, wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                payload=payload, cpu_threads=torch.get_num_threads())
    if args.profile == 'gpu':
        assert torch.cuda.is_available(), 'Actual GPU required; no skipped-device acceptance'
        import gffx.cuda
        report = gffx.cuda.capabilities()
        assert report['status'] == 'loaded', report
        info.update(gpu=torch.cuda.get_device_name(0), compute=list(torch.cuda.get_device_capability(0)), provider=report)
        torch.cuda.reset_peak_memory_stats()
        names = GPU + ['python/test_image_cuda.py::'+name for name in GPU_IMAGES]
    else:
        names = CPU
    selected = [str(root/'tests'/name) for name in names]
    arguments = ['-q', '--import-mode=importlib', '-o', 'junit_family=xunit1',
                 '--basetemp='+str(folder/'temp'), '--junitxml='+str(folder/'tests.xml'), *selected]
    if args.profile != 'gpu':
        arguments += ['-k', CPU_EXCLUSIONS]

    class Recorder:
        def __init__(self):
            self.outcomes = []
        def pytest_runtest_logreport(self, report):
            if report.when == 'call' or report.skipped or (report.when == 'setup' and report.failed):
                self.outcomes.append(dict(nodeid=report.nodeid, when=report.when, outcome=report.outcome))
        def pytest_sessionfinish(self, session, exitstatus):
            if any(row['outcome'] == 'skipped' for row in self.outcomes):
                session.exitstatus = 1
    recorder = Recorder()
    code = int(pytest.main(arguments, plugins=[recorder]))
    info.update(selection=names, exclusions=CPU_EXCLUSIONS if args.profile != 'gpu' else None,
                outcomes=recorder.outcomes, exit_code=code,
                passed=sum(row['outcome']=='passed' for row in recorder.outcomes),
                skipped=sum(row['outcome']=='skipped' for row in recorder.outcomes),
                failed=sum(row['outcome']=='failed' for row in recorder.outcomes))
    info['selected_nodeids_sha256'] = hashlib.sha256('\n'.join(sorted({row['nodeid'] for row in recorder.outcomes})).encode()).hexdigest()
    info['completed_at'] = datetime.now(timezone.utc).isoformat()
    if args.profile == 'gpu':
        torch.cuda.synchronize()
        info.update(allocated_peak=torch.cuda.max_memory_allocated(), reserved_peak=torch.cuda.max_memory_reserved())
    (folder/'result.json').write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key:info[key] for key in ['profile','python','torch','passed','skipped','failed','exit_code','wheel_sha256']}, indent=2))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
