"""Build/audit the Windows CPU + optional CUDA Driver-API alpha. Never publishes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_wheel import audit


def audit_gpu(wheel: Path) -> dict:
    result = audit(wheel, cuda_provider=True)
    if result["ptx"]["isa"] != "8.5" or result["ptx"]["target"] != "sm_75":
        raise ValueError("The embedded PTX differs from the experimental GPU profile")
    return result


def build(args) -> dict:
    toolkit = args.toolkit.resolve()
    nvcc = toolkit/'bin/nvcc.exe'
    compiler = subprocess.check_output([str(nvcc), '--version'], text=True, encoding='utf-8')
    if not re.search(r'V12\.6\.20\b', compiler):
        raise ValueError('This exact experimental profile requires nvcc 12.6.20')
    probe = 'import json,sys,struct,platform,torch;print(json.dumps(dict(python=sys.version,platform=platform.system(),bits=struct.calcsize("P")*8,torch=torch.__version__,cuda=torch.version.cuda,torch_lib=str(__import__("pathlib").Path(torch.__file__).parent/"lib"))))'
    profile = json.loads(subprocess.check_output([args.python, '-I', '-c', probe], text=True, encoding='utf-8'))
    if profile['platform'] != 'Windows' or profile['bits'] != 64 or profile['cuda'] is not None or profile['torch'].split('+')[0] != '2.10.0':
        raise ValueError('Use an explicit Windows x64 CPU-only PyTorch 2.10.0 builder')
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ['PYTHONPATH', 'PYTHONHOME', 'GFFX_CUDA_PLUGIN_PATH', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS']:
        env.pop(key, None)
    env.update(PYTHONUTF8='1', PIP_DISABLE_PIP_VERSION_CHECK='1', CMAKE_BUILD_PARALLEL_LEVEL='1', CMAKE_GENERATOR='Visual Studio 17 2022')
    commands = [[args.python, '-m', 'pip', 'wheel', str(args.source.resolve()), '--no-deps', '--no-build-isolation', '--no-index', '--wheel-dir', str(work/'raw'),
                 '-Ccmake.define.GFFX_BUILD_PYTORCH=ON', '-Ccmake.define.GFFX_ENABLE_CUDA=ON', '-Ccmake.define.GFFX_CUDA_ARCH_FLOOR=compute_75',
                 '-Ccmake.define.CUDAToolkit_ROOT='+str(toolkit), '-Cbuild-dir='+str(work/'cmake')]]
    with (work/'build.log').open('w', encoding='utf-8') as log:
        subprocess.run(commands[0], env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    raw = list((work/'raw').glob('*.whl'))
    if len(raw) != 1:
        raise ValueError('Expected one fresh wheel')
    repair = [args.python, '-m', 'delvewheel', 'repair']
    for library in ['gffx_core.dll', 'gffx_cuda12.dll', 'nvcuda.dll', 'c10.dll', 'torch.dll', 'torch_cpu.dll']:
        repair += ['--exclude', library]
    repair += ['--add-path', profile['torch_lib'], '-w', str(work/'repaired'), str(raw[0])]
    commands.append(repair)
    with (work/'repair.log').open('w', encoding='utf-8') as log:
        subprocess.run(repair, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    wheels = list((work/'repaired').glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError('Expected one repaired wheel')
    checked = audit_gpu(wheels[0])
    ptx = checked['ptx']
    if ptx['isa'] != '8.5' or ptx['target'] != 'sm_75':
        raise ValueError('The emitted PTX differs from the experimental profile')
    result = dict(build_profile=profile, toolkit=str(toolkit), nvcc=compiler, commands=commands, audit=checked)
    (work/'wheel-audit.json').write_text(json.dumps(checked, indent=2)+'\n', encoding='utf-8')
    (work/'build-result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    builder = commands.add_parser('build')
    builder.add_argument('--python', required=True)
    builder.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    builder.add_argument('--work-dir', type=Path, required=True)
    builder.add_argument('--toolkit', type=Path, required=True)
    checker = commands.add_parser('audit')
    checker.add_argument('wheel', type=Path)
    checker.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = build(args) if args.command == 'build' else audit_gpu(args.wheel)
    if args.command == 'audit' and args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
