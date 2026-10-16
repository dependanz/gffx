"""Build and audit the Windows CPU wheel without selecting a user's framework runtime.

Build inputs belong in an explicitly selected CPU-only PyTorch environment. The
0.2.1 Python distribution installs Torch; the native core remains framework-independent. This tool never publishes.
"""
from __future__ import annotations

import argparse
import base64
import csv
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile


def audit(wheel: Path, *, cuda_provider: bool = False) -> dict:
    import pefile  # Packaging-only dependency, provided by delvewheel.

    imports = {}
    ptx = None
    provider_name = "gffx/gffx_cuda12.dll"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(n for n in names if n.endswith('.dist-info/METADATA'))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        tags = archive.read(next(n for n in names if n.endswith('.dist-info/WHEEL'))).decode()
        if 'Tag: cp310-abi3-win_amd64' not in tags:
            raise ValueError('Expected the Windows x64 CPython 3.10 stable-ABI wheel')
        expected_dependencies = ['torch<=2.14.1,>=2.10.0'] if metadata['Version']=='0.2.1' else []
        if metadata['Name'] != 'gffx' or metadata.get_all('Requires-Dist', []) != expected_dependencies:
            raise ValueError('Unexpected versioned gffx runtime dependency policy')
        required = {'gffx/_core.pyd', 'gffx/_torch.pyd', 'gffx/gffx_core.dll'}
        if cuda_provider:
            required.add(provider_name)
        if not required.issubset(names):
            raise ValueError('CPU core, CPython bridge and optional PyTorch adapter must all be present')
        for name in names:
            if '..' in Path(name).parts or name.startswith('/'):
                raise ValueError('Unsafe wheel path')
            leaf = Path(name).name.lower()
            allowed_provider = cuda_provider and name == provider_name
            if leaf.endswith('.dll') and ((leaf.startswith('gffx_cuda') and not allowed_provider) or leaf.startswith(('torch', 'c10', 'cudart', 'cublas', 'cudnn', 'nvrtc')) or leaf == 'nvcuda.dll'):
                raise ValueError(f'Framework/CUDA library must not be bundled: {name}')
            if not name.lower().endswith(('.pyd', '.dll')):
                continue
            pe = pefile.PE(data=archive.read(name), fast_load=False)
            if pe.FILE_HEADER.Machine != 0x8664:
                raise ValueError(f'Non-x64 native binary: {name}')
            dependencies = sorted({entry.dll.decode('ascii').lower() for field in ['DIRECTORY_ENTRY_IMPORT', 'DIRECTORY_ENTRY_DELAY_IMPORT'] for entry in getattr(pe, field, [])})
            if allowed_provider:
                symbols = {entry.name for entry in pe.DIRECTORY_ENTRY_EXPORT.symbols}
                if b'gffx_cuda_plugin_handshake_v1' not in symbols:
                    raise ValueError('Missing CUDA provider ABI handshake')
                payloads = []
                for kind in getattr(pe, 'DIRECTORY_ENTRY_RESOURCE', []).entries:
                    if kind.id != 10:
                        continue
                    for resource in kind.directory.entries:
                        if resource.id == 101:
                            for language in resource.directory.entries:
                                info = language.data.struct
                                payloads.append(pe.get_data(info.OffsetToData, info.Size).rstrip(b'\0'))
                if len(payloads) != 1:
                    raise ValueError('Expected one embedded PTX resource')
                payload = payloads[0]
                text = payload.decode('utf-8')
                isa = re.search(r'^\s*\.version\s+(\S+)', text, re.MULTILINE)
                target = re.search(r'^\s*\.target\s+(\S+)', text, re.MULTILINE)
                if isa is None or target is None:
                    raise ValueError('Invalid embedded PTX')
                ptx = dict(isa=isa.group(1), target=target.group(1), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            pe.close()
            imports[name] = dependencies
            for library in dependencies:
                if allowed_provider and library == 'nvcuda.dll':
                    continue
                if allowed_provider and library.startswith(('torch', 'c10')):
                    raise ValueError('The CUDA provider must not depend on a framework')
                if any(part in library for part in ['torch_cuda', 'c10_cuda', 'gffx_cuda', 'cudart', 'cublas', 'cudnn', 'nvrtc', 'nvcuda']):
                    raise ValueError(f'Wheel imports a forbidden CUDA library: {name} -> {library}')
                if name in {'gffx/_core.pyd', 'gffx/gffx_core.dll'} and library.startswith(('torch', 'c10')):
                    raise ValueError('The base native core must not depend on PyTorch')
        record = archive.read(next(n for n in names if n.endswith('.dist-info/RECORD'))).decode()
        for name, digest, size in csv.reader(io.StringIO(record)):
            if not digest:
                continue
            algorithm, expected = digest.split('=', 1)
            data = archive.read(name)
            actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
            if algorithm != 'sha256' or actual != expected or len(data) != int(size):
                raise ValueError(f'Invalid wheel RECORD entry: {name}')
    return {'wheel': str(wheel.resolve()), 'version': metadata['Version'], 'bytes': wheel.stat().st_size, 'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(), 'requires_python': metadata['Requires-Python'], 'requires_dist': metadata.get_all('Requires-Dist', []), 'native_imports': imports, 'record_hashes_verified': True, 'gffx_cuda_provider': cuda_provider, 'ptx': ptx}


def build(args) -> dict:
    source = args.source.resolve()
    probe = 'import json,sys,struct,platform,torch;print(json.dumps(dict(python=sys.version,platform=platform.system(),bits=struct.calcsize("P")*8,torch=torch.__version__,cuda=torch.version.cuda,torch_lib=str(__import__("pathlib").Path(torch.__file__).parent/"lib"))))'
    profile = json.loads(subprocess.check_output([args.python, '-I', '-c', probe], text=True, encoding='utf-8'))
    if profile['platform'] != 'Windows' or profile['bits'] != 64:
        raise ValueError('This release profile requires Windows x64')
    if profile['cuda'] is not None or profile['torch'].split('+')[0] != '2.10.0':
        raise ValueError('Use the PyTorch 2.10.0 CPU-only build environment; CUDA-enabled build inputs are refused')
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    version = re.search(r'^version\s*=\s*"([^"]+)"', (source/'pyproject.toml').read_text(encoding='utf-8'), re.MULTILINE).group(1)
    environment = os.environ.copy()
    for key in ['PYTHONPATH', 'PYTHONHOME', 'GFFX_CUDA_PLUGIN_PATH']:
        environment.pop(key, None)
    environment.update(PYTHONUTF8='1', PIP_DISABLE_PIP_VERSION_CHECK='1', CMAKE_BUILD_PARALLEL_LEVEL='1')
    if args.generator:
        environment['CMAKE_GENERATOR'] = args.generator
    commands = [
        [args.python, '-m', 'pip', 'wheel', str(source), '--no-deps', '--no-build-isolation', '--no-index', '--wheel-dir', str(work/'raw'), '-Ccmake.define.GFFX_BUILD_PYTORCH=ON', '-Ccmake.define.GFFX_ENABLE_CUDA=OFF', '-Cbuild-dir='+str(work/'cmake')],
    ]
    with (work/'build.log').open('w', encoding='utf-8') as log:
        subprocess.run(commands[0], env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    raw = list((work/'raw').glob('*.whl'))
    if len(raw) != 1:
        raise ValueError('Expected one freshly built wheel')
    repair = [args.python, '-m', 'delvewheel', 'repair', '--exclude', 'gffx_core.dll', '--exclude', 'c10.dll', '--exclude', 'torch.dll', '--exclude', 'torch_cpu.dll', '--add-path', profile['torch_lib'], '-w', str(work/'repaired'), str(raw[0])]
    commands.append(repair)
    with (work/'repair.log').open('w', encoding='utf-8') as log:
        subprocess.run(repair, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    wheels = list((work/'repaired').glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError('Expected one repaired wheel')
    audit_command = [args.python, '-I', str(Path(__file__).resolve()), 'audit', str(wheels[0]), '--output', str(work/'wheel-audit.json')]
    commands.append(audit_command)
    subprocess.run(audit_command, env=environment, check=True, timeout=120)
    result = {'version': version, 'build_profile': profile, 'commands': commands, 'audit': json.loads((work/'wheel-audit.json').read_text(encoding='utf-8'))}
    (work/'build-result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    builder = commands.add_parser('build')
    builder.add_argument('--python', required=True, help='Explicit isolated CPU-only PyTorch 2.10.0 interpreter')
    builder.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    builder.add_argument('--work-dir', type=Path, required=True, help='New output directory; never overwritten')
    builder.add_argument('--generator', help='Optional explicit CMake generator')
    checker = commands.add_parser('audit')
    checker.add_argument('wheel', type=Path)
    checker.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = build(args) if args.command == 'build' else audit(args.wheel)
    if args.command == 'audit' and args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
