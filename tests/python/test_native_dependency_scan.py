"""Core implementation fragments receive the same dependency checks as C files."""
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('body,accepted', [('static int helper(void) { return 0; }', True),
                                        ('void *hidden(void) { return malloc(8); }', False)])
def test_native_fragment_dependency_scan(tmp_path, body, accepted):
    core = tmp_path / 'native' / 'core'
    core.mkdir(parents=True)
    (core / 'animation.c').write_text('#include "animation_cuda_dispatch.inc"\n', encoding='utf-8')
    (core / 'animation_cuda_dispatch.inc').write_text(body, encoding='utf-8')
    script = Path(__file__).resolve().parents[1] / 'abi' / 'dependency_inspection.cmake'
    result = subprocess.run(['cmake', '-DGFFX_SOURCE_DIR=' + str(tmp_path), '-P', str(script)],
                            capture_output=True, text=True, timeout=10)
    if accepted:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert "forbidden facility 'malloc'" in result.stderr
