"""Check strict transfer arithmetic in every CMake source-directory scope.

This configures three compiler-free targets, not a native accuracy experiment.
The real helper is extracted from the production build so a parent-only source
property cannot silently leave the two ABI-test providers on different flags.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _check_transfer_header_dependencies(cmake_source):
    command = re.search(
        r'add_custom_command\(.*?OUTPUT "\$\{gffx_cuda_ptx\}".*?\n\)',
        cmake_source, re.S,
    )
    assert command is not None, "PTX compilation command missing"
    dependencies = command.group(0).split("DEPENDS", 1)[1].split("COMMENT", 1)[0]
    pending = [ROOT / "native/core/image_transfer_scalar.h"]
    visited = set()
    while pending:
        header = pending.pop()
        if header in visited:
            continue
        visited.add(header)
        relative = header.relative_to(ROOT).as_posix()
        assert '"${CMAKE_CURRENT_SOURCE_DIR}/' + relative + '"' in dependencies, relative
        for included in re.findall(r'^\s*#include\s+"([^"]+)"', header.read_text(), re.M):
            pending.append(header.parent / included)


def test_image_ptx_tracks_transitive_transfer_headers():
    _check_transfer_header_dependencies((ROOT / "cmake/GffxCuda.cmake").read_text())


def test_image_ptx_rejects_missing_boundary_dependency():
    source = (ROOT / "cmake/GffxCuda.cmake").read_text()
    source = source.replace('"${CMAKE_CURRENT_SOURCE_DIR}/native/core/image_transfer_boundary.h"', '')
    with pytest.raises(AssertionError, match="image_transfer_boundary"):
        _check_transfer_header_dependencies(source)


@pytest.mark.parametrize("msvc", [False, True])
def test_image_strict_arithmetic_reaches_every_core_directory(tmp_path, msvc):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake required for build-contract configuration")
    source = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    helper = re.search(
        r"function\(gffx_link_core_runtime target\).*?endfunction\(\)",
        source, re.S,
    )
    assert helper is not None
    root_path = ROOT.as_posix()
    parent_only = re.search(
        r"# Shared sRGB arithmetic.*?endif\(\)", source, re.S,
    )
    parent_flags = "" if parent_only is None else parent_only.group(0).replace(
        "${CMAKE_CURRENT_SOURCE_DIR}", root_path,
    )
    expected = "/fp:strict" if msvc else "-ffp-contract=off;-fno-fast-math"
    configure = f'''cmake_minimum_required(VERSION 3.26)
project(image_source_scope NONE)
set(PROJECT_SOURCE_DIR "{root_path}")
set(MSVC {"TRUE" if msvc else "FALSE"})
set(UNIX FALSE)
{helper.group(0)}
function(check_flags target)
  get_source_file_property(actual "{root_path}/native/core/image.c"
      TARGET_DIRECTORY ${{target}} COMPILE_OPTIONS)
  if(NOT actual STREQUAL "{expected}")
    message(FATAL_ERROR "${{target}} transfer flags: '${{actual}}'")
  endif()
endfunction()
add_library(main_core STATIC IMPORTED)
gffx_link_core_runtime(main_core)
{parent_flags}
check_flags(main_core)
add_subdirectory(providers)
'''
    # Imported targets cannot accept PRIVATE runtime links; suppress only that
    # unrelated command in this compiler-free scope fixture.
    configure = "function(target_link_libraries)\nendfunction()\n" + configure
    (tmp_path / "CMakeLists.txt").write_text(configure, encoding="utf-8")
    providers = tmp_path / "providers"
    providers.mkdir()
    (providers / "CMakeLists.txt").write_text(
        "foreach(name IN ITEMS test_provider self_test)\n"
        "  add_library(${name} STATIC IMPORTED)\n"
        "  gffx_link_core_runtime(${name})\n"
        "  check_flags(${name})\n"
        "endforeach()\n", encoding="utf-8",
    )
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
