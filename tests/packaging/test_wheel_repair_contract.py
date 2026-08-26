"""Release-candidate wheel repair must preserve the PyTorch ownership boundary."""

from __future__ import annotations

from pathlib import Path
import shlex

try:
    import tomllib
except ModuleNotFoundError:  # CPython 3.10 test environments.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_REPAIR_COMMANDS = {
    "windows": (
        "delvewheel repair "
        "--exclude gffx_core.dll "
        "--exclude c10.dll "
        "--exclude torch.dll "
        "--exclude torch_cpu.dll "
        "-w {dest_dir} {wheel}"
    ),
    "linux": (
        "auditwheel repair "
        "--exclude libc10.so "
        "--exclude libtorch.so "
        "--exclude libtorch_cpu.so "
        "-w {dest_dir} {wheel}"
    ),
    "macos": (
        "env DYLD_LIBRARY_PATH=\"$(python -c 'from pathlib import Path; import torch; "
        "print(Path(torch.__file__).parent / \"lib\")')\" "
        "delocate-wheel "
        "--exclude libc10.dylib "
        "--exclude libtorch.dylib "
        "--exclude libtorch_cpu.dylib "
        "--require-archs {delocate_archs} "
        "-w {dest_dir} -v {wheel}"
    ),
}


def _cibuildwheel_config() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as source:
        return tomllib.load(source)["tool"]["cibuildwheel"]


def test_each_platform_has_an_exact_runtime_externalization_policy() -> None:
    config = _cibuildwheel_config()

    for platform, expected in EXPECTED_REPAIR_COMMANDS.items():
        assert config[platform]["repair-wheel-command"] == expected


def test_repair_policy_is_narrow_and_never_vendors_pytorch() -> None:
    config = _cibuildwheel_config()

    for platform, expected in EXPECTED_REPAIR_COMMANDS.items():
        command = config[platform]["repair-wheel-command"]
        tokens = shlex.split(command)
        exclusions = {
            tokens[index + 1]
            for index, token in enumerate(tokens)
            if token == "--exclude"
        }
        assert command == expected
        assert exclusions
        assert all("*" not in library for library in exclusions)
        assert "--ignore-missing-dependencies" not in tokens
        if platform != "windows":
            assert all("gffx_core" not in library for library in exclusions)
