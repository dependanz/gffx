"""Phase 1 Step 10 contracts for truthful package and repository metadata."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # CPython 3.10 test environments use the isolated test dependency.
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_pyproject() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as source:
        return tomllib.load(source)


def test_base_distribution_metadata_is_dependency_free_and_python_310_plus():
    document = load_pyproject()
    project = document["project"]

    assert project["name"] == "gffx"
    assert project["version"] == "0.2.0"
    assert project["requires-python"] == ">=3.10"
    assert project["dependencies"] == []
    assert project["license"] == {"file": "LICENSE"}
    assert project["description"] == (
        "Portable, dependency-light differentiable graphics and mesh operations"
    )
    assert project["keywords"] == [
        "autodiff",
        "differentiable-graphics",
        "geometry",
        "meshes",
    ]


def test_metadata_declares_supported_python_identity_without_claiming_operations():
    project = load_pyproject()["project"]
    classifiers = set(project["classifiers"])

    assert "Development Status :: 4 - Beta" in classifiers
    assert "License :: OSI Approved :: MIT License" in classifiers
    assert "Programming Language :: Python :: 3 :: Only" in classifiers
    for minor in range(10, 15):
        assert f"Programming Language :: Python :: 3.{minor}" in classifiers
    assert not any("Production/Stable" in value for value in classifiers)

    urls = project["urls"]
    assert urls == {
        "Documentation": "https://github.com/dependanz/gffx#readme",
        "Issues": "https://github.com/dependanz/gffx/issues",
        "Repository": "https://github.com/dependanz/gffx",
    }


def test_development_dependency_groups_are_explicit_and_do_not_leak_to_runtime():
    document = load_pyproject()
    groups = document["dependency-groups"]

    assert set(groups) == {
        "accelerator",
        "benchmark",
        "development",
        "example",
        "format",
        "framework",
        "packaging",
        "test",
        "testing-dashboard",
        "visualization",
    }
    assert groups["test"] == [
        "pytest==9.0.3",
        "tomli==2.2.1; python_version < '3.11'",
    ]
    assert groups["packaging"] == ["build==1.3.0", "cibuildwheel==4.2.0", "delvewheel==1.13.2; sys_platform == 'win32'", "twine==7.0.0"]
    assert groups["testing-dashboard"] == ["coverage==7.10.7"]
    assert groups["development"] == [
        {"include-group": "test"},
        {"include-group": "packaging"},
    ]
    for empty_group in (
        "accelerator",
        "benchmark",
        "example",
        "format",
        "framework",
        "visualization",
    ):
        assert groups[empty_group] == []

    assert document["project"]["dependencies"] == []
    assert document["build-system"]["requires"] == ["scikit-build-core==1.0.3"]


def _flow(text: str) -> str:
    """Collapse whitespace so a phrase assertion survives a line rewrap.

    These are documentation contracts about what the docs claim, not about where the author
    happened to wrap.  Matching raw text makes reflowing a paragraph a test failure, which
    trains people to edit the test rather than read it.
    """
    return " ".join(text.split())


def test_readme_states_the_current_product_and_non_support_boundaries():
    readme = _flow((REPO_ROOT / "README.md").read_text(encoding="utf-8"))

    for required in (
        "general-purpose graphics, geometry, and mesh toolkit",
        "differentiable wherever differentiation exists",
        "The public `0.1.x` releases are the inherited prototype",
        # The boundary section is load-bearing: a README that lists only capability overclaims
        # by omission, which is the failure this assertion exists to prevent.
        "## What does not work yet",
        "`0.2.0` release line",
        "import gffx",
        "gffx.capabilities()",
        "gffx.cuda.capabilities()",
        "Python 3.10+",
        "docs/INSTALLATION.md",
        "docs/BUILDING.md",
        "docs/DEPENDENCIES.md",
        "docs/SUPPORT_STATUS.md",
    ):
        assert required in readme


def test_install_build_dependency_and_support_documents_are_truthful():
    installation = (REPO_ROOT / "docs/INSTALLATION.md").read_text(encoding="utf-8")
    building = (REPO_ROOT / "docs/BUILDING.md").read_text(encoding="utf-8")
    dependencies = (REPO_ROOT / "docs/DEPENDENCIES.md").read_text(encoding="utf-8")
    support = (REPO_ROOT / "docs/SUPPORT_STATUS.md").read_text(encoding="utf-8")
    policy = (REPO_ROOT / "docs/RELEASE_POLICY_0_2_0.md").read_text(encoding="utf-8")
    assert "Windows x64" in installation and "0.2.0a2" in installation
    assert "GFFX_BUILD_PYTORCH" in building and "GFFX_ENABLE_CUDA" in building
    assert "zero mandatory third-party Python runtime dependencies" in dependencies
    assert "No third-party source is vendored" in dependencies and "MIT" in dependencies
    assert "candidate" in support and "experimental" in support
    assert "Explicit release sign-off for 0.2.0" in policy and "No fixed calendar wait" in policy
    assert "--check" in policy and "actual published-release event" in policy


def test_documented_dependency_categories_match_declared_groups():
    groups = set(load_pyproject()["dependency-groups"])
    dependencies = (REPO_ROOT / "docs" / "DEPENDENCIES.md").read_text(encoding="utf-8")

    for group in sorted(groups):
        assert f"`{group}`" in dependencies


def test_merged_documentation_has_no_branch_or_completed_step_staleness():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    support = (REPO_ROOT / "docs/SUPPORT_STATUS.md").read_text(encoding="utf-8")
    assert "from this branch" not in readme
    assert "RELEASE_POLICY_0_2_0.md" in readme
    assert "reference campaign" in support and "paused indefinitely" in support
    assert "ALPHA_SUPPORT_HISTORY.md" in support
