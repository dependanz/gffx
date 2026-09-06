"""Curated inspection pages. IDs are durable; selectors are explicit, first match wins.

created_on is the page's immutable creation date, not a test execution timestamp.
The initial thirteen pages were created together on 2026-09-05; exact times were not recorded.
New pages must declare their actual creation date explicitly. Renaming does not reset it.
This is a presentation catalog, not the development plan or a list of support claims.
Add a page here when adding a feature's fixtures; unassigned tests remain visible.
"""
from fnmatch import fnmatchcase

PAGES = (
    dict(created_on="2026-09-05", id="camera/projection", group="Camera", title="Projection & coordinates", kind="tests",
         description="Inspect world-to-clip matrices, pixel coordinates, depth conventions and batched camera rigs.",
         patterns=("tests/python/test_camera.py::test_cam1[01236]*", "tests/python/test_camera.py::test_cam20*")),
    dict(created_on="2026-09-05", id="camera/unprojection", group="Camera", title="Depth & unprojection", kind="tests",
         description="Check that depth returns to world positions and invalid backgrounds remain masked.",
         patterns=("tests/python/test_camera.py::test_cam1[459]*",)),
    dict(created_on="2026-09-05", id="camera/gradients", group="Camera", title="Camera gradients", kind="tests",
         description="Check finite-difference agreement and gradient flow to intrinsics and each camera pose. Expected failures are not verified support.",
         patterns=("tests/python/test_camera.py::test_cam17*",)),
    dict(created_on="2026-09-05", id="camera/validation", group="Camera", title="Invalid camera inputs", kind="tests",
         description="Check actionable rejection of invalid camera construction parameters.",
         patterns=("tests/python/test_camera.py::*",)),
    dict(created_on="2026-09-05", id="mesh/face-geometry", group="Geometry", title="Mesh face geometry", kind="tests",
         description="Inspect triangle areas, normals and their derivatives through the PyTorch adapter. A face means a mesh polygon, not a human face.",
         patterns=("tests/pytorch/test_face_geometry.py::*",)),
    dict(created_on="2026-09-05", id="geometry/operations", group="Geometry", title="Operation integration", kind="tests",
         description="Cross-operation adapter checks for mesh, transforms, proximity and rendering. Native CTest evidence is not imported by this Python reporter.",
         patterns=("tests/pytorch/test_all_operations.py::*",)),
    dict(created_on="2026-09-05", id="dashboard/panel-demo", group="Demonstrations", title="Mesh, image & history panels", kind="demo",
         description="Explore the shared viewer with a synthetic triangle, a tiny image and an illustrative history curve. This is not a graphics correctness test.", patterns=()),
    dict(created_on="2026-09-05", id="dashboard/viewer", group="Dashboard", title="Viewer & interaction checks", kind="tests",
         description="Inspect dashboard transport, displayed state and frontend interaction regressions. Physical touch and WebGPU framebuffer acceptance remain separate.",
         patterns=("tests/python/test_dashboard.py::*", "tests/python/test_dashboard_touch.py::*")),
    dict(created_on="2026-09-05", id="dashboard/retention", group="Dashboard", title="History, snapshots & references", kind="tests",
         description="Check bounded recent windows, sparse persistent snapshots, replay, resource limits and exact evidence references.",
         patterns=("tests/python/test_dashboard_*.py::*",)),
    dict(created_on="2026-09-05", id="dashboard/workspace", group="Dashboard", title="Testing workspace checks", kind="tests",
         description="Check stable pages, test-result reporting, coverage import and evidence isolation in this development application.",
         patterns=("tests/python/test_testing_dashboard.py::*",)),
    dict(created_on="2026-09-05", id="infrastructure/packaging", group="Infrastructure", title="Packaging & installation", kind="tests",
         description="Check installation, wheel contents and CI definitions. These are infrastructure checks, not geometry or platform-support acceptance.",
         patterns=("tests/packaging/*",)),
    dict(created_on="2026-09-05", id="infrastructure/loading", group="Infrastructure", title="Imports & backend loading", kind="tests",
         description="Check lazy imports, package selection, optional CUDA capability reports and PyTorch adapter activation.",
         patterns=("tests/python/test_base_import.py::*", "tests/python/test_package_under_test.py::*",
                   "tests/python/test_cuda_capabilities.py::*", "tests/pytorch/test_adapter_loading.py::*")),
    dict(created_on="2026-09-05", id="infrastructure/unassigned", group="Infrastructure", title="Unassigned checks", kind="tests",
         description="Reported tests without a curated page. Add explicit selectors to the catalog instead of hiding new evidence.", patterns=()),
)

def page_for(nodeid):
    for page in PAGES:
        if any(fnmatchcase(nodeid.replace('\\', '/'), p) for p in page['patterns']):
            return page['id']
    return 'infrastructure/unassigned'

def definition(identifier):
    for page in PAGES:
        if page['id'] == identifier:
            return {k:v for k,v in page.items() if k != 'patterns'}
    raise KeyError('Unknown testing page; no substitute selected')
