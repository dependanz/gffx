"""Independent existing integrals under the optional compiled arithmetic context."""
import pytest
import importlib.util
pytestmark=pytest.mark.skipif(importlib.util.find_spec("flint") is None,reason="isolated reference-only python-flint experiment is not installed")
from tests.packaging.test_opaque_boundary_taylor import *
from tests.packaging.test_opaque_boundary_material import *
from tests.packaging.test_opaque_high_peano import *
from tests.packaging.test_opaque_high_material import *

@pytest.fixture(autouse=True)
def compiled_context():
    from tools.opaque_arb_jets import backend
    with backend():yield
