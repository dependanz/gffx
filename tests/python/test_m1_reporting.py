"""M1 test evidence receives stable, explained pages without inventing results."""
import pytest
from tools.testing_dashboard.catalog import definition, page_for
from tools.testing_dashboard.explanations import explain


@pytest.mark.parametrize("node,page", [
    ("tests/pytorch/test_animation.py::test_animation_public_symbols", "animation/contract"),
    ("tests/python/test_image.py::test_image_namespace", "image/contract"),
    ("tests/python/test_image_transfer_oracle.py::test_any", "image/contract"),
    ("tests/packaging/test_image_build_contract.py::test_any", "infrastructure/m1-evidence-guards"),
    ("tests/python/test_image_cached_publication.py::test_any", "image/contract"),
    ("tests/pytorch/test_animation_cuda.py::test_any", "animation/contract"),
    ("tests/pytorch/test_animation_empty_context.py::test_any", "animation/contract"),
    ("tests/pytorch/test_animation_recovery.py::test_any", "animation/contract"),
    ("tests/pytorch/test_animation_edges.py::test_any", "animation/contract"),
    ("tests/pytorch/test_animation_validation.py::test_any", "animation/contract"),
    ("tests/pytorch/test_animation_empty_reductions.py::test_any", "animation/contract"),
    ("tests/pytorch/test_animation_finite_chunks.py::test_any", "animation/contract"),
    ("tests/python/test_animation_showcase.py::test_any", "animation/contract"),
    ("tests/python/test_animation_cached_publication.py::test_any", "infrastructure/m1-presentation-evidence"),
    ("tests/python/test_image_recovery_demo.py::test_any", "infrastructure/m1-presentation-evidence"),
    ("tests/packaging/test_deferred_head_tooling.py::test_any", "infrastructure/m1-presentation-evidence"),
    ("tests/python/test_animation_cost.py::test_any", "animation/resources"),
    ("tests/packaging/test_deferred_preflight.py::test_any", "render/resources"),
    ("tests/packaging/test_deferred_cached.py::test_any", "render/resources"),
    ("tests/python/test_deferred.py::test_deferred_contract_analytic_quad", "render/deferred"),
    ("tests/python/test_render_backward_defaults.py::test_raster_backward_all_missing_does_not_call_native", "render/backward-device"),
])
def test_m1_page_assignment_is_explicit(node, page):
    assert page_for(node) == page
    row = definition(page)
    assert row["kind"] == "tests"
    assert row["created_on"] == "2026-09-09"
    assert row["description"]


@pytest.mark.parametrize("name", [
    "raster_backward_missing_cotangent_device",
    "raster_backward_all_missing_does_not_call_native",
])
def test_m1_raster_regressions_have_authored_explanations(name):
    guide = explain("tests/python/test_render_backward_defaults.py::test_" + name)
    assert guide is not None
    assert guide["authored"] is True
    assert guide["purpose"] and guide["method"] and guide["limits"]


@pytest.mark.parametrize("node,page", [
    ("tests/python/test_image_loss_gradients.py::test_any", "shading/image-loss-gradients"),
    ("tests/python/test_image_loss_demo.py::test_any", "shading/image-loss-gradients"),
    ("tests/pytorch/test_animation_stream.py::test_any", "animation/contract"),
    ("tests/pytorch/test_animation_gradient_selection.py::test_any", "animation/contract"),
    ("tests/python/test_animation_lesson.py::test_any", "dashboard/learning-checks"),
])
def test_m1_routing_preserves_specific_older_pages(node, page):
    assert page_for(node) == page
