"""Published source must preserve raw-byte identities bound by saved evidence."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BOUND_PATHS=(
    'tests/python/test_classic_material_runtime.py',
    'tests/python/test_cuda_repetition.py',
    'tests/python/test_material_light_batches.py',
    'tests/python/test_material_public_contract.py',
    'tests/python/test_principled_bundle.py',
    'tests/python/test_principled_cuda_bundle.py',
    'tests/python/test_principled_cuda_physical.py',
    'tests/python/test_principled_cuda_qualification.py',
    'tests/python/test_principled_curve_reuse.py',
    'tests/python/test_principled_runtime.py',
    'tests/python/test_shading_boundary_contract.py',
    'tests/python/test_shading_custom_contract.py',
    'tests/python/test_shading_execution_boundary.py',
    'tests/python/test_shading_m1_gradients.py',
    'tests/python/test_shading_public_complete.py',
    'tests/python/test_shading_public_edges.py',
    'tests/python/test_shading_public_lobe_layouts.py',
    'tests/python/test_shading_public_lobes_lights.py',
    'tests/python/test_shading_public_sources.py',
    'tools/affine_environment.py',
    'tools/albedo_environment.py',
    'tools/albedo_image_assembly.py',
    'tools/albedo_image_sampling.py',
    'tools/albedo_images.py',
    'tools/albedo_sampling.py',
    'tools/albedo_tiles.py',
    'tools/constant_environment_reference.py',
    'tools/corpus_registry.py',
    'tools/cuda_budget_cost.py',
    'tools/cuda_budget_probe.py',
    'tools/cuda_bundle_probe.py',
    'tools/cuda_repetition.py',
    'tools/environment_adjoint_preflight.py',
    'tools/environment_candidate_point.py',
    'tools/environment_full_images.py',
    'tools/environment_image_adjoint.py',
    'tools/environment_image_preflight.py',
    'tools/environment_image_runner.py',
    'tools/environment_image_schedule.py',
    'tools/environment_loss.py',
    'tools/image_loss_gradients.py',
    'tools/image_loss_reference.py',
    'tools/material_loss_cost.py',
    'tools/material_loss_tensor.py',
    'tools/material_mean_validation.py',
    'tools/principled_direct_images.py',
    'tools/public_budget_probe.py',
    'tools/shading_cpu_regression.py',
    'tools/shading_cuda_closeout.py',
    'tools/shading_environment_reference.py',
    'tools/shading_image_bundle.py',
    'tools/shading_image_probe.py',
    'tools/shading_independent_evaluator.py',
    'tools/testing_dashboard/build_lessons.py',
    'tools/testing_dashboard/mathml.py',
    'tools/testing_dashboard/textbook.py',
)


def test_shading_hash_bound_files_preserve_exact_checkout_bytes():
    rules={line.strip() for line in (ROOT/'.gitattributes').read_text().splitlines()
           if line.strip() and not line.lstrip().startswith('#')}
    missing=[path for path in BOUND_PATHS if path+' -text' not in rules]
    assert not missing, 'Raw source SHA256 would be vulnerable to EOL conversion: '+repr(missing)
    assert all((ROOT/path).is_file() for path in BOUND_PATHS)
