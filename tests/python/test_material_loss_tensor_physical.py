"""Unchanged physical screens, now evaluated by the tensor resource.

These are regressions of existing cohorts, not fresh holdouts. The existing
independent references, thresholds and saved-row checks are reused verbatim.
"""
import importlib.util
import os
from pathlib import Path
import time
import pytest
torch=pytest.importorskip('torch')

spec=importlib.util.spec_from_file_location('tensor_physical_controls',Path(__file__).with_name('test_material_loss_resource.py'))
controls=importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


@pytest.fixture(autouse=True)
def bounded_case():
    deadline=float(os.environ.get('GFFX_TENSOR_DEADLINE','inf'))
    assert time.perf_counter()<deadline,'physical worker deadline before case'
    prior=torch.get_num_threads();torch.set_num_threads(1)
    try: yield
    finally: torch.set_num_threads(prior)
    assert time.perf_counter()<deadline,'physical worker deadline after case'


@pytest.fixture(scope='module')
def resources(tmp_path_factory):
    from tools.material_loss_cost import load_json,construct,HASHES
    from tools.material_loss_tensor import TensorLossResource
    location=os.environ.get('GFFX_LOSS_RESOURCE_FIXTURE_DIR')
    if not location: pytest.skip('explicit hash-verified saved physical fixtures required')
    result=[]
    for name,digest in HASHES.items():
        reference=construct(load_json(Path(location)/(name+'.json'),digest))
        result.append(TensorLossResource.from_resource(reference,evidence_id='regression-'+digest,
            generator=dict(source_sha256='62df9a500bca26e9eeb79fef05275e42567de45eef301a369ba7d6fa838839fd',policy='conditional-allocation-v1')))
    return *result,tmp_path_factory.mktemp('tensor-physical')


test_tensor_directional_physical=controls.test_full_resource_held_out_directional_accuracy_and_gradients
test_tensor_mean_physical=controls.test_full_resource_mean_matches_reference_and_its_gradients
test_tensor_furnace_physical=controls.test_full_resource_independently_integrated_furnace
